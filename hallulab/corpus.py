"""PDF acquisition and page-aware chunk indexing (no LLM)."""
import json
import re
from pathlib import Path

from .io_utils import write_jsonl


def download_papers(manifest, pdf_dir, timeout=60):
    import requests
    papers = json.loads(Path(manifest).read_text(encoding='utf-8'))
    out = Path(pdf_dir)
    out.mkdir(parents=True, exist_ok=True)
    success = []
    for paper in papers:
        target = out / f"{paper['id']}.pdf"
        if target.exists() and target.stat().st_size > 1024:
            print(f"[cached] {target}")
            success.append(paper['id'])
            continue
        try:
            response = requests.get(paper['url'], timeout=timeout,
                                    headers={'User-Agent':'HallucinationMidterm/0.1 (academic research)'})
            response.raise_for_status()
            if not response.content.startswith(b'%PDF'):
                raise ValueError('URL did not return PDF (possibly blocked)')
            target.write_bytes(response.content)
            success.append(paper['id'])
            print(f"[downloaded] {target} ({len(response.content) / 1024:.0f} KiB)")
        except Exception as exc:
            print(f"[warning] {paper['id']}: {exc}")
            print(f"          Manual PDF source: {paper['paper_page']}")
    print(f"Downloaded/available: {len(success)}/{len(papers)}")
    return success


def normalize_pdf_text(s):
    s = s.replace('\u00ad', '')
    s = re.sub(r'-\s*\n\s*(?=[a-z])', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def chunks_from_pages(paper_id, pages, chunk_words=210, overlap_words=35):
    if chunk_words < 50 or not (0 <= overlap_words < chunk_words):
        raise ValueError('Require chunk_words >= 50 and 0 <= overlap_words < chunk_words')
    rows = []
    for page_num, page_text in enumerate(pages, start=1):
        words = normalize_pdf_text(page_text).split()
        if len(words) < 30:
            continue
        stride = chunk_words - overlap_words
        for j, start in enumerate(range(0, len(words), stride)):
            subset = words[start:start + chunk_words]
            if len(subset) < 30:
                continue
            rows.append({
                'id': f'{paper_id}_p{page_num:03d}_c{j:02d}',
                'paper_id': paper_id, 'page': page_num,
                'text': ' '.join(subset)
            })
    return rows


def build_corpus(manifest, pdf_dir, corpus_path, chunk_words=210, overlap_words=35):
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError('Install PyMuPDF: pip install -r requirements.txt') from exc
    papers = json.loads(Path(manifest).read_text(encoding='utf-8'))
    rows, included = [], []
    for paper in papers:
        file = Path(pdf_dir) / f"{paper['id']}.pdf"
        if not file.exists():
            print(f"[skipped] {paper['id']} PDF missing: {file}")
            continue
        try:
            with fitz.open(file) as doc:
                if len(doc) == 0:
                    raise ValueError('PDF has no pages')
                pages = [page.get_text('text') for page in doc]
            chunks = chunks_from_pages(paper['id'], pages, chunk_words, overlap_words)
            if not chunks:
                raise ValueError('No readable text (scanned PDF?)')
            for row in chunks:
                row.update({'title': paper['title'], 'source_url': paper['paper_page']})
            rows.extend(chunks)
            included.append(paper['id'])
            print(f"[indexed] {paper['id']}: {len(chunks)} chunks")
        except Exception as exc:
            print(f"[warning] {paper['id']}: {exc}")
    if not rows:
        raise RuntimeError('No PDF text indexed. Run download first or add named PDFs to data/papers/.')
    if len(included) < len(papers):
        print('[warning] Missing papers may cause false unanswerability; inspect corpus coverage.')
    write_jsonl(corpus_path, rows)
    print(f"Wrote {len(rows)} chunks to {corpus_path}")
    return rows
