"""Download official benchmark files at pinned Git commits; keep byte hashes."""
import hashlib
import json
from pathlib import Path
import urllib.request
import urllib.error
import time

ROOT = Path(__file__).resolve().parents[1] / 'data' / 'benchmarks_20261111'
REPOS = {
    'ragtruth': ('ParticleMedia/RAGTruth', 'main', [
        'README.md', 'dataset/response.jsonl', 'dataset/source_info.jsonl',
        'baseline/evaluation.py']),
    'rgb': ('chen700564/RGB', 'master', [
        'readme.md', 'evalue.py', 'fact_evalue.py', 'reject_evalue.py',
        'data/en_refine.json', 'data/en_fact.json',
        'config/instruction.yaml']),
}


def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'HalluLab-course-benchmark'})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                return response.read()
        except urllib.error.HTTPError:
            raise
        except (OSError, urllib.error.URLError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / 'DOWNLOAD_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, (repo, branch, files) in REPOS.items():
        if name not in manifest:
            commit = json.loads(fetch(f'https://api.github.com/repos/{repo}/commits/{branch}'))['sha']
            manifest[name] = {'repo': repo, 'commit': commit, 'files': {}}
            manifest_path.write_text(json.dumps(manifest, indent=2))
        entry = manifest[name]
        for relative in files:
            target = ROOT / name / relative
            if relative in entry['files'] and target.exists():
                assert hashlib.sha256(target.read_bytes()).hexdigest() == entry['files'][relative]['sha256']
                continue
            url = f'https://raw.githubusercontent.com/{repo}/{entry["commit"]}/{relative}'
            try:
                data = fetch(url)
            except urllib.error.HTTPError as exc:
                if relative == 'baseline/evaluation.py' and exc.code == 404:
                    continue
                raise
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            entry['files'][relative] = {'url': url, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(name, relative, len(data), flush=True)


if __name__ == '__main__':
    main()
