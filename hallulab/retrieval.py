"""Reproducible BM25 retrieval; optional neural fusion."""
import math
import re
from collections import Counter


def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())


class BM25Retriever:
    def __init__(self, corpus, k1=1.5, b=.75):
        if not corpus:
            raise ValueError('Empty corpus')
        self.corpus = corpus
        self.freqs = [Counter(tokenize(c['text'])) for c in corpus]
        self.lens = [sum(f.values()) for f in self.freqs]
        self.avgdl = max(sum(self.lens)/len(self.lens), 1)
        df = Counter()
        for row in self.freqs:
            df.update(row.keys())
        n = len(corpus)
        self.idf = {t: math.log(1 + (n - v + .5)/(v + .5)) for t,v in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query):
        tokens = set(tokenize(query))
        scores = []
        for counts,dl in zip(self.freqs, self.lens):
            s = 0.0
            for t in tokens:
                tf = counts.get(t, 0)
                if tf:
                    s += self.idf.get(t,0) * tf * (self.k1 + 1)/(tf + self.k1 * (1-self.b+self.b*dl/self.avgdl))
            scores.append(s)
        return scores

    def search(self, query, top_k=4):
        ranks = sorted(enumerate(self.scores(query)), key=lambda z: (-z[1], z[0]))
        return [dict(self.corpus[i], score=round(score,6)) for i,score in ranks[:top_k]]


class HybridRetriever(BM25Retriever):
    """Reciprocal rank fusion (BM25 + embeddings); first call downloads HF weights."""
    def __init__(self, corpus, model_name='BAAI/bge-small-en-v1.5'):
        super().__init__(corpus)
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError('Install optional dependencies from requirements-hybrid.txt') from exc
        self.np = np
        self.embedder = SentenceTransformer(model_name)
        self.vectors = self.embedder.encode([x['text'] for x in corpus],
                                             normalize_embeddings=True, show_progress_bar=True)

    def search(self, query, top_k=4):
        scores = self.scores(query)
        bm_order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        dense_scores = self.vectors @ q
        dense_order = sorted(range(len(scores)), key=lambda i: (-float(dense_scores[i]), i))
        final = Counter()
        for order in (bm_order, dense_order):
            for rank, idx in enumerate(order):
                final[idx] += 1.0/(60 + rank + 1)
        ranks = sorted(range(len(scores)), key=lambda i: (-final[i], i))[:top_k]
        return [dict(self.corpus[i], score=round(final[i],6)) for i in ranks]
