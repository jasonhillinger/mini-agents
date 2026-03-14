import math
import re
from collections import Counter, defaultdict


class RAG:
    def __init__(self, documents: dict[str, str]):
        self.documents = documents
        self.keys = list(documents.keys())

        self.keyTokens = [self.tokenize(k) for k in self.keys]

        self.vocab = set()
        for tokens in self.keyTokens:
            self.vocab.update(tokens)

        self.idf = self.computeIdf()
        self.keyVectors = [self.computeTfidf(tokens) for tokens in self.keyTokens]

    def tokenize(self, text: str):
        text = text.lower()
        return re.findall(r"\b[a-z0-9_]+\b", text)

    def computeIdf(self):
        N = len(self.keyTokens)
        df = defaultdict(int)

        for tokens in self.keyTokens:
            for word in set(tokens):
                df[word] += 1

        idf = {}
        for word in self.vocab:
            idf[word] = math.log(N / (1 + df[word]))

        return idf

    def computeTfidf(self, tokens):
        tf = Counter(tokens)
        vec = {}
        for word in tf:
            if word in self.idf:
                vec[word] = tf[word] * self.idf[word]

        return vec

    def cosineSimilarity(self, v1, v2):
        dot = 0

        for word in v1:
            if word in v2:
                dot += v1[word] * v2[word]

        mag1 = math.sqrt(sum(v * v for v in v1.values()))
        mag2 = math.sqrt(sum(v * v for v in v2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0

        return dot / (mag1 * mag2)

    def search(self, query: str, maxResults: int = 3):
        queryTokens = self.tokenize(query)

        queryVec = {}
        tf = Counter(queryTokens)

        for word in tf:
            if word in self.idf:
                queryVec[word] = tf[word] * self.idf[word]

        scores = []

        for i, keyVec in enumerate(self.keyVectors):
            score = self.cosineSimilarity(queryVec, keyVec)

            scores.append(
                (
                    score,
                    self.keys[i],
                    self.documents[self.keys[i]],
                )
            )

        scores.sort(key=lambda x: x[0], reverse=True)

        return scores[:maxResults]
