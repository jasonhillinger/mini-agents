import math
import re
from collections import Counter, defaultdict


class RAG:
    STOPWORDS = {
        "the",
        "is",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "this",
        "that",
        "it",
        "as",
        "at",
        "be",
        "by",
        "are",
        "was",
        "were",
    }

    def __init__(self, documents: dict[str, str]):
        self.documents = documents
        self.keys = list(documents.keys())
        self.values = list(documents.values())

        self.tokenized = [self.tokenize(v) for v in self.values]

        self.docFreq = defaultdict(int)

        for tokens in self.tokenized:
            for t in set(tokens):
                self.docFreq[t] += 1

        self.N = len(self.tokenized)

        self.avgDocLen = sum(len(t) for t in self.tokenized) / self.N

        self.docLengths = [len(t) for t in self.tokenized]

    # ----------------------------
    # Tokenization
    # ----------------------------

    def tokenize(self, text: str):
        text = text.lower()

        words = re.findall(r"[a-z0-9_]+", text)

        words = [w for w in words if w not in self.STOPWORDS]

        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]

        return words + bigrams

    # ----------------------------
    # BM25
    # ----------------------------

    def bm25(self, queryTokens, docTokens, docIndex):
        k1 = 1.5
        b = 0.75

        score = 0

        tf = Counter(docTokens)

        docLen = self.docLengths[docIndex]

        for word in queryTokens:
            if word not in tf:
                continue

            df = self.docFreq.get(word, 0)

            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

            freq = tf[word]

            denom = freq + k1 * (1 - b + b * docLen / self.avgDocLen)

            score += idf * ((freq * (k1 + 1)) / denom)

        return score

    # ----------------------------
    # Search
    # ----------------------------

    def search(self, query: str, maxResults: int = 5):
        queryTokens = self.tokenize(query)

        scores = []

        for i, tokens in enumerate(self.tokenized):
            score = self.bm25(queryTokens, tokens, i)

            # path boosting
            key = self.keys[i].lower()
            for q in queryTokens:
                if q in key:
                    score += 0.5

            scores.append((score, self.keys[i], self.documents[self.keys[i]]))

        scores.sort(reverse=True)

        return scores[:maxResults]
