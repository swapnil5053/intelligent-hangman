from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import math
import re

from ..utils import ALPHABET, load_corpus, group_by_length, match_candidates, letter_histogram

START = '^'
END = '$'


class HMMOracle:
    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.words: List[str] = []
        self.by_len: Dict[int, List[str]] = {}
        self.bigram_counts: Dict[str, Counter] = defaultdict(Counter)
        self.bigram_probs: Dict[str, Dict[str, float]] = {}
        self.global_letter_freq: Counter = Counter()

    def fit(self, corpus_path: str):
        words = load_corpus(corpus_path)
        self.words = words
        self.by_len = group_by_length(words)
        self.global_letter_freq = letter_histogram(words)
        # bigram counts with start and end
        for w in words:
            seq = [START] + list(w) + [END]
            for a, b in zip(seq[:-1], seq[1:]):
                self.bigram_counts[a][b] += 1
        # convert to probs with Laplace smoothing
        vocab = ALPHABET + [END]
        V = len(vocab)
        for a, cnts in self.bigram_counts.items():
            total = sum(cnts.values()) + self.smoothing * V
            self.bigram_probs[a] = {}
            for b in vocab:
                self.bigram_probs[a][b] = (cnts.get(b, 0) + self.smoothing) / total
        # ensure unseen predecessor has a row
        if START not in self.bigram_probs:
            total = self.smoothing * V
            self.bigram_probs[START] = {b: self.smoothing / total for b in vocab}

    def candidate_words(self, pattern: str, guessed: Set[str]) -> List[str]:
        L = len(pattern.replace(' ', ''))
        pool = self.by_len.get(L, [])
        return match_candidates(pool, pattern, guessed)

    def letter_distribution(self, pattern: str, guessed: Set[str]) -> Dict[str, float]:
        # posterior over letters using candidate words and bigram context
        cands = self.candidate_words(pattern, guessed)
        masked = pattern.replace(' ', '')
        unknown_positions = [i for i, ch in enumerate(masked) if ch == '_']
        if not unknown_positions:
            return {a: 0.0 for a in ALPHABET}
        # frequency-based posterior across candidates
        letter_scores = Counter()
        for w in cands:
            for i in unknown_positions:
                letter_scores[w[i]] += 1
        # integrate bigram context per position
        context_scores = Counter()
        for i in unknown_positions:
            prev = masked[i-1] if i-1 >= 0 else START
            next_known = masked[i+1] if i+1 < len(masked) and masked[i+1] != '_' else None
            for a in ALPHABET:
                p = self.bigram_probs.get(prev, {}).get(a, 1e-8)
                if next_known is not None:
                    p *= self.bigram_probs.get(a, {}).get(next_known, 1e-8)
                context_scores[a] += p
        # combine
        combined = {}
        # fallback if no candidates
        if len(cands) == 0:
            total_ctx = sum(context_scores[a] for a in ALPHABET)
            for a in ALPHABET:
                combined[a] = (context_scores[a] / total_ctx) if total_ctx > 0 else 1.0 / len(ALPHABET)
        else:
            # normalize both and take geometric mean to balance
            total_freq = sum(letter_scores[a] for a in ALPHABET) or 1.0
            total_ctx = sum(context_scores[a] for a in ALPHABET) or 1.0
            for a in ALPHABET:
                pf = (letter_scores[a] / total_freq)
                pc = (context_scores[a] / total_ctx)
                combined[a] = math.sqrt(pf * pc)
        # mask guessed letters
        for g in guessed:
            combined[g] = 0.0
        # normalize to prob dist
        s = sum(combined.values()) or 1.0
        return {a: combined[a] / s for a in ALPHABET}
