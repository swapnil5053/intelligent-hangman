from typing import List, Dict, Set, Tuple
import numpy as np

from .utils import ALPHABET, update_masked, is_solved


class HangmanEnv:
    def __init__(self, word: str, oracle, lives: int = 6, max_len: int = 20):
        self.word = word
        self.oracle = oracle
        self.lives_init = lives
        self.lives = lives
        self.max_len = max_len
        self.pattern = ' '.join(['_'] * len(word))
        self.guessed: Set[str] = set()
        self.done = False

    def reset(self):
        self.lives = self.lives_init
        self.pattern = ' '.join(['_'] * len(self.word))
        self.guessed = set()
        self.done = False
        return self._state()

    def _state(self) -> np.ndarray:
        masked = self.pattern.replace(' ', '')
        L = len(masked)
        # encode word letters one-hot; unknown -> all zeros
        word_enc = np.zeros((self.max_len, len(ALPHABET)), dtype=np.float32)
        for i in range(min(L, self.max_len)):
            ch = masked[i]
            if ch != '_':
                idx = ord(ch) - ord('a')
                word_enc[i, idx] = 1.0
        word_enc = word_enc.flatten()
        # guessed mask
        guessed_mask = np.zeros(len(ALPHABET), dtype=np.float32)
        for g in self.guessed:
            guessed_mask[ord(g) - ord('a')] = 1.0
        # oracle probs
        probs = self.oracle.letter_distribution(self.pattern, self.guessed)
        prob_vec = np.array([probs[a] for a in ALPHABET], dtype=np.float32)
        # lives normalized
        lives_norm = np.array([self.lives / max(1, self.lives_init)], dtype=np.float32)
        return np.concatenate([word_enc, guessed_mask, prob_vec, lives_norm])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            return self._state(), 0.0, True, {}
        guess = ALPHABET[action]
        reward = -0.1
        repeated = guess in self.guessed
        if repeated:
            reward -= 0.5
        self.guessed.add(guess)
        self.pattern, revealed = update_masked(self.word, self.pattern, guess)
        if revealed:
            reward += 1.0
        else:
            reward -= 2.0
            self.lives -= 1
        if is_solved(self.pattern):
            reward += 10.0
            self.done = True
        elif self.lives <= 0:
            reward -= 10.0
            self.done = True
        return self._state(), reward, self.done, {'revealed': revealed, 'pattern': self.pattern}
