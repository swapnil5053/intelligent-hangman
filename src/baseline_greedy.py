from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
from collections import Counter
import random
import os

from .deprecated.hmm_oracle import HMMOracle
from .utils import update_masked, is_solved, ALPHABET


@dataclass
class GameResult:
    word: str
    success: bool
    wrong_guesses: int
    repeated_guesses: int
    guesses: List[str]
    steps: List[str]


def play_game(word: str, oracle: HMMOracle, lives: int = 6, seed: int = 42) -> GameResult:
    random.seed(seed)
    pattern = ' '.join(['_'] * len(word))
    guessed: Set[str] = set()
    wrong = 0
    repeated = 0
    guesses = []
    steps = [f"Word: {pattern} | Lives: {lives}"]

    while lives > 0 and not is_solved(pattern):
        probs = oracle.letter_distribution(pattern, guessed)
        # Top-k expected-reveals selection to strengthen baseline
        sorted_letters = sorted(probs.items(), key=lambda x: (-x[1], x[0]))
        top_k = int(os.getenv('BASELINE_TOPK', '3'))
        # candidate words for expected reveals computation
        cands = oracle.candidate_words(pattern, guessed)
        def expected_reveals(letter: str) -> float:
            if not cands:
                return probs.get(letter, 0.0)
            masked = pattern.replace(' ', '')
            unknown_positions = [i for i, ch in enumerate(masked) if ch == '_']
            reveals = 0
            for w in cands:
                for i in unknown_positions:
                    if w[i] == letter:
                        reveals += 1
            # average reveals over candidates
            return reveals / max(1, len(cands))
        # choose best among top-k unguessed by expected reveals, tie-break by prob
        candidates = [(a, p) for a, p in sorted_letters if a not in guessed][:top_k]
        guess = None
        if candidates:
            best = None
            best_er = -1.0
            for a, p in candidates:
                er = expected_reveals(a)
                if er > best_er or (abs(er - best_er) < 1e-9 and (best is None or p > best[1])):
                    best_er = er
                    best = (a, p)
            guess = best[0] if best else None
        # fallback to any highest prob if needed
        if guess is None:
            for a, _ in sorted_letters:
                if a not in guessed:
                    guess = a
                    break
        if guess is None:
            # fallback to any remaining letter
            remaining = [a for a in ALPHABET if a not in guessed]
            if not remaining:
                break
            guess = remaining[0]
        if guess in guessed:
            repeated += 1
        guessed.add(guess)
        pattern, revealed = update_masked(word, pattern, guess)
        guesses.append(guess)
        if not revealed:
            lives -= 1
            wrong += 1
        steps.append(f"Word: {pattern} | Lives: {lives} | Guess: {guess}")

    return GameResult(
        word=word,
        success=is_solved(pattern),
        wrong_guesses=wrong,
        repeated_guesses=repeated,
        guesses=guesses,
        steps=steps,
    )


def evaluate_greedy(words: List[str], oracle: HMMOracle, lives: int = 6, seed: int = 42) -> Dict:
    rng = random.Random(seed)
    results = []
    for w in words:
        res = play_game(w, oracle, lives=lives, seed=rng.randint(0, 10**9))
        results.append(res)
    wins = sum(1 for r in results if r.success)
    total = len(results)
    wrong = sum(r.wrong_guesses for r in results)
    repeated = sum(r.repeated_guesses for r in results)
    success_rate = (wins / total * 100) if total else 0.0
    final_score = (success_rate * 2000) - (wrong * 5) - (repeated * 2)
    avg_wrong = wrong / total if total else 0.0
    avg_repeated = repeated / total if total else 0.0
    return {
        'total_games': total,
        'wins': wins,
        'success_rate': success_rate,
        'wrong_total': wrong,
        'repeated_total': repeated,
        'avg_wrong_per_game': avg_wrong,
        'avg_repeated_per_game': avg_repeated,
        'final_score': final_score,
        'results': results,
    }
