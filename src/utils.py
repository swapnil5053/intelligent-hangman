import os
import re
import random
import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ALPHABET = [chr(i) for i in range(ord('a'), ord('z') + 1)]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clean_word(w: str) -> str:
    w = w.strip().lower()
    w = re.sub(r'[^a-z]', '', w)
    return w


def load_corpus(path: str) -> List[str]:
    words = []
    if not os.path.exists(path):
        return words
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            w = clean_word(line)
            if w:
                words.append(w)
    return list(dict.fromkeys(words))


def group_by_length(words: List[str]) -> Dict[int, List[str]]:
    groups = defaultdict(list)
    for w in words:
        groups[len(w)].append(w)
    return groups


def pattern_to_regex(pattern: str) -> re.Pattern:
    # pattern like _ p p _ e -> "^.pp.e$" (underscore is unknown)
    s = pattern.replace(' ', '')
    s = ''.join(['.' if c == '_' else c for c in s])
    return re.compile('^' + s + '$')


def match_candidates(words: List[str], pattern: str, guessed: Set[str]) -> List[str]:
    rx = pattern_to_regex(pattern)
    cands = [w for w in words if rx.match(w)]
    # prune words containing already-guessed wrong letters
    fixed = set([c for c in pattern if c != '_' and c != ' '])
    wrong = guessed - fixed
    pruned = []
    for w in cands:
        ok = True
        for g in wrong:
            if g in w:
                ok = False
                break
        if ok:
            pruned.append(w)
    return pruned


def update_masked(word: str, pattern: str, guess: str) -> Tuple[str, bool]:
    s = pattern.replace(' ', '')
    revealed = False
    out = []
    for i, ch in enumerate(word):
        if s[i] != '_':
            out.append(s[i])
        elif ch == guess:
            out.append(guess)
            revealed = True
        else:
            out.append('_')
    return ' '.join(out), revealed


def is_solved(pattern: str) -> bool:
    return '_' not in pattern


def letter_histogram(words: List[str]) -> Counter:
    c = Counter()
    for w in words:
        c.update(list(w))
    return c


def save_json(obj, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def plot_histogram(counter: Counter, title: str, path: str):
    ensure_dir(os.path.dirname(path))
    letters = ALPHABET
    vals = [counter.get(a, 0) for a in letters]
    plt.figure(figsize=(10, 4))
    sns.barplot(x=letters, y=vals, color="#4C78A8")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_curve(values: List[float], title: str, ylabel: str, path: str):
    ensure_dir(os.path.dirname(path))
    plt.figure(figsize=(6, 4))
    plt.plot(values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Episode')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
