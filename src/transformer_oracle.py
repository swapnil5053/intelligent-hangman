import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, Set, List

from src.transformer_model import HangmanTransformer, VOCAB, ALPHABET, MAX_WORD_LEN, SEQ_LEN
from src.utils import load_corpus, group_by_length, match_candidates


class TransformerOracle:
    def __init__(self, model_path: str = os.path.join('data', 'transformer_agent.pth'),
                 corpus_path: str = os.path.join('data', 'corpus.txt'), device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = HangmanTransformer().to(self.device)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded Transformer model from {model_path}")
        else:
            print(f"Warning: No weights found at {model_path}. Model is randomly initialized.")

        self.model.eval()

        # Load corpus for candidate words if needed by the evaluator
        self.words: List[str] = []
        self.by_len: Dict[int, List[str]] = {}
        if os.path.exists(corpus_path):
            self.words = load_corpus(corpus_path)
            self.by_len = group_by_length(self.words)

    def candidate_words(self, pattern: str, guessed: Set[str]) -> List[str]:
        L = len(pattern.replace(' ', ''))
        pool = self.by_len.get(L, [])
        return match_candidates(pool, pattern, guessed)

    def letter_distribution(self, pattern: str, guessed: Set[str]) -> Dict[str, float]:
        """
        Computes the probability distribution over the 26 letters of the alphabet
        for the masked positions in the pattern, with already guessed letters masked to 0.0.
        """
        masked = pattern.replace(' ', '')
        masked_positions = [i for i, c in enumerate(masked) if c == '_']

        # If there are no masked positions or the word is solved, return zero distribution
        if not masked_positions:
            return {c: 0.0 for c in ALPHABET}

        # Truncate pattern to MAX_WORD_LEN
        masked_list = list(masked)[:MAX_WORD_LEN]
        
        # 1. Construct input tokens
        input_tokens = [VOCAB[c] for c in masked_list]
        input_tokens += [VOCAB['<pad>']] * (MAX_WORD_LEN - len(masked_list))
        
        # Separator
        input_tokens.append(VOCAB['<sep>'])
        
        # Wrong guesses
        wrong_guesses = guessed - set([c for c in masked_list if c != '_'])
        wrong_tokens = [VOCAB[c] for c in sorted(wrong_guesses)]
        wrong_tokens += [VOCAB['<pad>']] * (26 - len(wrong_tokens))
        
        input_ids = input_tokens + wrong_tokens
        
        # Convert to tensor
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(input_tensor)  # Shape: [1, SEQ_LEN, VOCAB_SIZE]
            logits = logits[0]  # Shape: [SEQ_LEN, VOCAB_SIZE]
            
            # Aggregate probabilities across all masked positions
            probs_sum = np.zeros(26, dtype=np.float32)
            for pos in masked_positions:
                if pos < MAX_WORD_LEN:
                    # Get logits for the 26 alphabet characters at this position
                    pos_logits = logits[pos, :26]
                    pos_probs = torch.softmax(pos_logits, dim=-1).cpu().numpy()
                    probs_sum += pos_probs
            
            # Average probabilities
            probs_avg = probs_sum / len(masked_positions)

        # Build letter distribution
        dist = {}
        for idx, char in enumerate(ALPHABET):
            if char in guessed:
                dist[char] = 0.0
            else:
                dist[char] = float(probs_avg[idx])
                
        # Normalize the distribution over the remaining letters
        total_p = sum(dist.values())
        if total_p > 0:
            dist = {char: p / total_p for char, p in dist.items()}
        else:
            # Fallback to uniform distribution over unguessed letters if total prob is 0
            unguessed = [c for c in ALPHABET if c not in guessed]
            if unguessed:
                dist = {char: (1.0 / len(unguessed) if char not in guessed else 0.0) for char in ALPHABET}
            else:
                dist = {char: 0.0 for char in ALPHABET}
                
        return dist

    def predict(self, pattern: str, guessed: Set[str]) -> str:
        """
        Convenience method to predict the next single character to guess.
        """
        dist = self.letter_distribution(pattern, guessed)
        # Choose the unguessed character with the highest probability
        sorted_chars = sorted(dist.items(), key=lambda x: (-x[1], x[0]))
        return sorted_chars[0][0]
