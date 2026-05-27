import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

from src.utils import load_corpus, ALPHABET, set_seed
from src.transformer_model import HangmanTransformer, VOCAB, VOCAB_SIZE, MAX_WORD_LEN, SEQ_LEN


class HangmanDataset(Dataset):
    def __init__(self, words: List[str]):
        # Filter words to make sure they fit within max length
        self.words = [w for w in words if 0 < len(w) <= MAX_WORD_LEN]

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        word = self.words[idx]
        word_len = len(word)
        word_chars = set(word)

        # 1. Randomly decide how many correct letters to reveal
        # Ensure we keep at least one letter masked
        num_to_reveal = random.randint(0, max(0, len(word_chars) - 1))
        revealed_chars = set(random.sample(list(word_chars), num_to_reveal)) if num_to_reveal > 0 else set()

        # 2. Randomly decide wrong guesses (0 to 6 wrong guesses)
        wrong_pool = [c for c in ALPHABET if c not in word_chars]
        num_wrong = random.randint(0, min(6, len(wrong_pool)))
        wrong_guesses = set(random.sample(wrong_pool, num_wrong)) if num_wrong > 0 else set()

        # 3. Construct input sequence
        # Masked word representation
        masked_word = [c if c in revealed_chars else '_' for c in word]
        input_tokens = [VOCAB[c] for c in masked_word]
        # Pad word part to MAX_WORD_LEN
        input_tokens += [VOCAB['<pad>']] * (MAX_WORD_LEN - len(masked_word))

        # Separator
        input_tokens.append(VOCAB['<sep>'])

        # Wrong guesses part
        wrong_tokens = [VOCAB[c] for c in sorted(wrong_guesses)]
        # Pad wrong guesses part to 26
        wrong_tokens += [VOCAB['<pad>']] * (26 - len(wrong_tokens))

        # Combine
        input_ids = input_tokens + wrong_tokens

        # 4. Construct target sequence
        # Target has the same length as input. Masked positions target the original character,
        # all other positions have -100 (ignored by CrossEntropyLoss)
        targets = [-100] * SEQ_LEN
        for i, c in enumerate(masked_word):
            if c == '_':
                targets[i] = VOCAB[word[i]]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def train(corpus_path: str, save_path: str, epochs: int, batch_size: int, lr: float, seed: int):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load corpus
    words = load_corpus(corpus_path)
    if not words:
        raise ValueError(f"Corpus at {corpus_path} is empty or does not exist.")
    print(f"Loaded {len(words)} words from corpus.")

    dataset = HangmanDataset(words)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize model
    model = HangmanTransformer().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct_masked = 0
        total_masked = 0

        for input_ids, targets in dataloader:
            input_ids, targets = input_ids.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)

            # Flatten outputs and targets for Loss computation
            # logits: [batch, seq, vocab] -> [batch * seq, vocab]
            # targets: [batch, seq] -> [batch * seq]
            loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate character accuracy on masked tokens
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                mask = targets != -100
                correct_masked += (preds[mask] == targets[mask]).sum().item()
                total_masked += mask.sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = (correct_masked / total_masked) * 100 if total_masked > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Masked Char Accuracy: {accuracy:.2f}%")

    # Save the model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': VOCAB,
        'max_word_len': MAX_WORD_LEN,
        'seq_len': SEQ_LEN
    }, save_path)
    print(f"Model saved successfully to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, default=os.path.join('data', 'corpus.txt'))
    parser.add_argument('--save_path', type=str, default=os.path.join('data', 'transformer_agent.pth'))
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train(args.corpus, args.save_path, args.epochs, args.batch_size, args.lr, args.seed)
