# Model Evaluation & Performance Analysis Report

This report presents a detailed comparative study of the different algorithmic approaches used to solve the Hangman game: the **HMM Bigram + Greedy baseline**, the **Deep Q-Network (DQN)** reinforcement learning agent, and the **PyTorch Transformer** oracle.

---

## 1. Algorithmic Strategies

### 1.1 Hidden Markov Model (HMM) Bigram + Greedy Baseline
The HMM approach treats words as a sequence of letters with transition probabilities. 
- **Learning**: Calculated bigram transition counts from the 50k-word `corpus.txt` with Laplace smoothing to handle zero counts.
- **Inference**: Given a pattern like `_ p p _ e` and the set of wrong guesses:
  1. Filters the corpus for words matching the length and reveal pattern.
  2. Combines letter frequency within matches with bigram transition probabilities from neighbors (previous and next characters).
  3. Outputs a posterior probability distribution `P(letter | pattern, context)`.
- **Action**: The Greedy agent picks the letter with the highest probability.

### 1.2 Deep Q-Network (DQN) Reinforcement Learning
The DQN agent learns a policy mapping Hangman states to character action values (Q-values) to optimize long-term game success.
- **State Vector Configuration**:
  1. Masked word state (max_len × 26, one-hot encoded, flattened).
  2. Guessed letters mask (26-dimensional binary vector).
  3. HMM oracle recommendations (26-dimensional probability distribution).
  4. Lives left (1-dimensional, normalized: `lives_remaining / total_lives`).
- **Action Space**: 26 discrete actions (letters `a` through `z`).
- **Q-Network (QNet)**: A Multi-Layer Perceptron (MLP) with layers: `Input` $\rightarrow$ `256 (ReLU)` $\rightarrow$ `256 (ReLU)` $\rightarrow$ `26 (Outputs)`.
- **Replay Buffer & Target Network**: Experience replay stabilizes updates, and a separate target network coordinates gradient updates.
- **Exploration Policy**: $\epsilon$-greedy exploration with decay:
  - $\epsilon_{\text{start}} = 1.0$
  - $\epsilon_{\text{end}} = 0.05$
  - $\epsilon_{\text{decay}} = 0.995$

---

## 2. Reward Shaping & Scoring

### 2.1 DQN Training Rewards
To encourage sample efficiency and correctness, the environment provides the following feedback signals:
* **Correct Guess**: `+1.0`
* **Wrong Guess**: `-2.0`
* **Game Won (Solved)**: `+10.0`
* **Game Lost (Out of Lives)**: `-10.0`
* **Step Penalty**: `-0.1` per step (encourages speed and penalizes stalling).

### 2.2 Benchmarking Score Formula
All agents are evaluated against the official competition scoring metric:
$$\text{Final Score} = (\text{Success Rate} \times 2000) - (\text{Wrong Guesses} \times 5) - (\text{Repeated Guesses} \times 2)$$

*This score penalizes wrong guesses severely, which can lead to large negative scores if the success rate is low.*

---

## 3. Benchmark Results

Both agents were evaluated over 2,000 game episodes under identical settings (6 lives, seed=42).

### 3.1 In-Distribution Evaluation (sampled from `corpus.txt`)
This benchmark tests performance on words the models were trained on (in-distribution). 

| Model | Dataset | Games | Success Rate | Wrong Guesses | Repeated Guesses | Final Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **HMM + Greedy** | `corpus.txt` sample | 2000 | **95.00%** | **3,956** | 0 | **170,220** |
| **DQN (Hybrid top-5)** | `corpus.txt` sample | 2000 | 58.45% | 9,093 | 0 | 71,435 |

*Note: The DQN Hybrid evaluation filters agent choices using the top-5 letters recommended by the HMM oracle to prevent extreme outliers.*

### 3.2 Out-of-Distribution / Held-out Evaluation (from `test_words.txt`)
This benchmark represents the **official test results** on unseen vocabulary words.

| Model | Dataset | Games | Success Rate | Wrong Guesses | Repeated Guesses | Final Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **HMM + Greedy** | `test_words.txt` | 2000 | **32.00%** | **10,477** | 0 | **11,615** |
| **DQN (Strict)** | `test_words.txt` | 2000 | 2.45% | 11,939 | 0 | -54,795 |

---

## 4. Key Performance Observations

### 4.1 Dominance of the HMM Baseline
The statistical HMM baseline completely dominated the DQN model in both settings. The HMM + Greedy approach achieves a **95% win rate** on the corpus sample. 

* **Why HMM is Strong**: Simple bigram statistics leverage spelling syntax and local letter relationships directly. By filtering candidate words on the fly, the model bounds its search space logically.
* **Why DQN Underperformed**: 
  1. The game state space in Hangman is enormous and sparse. Learning this pattern mapping using a basic MLP with only 2,000 training episodes is insufficient.
  2. The DQN struggles to capture positional details since states are flattened.
  3. $\epsilon$-greedy exploration often chooses low-probability letters early in training, slowing convergence.

### 4.2 Negative Scores on Held-Out Test Set
On the held-out `test_words.txt` dataset, scores dropped drastically:
- **HMM + Greedy** dropped to **32% success rate** with a final score of **11,615**.
- **DQN** fell to **2.45% success rate** with a negative score of **-54,795**.

**Why the drop?**
1. **Out-of-Distribution Vocabulary**: Held-out words do not match the character distributions or spelling patterns trained on.
2. **Score Penalties**: Because each wrong guess incurs a $-5$ points penalty, the accumulated 11,939 wrong guesses of the DQN completely overwhelmed the success reward, dragging the score deep into the negative region. This shows that accuracy (success rate) alone does not save an agent from poor performance if it makes a high number of incorrect guesses.

---

## 5. Training Visualizations
The model training outputs curves indicating progress across episodes:
- `plots/dqn_reward.png`: Visualizes average episode reward climbing over training epochs as the agent starts avoiding wrong letters.
- `plots/dqn_success_rate.png`: Displays success rate progression, showing how the model begins solving words once exploration ($\epsilon$) decays below 0.2.

---

## 6. Recommendations & Future Enhancements

To improve performance, future iterations should incorporate the following designs:
1. **Extended RL Training**: Increase training length to 10k–50k episodes with a larger replay buffer size (e.g., 50k experiences) to cover vocabulary permutations.
2. **Curriculum Learning**: Train the agent on shorter/simpler words first, gradually scaling word length and lexical complexity.
3. **Hybrid Action Filtering**: Restrict DQN actions to the top-k letters suggested by the HMM/Transformer oracle rather than the full 26-letter space. This prevents the agent from making highly improbable guesses.
4. **Enriched Context Models**: Replace the HMM bigram oracle with trigram transition tables or sub-word sequence models to capture longer-range dependencies.
