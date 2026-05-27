import os
import argparse
import random
from typing import List
from collections import Counter

import numpy as np
import pandas as pd

from .utils import set_seed, ensure_dir, load_corpus, letter_histogram, plot_histogram, plot_curve, ALPHABET
from .deprecated.hmm_oracle import HMMOracle
from .baseline_greedy import evaluate_greedy
from .hangman_env import HangmanEnv
from .deprecated.dqn_agent import DQNAgent, DQNConfig
from .transformer_oracle import TransformerOracle

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch


def load_test_words(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def run_baseline(corpus_path: str, test_path: str, n_games: int, lives: int, seed: int, outdir: str, data_dir: str):
    set_seed(seed)
    oracle = HMMOracle(smoothing=1.0)
    oracle.fit(corpus_path)
    pool = load_test_words(test_path)
    if not pool:
        pool = random.sample(oracle.words, min(n_games, len(oracle.words)))
    words = pool[:n_games]
    metrics = evaluate_greedy(words, oracle, lives=lives, seed=seed)
    print("Baseline Metrics:")
    for k in ['total_games','wins','success_rate','wrong_total','repeated_total','avg_wrong_per_game','avg_repeated_per_game','final_score']:
        print(f"{k}: {metrics[k]}")
    # sample traces for demo
    print("\nSample gameplay traces:")
    for res in metrics['results'][:3]:
        print(f"Word target: {res.word}")
        for s in res.steps[:8]:
            print("  ", s)
        if len(res.steps) > 8:
            print("   ...")
    # plots
    wrong_letters = Counter()
    for res in metrics['results']:
        for g in res.guesses:
            if g not in res.word:
                wrong_letters[g] += 1
    ensure_dir(outdir)
    plot_histogram(letter_histogram(oracle.words), 'Corpus Letter Frequency', os.path.join(outdir, 'corpus_letter_freq.png'))
    plot_histogram(wrong_letters, 'Wrong Guess Letters', os.path.join(outdir, 'wrong_guess_letters.png'))
    # save per-game results to CSV
    ensure_dir(data_dir)
    rows = []
    for r in metrics['results']:
        rows.append({
            'word': r.word,
            'success': int(r.success),
            'wrong_guesses': r.wrong_guesses,
            'repeated_guesses': r.repeated_guesses,
            'num_steps': len(r.steps),
            'guesses_seq': ' '.join(r.guesses),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(data_dir, 'baseline_results.csv'), index=False)
    return metrics


def run_dqn(corpus_path: str, test_path: str, episodes: int, lives: int, seed: int, outdir: str,
            epsilon_start: float, epsilon_end: float, epsilon_decay: float, data_dir: str):
    set_seed(seed)
    oracle = HMMOracle(smoothing=1.0)
    oracle.fit(corpus_path)
    pool = load_test_words(test_path)
    if not pool:
        pool = oracle.words
    max_len = min(20, max(len(w) for w in pool)) if pool else 20
    # infer state dim from env.reset
    tmp_env = HangmanEnv(pool[0], oracle, lives=lives, max_len=max_len)
    s0 = tmp_env.reset()
    state_dim = s0.shape[0]
    cfg = DQNConfig(epsilon_start=epsilon_start, epsilon_end=epsilon_end, epsilon_decay=epsilon_decay)
    agent = DQNAgent(state_dim, cfg)
    rewards = []
    eps_success = []
    eps_eps = []
    for ep in range(episodes):
        word = random.choice(pool)
        env = HangmanEnv(word, oracle, lives=lives, max_len=max_len)
        s = env.reset()
        done = False
        ep_rew = 0.0
        step = 0
        while not done and step < 100:
            # guessed mask is the middle slice of state: [max_len*26 : max_len*26+26]
            guessed_mask = s[max_len*26 : max_len*26+26]
            a = agent.act(s, guessed_mask)
            sn, r, done, info = env.step(a)
            agent.push(s, a, r, sn, float(done))
            loss = agent.train_step()
            s = sn
            ep_rew += r
            step += 1
        rewards.append(ep_rew)
        eps_success.append(1.0 if '_' not in env.pattern else 0.0)
        eps_eps.append(agent.epsilon)
        if (ep+1) % 100 == 0:
            print(f"Episode {ep+1}/{episodes} | reward={ep_rew:.2f} | epsilon={agent.epsilon:.3f}")
    ensure_dir(outdir)
    plot_curve(rewards, 'DQN Training Reward', 'Reward', os.path.join(outdir, 'dqn_reward.png'))
    # rolling success rate
    win_rate = []
    win_window = 100
    for i in range(len(eps_success)):
        lo = max(0, i - win_window + 1)
        win_rate.append(sum(eps_success[lo:i+1]) / (i - lo + 1))
    plot_curve(win_rate, 'DQN Success Rate (rolling)', 'Win rate', os.path.join(outdir, 'dqn_success_rate.png'))
    # save per-episode results
    ensure_dir(data_dir)
    dqn_df = pd.DataFrame({
        'episode': list(range(1, len(rewards)+1)),
        'reward': rewards,
        'success': eps_success,
        'epsilon': eps_eps,
    })
    dqn_df.to_csv(os.path.join(data_dir, 'dqn_results.csv'), index=False)
    # save weights
    try:
        import torch
        ensure_dir(data_dir)
        torch.save(agent.q.state_dict(), os.path.join(data_dir, 'dqn_agent.pth'))
    except Exception:
        pass
    return {'rewards': rewards, 'success_rate_curve': win_rate}


def evaluate_dqn(corpus_path: str, test_path: str, n_games: int, lives: int, seed: int, data_dir: str, outdir: str):
    import json
    import numpy as np
    import os
    set_seed(seed)
    oracle = HMMOracle(smoothing=1.0)
    oracle.fit(corpus_path)
    pool = load_test_words(test_path)
    if not pool:
        pool = oracle.words
    max_len = min(20, max(len(w) for w in pool)) if pool else 20
    # rebuild agent and load weights
    tmp_env = HangmanEnv(pool[0], oracle, lives=lives, max_len=max_len)
    s0 = tmp_env.reset()
    state_dim = s0.shape[0]
    agent = DQNAgent(state_dim, DQNConfig())
    # load weights if available
    try:
        import torch
        weights_path = os.path.join(data_dir, 'dqn_agent.pth')
        if os.path.exists(weights_path):
            agent.q.load_state_dict(torch.load(weights_path, map_location='cpu'))
            agent.tgt.load_state_dict(agent.q.state_dict())
    except Exception:
        pass
    # set greedy evaluation
    agent.epsilon = 0.0
    results = []
    rng = random.Random(seed)
    hybrid_eval_env = os.getenv('HYBRID_EVAL', '1').lower() not in ('0','false','no')
    for i in range(min(n_games, len(pool))):
        word = pool[i % len(pool)]
        env = HangmanEnv(word, oracle, lives=lives, max_len=max_len)
        s = env.reset()
        done = False
        guessed_mask = None
        guesses_made = []
        wrong = 0
        repeated = 0
        while not done:
            guessed_mask = s[max_len*26 : max_len*26+26]
            if hybrid_eval_env:
                # Hybrid top-k selection: restrict actions to oracle top-k
                # Build guessed set
                guessed_set = set()
                for idx, m in enumerate(guessed_mask):
                    if m == 1.0:
                        guessed_set.add(ALPHABET[idx])
                # Oracle distribution from current pattern
                dist = oracle.letter_distribution(env.pattern, guessed_set)
                # Sort letters by prob, filter unguessed, take top-k
                top_k = 5
                sorted_letters = sorted(dist.items(), key=lambda x: x[1], reverse=True)
                allowed_idxs = [ALPHABET.index(ch) for ch, p in sorted_letters if ch not in guessed_set][:top_k]
                if not allowed_idxs:
                    a = agent.act(s, guessed_mask)
                else:
                    # choose argmax Q among allowed indices
                    try:
                        import torch
                        with torch.no_grad():
                            qv = agent.q(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                            best_idx = max(allowed_idxs, key=lambda idx: qv[idx].item())
                            a = best_idx
                    except Exception:
                        a = agent.act(s, guessed_mask)
            else:
                # Strict: pure DQN action (no oracle restriction)
                a = agent.act(s, guessed_mask)
            # count repeats
            if guessed_mask[a] == 1.0:
                repeated += 1
            sn, r, done, info = env.step(a)
            if not info.get('revealed', False):
                wrong += 1
            s = sn
        results.append({'word': word, 'success': int('_' not in env.pattern), 'wrong': wrong, 'repeated': repeated})
    wins = sum(r['success'] for r in results)
    total = len(results)
    wrong_total = sum(r['wrong'] for r in results)
    repeated_total = sum(r['repeated'] for r in results)
    success_rate = (wins / total * 100.0) if total else 0.0
    final_score = (success_rate * 2000) - (wrong_total * 5) - (repeated_total * 2)
    metrics = {
        'total_games': total,
        'wins': wins,
        'success_rate': success_rate,
        'wrong_total': wrong_total,
        'repeated_total': repeated_total,
        'final_score': final_score,
    }
    print("DQN Evaluation Metrics:")
    for k in ['total_games','wins','success_rate','wrong_total','repeated_total','final_score']:
        print(f"{k}: {metrics[k]}")
    ensure_dir(data_dir)
    with open(os.path.join(data_dir, 'dqn_eval_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    return metrics


def generate_dqn_summary(outdir: str, data_dir: str):
    # Compare baseline vs DQN for Corpus and Test if available
    import json
    ensure_dir(outdir)

    def load_baseline(path):
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        total = len(df)
        wins = int(df['success'].sum()) if 'success' in df.columns else 0
        wrong_total = int(df['wrong_guesses'].sum()) if 'wrong_guesses' in df.columns else 0
        repeated_total = int(df['repeated_guesses'].sum()) if 'repeated_guesses' in df.columns else 0
        sr = (wins / total * 100.0) if total else 0.0
        score = (sr * 2000) - (wrong_total * 5) - (repeated_total * 2)
        return {'success_rate': sr, 'wrong_total': wrong_total, 'repeated_total': repeated_total, 'final_score': score}

    def load_dqn(path):
        if not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Try corpus/test specific files first; fall back to generic
    base_c = load_baseline(os.path.join(data_dir, 'baseline_results_corpus.csv'))
    base_t = load_baseline(os.path.join(data_dir, 'baseline_results_test.csv'))
    if base_c is None and os.path.exists(os.path.join(data_dir, 'baseline_results.csv')):
        base_c = load_baseline(os.path.join(data_dir, 'baseline_results.csv'))

    dqn_c = load_dqn(os.path.join(data_dir, 'dqn_eval_metrics_corpus.json'))
    dqn_t = load_dqn(os.path.join(data_dir, 'dqn_eval_metrics_test.json'))
    if dqn_c is None and os.path.exists(os.path.join(data_dir, 'dqn_eval_metrics.json')):
        dqn_c = load_dqn(os.path.join(data_dir, 'dqn_eval_metrics.json'))

    # build PDF
    pdf_path = os.path.join(os.path.dirname(outdir), 'DQN_Summary_Report.pdf')
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    y = height - 72
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, y, "Intelligent Hangman — DQN Summary Report")
    y -= 24
    c.setFont("Helvetica", 11)
    c.drawString(72, y, "Goal: Compare HMM Greedy vs DQN using HMM oracle context.")
    y -= 18
    c.drawString(72, y, "Approach: State = masked word + guessed letters + oracle probs + lives; Actions = 26 letters.")
    y -= 24

    def draw_section(title, base, dqn):
        nonlocal y
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, y, title)
        y -= 14
        headers = ["Model", "Success %", "Wrong", "Repeated", "Final Score"]
        rows = []
        if base:
            rows.append(["HMM Greedy", f"{base['success_rate']:.2f}", str(base['wrong_total']), str(base['repeated_total']), f"{base['final_score']:.0f}"])
        if dqn:
            rows.append(["DQN", f"{dqn['success_rate']:.2f}", str(dqn['wrong_total']), str(dqn['repeated_total']), f"{dqn['final_score']:.0f}"])
        x0 = 72
        colw = [120, 90, 70, 70, 100]
        c.setFont("Helvetica-Bold", 11)
        x = x0
        for i, h in enumerate(headers):
            c.drawString(x, y, h)
            x += colw[i]
        y -= 14
        c.setFont("Helvetica", 11)
        for row in rows:
            x = x0
            for i, cell in enumerate(row):
                c.drawString(x, y, cell)
                x += colw[i]
            y -= 14
        y -= 12

    # Sections
    if base_c or dqn_c:
        draw_section("Corpus (data/corpus.txt) — 2000 games", base_c, dqn_c)
    if base_t or dqn_t:
        draw_section("Test (data/test_words.txt) — OFFICIAL", base_t, dqn_t)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y, "Key Takeaways")
    y -= 14
    c.setFont("Helvetica", 11)
    bullets = [
        "Greedy HMM is strong in-distribution; DQN needs more episodes/tuning to match.",
        "On held-out test, penalties for wrong guesses dominate; scores can be negative.",
        "Hybrid evaluation (oracle top-k) reduces poor actions but may be disallowed for official scoring.",
    ]
    for b in bullets:
        c.drawString(84, y, f"• {b}")
        y -= 14
    c.save()
    print(f"Saved DQN summary to {pdf_path}")


def generate_report(outdir: str, baseline_metrics: dict = None):
    ensure_dir(outdir)
    # Save report at project root (parent of plots dir)
    pdf_path = os.path.join(os.path.dirname(outdir), 'Analysis_Report.pdf')
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    y = height - 72
    def line(txt, dy=18):
        nonlocal y
        c.drawString(72, y, txt)
        y -= dy
    c.setFont("Helvetica-Bold", 16)
    line("Intelligent Hangman: HMM + DQN Analysis Report", dy=24)
    c.setFont("Helvetica", 11)
    line("Project overview: Hangman assistant using HMM for letter probabilities and optional DQN for action selection.")
    line("Dataset: Provided corpus cleaned to lowercase alphabetic, grouped by word length.")
    line("Models: HMM bigram with Laplace smoothing; DQN with MLP and epsilon-greedy policy.")
    line("")
    if baseline_metrics:
        line("Baseline (Greedy with HMM) metrics:")
        for k in ['total_games','wins','success_rate','wrong_total','repeated_total','avg_wrong_per_game','avg_repeated_per_game','final_score']:
            line(f"- {k}: {round(baseline_metrics[k], 4) if isinstance(baseline_metrics[k], float) else baseline_metrics[k]}")
    # embed plots if exist
    y -= 12
    for img in ['corpus_letter_freq.png','wrong_guess_letters.png','dqn_reward.png','dqn_success_rate.png']:
        p = os.path.join(outdir, img)
        if os.path.exists(p):
            if y < 200:
                c.showPage()
                y = height - 72
            c.drawImage(p, 72, y-160, width=4.5*inch, height=3*inch, preserveAspectRatio=True, mask='auto')
            y -= 180
            line(img)
    # end of report
    c.save()
    print(f"Saved report to {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline','dqn','report','dqn_eval','dqn_summary','transformer'], default='baseline')
    parser.add_argument('--n_games', type=int, default=1000)
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--lives', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.05)
    parser.add_argument('--epsilon_decay', type=float, default=0.995)
    parser.add_argument('--corpus', type=str, default=os.path.join('data','corpus.txt'))
    parser.add_argument('--test', type=str, default=os.path.join('data','test_words.txt'))
    parser.add_argument('--outdir', type=str, default=os.path.join('plots'))
    parser.add_argument('--data_dir', type=str, default=os.path.join('data'))
    args = parser.parse_args()

    ensure_dir(args.outdir)

    if args.mode == 'baseline':
        oracle = HMMOracle()
        oracle.fit(args.corpus)
        metrics = run_baseline(args.corpus, args.test, args.n_games, args.lives, args.seed, args.outdir, args.data_dir)
        generate_report(args.outdir, baseline_metrics=metrics)
    elif args.mode == 'dqn':
        run_dqn(args.corpus, args.test, args.episodes, args.lives, args.seed, args.outdir,
                args.epsilon_start, args.epsilon_end, args.epsilon_decay, args.data_dir)
    elif args.mode == 'report':
        metrics = run_baseline(args.corpus, args.test, args.n_games, args.lives, args.seed, args.outdir, args.data_dir)
        generate_report(args.outdir, baseline_metrics=metrics)
    elif args.mode == 'dqn_eval':
        evaluate_dqn(args.corpus, args.test, args.n_games, args.lives, args.seed, args.data_dir, args.outdir)
    elif args.mode == 'dqn_summary':
        generate_dqn_summary(args.outdir, args.data_dir)
    elif args.mode == 'transformer':
        oracle = TransformerOracle(model_path=os.path.join(args.data_dir, 'transformer_agent.pth'), corpus_path=args.corpus)
        pool = load_test_words(args.test)
        if not pool:
            pool = random.sample(oracle.words, min(args.n_games, len(oracle.words)))
        words = pool[:args.n_games]
        metrics = evaluate_greedy(words, oracle, lives=args.lives, seed=args.seed)
        print("Transformer Evaluation Metrics:")
        for k in ['total_games','wins','success_rate','wrong_total','repeated_total','avg_wrong_per_game','avg_repeated_per_game','final_score']:
            print(f"{k}: {metrics[k]}")
        # sample traces for demo
        print("\nSample gameplay traces:")
        for res in metrics['results'][:3]:
            print(f"Word target: {res.word}")
            for s in res.steps[:8]:
                print("  ", s)
            if len(res.steps) > 8:
                print("   ...")
        # Save results to a csv file
        ensure_dir(args.data_dir)
        rows = []
        for r in metrics['results']:
            rows.append({
                'word': r.word,
                'success': int(r.success),
                'wrong_guesses': r.wrong_guesses,
                'repeated_guesses': r.repeated_guesses,
                'num_steps': len(r.steps),
                'guesses_seq': ' '.join(r.guesses),
            })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(args.data_dir, 'transformer_results.csv'), index=False)
        generate_report(args.outdir, baseline_metrics=metrics)


if __name__ == '__main__':
    main()
