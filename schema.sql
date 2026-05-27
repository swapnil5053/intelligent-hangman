-- schema.sql
-- SQL script to initialize the Hangman database schema in Supabase.

-- 1. Create the games table to log every played game
CREATE TABLE IF NOT EXISTS games (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    word TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    wrong_guesses INTEGER NOT NULL,
    guesses TEXT[] NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable row-level security (optional, disabled by default for simplicity in university labs,
-- but we keep it open for public insert if direct, or we can use the service role key)
-- ALTER TABLE games ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY "Allow public insert and read" ON games FOR ALL USING (true) WITH CHECK (true);

-- 2. Create a view for the hardest words leaderboard
-- Aggregates game results to calculate times played, win count, win rate, and average wrong guesses.
CREATE OR REPLACE VIEW hardest_words AS
SELECT
    word,
    COUNT(*) AS play_count,
    SUM(CASE WHEN success THEN 1 ELSE 0 END) AS win_count,
    ROUND((SUM(CASE WHEN success THEN 1 ELSE 0 END)::DECIMAL / COUNT(*)) * 100, 2) AS win_rate,
    ROUND(AVG(wrong_guesses), 2) AS avg_wrong_guesses
FROM
    games
GROUP BY
    word;
