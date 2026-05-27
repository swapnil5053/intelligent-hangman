"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { api, LeaderboardItem } from "@/utils/api";

const ALPHABET = "abcdefghijklmnopqrstuvwxyz".split("");

export default function Home() {
  // Game state
  const [word, setWord] = useState<string>("");
  const [pattern, setPattern] = useState<string>("");
  const [guessedLetters, setGuessedLetters] = useState<Set<string>>(new Set());
  const [lives, setLives] = useState<number>(6);
  const [wrongGuessesCount, setWrongGuessesCount] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [isGameOver, setIsGameOver] = useState<boolean>(false);
  const [success, setSuccess] = useState<boolean>(false);

  // Predictions state
  const [predictions, setPredictions] = useState<Record<string, number>>({});
  const [suggestedLetter, setSuggestedLetter] = useState<string>("");
  const [isAutoplayRunning, setIsAutoplayRunning] = useState<boolean>(false);

  // Leaderboard & global state
  const [leaderboard, setLeaderboard] = useState<LeaderboardItem[]>([]);
  const [isLoadingLeaderboard, setIsLoadingLeaderboard] = useState<boolean>(false);
  const [customWordInput, setCustomWordInput] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  // Autoplay timer reference
  const autoplayTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch leaderboard statistics
  const fetchLeaderboard = async () => {
    setIsLoadingLeaderboard(true);
    try {
      const data = await api.getLeaderboard(8);
      setLeaderboard(data);
    } catch (err: any) {
      console.error("Failed to load leaderboard:", err);
    } finally {
      setIsLoadingLeaderboard(false);
    }
  };

  // Fetch leaderboard on load
  useEffect(() => {
    fetchLeaderboard();
  }, []);

  // Fetch letter predictions from FastAPI server
  const fetchPredictions = useCallback(async (currentPattern: string, guessedList: string[]) => {
    setError(null);
    try {
      const res = await api.guessLetter(currentPattern, guessedList);
      setPredictions(res.predictions);
      setSuggestedLetter(res.suggested_letter);
    } catch (err: any) {
      console.error("Failed to fetch predictions:", err);
      setError("Failed to fetch model predictions. Is the backend running?");
    }
  }, []);

  // Log game result to Supabase
  const logGameToDatabase = async (targetWord: string, isWin: boolean, wrongCount: number, allGuesses: string[]) => {
    try {
      await api.logGame({
        word: targetWord,
        success: isWin,
        wrong_guesses: wrongCount,
        guesses: allGuesses,
      });
      fetchLeaderboard(); // refresh stats
    } catch (err) {
      console.error("Failed to log game to database:", err);
    }
  };

  // Start new game
  const startNewGame = async (customWord?: string) => {
    // Reset autoplay
    setIsAutoplayRunning(false);
    if (autoplayTimerRef.current) {
      clearTimeout(autoplayTimerRef.current);
    }

    setError(null);
    setGuessedLetters(new Set());
    setLives(6);
    setWrongGuessesCount(0);
    setSuccess(false);
    setIsGameOver(false);

    if (customWord) {
      const cleaned = customWord.trim().toLowerCase().replace(/[^a-z]/g, "");
      if (!cleaned) {
        setError("Please enter a valid word (a-z letters only).");
        return;
      }
      setWord(cleaned);
      const initialPattern = Array(cleaned.length).fill("_").join(" ");
      setPattern(initialPattern);
      setIsPlaying(true);
      setCustomWordInput("");
      // Fetch initial predictions
      fetchPredictions(initialPattern, []);
    } else {
      try {
        const game = await api.startGame();
        setWord(game.word);
        setPattern(game.pattern);
        setIsPlaying(true);
        // Fetch initial predictions
        fetchPredictions(game.pattern, []);
      } catch (err: any) {
        console.error("Failed to start game:", err);
        setError("Failed to start game. Make sure the backend server is running on http://127.0.0.1:8000");
      }
    }
  };

  // Handle single letter guess
  const makeGuess = useCallback(
    async (letter: string) => {
      if (isGameOver || guessedLetters.has(letter)) return;

      const newGuessed = new Set(guessedLetters);
      newGuessed.add(letter);
      setGuessedLetters(newGuessed);

      const guessedList = Array.from(newGuessed);

      if (word.includes(letter)) {
        // Correct guess: Update masked pattern
        const newPatternArr = word.split("").map((char) => (newGuessed.has(char) ? char : "_"));
        const newPattern = newPatternArr.join(" ");
        setPattern(newPattern);

        const isSolved = !newPatternArr.includes("_");
        if (isSolved) {
          setIsGameOver(true);
          setSuccess(true);
          setIsPlaying(false);
          setIsAutoplayRunning(false);
          logGameToDatabase(word, true, wrongGuessesCount, guessedList);
        } else {
          fetchPredictions(newPattern, guessedList);
        }
      } else {
        // Wrong guess: Deduct life
        const newWrongGuesses = wrongGuessesCount + 1;
        setWrongGuessesCount(newWrongGuesses);
        const newLives = lives - 1;
        setLives(newLives);

        if (newLives <= 0) {
          setIsGameOver(true);
          setSuccess(false);
          setIsPlaying(false);
          setIsAutoplayRunning(false);
          // Reveal full word
          setPattern(word.split("").join(" "));
          logGameToDatabase(word, false, newWrongGuesses, guessedList);
        } else {
          fetchPredictions(pattern, guessedList);
        }
      }
    },
    [word, pattern, guessedLetters, lives, wrongGuessesCount, isGameOver, fetchPredictions]
  );

  // Autoplay handler
  useEffect(() => {
    if (isAutoplayRunning && isPlaying && suggestedLetter && !isGameOver) {
      autoplayTimerRef.current = setTimeout(() => {
        makeGuess(suggestedLetter);
      }, 900); // 900ms delay between guesses for nice visual speed
    }

    return () => {
      if (autoplayTimerRef.current) {
        clearTimeout(autoplayTimerRef.current);
      }
    };
  }, [isAutoplayRunning, isPlaying, suggestedLetter, makeGuess, isGameOver]);

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (autoplayTimerRef.current) {
        clearTimeout(autoplayTimerRef.current);
      }
    };
  }, []);

  // Format probabilities
  const topPredictions = Object.entries(predictions)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  return (
    <div className="bg-slate-950 text-slate-100 min-h-screen font-sans">
      {/* Top Banner */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-slate-700 to-slate-500 flex items-center justify-center font-bold text-white shadow-md">
              H
            </div>
            <h1 className="text-xl font-bold tracking-tight text-slate-200">
              Hangman Control Panel
            </h1>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-800/80 border border-slate-700 text-xs">
              <span className={`w-2.5 h-2.5 rounded-full ${isPlaying ? "bg-sky-500 animate-pulse" : "bg-emerald-500"}`} />
              <span className="font-semibold text-slate-300">
                {isGameOver ? "Session Ended" : isPlaying ? "Active Session" : "Ready"}
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8 lg:py-12">
        {error && (
          <div className="mb-6 p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm flex items-center gap-3 animate-fadeIn">
            <span className="font-bold">Error:</span> {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Columns: Play Zone */}
          <div className="lg:col-span-2 space-y-8">
            {/* Setup Options */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6 backdrop-blur-md shadow-xl">
              <h2 className="text-lg font-semibold text-slate-200 mb-4">
                Game Configuration
              </h2>
              <div className="flex flex-col md:flex-row items-stretch gap-4">
                <button
                  onClick={() => startNewGame()}
                  className="flex-1 px-6 py-3 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 font-semibold text-slate-200 shadow-md active:scale-95 transition-all text-center"
                >
                  Generate Random Word
                </button>
                <div className="flex-1 flex items-stretch gap-2 bg-slate-950 border border-slate-800 rounded-xl p-1">
                  <input
                    type="password"
                    placeholder="Enter custom target word..."
                    value={customWordInput}
                    onChange={(e) => setCustomWordInput(e.target.value)}
                    className="flex-1 bg-transparent px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none"
                  />
                  <button
                    onClick={() => startNewGame(customWordInput)}
                    className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 font-medium text-xs text-slate-200 active:scale-95 transition-all"
                  >
                    Start Session
                  </button>
                </div>
              </div>
            </div>

            {/* Active Game Console */}
            {isPlaying || isGameOver ? (
              <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6 md:p-8 backdrop-blur-md shadow-xl space-y-8 relative overflow-hidden">
                {/* Score & Lives Info */}
                <div className="flex items-center justify-between gap-4 border-b border-slate-800/80 pb-6">
                  <div>
                    <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Metadata</h3>
                    <div className="text-lg font-bold text-slate-200 mt-1">
                      Length: <span className="text-sky-400">{word.length}</span> characters
                    </div>
                  </div>
                  <div className="text-right">
                    <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Lives Remaining</h3>
                    <div className="flex items-center justify-end gap-1.5 mt-1">
                      {Array.from({ length: 6 }).map((_, idx) => (
                        <span
                          key={idx}
                          className={`w-3.5 h-3.5 rounded-full border transition-all ${
                            idx < lives
                              ? "bg-sky-500 border-sky-400 shadow-md shadow-sky-500/20 scale-100"
                              : "bg-slate-900 border-slate-800 scale-90"
                          }`}
                        />
                      ))}
                    </div>
                  </div>
                </div>

                {/* Game Area Grid (Visual Hangman + Word Slots) */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
                  {/* SVG Hangman */}
                  <div className="flex justify-center items-center">
                    <svg viewBox="0 0 200 250" className="w-full max-w-[160px] stroke-slate-500 stroke-[5] fill-none stroke-linecap-round">
                      {/* Gallows (Fixed) */}
                      <path d="M 20 230 L 100 230" className="stroke-slate-700" /> {/* Base */}
                      <path d="M 40 230 L 40 30" className="stroke-slate-700" />   {/* Vertical Pole */}
                      <path d="M 40 30 L 130 30" className="stroke-slate-700" />   {/* Crossbar */}
                      <path d="M 40 70 L 80 30" className="stroke-slate-700" />    {/* Brace */}
                      <path d="M 130 30 L 130 65" className="stroke-slate-700 stroke-[3]" /> {/* Rope */}

                      {/* Head */}
                      {wrongGuessesCount >= 1 && (
                        <circle cx="130" cy="85" r="20" className="stroke-sky-500 stroke-[4]" />
                      )}
                      {/* Torso */}
                      {wrongGuessesCount >= 2 && (
                        <line x1="130" y1="105" x2="130" y2="165" className="stroke-sky-500 stroke-[4]" />
                      )}
                      {/* Left Arm */}
                      {wrongGuessesCount >= 3 && (
                        <line x1="130" y1="120" x2="100" y2="145" className="stroke-sky-500 stroke-[4]" />
                      )}
                      {/* Right Arm */}
                      {wrongGuessesCount >= 4 && (
                        <line x1="130" y1="120" x2="160" y2="145" className="stroke-sky-500 stroke-[4]" />
                      )}
                      {/* Left Leg */}
                      {wrongGuessesCount >= 5 && (
                        <line x1="130" y1="165" x2="105" y2="215" className="stroke-sky-500 stroke-[4]" />
                      )}
                      {/* Right Leg */}
                      {wrongGuessesCount >= 6 && (
                        <line x1="130" y1="165" x2="155" y2="215" className="stroke-sky-500 stroke-[4]" />
                      )}
                    </svg>
                  </div>

                  {/* Word Slot Patterns */}
                  <div className="md:col-span-2 flex flex-col justify-center items-center md:items-start space-y-6">
                    <div className="flex flex-wrap justify-center gap-2 md:gap-3 py-4">
                      {pattern.split(" ").map((char, idx) => (
                        <div
                          key={idx}
                          className={`w-10 h-14 md:w-12 md:h-16 rounded-xl flex items-center justify-center font-bold text-lg md:text-xl border transition-all ${
                            char !== "_"
                              ? "bg-slate-900 border-sky-500/60 text-sky-200 shadow-md shadow-sky-500/10 scale-105"
                              : "bg-slate-950/80 border-slate-800 text-slate-500"
                          }`}
                        >
                          {char}
                        </div>
                      ))}
                    </div>

                    {isGameOver && (
                      <div className="animate-fadeIn">
                        {success ? (
                          <div className="px-4 py-2 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-semibold text-sm">
                            Word Solved Successfully
                          </div>
                        ) : (
                          <div className="px-4 py-2 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 font-semibold text-sm">
                            Failed to Solve. Target Word was &ldquo;{word}&rdquo;
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>

                {/* Letters Input Keyboard */}
                <div className="space-y-4">
                  <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest text-center md:text-left">
                    Input Keyboard
                  </h3>
                  <div className="grid grid-cols-6 sm:grid-cols-9 md:grid-cols-10 gap-2">
                    {ALPHABET.map((letter) => {
                      const hasGuessed = guessedLetters.has(letter);
                      const isCorrect = hasGuessed && word.includes(letter);
                      return (
                        <button
                          key={letter}
                          disabled={hasGuessed || isGameOver}
                          onClick={() => makeGuess(letter)}
                          className={`h-11 md:h-12 rounded-lg font-bold text-sm uppercase transition-all flex items-center justify-center ${
                            isCorrect
                              ? "bg-emerald-950/30 border border-emerald-500/50 text-emerald-400"
                              : hasGuessed
                              ? "bg-slate-950 border border-slate-900 text-slate-600 line-through"
                              : "bg-slate-950 border border-slate-800 text-slate-300 hover:border-slate-600 hover:bg-slate-900 active:scale-95"
                          } disabled:cursor-not-allowed`}
                        >
                          {letter}
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>
            ) : (
              // Empty State
              <div className="rounded-2xl border border-dashed border-slate-800 p-12 text-center text-slate-500 space-y-3">
                <div className="font-medium text-slate-400">No Active Session</div>
                <p className="text-xs text-slate-600 max-w-sm mx-auto">
                  Generate a random word or enter a custom target word to begin prediction testing.
                </p>
              </div>
            )}
          </div>

          {/* Right Columns: Inference Engine & Leaderboard */}
          <div className="space-y-8">
            {/* Predictions Visualizer */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6 backdrop-blur-md shadow-xl flex flex-col">
              <h2 className="text-lg font-semibold text-slate-200 mb-4">
                Prediction Engine
              </h2>

              {isPlaying && suggestedLetter ? (
                <div className="space-y-6 flex-1 flex flex-col">
                  {/* Suggested letter card */}
                  <div className="rounded-xl border border-slate-700 bg-slate-950/40 p-4 text-center shadow-inner">
                    <div className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
                      Top Recommendation
                    </div>
                    <div className="text-5xl font-black text-slate-100 py-2 uppercase">
                      {suggestedLetter}
                    </div>
                    <p className="text-[10px] text-slate-500 mt-1">
                      Highest probability letter matching the current context.
                    </p>
                  </div>

                  {/* Actions */}
                  <div className="flex gap-2">
                    <button
                      onClick={() => makeGuess(suggestedLetter)}
                      className="flex-1 px-3 py-2.5 rounded-lg bg-sky-600 hover:bg-sky-500 font-semibold text-xs text-white transition-all active:scale-95"
                    >
                      Apply Suggested Guess
                    </button>
                    <button
                      onClick={() => setIsAutoplayRunning(!isAutoplayRunning)}
                      className={`flex-1 px-3 py-2.5 rounded-lg font-semibold text-xs transition-all active:scale-95 border ${
                        isAutoplayRunning
                          ? "bg-rose-950/20 border-rose-500/50 text-rose-400 hover:bg-rose-950/40"
                          : "bg-slate-950 border-slate-850 text-slate-300 hover:bg-slate-900"
                      }`}
                    >
                      {isAutoplayRunning ? "Stop Autoplay" : "Run Autoplay"}
                    </button>
                  </div>

                  {/* Probability Chart */}
                  <div className="space-y-3 flex-1">
                    <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
                      Probability Metrics
                    </h3>
                    <div className="space-y-2.5">
                      {topPredictions.map(([char, prob]) => {
                        const pct = (prob * 100).toFixed(1);
                        return (
                          <div key={char} className="space-y-1">
                            <div className="flex items-center justify-between text-xs font-medium text-slate-300">
                              <span className="uppercase text-slate-100 font-bold">{char}</span>
                              <span>{pct}%</span>
                            </div>
                            <div className="h-2 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800/40">
                              <div
                                style={{ width: `${pct}%` }}
                                className="h-full bg-sky-500 rounded-full transition-all duration-300"
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex-1 flex flex-col items-center justify-center text-center p-8 text-slate-600 space-y-2">
                  <p className="text-xs">Start a game session to view model predictions and letter probabilities.</p>
                </div>
              )}
            </div>

            {/* Leaderboard */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6 backdrop-blur-md shadow-xl">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-md font-semibold text-slate-200">
                  Difficult Words Leaderboard
                </h2>
                <button
                  onClick={fetchLeaderboard}
                  disabled={isLoadingLeaderboard}
                  className="p-1.5 rounded-lg border border-slate-800 hover:bg-slate-800 text-[10px] text-slate-400 transition-all active:scale-95 disabled:opacity-50"
                  title="Reload Stats"
                >
                  {isLoadingLeaderboard ? "Loading..." : "Refresh"}
                </button>
              </div>

              {leaderboard && leaderboard.length > 0 ? (
                <div className="overflow-x-auto rounded-lg border border-slate-850">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="bg-slate-950/60 border-b border-slate-800 text-slate-400 font-bold uppercase tracking-wider">
                        <th className="px-3 py-2.5">Word</th>
                        <th className="px-3 py-2.5 text-center">Plays</th>
                        <th className="px-3 py-2.5 text-center">Win Rate</th>
                        <th className="px-3 py-2.5 text-center">Avg Wrong</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-850">
                      {leaderboard.map((item, idx) => (
                        <tr
                          key={idx}
                          className="hover:bg-slate-900/40 transition-colors"
                        >
                          <td className="px-3 py-2.5 font-semibold text-slate-300 truncate max-w-[100px]" title={item.word}>
                            {item.word}
                          </td>
                          <td className="px-3 py-2.5 text-center text-slate-400">{item.play_count}</td>
                          <td className={`px-3 py-2.5 text-center font-bold ${
                            item.win_rate <= 30
                              ? "text-rose-400"
                              : item.win_rate <= 60
                              ? "text-amber-400"
                              : "text-emerald-400"
                          }`}>
                            {item.win_rate}%
                          </td>
                          <td className="px-3 py-2.5 text-center text-slate-400">{item.avg_wrong_guesses}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="p-8 text-center text-xs text-slate-600">
                  No statistical data logged yet.
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
