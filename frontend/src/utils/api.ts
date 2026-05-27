const API_BASE_URL = "http://127.0.0.1:8000/api";

export interface StartGameResponse {
  word: string;
  pattern: string;
  word_length: number;
}

export interface GuessResponse {
  predictions: Record<string, number>;
  suggested_letter: string;
}

export interface LogGameRequest {
  word: string;
  success: boolean;
  wrong_guesses: number;
  guesses: string[];
}

export interface LogGameResponse {
  status: string;
  message: string;
  game_id?: string;
}

export interface LeaderboardItem {
  word: string;
  play_count: number;
  win_count: number;
  win_rate: number;
  avg_wrong_guesses: number;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${path}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers || {}),
    },
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`API error (${response.status}): ${errText || response.statusText}`);
  }

  return response.json() as Promise<T>;
}

export const api = {
  startGame: async (): Promise<StartGameResponse> => {
    return request<StartGameResponse>("/game/start", { method: "POST" });
  },

  guessLetter: async (pattern: string, guessed: string[]): Promise<GuessResponse> => {
    return request<GuessResponse>("/game/guess", {
      method: "POST",
      body: JSON.stringify({ pattern, guessed }),
    });
  },

  logGame: async (data: LogGameRequest): Promise<LogGameResponse> => {
    return request<LogGameResponse>("/game/log", {
      method: "POST",
      body: JSON.stringify(data),
    });
  },

  getLeaderboard: async (limit: number = 10): Promise<LeaderboardItem[]> => {
    return request<LeaderboardItem[]>(`/leaderboard?limit=${limit}`, { method: "GET" });
  },
};
