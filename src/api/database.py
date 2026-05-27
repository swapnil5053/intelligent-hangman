import os
import uuid
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase_client = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Successfully connected to Supabase.")
    except Exception as e:
        print(f"Error initializing Supabase client: {e}")
        print("Database queries will run in mock mode.")
else:
    print("Warning: SUPABASE_URL or SUPABASE_KEY environment variable is missing.")
    print("Database queries will run in mock mode.")


def log_game_result(word: str, success: bool, wrong_guesses: int, guesses: List[str]) -> str:
    """
    Logs the outcome of a finished Hangman game to the games table in Supabase.
    Returns the game record UUID.
    """
    if supabase_client is not None:
        try:
            response = supabase_client.table("games").insert({
                "word": word,
                "success": success,
                "wrong_guesses": wrong_guesses,
                "guesses": guesses
            }).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0].get("id")
            
        except Exception as e:
            print(f"Failed to log game to Supabase: {e}")
            
    # Mock fallback
    mock_id = str(uuid.uuid4())
    print(f"[MOCK DB] Logged game (ID={mock_id}): word='{word}', success={success}, wrong={wrong_guesses}, guesses={guesses}")
    return mock_id


def get_leaderboard_stats(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieves statistics on the hardest words from the Supabase view.
    """
    if supabase_client is not None:
        try:
            response = supabase_client.table("hardest_words") \
                .select("*") \
                .order("win_rate", desc=False) \
                .order("avg_wrong_guesses", desc=True) \
                .limit(limit) \
                .execute()
            
            if response.data is not None:
                return response.data
            
        except Exception as e:
            print(f"Failed to fetch leaderboard from Supabase: {e}")

    # Mock fallback for leaderboard
    print("[MOCK DB] Fetching mock leaderboard stats")
    return [
        {
            "word": "gynandromorphous",
            "play_count": 5,
            "win_count": 1,
            "win_rate": 20.0,
            "avg_wrong_guesses": 5.2
        },
        {
            "word": "troveless",
            "play_count": 3,
            "win_count": 0,
            "win_rate": 0.0,
            "avg_wrong_guesses": 6.0
        },
        {
            "word": "marmar",
            "play_count": 10,
            "win_count": 4,
            "win_rate": 40.0,
            "avg_wrong_guesses": 4.5
        },
        {
            "word": "gastrostenosis",
            "play_count": 8,
            "win_count": 4,
            "win_rate": 50.0,
            "avg_wrong_guesses": 4.0
        },
        {
            "word": "unnotify",
            "play_count": 2,
            "win_count": 1,
            "win_rate": 50.0,
            "avg_wrong_guesses": 5.0
        }
    ]
