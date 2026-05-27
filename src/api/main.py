import os
import random
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.transformer_oracle import TransformerOracle
from src.utils import load_corpus
from src.api import schemas, database

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and corpus on startup
    model_path = os.path.join("data", "transformer_agent.pth")
    corpus_path = os.path.join("data", "corpus.txt")
    
    print(f"Startup: Loading Hangman Transformer from {model_path}...")
    try:
        # Load the oracle with weights
        state["oracle"] = TransformerOracle(model_path=model_path, corpus_path=corpus_path)
        # Load corpus for random word picking
        state["words"] = load_corpus(corpus_path)
        print("Startup: Transformer model and corpus successfully loaded.")
    except Exception as e:
        print(f"Startup Error: Failed to load Transformer or corpus: {e}")
        state["oracle"] = None
        state["words"] = []
        
    yield
    # Clean up state on shutdown
    state.clear()
    print("Shutdown: Application state cleared.")


app = FastAPI(
    title="Intelligent Hangman API",
    description="FastAPI backend serving PyTorch Hangman predictions and logging game history to Supabase.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/game/start", response_model=schemas.StartGameResponse, summary="Start a new game session")
def start_game():
    """
    Selects a random word from the corpus and returns its initial masked state.
    """
    words = state.get("words", [])
    if not words:
        # Fallback dictionary if corpus is unavailable
        words = ["production", "backend", "database", "supabase", "transformer", "neuralnetwork", "marmar"]
        
    word = random.choice(words)
    pattern = " ".join(["_"] * len(word))
    
    return {
        "word": word,
        "pattern": pattern,
        "word_length": len(word)
    }


@app.post("/api/game/guess", response_model=schemas.GuessResponse, summary="Get predictions for a given game state")
def submit_guess(req: schemas.GuessRequest):
    """
    Given a current word pattern (e.g. '_ p p _ _') and list of guessed letters,
    returns character probabilities from the Transformer and recommends the best next guess.
    """
    oracle = state.get("oracle")
    if oracle is None:
        raise HTTPException(
            status_code=503,
            detail="The Transformer model is not loaded. Ensure that data/transformer_agent.pth is present."
        )
        
    try:
        # Clean pattern and guessed letters input
        cleaned_pattern = req.pattern.strip().lower()
        cleaned_guessed = {g.strip().lower() for g in req.guessed if g.strip()}
        
        # Get distribution and prediction
        predictions = oracle.letter_distribution(cleaned_pattern, cleaned_guessed)
        suggested_letter = oracle.predict(cleaned_pattern, cleaned_guessed)
        
        return {
            "predictions": predictions,
            "suggested_letter": suggested_letter
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(e)}")


@app.post("/api/game/log", response_model=schemas.LogGameResponse, summary="Log a completed game result")
def log_game(req: schemas.LogGameRequest):
    """
    Logs word, success, wrong guesses count, and guesses list to Supabase.
    """
    try:
        game_id = database.log_game_result(
            word=req.word.strip().lower(),
            success=req.success,
            wrong_guesses=req.wrong_guesses,
            guesses=[g.strip().lower() for g in req.guesses]
        )
        return {
            "status": "success",
            "message": "Game result successfully logged.",
            "game_id": game_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log game to database: {str(e)}")


@app.get("/api/leaderboard", response_model=List[schemas.LeaderboardItem], summary="Get the hardest words leaderboard")
def get_leaderboard(limit: int = 10):
    """
    Retrieves the global leaderboard containing the hardest words based on play history.
    """
    try:
        leaderboard = database.get_leaderboard_stats(limit=limit)
        return leaderboard
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch leaderboard: {str(e)}")
