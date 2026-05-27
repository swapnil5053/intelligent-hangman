from pydantic import BaseModel, Field
from typing import List, Dict


class StartGameResponse(BaseModel):
    word: str = Field(..., description="The chosen target word")
    pattern: str = Field(..., description="The initial masked pattern of the word, e.g. '_ _ _ _ _'")
    word_length: int = Field(..., description="The length of the word")


class GuessRequest(BaseModel):
    pattern: str = Field(..., example="_ p p _ _", description="The current game pattern with spaces")
    guessed: List[str] = Field(default_factory=list, example=["a", "e"], description="List of letters guessed so far")


class GuessResponse(BaseModel):
    predictions: Dict[str, float] = Field(..., description="Probability distribution over 26 letters of the alphabet")
    suggested_letter: str = Field(..., description="The highest probability unguessed letter")


class LogGameRequest(BaseModel):
    word: str = Field(..., example="apple")
    success: bool = Field(..., example=True)
    wrong_guesses: int = Field(..., example=2)
    guesses: List[str] = Field(..., example=["a", "e", "p", "l"])


class LogGameResponse(BaseModel):
    status: str = Field("success")
    message: str = Field(..., description="Success or error description message")
    game_id: str = Field(None, description="The logged game UUID from the database")


class LeaderboardItem(BaseModel):
    word: str
    play_count: int
    win_count: int
    win_rate: float
    avg_wrong_guesses: float
