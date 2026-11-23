from datetime import datetime

from pydantic import BaseModel, Field


class MCQ(BaseModel):
    stem: str
    key: str
    distractors: list[str]


class Question(BaseModel):
    question: MCQ
    topic: str


class HistoryRecord(Question):
    response: str = None
    confidence: int | None = Field(None, ge=1, le=5)
    correct: bool = None
    skipped: bool
    date: datetime


class Inputs(BaseModel):
    candidate_questions: list[Question]
    history: list[HistoryRecord]
