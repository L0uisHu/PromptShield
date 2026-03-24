from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine

DATABASE_URL = "sqlite:///./promptshield.db"
engine = create_engine(DATABASE_URL)


class RequestLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    flagged: bool
    score: float
    flagged_content: Optional[str] = None
    reason: Optional[str] = None
    detection_method: Optional[str] = None
    request_preview: str
    latency_ms: int


def create_db():
    SQLModel.metadata.create_all(engine)


def log_request(
    flagged: bool,
    score: float,
    flagged_content: str,
    reason: str,
    detection_method: str,
    request_preview: str,
    latency_ms: int,
):
    record = RequestLog(
        flagged=flagged,
        score=score,
        flagged_content=flagged_content or None,
        reason=reason or None,
        detection_method=detection_method or None,
        request_preview=request_preview,
        latency_ms=latency_ms,
    )
    with Session(engine) as session:
        session.add(record)
        session.commit()
