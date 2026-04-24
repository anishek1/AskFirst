from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class Session:
    session_id: str
    timestamp: datetime
    user_message: str
    clary_questions: list[str]
    user_followup: str
    clary_response: str
    severity: str
    tags: list[str]


@dataclass
class User:
    user_id: str
    name: str
    age: int
    gender: str
    location: str
    occupation: str
    onboarding_notes: str
    sessions: list[Session]


def load_dataset(path: str) -> list[User]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    users = []
    for u in raw["users"]:
        sessions = []
        for c in u["conversations"]:
            questions = c.get("clary_questions", [])
            if isinstance(questions, str):
                questions = [questions]
            sessions.append(Session(
                session_id=c["session_id"],
                timestamp=datetime.fromisoformat(c["timestamp"]),
                user_message=c.get("user_message", ""),
                clary_questions=questions,
                user_followup=c.get("user_followup", ""),
                clary_response=c.get("clary_response", ""),
                severity=c.get("severity", ""),
                tags=c.get("tags", []),
            ))
        sessions.sort(key=lambda s: s.timestamp)
        users.append(User(
            user_id=u["user_id"],
            name=u["name"],
            age=u.get("age", 0),
            gender=u.get("gender", ""),
            location=u.get("location", ""),
            occupation=u.get("occupation", ""),
            onboarding_notes=u.get("onboarding_notes", ""),
            sessions=sessions,
        ))
    return users
