from datetime import datetime
from src.data_loader import Session, User


def compute_reference_date(sessions: list[Session]) -> datetime:
    return min(s.timestamp for s in sessions)


def get_week_number(session: Session, reference: datetime) -> int:
    delta = (session.timestamp.date() - reference.date()).days
    return delta // 7 + 1


def get_day_delta(session: Session, reference: datetime) -> int:
    return (session.timestamp.date() - reference.date()).days


def get_time_of_day_label(session: Session) -> str:
    hour = session.timestamp.hour
    if hour >= 23 or hour < 4:
        return "LATE-NIGHT (11pm-4am)"
    elif hour < 12:
        return "MORNING"
    elif hour < 17:
        return "AFTERNOON"
    else:
        return "EVENING"


def build_temporal_timeline(user: User) -> str:
    sessions = sorted(user.sessions, key=lambda s: s.timestamp)
    reference = compute_reference_date(sessions)
    lines = []

    for s in sessions:
        week = get_week_number(s, reference)
        day = get_day_delta(s, reference)
        tod = get_time_of_day_label(s)
        dt_str = s.timestamp.strftime("%Y-%m-%d %a %H:%M")

        lines.append(f"[Week {week} | Day {day} | {dt_str} | {tod}] {s.session_id}")
        lines.append(f'User: "{s.user_message}"')
        if s.clary_questions:
            lines.append("Questions: " + " | ".join(s.clary_questions))
        if s.user_followup:
            lines.append(f'Followup: "{s.user_followup}"')
        lines.append(f'Response: "{s.clary_response}"')
        lines.append(f"Severity: {s.severity} | Tags: {', '.join(s.tags)}")
        lines.append("---")

    return "\n".join(lines)
