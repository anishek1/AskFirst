import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.data_loader import load_dataset, User
from src.llm_provider import get_provider
from src.pattern_detector import stream_pattern_detection
from src.temporal_engine import build_temporal_timeline

DATASET_PATH = Path(__file__).parent / "Task" / "askfirst_synthetic_dataset.json"

st.set_page_config(
    page_title="Clary — Health Pattern Detector",
    page_icon="",
    layout="wide",
)


@st.cache_data
def get_dataset() -> list[User]:
    return load_dataset(str(DATASET_PATH))


def confidence_badge(level: str) -> str:
    icons = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}
    colors = {"high": "green", "medium": "orange", "low": "red"}
    label = icons.get(level.lower(), level.upper())
    color = colors.get(level.lower(), "gray")
    return f":{color}[{label}]"


def render_patterns(patterns_json: str) -> None:
    try:
        data = json.loads(patterns_json)
    except json.JSONDecodeError as e:
        st.error(f"JSON parse error: {e}")
        st.code(patterns_json, language="json")
        return

    patterns = data.get("patterns", [])
    if not patterns:
        st.warning("No patterns detected.")
        return

    st.success(f"Detected {len(patterns)} pattern(s)")

    for p in patterns:
        conf = p.get("confidence", "low").lower()
        badge = confidence_badge(conf)
        pid = p.get("pattern_id", "?")
        title = p.get("title", "")

        with st.expander(f"**{pid}** — {title}   [{badge}]", expanded=True):
            col1, col2 = st.columns([3, 2])

            with col1:
                st.markdown("**Temporal Reasoning**")
                st.markdown(p.get("temporal_reasoning", ""))
                st.markdown("**Confidence Justification**")
                st.info(p.get("confidence_justification", ""))

            with col2:
                st.markdown("**Sessions Involved**")
                for sid in p.get("sessions_involved", []):
                    st.code(sid, language=None)
                st.markdown("**Reasoning Trace**")
                for i, step in enumerate(p.get("trace", []), 1):
                    st.markdown(f"`{i}.` {step}")


def run_analysis(user: User) -> None:
    try:
        provider = get_provider()
    except Exception as e:
        st.error(f"Provider error: {e}\n\nCheck your .env file matches .env.example")
        return

    with st.expander("Temporal timeline sent to LLM", expanded=False):
        st.code(build_temporal_timeline(user), language=None)

    st.markdown("### Reasoning trace")
    reasoning_box = st.empty()

    st.markdown("### Detected patterns")
    patterns_box = st.empty()

    reasoning_text = ""
    json_str = None

    for event_type, content in stream_pattern_detection(user, provider):
        if event_type == "reasoning":
            reasoning_text += content
            reasoning_box.markdown(
                "<div style='font-size:0.82em;color:#555;font-family:monospace;"
                f"white-space:pre-wrap;border-left:3px solid #ccc;padding-left:10px'>"
                f"{reasoning_text}</div>",
                unsafe_allow_html=True,
            )
        elif event_type == "reasoning_complete":
            reasoning_box.markdown(
                "<div style='font-size:0.82em;color:#555;font-family:monospace;"
                f"white-space:pre-wrap;border-left:3px solid #ccc;padding-left:10px'>"
                f"{content}</div>",
                unsafe_allow_html=True,
            )
        elif event_type == "complete":
            json_str = content
            with patterns_box.container():
                render_patterns(json_str)
        elif event_type == "error":
            patterns_box.error(content)

    if json_str:
        with st.expander("Raw JSON output", expanded=False):
            st.code(json_str, language="json")


def main() -> None:
    st.title("Clary — Health Pattern Detector")
    st.caption(
        "Ask First · AI Intern Assignment · "
        "Cross-session health pattern detection with temporal reasoning"
    )

    # API key warning
    if not os.getenv("LLM_API_KEY"):
        st.warning(
            "No LLM_API_KEY found. Copy `.env.example` to `.env` and set your key.",
            icon="",
        )

    try:
        users = get_dataset()
    except FileNotFoundError:
        st.error(f"Dataset not found at: {DATASET_PATH}")
        return

    user_map = {u.user_id: u for u in users}

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    selected_user = None
    analyze_all = False

    with col1:
        if st.button("Analyze Arjun (USR001)", use_container_width=True):
            selected_user = user_map.get("USR001")
    with col2:
        if st.button("Analyze Meera (USR002)", use_container_width=True):
            selected_user = user_map.get("USR002")
    with col3:
        if st.button("Analyze Priya (USR003)", use_container_width=True):
            selected_user = user_map.get("USR003")
    with col4:
        if st.button("Analyze All Users", use_container_width=True):
            analyze_all = True

    st.markdown("---")

    if analyze_all:
        tabs = st.tabs([u.name for u in users])
        for tab, user in zip(tabs, users):
            with tab:
                st.subheader(f"{user.name} ({user.user_id})")
                st.caption(f"{user.occupation} · {user.location} · Age {user.age}")
                run_analysis(user)
    elif selected_user:
        st.subheader(f"{selected_user.name} ({selected_user.user_id})")
        st.caption(
            f"{selected_user.occupation} · {selected_user.location} · Age {selected_user.age}"
        )
        run_analysis(selected_user)
    else:
        st.info(
            "Select a user above to start pattern analysis. "
            "The reasoning trace will stream live, followed by structured JSON pattern cards."
        )

        st.markdown("#### Dataset overview")
        for u in users:
            with st.expander(f"{u.name} ({u.user_id}) — {len(u.sessions)} sessions"):
                st.markdown(f"**Profile**: {u.onboarding_notes}")
                sessions = sorted(u.sessions, key=lambda s: s.timestamp)
                st.markdown(
                    f"**Date range**: {sessions[0].timestamp.strftime('%b %d')} "
                    f"– {sessions[-1].timestamp.strftime('%b %d, %Y')}"
                )
                for s in sessions:
                    st.markdown(
                        f"- `{s.session_id}` {s.timestamp.strftime('%b %d')} "
                        f"— {s.tags}"
                    )


if __name__ == "__main__":
    main()
