import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.data_loader import load_dataset, User
from src.llm_provider import get_provider
from src.chat_engine import (
    build_system_prompt,
    build_initial_user_message,
    build_all_patients_system_prompt,
    build_cross_patient_user_message,
    stream_response,
)
from src.temporal_engine import (
    compute_reference_date,
    get_week_number,
    get_day_delta,
    get_time_of_day_label,
    build_temporal_timeline,
)

DATASET_PATH = Path(__file__).parent / "dataset" / "askfirst_synthetic_dataset.json"
CACHE_DIR    = Path(__file__).parent / ".clary_cache"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clary · Ask First",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* ── Root overrides ─────────────────────────────────────── */
html, body, [class*="css"] { 
    font-family: 'Inter', sans-serif !important; 
    color: #e2e8f0;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: -0.02em;
}

/* Dark app background with subtle gradient */
.stApp { 
    background: radial-gradient(circle at top right, #0f172a 0%, #020617 100%);
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.4) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}
[data-testid="stSidebar"] .block-container { padding: 2rem 1.5rem; }

/* ── Main content ────────────────────────────────────────── */
.main .block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 5rem;
}

/* ── Chat messages (Glassmorphism) ───────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin: 0.75rem 0;
    border: 1px solid rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    animation: fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

@keyframes fadeIn { 
    from { opacity: 0; transform: translateY(12px); } 
    to { opacity: 1; transform: translateY(0); } 
}

/* Assistant Bubble */
[data-testid="stChatMessage"][data-test-message-author-type="assistant"] {
    background: linear-gradient(145deg, rgba(14, 165, 233, 0.08) 0%, rgba(2, 132, 199, 0.02) 100%);
    border-left: 2px solid #0ea5e9;
}

/* User Bubble */
[data-testid="stChatMessage"][data-test-message-author-type="human"] {
    background: rgba(255, 255, 255, 0.02);
    border-right: 2px solid #64748b;
}

/* ── Chat Input ──────────────────────────────────────────── */
[data-testid="stChatInput"] {
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    background: rgba(2, 6, 23, 0.8);
    backdrop-filter: blur(16px);
    padding-bottom: 2rem;
}
[data-testid="stChatInput"] textarea {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    color: #f8fafc !important;
    font-size: 1.05em !important;
    padding: 0.8rem 1.2rem !important;
    transition: all 0.2s ease;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15), 0 0 20px rgba(56, 189, 248, 0.1) !important;
    background: rgba(15, 23, 42, 0.8) !important;
}

/* ── Buttons (Premium Glow) ──────────────────────────────── */
.stButton > button {
    border-radius: 12px;
    font-family: 'Outfit', sans-serif;
    font-weight: 500;
    letter-spacing: 0.01em;
    font-size: 0.95em;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #cbd5e1;
    padding: 0.6rem 1.2rem;
}
.stButton > button:hover {
    background: rgba(30, 41, 59, 0.8);
    border-color: rgba(255, 255, 255, 0.2);
    color: #f8fafc;
    transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    border: 1px solid #38bdf8;
    color: #ffffff;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.2);
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(2, 132, 199, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2);
    transform: translateY(-2px);
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
}

/* ── Expanders (Pattern Cards) ───────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    background: rgba(15, 23, 42, 0.3) !important;
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    overflow: hidden;
    margin-bottom: 1rem;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(14, 165, 233, 0.3) !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
}
[data-testid="stExpander"] summary { 
    color: #f1f5f9; 
    font-family: 'Outfit', sans-serif;
    padding: 1rem 1.2rem !important;
    background: rgba(255, 255, 255, 0.02);
}
[data-testid="stExpander"] summary:hover { color: #38bdf8; }

/* ── High/Med/Low Confidence Accents ─────────────────────── */
[data-testid="stExpander"]:has([data-confidence="high"]) { border-left: 3px solid #10b981 !important; }
[data-testid="stExpander"]:has([data-confidence="medium"]) { border-left: 3px solid #f59e0b !important; }
[data-testid="stExpander"]:has([data-confidence="low"]) { border-left: 3px solid #ef4444 !important; }

/* ── Code & Inline Code ──────────────────────────────────── */
code {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    padding: 0.15rem 0.4rem;
    font-size: 0.85em;
    color: #7dd3fc;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
}

/* ── Status Widget ───────────────────────────────────────── */
[data-testid="stStatusWidget"] {
    background: rgba(15, 23, 42, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(8px);
}

/* ── Info / Success Boxes ────────────────────────────────── */
[data-testid="stInfo"] {
    background: rgba(14, 165, 233, 0.05) !important;
    border: 1px solid rgba(14, 165, 233, 0.2) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
[data-testid="stSuccess"] {
    background: rgba(16, 185, 129, 0.05) !important;
    border: 1px solid rgba(16, 185, 129, 0.2) !important;
    border-radius: 10px !important;
}

/* ── Reasoning Trace (Details/Summary) ───────────────────── */
details > summary {
    cursor: pointer;
    font-size: 0.85em;
    color: #94a3b8;
    padding: 6px 0;
    user-select: none;
    list-style: none;
    font-family: 'Outfit', sans-serif;
    letter-spacing: 0.02em;
    transition: color 0.2s ease;
}
details > summary::before { content: "⚡ "; font-size: 1.1em; vertical-align: middle; }
details > summary:hover { color: #38bdf8; }
details pre {
    background: rgba(2, 6, 23, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.78em;
    color: #94a3b8;
    white-space: pre-wrap;
    max-height: 250px;
    overflow-y: auto;
    margin-top: 8px;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    line-height: 1.6;
    box-shadow: inset 0 2px 10px rgba(0,0,0,0.2);
}

/* ── Metrics ──────────────────────────────────────────────── */
[data-testid="stMetricValue"] { 
    color: #e2e8f0; 
    font-family: 'Outfit', sans-serif;
    font-weight: 700; 
    font-size: 2.2rem;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8;
    font-weight: 500;
}

/* ── Divider ──────────────────────────────────────────────── */
hr { 
    border-color: rgba(255, 255, 255, 0.08); 
    margin: 1.5rem 0; 
}

/* ── Typographic elements ────────────────────────────────── */
.gradient-text {
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Analysis cache helpers ────────────────────────────────────────────────────

def _compute_user_hash(user: User) -> str:
    key = "|".join(
        f"{s.session_id}:{s.timestamp.isoformat()}"
        for s in sorted(user.sessions, key=lambda x: x.timestamp)
    )
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _compute_all_hash(users: list) -> str:
    key = "||".join(
        "|".join(
            f"{s.session_id}:{s.timestamp.isoformat()}"
            for s in sorted(u.sessions, key=lambda x: x.timestamp)
        )
        for u in sorted(users, key=lambda u: u.user_id)
    )
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _cache_path(uid: str, cache_hash: str) -> Path:
    return CACHE_DIR / f"{uid}_{cache_hash}.json"


def _load_analysis_cache(uid: str, cache_hash: str) -> dict | None:
    path = _cache_path(uid, cache_hash)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_analysis_cache(uid: str, cache_hash: str, content: str, thinking: str) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    _cache_path(uid, cache_hash).write_text(
        json.dumps({"content": content, "thinking": thinking}),
        encoding="utf-8",
    )


# ── Session-state helpers ─────────────────────────────────────────────────────

def _init_state() -> None:
    if "selected_uid" not in st.session_state:
        st.session_state.selected_uid = None
    if "chats" not in st.session_state:
        st.session_state.chats = {}  # uid → chat dict


def _get_chat(uid: str) -> dict:
    """Return (creating if necessary) the chat state dict for a user."""
    if uid not in st.session_state.chats or not st.session_state.chats[uid]:
        st.session_state.chats[uid] = {
            "system":       "",    # full system prompt with timeline
            "messages":     [],    # display dicts: {role, content, thinking, type}
            "llm_messages": [],    # sent to LLM: [{role, content}]
            "done":         False, # True after initial analysis completes
        }
    return st.session_state.chats[uid]


# ── Dataset loader (cached) ───────────────────────────────────────────────────

@st.cache_data
def _load_users() -> list[User]:
    return load_dataset(str(DATASET_PATH))


@st.cache_resource
def _get_provider():
    return get_provider()


# ── JSON extraction helper ────────────────────────────────────────────────────

def _extract_json(text: str) -> str | None:
    """Extract JSON from ```json ... ``` fences, or return bare object if present."""
    if "```json" in text:
        start = text.index("```json") + len("```json")
        remainder = text[start:]
        end = remainder.index("```") if "```" in remainder else len(remainder)
        return remainder[:end].strip()
    stripped = text.strip()
    if stripped.startswith("{"):
        return stripped
    return None


# ── Pattern card renderer ─────────────────────────────────────────────────────

_CONF_ICON  = {"high": "🟢", "medium": "🟡", "low": "🔴"}
_CONF_BADGE = {"high": "HIGH ▲", "medium": "MED  ◆", "low": "LOW  ▼"}


def _parse_pattern_json(json_str: str) -> tuple[dict | None, str | None]:
    """Parse model pattern JSON and return a user-facing validation error if invalid."""
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}."

    if not isinstance(data, dict):
        return None, "Pattern output must be a JSON object."
    if "patterns" not in data or not isinstance(data["patterns"], list):
        return None, "Pattern output must contain a top-level `patterns` list."
    return data, None


def _session_lookup(user: User | None) -> dict[str, object]:
    """Map both full and short session IDs to Session objects for citation lookup."""
    if user is None:
        return {}

    lookup = {}
    for session in user.sessions:
        lookup[session.session_id] = session
        lookup[session.session_id.split("_")[-1]] = session
    return lookup


def _render_session_citation(sid: str, lookup: dict[str, object]) -> None:
    """Render a cited session with original transcript context when available."""
    session = lookup.get(sid)
    if session is None:
        st.code(sid, language=None)
        return

    label = session.session_id
    if hasattr(st, "popover"):
        with st.popover(label, use_container_width=True):
            st.caption(session.timestamp.strftime("%Y-%m-%d %a %H:%M"))
            st.markdown(f"**User:** {session.user_message}")
            if session.user_followup:
                st.markdown(f"**Follow-up:** {session.user_followup}")
            st.markdown(f"**Clary:** {session.clary_response}")
            st.caption(f"Severity: {session.severity or '?'} | Tags: {', '.join(session.tags)}")
    else:
        st.code(label, language=None)
        st.caption(session.timestamp.strftime("%Y-%m-%d %a %H:%M"))


def _render_patterns(json_str: str, user: User | None = None) -> bool:
    """Render interactive pattern cards from a JSON string."""
    data, error = _parse_pattern_json(json_str)
    if error:
        st.error("The model returned output that was not valid assignment JSON.")
        st.caption(error)
        st.info("Use Regenerate Analysis to ask the model for the required JSON format again.")
        st.code(json_str, language="json")
        return False

    patterns = data.get("patterns", [])
    if not patterns:
        st.warning("No patterns were detected in this analysis run.")
        return False

    sessions = _session_lookup(user)

    conf_counts = {}
    for p in patterns:
        c = p.get("confidence", "low").lower()
        conf_counts[c] = conf_counts.get(c, 0) + 1

    # Summary header
    cols = st.columns(4)
    cols[0].metric("Patterns found", len(patterns))
    cols[1].metric("🟢 High", conf_counts.get("high", 0))
    cols[2].metric("🟡 Medium", conf_counts.get("medium", 0))
    cols[3].metric("🔴 Low", conf_counts.get("low", 0))

    st.markdown("")

    for p in patterns:
        conf   = p.get("confidence", "low").lower()
        icon   = _CONF_ICON.get(conf, "⚪")
        badge  = _CONF_BADGE.get(conf, conf.upper())
        pid    = p.get("pattern_id", "?")
        title  = p.get("title", "Untitled pattern")

        with st.expander(f"{icon} **{pid}** · {title}  `{badge}`", expanded=True):
            st.markdown(
                f"<span data-confidence='{conf}' style='display:none'></span>",
                unsafe_allow_html=True,
            )
            left, right = st.columns([3, 2])

            with left:
                st.markdown("**📅 Temporal Reasoning**")
                st.markdown(p.get("temporal_reasoning", "—"))
                st.markdown("**🔍 Confidence Justification**")
                st.info(p.get("confidence_justification", "—"))

            with right:
                st.markdown("**📋 Sessions Involved**")
                for sid in p.get("sessions_involved", []):
                    _render_session_citation(sid, sessions)
                st.markdown("**🧠 Reasoning Trace**")
                for i, step in enumerate(p.get("trace", []), 1):
                    st.caption(f"{i}. {step}")

    # Pattern timeline visualisation
    if patterns and user is not None:
        with st.expander("📅 Pattern Timeline", expanded=True):
            _render_pattern_timeline(data, user, sessions)

    return True


# ── Plotly pattern timeline ───────────────────────────────────────────────────

def _render_pattern_timeline(patterns_data: dict, user: User, session_lookup: dict) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.caption("Install plotly (`pip install plotly`) to enable the pattern timeline.")
        return

    patterns = patterns_data.get("patterns", [])
    if not patterns:
        return

    conf_colors = {"high": "#10b981", "medium": "#f59e0b", "low": "#ef4444"}
    rows, x_vals, hovers, colors = [], [], [], []

    for p in patterns:
        conf  = p.get("confidence", "low").lower()
        color = conf_colors.get(conf, "#94a3b8")
        label = f"{p.get('pattern_id', '?')}: {p.get('title', '')[:48]}"
        for sid in p.get("sessions_involved", []):
            sess = session_lookup.get(sid)
            if sess:
                rows.append(label)
                x_vals.append(sess.timestamp)
                hovers.append(
                    f"<b>{sess.session_id}</b><br>"
                    f"{sess.timestamp.strftime('%b %d, %Y %H:%M')}<br>"
                    f"Severity: {sess.severity or '?'}<br>"
                    f"Tags: {', '.join(sess.tags[:3]) or '—'}"
                )
                colors.append(color)

    if not x_vals:
        st.caption("No session dates found for timeline.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=rows, mode="markers",
        marker=dict(size=14, color=colors, symbol="circle",
                    line=dict(width=1, color="rgba(255,255,255,0.2)")),
        hovertext=hovers, hoverinfo="text", showlegend=False,
    ))
    for conf, color in conf_colors.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=conf.capitalize(),
            marker=dict(size=10, color=color), showlegend=True,
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        font=dict(color="#e2e8f0", family="Inter, sans-serif", size=12),
        xaxis=dict(title="", gridcolor="rgba(255,255,255,0.05)", tickformat="%b %d",
                   tickfont=dict(size=11, color="#64748b"), zeroline=False),
        yaxis=dict(title="", gridcolor="rgba(255,255,255,0.04)",
                   tickfont=dict(size=10), autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=30),
        height=max(180, len(set(rows)) * 55 + 100),
        hoverlabel=dict(bgcolor="rgba(15,23,42,0.95)",
                        bordercolor="rgba(255,255,255,0.1)",
                        font=dict(color="#e2e8f0")),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── HTML report generator ─────────────────────────────────────────────────────

_SHARED_REPORT_CSS = (
    "*{margin:0;padding:0;box-sizing:border-box}"
    "body{background:#020617;color:#e2e8f0;font-family:'Segoe UI',system-ui,sans-serif;line-height:1.6}"
    ".container{max-width:900px;margin:0 auto;padding:2rem 1.5rem 4rem}"
    "header{border-bottom:1px solid rgba(255,255,255,.08);padding-bottom:1.5rem;margin-bottom:2rem}"
    ".logo{font-size:1rem;color:#38bdf8;font-weight:700;letter-spacing:.06em;margin-bottom:.4rem}"
    ".pname{font-size:1.7rem;font-weight:600;color:#f8fafc}"
    ".pmeta{color:#64748b;font-size:.82rem;margin-top:.3rem}"
    ".pnotes{color:#94a3b8;font-size:.8rem;margin-top:.5rem;font-style:italic}"
    ".summary{display:flex;gap:1rem;background:rgba(15,23,42,.5);border:1px solid rgba(255,255,255,.06);"
    "border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:2rem}"
    ".metric{flex:1;text-align:center}"
    ".mv{display:block;font-size:1.8rem;font-weight:700}"
    ".ml{display:block;font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em;margin-top:.1rem}"
    ".card{background:rgba(15,23,42,.4);border:1px solid rgba(255,255,255,.07);"
    "border-radius:14px;padding:1.4rem;margin-bottom:1.2rem;page-break-inside:avoid}"
    ".card-header{display:flex;align-items:baseline;gap:.7rem;margin-bottom:.9rem}"
    ".pid{font-size:.68rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.1em}"
    ".ptitle{font-size:1rem;font-weight:600;color:#f8fafc;flex:1}"
    ".conf{font-size:.76rem;font-weight:700}"
    ".two-col{display:grid;grid-template-columns:3fr 2fr;gap:1.5rem}"
    ".lbl{font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;"
    "font-weight:600;margin-bottom:.25rem;margin-top:.75rem}"
    ".body{color:#cbd5e1;font-size:.85rem;line-height:1.55}"
    ".cbox{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);"
    "border-radius:8px;padding:.55rem .8rem}"
    ".badges{display:flex;flex-wrap:wrap;gap:.3rem;margin-bottom:.2rem}"
    ".badge{background:rgba(30,41,59,.7);border:1px solid rgba(255,255,255,.08);border-radius:6px;"
    "padding:.12rem .5rem;font-size:.73rem;font-family:monospace;color:#7dd3fc}"
    ".trace{list-style:decimal;padding-left:1.1rem;color:#64748b;font-size:.79rem}"
    ".trace li{padding:.18rem 0}"
    "footer{margin-top:3rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,.06);"
    "text-align:center;color:#334155;font-size:.74rem}"
    "@media print{body{background:#fff;color:#111}}"
)


def _generate_html_report(patterns_data: dict, user: User, session_lookup: dict) -> str:
    patterns    = patterns_data.get("patterns", [])
    conf_colors = {"high": "#10b981", "medium": "#f59e0b", "low": "#ef4444"}
    conf_icons  = {"high": "🟢", "medium": "🟡", "low": "🔴"}

    cards_html = ""
    for p in patterns:
        conf  = p.get("confidence", "low").lower()
        color = conf_colors.get(conf, "#94a3b8")
        icon  = conf_icons.get(conf, "⚪")

        sessions_html = "".join(
            f'<span class="badge">{session_lookup[sid].session_id} · '
            f'{session_lookup[sid].timestamp.strftime("%b %d")}</span>'
            if sid in session_lookup else f'<span class="badge">{sid}</span>'
            for sid in p.get("sessions_involved", [])
        )
        trace_html = "".join(f"<li>{s}</li>" for s in p.get("trace", []))

        cards_html += (
            f'<div class="card" style="border-left:3px solid {color}">'
            f'<div class="card-header">'
            f'<span class="pid">{p.get("pattern_id","?")}</span>'
            f'<span class="ptitle">{p.get("title","")}</span>'
            f'<span class="conf" style="color:{color}">{icon} {conf.upper()}</span>'
            f'</div>'
            f'<div class="two-col">'
            f'<div><p class="lbl">Temporal Reasoning</p>'
            f'<p class="body">{p.get("temporal_reasoning","—")}</p>'
            f'<p class="lbl">Confidence Justification</p>'
            f'<p class="body cbox">{p.get("confidence_justification","—")}</p></div>'
            f'<div><p class="lbl">Sessions Involved</p>'
            f'<div class="badges">{sessions_html}</div>'
            f'<p class="lbl" style="margin-top:1rem">Reasoning Trace</p>'
            f'<ol class="trace">{trace_html}</ol></div>'
            f'</div></div>'
        )

    n_high = sum(1 for p in patterns if p.get("confidence","").lower() == "high")
    n_med  = sum(1 for p in patterns if p.get("confidence","").lower() == "medium")
    n_low  = sum(1 for p in patterns if p.get("confidence","").lower() == "low")

    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>Clary · {user.name} Report</title>'
        f'<style>{_SHARED_REPORT_CSS}</style></head><body>'
        f'<div class="container">'
        f'<header>'
        f'<div class="logo">🩺 CLARY · HEALTH PATTERN REPORT</div>'
        f'<div class="pname">{user.name}</div>'
        f'<div class="pmeta">{user.user_id} · {user.age}y · {user.gender}'
        f' · {user.occupation} · {user.location}</div>'
        f'<div class="pnotes">{user.onboarding_notes}</div>'
        f'</header>'
        f'<div class="summary">'
        f'<div class="metric"><span class="mv">{len(patterns)}</span><span class="ml">Patterns</span></div>'
        f'<div class="metric"><span class="mv" style="color:#10b981">{n_high}</span><span class="ml">High</span></div>'
        f'<div class="metric"><span class="mv" style="color:#f59e0b">{n_med}</span><span class="ml">Medium</span></div>'
        f'<div class="metric"><span class="mv" style="color:#ef4444">{n_low}</span><span class="ml">Low</span></div>'
        f'</div>'
        f'{cards_html}'
        f'<footer>Generated by Clary · Ask First · {datetime.now().strftime("%Y-%m-%d %H:%M")}</footer>'
        f'</div></body></html>'
    )


# ── Cross-patient renderers ───────────────────────────────────────────────────

_USER_COLORS = {"USR001": "#38bdf8", "USR002": "#a78bfa", "USR003": "#34d399"}


def _render_cross_patient_patterns(json_str: str, users: list[User]) -> bool:
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON from model: {exc.msg}")
        st.code(json_str, language="json")
        return False

    if not isinstance(data, dict) or "cross_patient_patterns" not in data:
        st.error("Expected `cross_patient_patterns` key in model output.")
        st.code(json_str, language="json")
        return False

    patterns = data.get("cross_patient_patterns", [])
    if not patterns:
        st.warning("No cross-patient patterns detected.")
        return False

    combined_lookup: dict = {}
    for u in users:
        combined_lookup.update(_session_lookup(u))

    conf_counts: dict = {}
    for p in patterns:
        c = p.get("confidence", "low").lower()
        conf_counts[c] = conf_counts.get(c, 0) + 1

    cols = st.columns(4)
    cols[0].metric("Cross-patient patterns", len(patterns))
    cols[1].metric("🟢 High",   conf_counts.get("high",   0))
    cols[2].metric("🟡 Medium", conf_counts.get("medium", 0))
    cols[3].metric("🔴 Low",    conf_counts.get("low",    0))
    st.markdown("")

    for p in patterns:
        conf  = p.get("confidence", "low").lower()
        icon  = _CONF_ICON.get(conf, "⚪")
        badge = _CONF_BADGE.get(conf, conf.upper())
        pid   = p.get("pattern_id", "?")
        title = p.get("title", "Untitled")

        with st.expander(f"{icon} **{pid}** · {title}  `{badge}`", expanded=True):
            st.markdown(
                f"<span data-confidence='{conf}' style='display:none'></span>",
                unsafe_allow_html=True,
            )
            _affected_parts = []
            for _ua in p.get("users_affected", []):
                _uc = _USER_COLORS.get(_ua.split()[0], "#94a3b8")
                _affected_parts.append(
                    f"<span style='background:rgba(30,41,59,.6);border:1px solid rgba(255,255,255,.1);"
                    f"border-radius:6px;padding:.15rem .6rem;font-size:.8rem;color:{_uc}'>{_ua}</span>"
                )
            affected_html = " ".join(_affected_parts)
            st.markdown(
                f"<div style='margin-bottom:.8rem'>👥 <strong>Patients:</strong> {affected_html}</div>",
                unsafe_allow_html=True,
            )
            left, right = st.columns([3, 2])
            with left:
                st.markdown("**📅 Temporal Reasoning**")
                st.markdown(p.get("temporal_reasoning", "—"))
                st.markdown("**🔍 Confidence Justification**")
                st.info(p.get("confidence_justification", "—"))
            with right:
                st.markdown("**📋 Sessions Involved**")
                for sid in p.get("sessions_involved", []):
                    _render_session_citation(sid, combined_lookup)
                st.markdown("**🧠 Reasoning Trace**")
                for i, step in enumerate(p.get("trace", []), 1):
                    st.caption(f"{i}. {step}")

    with st.expander("📅 Cross-Patient Pattern Timeline", expanded=True):
        _render_cross_patient_timeline(data, users, combined_lookup)

    return True


def _render_cross_patient_timeline(
    patterns_data: dict, users: list[User], session_lookup: dict
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.caption("Install plotly to enable the timeline.")
        return

    patterns      = patterns_data.get("cross_patient_patterns", [])
    user_name_map = {u.user_id: u.name for u in users}

    fig = go.Figure()
    for uid, color in _USER_COLORS.items():
        name = user_name_map.get(uid, uid)
        x_vals, y_vals, hovers = [], [], []
        for p in patterns:
            label = f"{p.get('pattern_id','?')}: {p.get('title','')[:40]}"
            for sid in p.get("sessions_involved", []):
                if not sid.startswith(uid):
                    continue
                sess = session_lookup.get(sid)
                if sess:
                    x_vals.append(sess.timestamp)
                    y_vals.append(label)
                    hovers.append(
                        f"<b>{name}</b> · {sess.session_id}<br>"
                        f"{sess.timestamp.strftime('%b %d, %Y')}<br>"
                        f"Tags: {', '.join(sess.tags[:3]) or '—'}"
                    )
        if x_vals:
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode="markers", name=name,
                marker=dict(size=14, color=color, symbol="circle",
                            line=dict(width=1, color="rgba(255,255,255,.2)")),
                hovertext=hovers, hoverinfo="text",
            ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        font=dict(color="#e2e8f0", family="Inter, sans-serif", size=12),
        xaxis=dict(title="", gridcolor="rgba(255,255,255,0.05)", tickformat="%b %d",
                   tickfont=dict(size=11, color="#64748b"), zeroline=False),
        yaxis=dict(title="", gridcolor="rgba(255,255,255,0.04)",
                   tickfont=dict(size=10), autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=30),
        height=max(180, len(patterns) * 60 + 100),
        hoverlabel=dict(bgcolor="rgba(15,23,42,.95)",
                        bordercolor="rgba(255,255,255,.1)",
                        font=dict(color="#e2e8f0")),
    )
    st.plotly_chart(fig, use_container_width=True)


def _generate_cp_html_report(patterns_data: dict, users: list[User]) -> str:
    patterns    = patterns_data.get("cross_patient_patterns", [])
    conf_colors = {"high": "#10b981", "medium": "#f59e0b", "low": "#ef4444"}
    conf_icons  = {"high": "🟢", "medium": "🟡", "low": "🔴"}

    combined_lookup: dict = {}
    for u in users:
        combined_lookup.update(_session_lookup(u))

    cards_html = ""
    for p in patterns:
        conf     = p.get("confidence", "low").lower()
        color    = conf_colors.get(conf, "#94a3b8")
        icon     = conf_icons.get(conf, "⚪")
        affected = " · ".join(p.get("users_affected", []))

        sessions_html = "".join(
            f'<span class="badge">{combined_lookup[sid].session_id} · '
            f'{combined_lookup[sid].timestamp.strftime("%b %d")}</span>'
            if sid in combined_lookup else f'<span class="badge">{sid}</span>'
            for sid in p.get("sessions_involved", [])
        )
        trace_html = "".join(f"<li>{s}</li>" for s in p.get("trace", []))

        cards_html += (
            f'<div class="card" style="border-left:3px solid {color}">'
            f'<div class="card-header">'
            f'<span class="pid">{p.get("pattern_id","?")}</span>'
            f'<span class="ptitle">{p.get("title","")}</span>'
            f'<span class="conf" style="color:{color}">{icon} {conf.upper()}</span>'
            f'</div>'
            f'<p class="lbl">Patients Affected</p>'
            f'<p class="body" style="margin-bottom:.8rem">👥 {affected}</p>'
            f'<div class="two-col">'
            f'<div><p class="lbl">Temporal Reasoning</p>'
            f'<p class="body">{p.get("temporal_reasoning","—")}</p>'
            f'<p class="lbl">Confidence Justification</p>'
            f'<p class="body cbox">{p.get("confidence_justification","—")}</p></div>'
            f'<div><p class="lbl">Sessions Involved</p>'
            f'<div class="badges">{sessions_html}</div>'
            f'<p class="lbl" style="margin-top:1rem">Reasoning Trace</p>'
            f'<ol class="trace">{trace_html}</ol></div>'
            f'</div></div>'
        )

    n_high     = sum(1 for p in patterns if p.get("confidence","").lower() == "high")
    n_med      = sum(1 for p in patterns if p.get("confidence","").lower() == "medium")
    n_low      = sum(1 for p in patterns if p.get("confidence","").lower() == "low")
    user_names = ", ".join(u.name for u in users)

    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        f'<title>Clary · Population Report</title>'
        f'<style>{_SHARED_REPORT_CSS}</style></head><body>'
        f'<div class="container">'
        f'<header>'
        f'<div class="logo">🩺 CLARY · POPULATION ANALYSIS REPORT</div>'
        f'<div class="pname">All Patients</div>'
        f'<div class="pmeta">{user_names} · {len(users)} patients · Cross-patient pattern detection</div>'
        f'</header>'
        f'<div class="summary">'
        f'<div class="metric"><span class="mv">{len(patterns)}</span><span class="ml">Patterns</span></div>'
        f'<div class="metric"><span class="mv" style="color:#10b981">{n_high}</span><span class="ml">High</span></div>'
        f'<div class="metric"><span class="mv" style="color:#f59e0b">{n_med}</span><span class="ml">Medium</span></div>'
        f'<div class="metric"><span class="mv" style="color:#ef4444">{n_low}</span><span class="ml">Low</span></div>'
        f'</div>'
        f'{cards_html}'
        f'<footer>Generated by Clary · Ask First · {datetime.now().strftime("%Y-%m-%d %H:%M")}</footer>'
        f'</div></body></html>'
    )


# ── Data-ingestion animation ──────────────────────────────────────────────────

def _animate_data_load(user: User) -> None:
    """
    Show each session being inserted in chronological order with temporal labels.
    This visually represents the temporal indexing that underpins Clary's reasoning.
    """
    sessions = sorted(user.sessions, key=lambda s: s.timestamp)
    ref = compute_reference_date(sessions)

    with st.status(
        f"📥  Ingesting temporal data for **{user.name}** ({len(sessions)} sessions)…",
        expanded=True,
    ) as status:
        for s in sessions:
            week = get_week_number(s, ref)
            day  = get_day_delta(s, ref)
            tod  = get_time_of_day_label(s)
            tags = " · ".join(s.tags[:3]) if s.tags else "—"

            st.markdown(
                f"<span style='font-family:monospace;font-size:0.82em;"
                f"color:#005f73'>[W{week:02d}/D{day:03d}]</span>&nbsp;"
                f"<strong style='color:#c9d1d9'>{s.session_id}</strong>&nbsp;"
                f"<span style='color:#4a5568;font-size:0.82em'>"
                f"{s.timestamp.strftime('%b %d, %Y')} · {tod[:9]}"
                f" · sev:{s.severity or '?'} · {tags}</span>",
                unsafe_allow_html=True,
            )
            time.sleep(0.14)

        status.update(
            label=f"✅  {len(sessions)} sessions ingested · Temporal index ready",
            state="complete",
            expanded=False,
        )
        time.sleep(0.25)


# ── Core streaming render ─────────────────────────────────────────────────────

def _render_assignment_checklist() -> None:
    st.markdown(
        "<p style='font-size:0.8em;color:#8b949e;margin:0 0 8px'>FEATURES</p>",
        unsafe_allow_html=True,
    )
    st.caption("[x] Streamlit interface")
    st.caption("[x] Chat streaming")
    st.caption("[x] Per-patient session ingestion")
    st.caption("[x] Temporal history reasoning")
    st.caption("[x] JSON + HTML report export")
    st.caption("[x] Confidence scoring & trace")
    st.caption("[x] No hardcoded patterns")
    st.caption("[x] Analysis cache (page-reload safe)")
    st.caption("[x] Pattern timeline (Plotly)")
    st.caption("[x] Cross-patient population view")


def _render_reasoning_panel(user: User) -> None:
    with st.expander("How Clary reasons", expanded=False):
        st.markdown(
            f"""
1. Loads only **{user.name}'s** sessions and sorts them chronologically.
2. Adds week numbers, day deltas, dates, time-of-day labels, severity, and tags.
3. Sends the full temporal timeline to the configured LLM in one reasoning call.
4. Asks the model to check recurrence, temporal direction, negative evidence, resolution, and cascades.
5. Requires JSON patterns with confidence scores and trace steps.
"""
        )


def _render_raw_timeline(user: User) -> None:
    """Expose the exact temporal timeline used by the model."""
    with st.expander("View raw temporal timeline sent to Clary", expanded=False):
        st.code(build_temporal_timeline(user), language="text")


def _render_analysis_controls(user: User, content: str) -> None:
    """Render download (JSON + HTML) and regenerate actions for the selected user's analysis."""
    json_str = _extract_json(content) if content else None
    col1, col2, col3 = st.columns([1, 1, 1])

    data: dict | None = None
    if json_str:
        parsed, err = _parse_pattern_json(json_str)
        if not err:
            data = parsed
    sessions = _session_lookup(user)

    with col1:
        if data:
            st.download_button(
                "⬇ Download JSON",
                data=json.dumps(data, indent=2),
                file_name=f"{user.user_id}_patterns.json",
                mime="application/json",
                use_container_width=True,
                key=f"download_json_{user.user_id}",
            )
        else:
            st.button("⬇ Download JSON", disabled=True, use_container_width=True,
                      key=f"download_json_{user.user_id}")

    with col2:
        if data:
            html_report = _generate_html_report(data, user, sessions)
            st.download_button(
                "⬇ Download HTML",
                data=html_report,
                file_name=f"{user.user_id}_report.html",
                mime="text/html",
                use_container_width=True,
                key=f"download_html_{user.user_id}",
            )
        else:
            st.button("⬇ Download HTML", disabled=True, use_container_width=True,
                      key=f"download_html_{user.user_id}")

    with col3:
        if st.button("↺ Regenerate", use_container_width=True, key=f"regen_{user.user_id}"):
            st.session_state.chats[user.user_id] = {}
            st.rerun()


def _render_cp_analysis_controls(users: list[User], content: str) -> None:
    """Render download (JSON + HTML) for cross-patient analysis."""
    json_str = _extract_json(content) if content else None
    col1, col2 = st.columns([1, 1])

    data: dict | None = None
    if json_str:
        try:
            parsed = json.loads(json_str)
            if "cross_patient_patterns" in parsed:
                data = parsed
        except Exception:
            pass

    with col1:
        if data:
            st.download_button(
                "⬇ Download JSON",
                data=json.dumps(data, indent=2),
                file_name="population_patterns.json",
                mime="application/json",
                use_container_width=True,
                key="download_json_ALL",
            )
        else:
            st.button("⬇ Download JSON", disabled=True, use_container_width=True,
                      key="download_json_ALL")

    with col2:
        if data:
            html_report = _generate_cp_html_report(data, users)
            st.download_button(
                "⬇ Download HTML",
                data=html_report,
                file_name="population_report.html",
                mime="text/html",
                use_container_width=True,
                key="download_html_ALL",
            )
        else:
            st.button("⬇ Download HTML", disabled=True, use_container_width=True,
                      key="download_html_ALL")


def _do_stream(chat: dict, is_initial: bool = False) -> tuple[str, str]:
    """
    Stream the next LLM response inside the **current** st.chat_message context.
    Updates live placeholders for thinking and content.
    Returns (full_content_text, full_thinking_text).
    """
    provider = _get_provider()

    thinking_ph = st.empty()
    content_ph  = st.empty()

    thinking = ""
    content  = ""

    for chunk_type, chunk in stream_response(
        chat["system"], chat["llm_messages"], provider
    ):
        if chunk_type == "thinking":
            thinking += chunk
            thinking_ph.markdown(
                f"<details open><summary>💭 Clary is reasoning… "
                f"({len(thinking.split())} words)</summary>"
                f"<pre>{thinking[-700:]}</pre></details>",
                unsafe_allow_html=True,
            )
        elif chunk_type == "content":
            content += chunk
            if is_initial:
                content_ph.markdown(
                    "<span style='color:#6e7681;font-style:italic'>"
                    "⬛ Generating structured analysis…</span>",
                    unsafe_allow_html=True,
                )
            else:
                content_ph.markdown(content + " ⬛")
        elif chunk_type == "error":
            content_ph.error(f"❌ Provider error: {chunk}")
            return "", thinking

    # Finalise
    content_ph.empty()
    if thinking:
        thinking_ph.markdown(
            f"<details><summary>💭 Reasoning trace "
            f"({len(thinking.split())} words) — click to expand</summary>"
            f"<pre>{thinking}</pre></details>",
            unsafe_allow_html=True,
        )
    else:
        thinking_ph.empty()

    return content, thinking


# ── Message history renderer (for chat replay) ────────────────────────────────

def _display_past_message(msg: dict, user: User | None = None) -> None:
    """Render a previously stored message (user or assistant)."""
    role    = msg["role"]
    content = msg["content"]
    thinking = msg.get("thinking", "")
    mtype   = msg.get("type", "chat")

    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
        return

    with st.chat_message("assistant", avatar="🩺"):
        if thinking:
            st.markdown(
                f"<details><summary>💭 Reasoning trace "
                f"({len(thinking.split())} words) — click to expand</summary>"
                f"<pre>{thinking}</pre></details>",
                unsafe_allow_html=True,
            )
        if mtype == "analysis":
            json_str = _extract_json(content)
            if json_str:
                _render_patterns(json_str, user)
                if user is not None:
                    _render_analysis_controls(user, content)
            else:
                st.markdown(content)
                if user is not None:
                    _render_analysis_controls(user, content)
        else:
            st.markdown(content)


# ── Follow-up handler ─────────────────────────────────────────────────────────

def _handle_followup(chat: dict, user_input: str) -> None:
    """Append a user turn, stream Clary's response, and persist both."""
    # Display user bubble
    with st.chat_message("user"):
        st.markdown(user_input)

    # Update state
    chat["messages"].append({"role": "user", "content": user_input})
    chat["llm_messages"].append({"role": "user", "content": user_input})

    # Stream Clary's reply
    with st.chat_message("assistant", avatar="🩺"):
        content, thinking = _do_stream(chat, is_initial=False)

    if content:
        chat["messages"].append({
            "role":     "assistant",
            "content":  content,
            "thinking": thinking,
            "type":     "chat",
        })
        chat["llm_messages"].append({"role": "assistant", "content": content})

    st.rerun()


# ── All-patients view ────────────────────────────────────────────────────────

def _render_all_patients_view(users: list[User], chat: dict) -> None:
    st.markdown(
        "<h2 style='color:#e6edf3;margin-bottom:2px'>🔬 Population Analysis</h2>"
        "<p style='color:#6e7681;margin:0;font-size:.85em'>"
        "Cross-patient pattern detection across all patients</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    all_hash = _compute_all_hash(users)

    # Restore from file cache if available
    if not chat["done"]:
        cached = _load_analysis_cache("ALL", all_hash)
        if cached:
            chat["system"] = build_all_patients_system_prompt(users)
            init_msg = build_cross_patient_user_message(users)
            chat["llm_messages"] = [
                {"role": "user",      "content": init_msg},
                {"role": "assistant", "content": cached["content"]},
            ]
            chat["messages"] = [{
                "role":     "assistant",
                "content":  cached["content"],
                "thinking": cached.get("thinking", ""),
                "type":     "cp_analysis",
            }]
            chat["done"] = True
            st.rerun()

    if not chat["done"]:
        chat["system"] = build_all_patients_system_prompt(users)
        for u in users:
            _animate_data_load(u)
        st.divider()

        st.markdown(
            "<p style='color:#8b949e;font-size:.85em;margin-bottom:4px'>"
            "🔬&nbsp; CROSS-PATIENT ANALYSIS — streaming</p>",
            unsafe_allow_html=True,
        )

        init_msg = build_cross_patient_user_message(users)
        chat["llm_messages"] = [{"role": "user", "content": init_msg}]

        with st.chat_message("assistant", avatar="🩺"):
            content, thinking = _do_stream(chat, is_initial=True)

        if content:
            json_str = _extract_json(content)
            if json_str:
                _render_cross_patient_patterns(json_str, users)
                _render_cp_analysis_controls(users, content)
            else:
                st.markdown(content)
                st.warning(
                    "Cross-patient JSON not detected. You can still ask follow-up questions."
                )
                _render_cp_analysis_controls(users, content)

            chat["messages"] = [{
                "role":     "assistant",
                "content":  content,
                "thinking": thinking,
                "type":     "cp_analysis",
            }]
            chat["llm_messages"].append({"role": "assistant", "content": content})
            chat["done"] = True
            _save_analysis_cache("ALL", all_hash, content, thinking)

        st.divider()
        st.success("✅ Population analysis complete. Ask Clary about cross-patient findings.")

        prompt = st.chat_input("Ask about cross-patient patterns…", key="chat_input_ALL")
        if prompt:
            _handle_followup(chat, prompt)

    else:
        for msg in chat["messages"]:
            if msg.get("type") == "cp_analysis":
                with st.chat_message("assistant", avatar="🩺"):
                    if msg.get("thinking"):
                        st.markdown(
                            f"<details><summary>💭 Reasoning trace "
                            f"({len(msg['thinking'].split())} words) — click to expand</summary>"
                            f"<pre>{msg['thinking']}</pre></details>",
                            unsafe_allow_html=True,
                        )
                    json_str = _extract_json(msg["content"])
                    if json_str:
                        _render_cross_patient_patterns(json_str, users)
                        _render_cp_analysis_controls(users, msg["content"])
                    else:
                        st.markdown(msg["content"])
                        _render_cp_analysis_controls(users, msg["content"])
            else:
                _display_past_message(msg, None)

        prompt = st.chat_input("Ask about cross-patient patterns…", key="chat_input_ALL")
        if prompt:
            _handle_followup(chat, prompt)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar(users: list[User]) -> None:
    with st.sidebar:
        st.markdown(
            "<h2 style='color:#00b4d8;margin:0;font-size:1.4rem'>🩺 Clary</h2>",
            unsafe_allow_html=True,
        )
        st.caption("Ask First · Health Pattern Analyst")
        st.divider()

        st.markdown(
            "<p style='font-size:0.8em;color:#8b949e;margin:0 0 8px'>SELECT PATIENT</p>",
            unsafe_allow_html=True,
        )

        for u in users:
            chat_state = st.session_state.chats.get(u.user_id, {})
            is_done     = chat_state.get("done", False)
            n_questions = len([m for m in chat_state.get("messages", []) if m["role"] == "user"])
            is_selected = st.session_state.selected_uid == u.user_id

            suffix = ""
            if is_done and n_questions > 0:
                suffix = f"  · {n_questions}Q"
            elif is_done:
                suffix = "  · ✓"

            if st.button(
                f"**{u.name}** ({u.user_id}){suffix}",
                key=f"sel_{u.user_id}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                if st.session_state.selected_uid != u.user_id:
                    st.session_state.selected_uid = u.user_id
                    st.rerun()

        st.markdown(
            "<p style='font-size:0.8em;color:#8b949e;margin:12px 0 8px'>POPULATION</p>",
            unsafe_allow_html=True,
        )
        _is_all = st.session_state.selected_uid == "ALL"
        _all_cs = st.session_state.chats.get("ALL", {})
        _all_sfx = "  · ✓" if _all_cs.get("done") else ""
        if st.button(
            f"🔬 **All Patients**{_all_sfx}",
            key="sel_ALL",
            use_container_width=True,
            type="primary" if _is_all else "secondary",
        ):
            if st.session_state.selected_uid != "ALL":
                st.session_state.selected_uid = "ALL"
                st.rerun()

        # ── Selected user profile ──────────────────────────────────────────
        st.divider()
        _render_assignment_checklist()

        if st.session_state.selected_uid == "ALL":
            if _all_cs.get("done"):
                st.divider()
                if st.button("🔄 Reset Population Analysis", use_container_width=True):
                    st.session_state.chats["ALL"] = {}
                    st.rerun()

        elif st.session_state.selected_uid:
            uid   = st.session_state.selected_uid
            umap  = {u.user_id: u for u in users}
            user  = umap[uid]
            chat_state = st.session_state.chats.get(uid, {})

            st.divider()
            st.markdown(
                f"<p style='font-size:0.8em;color:#8b949e;margin:0 0 4px'>PATIENT PROFILE</p>"
                f"<strong style='color:#e6edf3'>{user.name}</strong>, {user.age}y · {user.gender}",
                unsafe_allow_html=True,
            )
            st.caption(f"📍 {user.location}")
            st.caption(f"💼 {user.occupation}")
            st.caption(user.onboarding_notes)

            sessions = sorted(user.sessions, key=lambda s: s.timestamp)
            ref = compute_reference_date(sessions)

            st.markdown(
                f"<p style='font-size:0.8em;color:#8b949e;margin:8px 0 4px'>"
                f"SESSION TIMELINE ({len(sessions)} sessions)</p>",
                unsafe_allow_html=True,
            )
            for s in sessions:
                w = get_week_number(s, ref)
                d = get_day_delta(s, ref)
                tags = ", ".join(s.tags[:2]) if s.tags else "—"
                st.markdown(
                    f"<div style='font-size:0.72em;color:#8b949e;line-height:1.7;"
                    f"border-left:2px solid #1f2937;padding-left:8px;margin-bottom:2px'>"
                    f"<span style='color:#005f73;font-family:monospace'>W{w:02d}/D{d:03d}</span>"
                    f" · {s.timestamp.strftime('%b %d')}"
                    f" · <code style='font-size:0.85em'>{s.session_id[-4:]}</code>"
                    f" · {tags}</div>",
                    unsafe_allow_html=True,
                )

            # Reset button (only if analysis has been run)
            if chat_state.get("done"):
                st.divider()
                if st.button("🔄 Reset Analysis", use_container_width=True):
                    st.session_state.chats[uid] = {}
                    st.rerun()


# ── Welcome screen ────────────────────────────────────────────────────────────

def _render_welcome(users: list[User]) -> None:
    st.markdown(
        "<div style='padding: 2rem 0 1rem 0; text-align: center;'><h1 style='font-size:3.5rem; margin-bottom:0;'><span style='font-size: 1.1em; vertical-align: text-bottom;'>🩺</span> <span class='gradient-text'>Clary</span></h1><p style='color:#94a3b8; font-size:1.2rem; margin-top:0.5rem; font-family:\"Outfit\", sans-serif; letter-spacing: 0.5px;'>Intelligent Health Pattern Analyst</p></div>",
        unsafe_allow_html=True,
    )
    
    st.markdown(
        "<div style='background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 16px; padding: 2rem; margin: 1rem 0 2.5rem 0; backdrop-filter: blur(8px); text-align: center; font-size: 1.1em; color: #cbd5e1; line-height: 1.7;'><p style='margin-bottom: 1.5rem; color: #f8fafc; font-weight: 500;'>Please select a patient from the sidebar to begin analysis.</p><div style='display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem; font-size: 0.9em; text-align: left;'><div style='flex: 1; min-width: 200px; background: rgba(0,0,0,0.2); padding: 1.2rem; border-radius: 12px;'><span style='font-size: 1.5em; display: block; margin-bottom: 0.5rem;'>⏱️</span><strong>Temporal Reasoning</strong><br><span style='color: #94a3b8;'>Analyzes events across weeks using precise day-deltas.</span></div><div style='flex: 1; min-width: 200px; background: rgba(0,0,0,0.2); padding: 1.2rem; border-radius: 12px;'><span style='font-size: 1.5em; display: block; margin-bottom: 0.5rem;'>🧠</span><strong>Causal Analysis</strong><br><span style='color: #94a3b8;'>Checks negative evidence and cascades, not just keywords.</span></div><div style='flex: 1; min-width: 200px; background: rgba(0,0,0,0.2); padding: 1.2rem; border-radius: 12px;'><span style='font-size: 1.5em; display: block; margin-bottom: 0.5rem;'>💬</span><strong>Persistent Context</strong><br><span style='color: #94a3b8;'>Ask follow-up questions using the full chronological history.</span></div></div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='font-family:\"Outfit\", sans-serif; font-size:0.95em; color:#94a3b8; font-weight: 600; letter-spacing: 1px; margin-bottom: 1rem;'>PATIENT COHORT</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(len(users))
    for col, u in zip(cols, users):
        sessions = sorted(u.sessions, key=lambda s: s.timestamp)
        with col:
            st.markdown(
                f"<div style='background:rgba(15,23,42,0.4);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:1.5rem;transition:transform 0.2s ease, box-shadow 0.2s ease;cursor:default;' onmouseover=\"this.style.transform='translateY(-4px)'; this.style.boxShadow='0 10px 25px rgba(0,0,0,0.2)';\" onmouseout=\"this.style.transform='translateY(0)'; this.style.boxShadow='none';\"><div style='display:flex;justify-content:space-between;align-items:flex-start;'><div><div style='color:#f8fafc;font-weight:600;font-size:1.2em;font-family:\"Outfit\",sans-serif;'>{u.name}</div><div style='color:#64748b;font-size:0.85em;margin-top:2px;'><code style='background:transparent;border:none;padding:0;'>{u.user_id}</code> · {u.age}y</div></div><div style='background:rgba(56,189,248,0.1);color:#38bdf8;border-radius:8px;padding:0.4rem 0.8rem;text-align:center;'><div style='font-size:1.2rem;font-weight:700;font-family:\"Outfit\",sans-serif;line-height:1;'>{len(sessions)}</div><div style='font-size:0.65em;font-weight:600;text-transform:uppercase;margin-top:2px;'>Sessions</div></div></div><div style='margin-top:1.2rem;padding-top:1.2rem;border-top:1px solid rgba(255,255,255,0.05);'><div style='color:#94a3b8;font-size:0.8em;margin-bottom:0.5rem;'><span style='color:#475569;'>Timeline:</span><br>{sessions[0].timestamp.strftime('%b %d')} – {sessions[-1].timestamp.strftime('%b %d, %Y')}</div><div style='color:#cbd5e1;font-size:0.85em;line-height:1.5;'>\"{u.onboarding_notes[:90]}…\"</div></div></div>",
                unsafe_allow_html=True,
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_state()

    if not os.getenv("LLM_API_KEY"):
        st.warning(
            "⚠️ No `LLM_API_KEY` found. Copy `.env.example` to `.env` and set your key.",
            icon="🔑",
        )

    try:
        users = _load_users()
    except FileNotFoundError:
        st.error(f"Dataset not found at: `{DATASET_PATH}`")
        st.stop()

    user_map = {u.user_id: u for u in users}

    _render_sidebar(users)

    # ── Welcome screen ────────────────────────────────────────────────────────
    if not st.session_state.selected_uid:
        _render_welcome(users)
        return

    # ── All-patients population view ──────────────────────────────────────────
    uid = st.session_state.selected_uid
    if uid == "ALL":
        chat = _get_chat("ALL")
        _render_all_patients_view(users, chat)
        return

    # ── Single patient selected ───────────────────────────────────────────────
    user = user_map[uid]
    chat = _get_chat(uid)

    # Header
    st.markdown(
        f"<h2 style='color:#e6edf3;margin-bottom:2px'>{user.name} "
        f"<span style='color:#4a5568;font-size:0.6em;font-weight:400'>{user.user_id}</span></h2>"
        f"<p style='color:#6e7681;margin:0;font-size:0.85em'>"
        f"{user.occupation} · {user.location} · Age {user.age}</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Phase 1: Data ingestion + initial analysis ────────────────────────────
    _render_reasoning_panel(user)
    _render_raw_timeline(user)

    # ── File-based cache restore (survives page reload) ───────────────────────
    _user_hash = _compute_user_hash(user)
    if not chat["done"]:
        _cached = _load_analysis_cache(uid, _user_hash)
        if _cached:
            chat["system"] = build_system_prompt(user)
            _init_msg = build_initial_user_message(user)
            chat["llm_messages"] = [
                {"role": "user",      "content": _init_msg},
                {"role": "assistant", "content": _cached["content"]},
            ]
            chat["messages"] = [{
                "role":     "assistant",
                "content":  _cached["content"],
                "thinking": _cached.get("thinking", ""),
                "type":     "analysis",
            }]
            chat["done"] = True
            st.rerun()

    if not chat["done"]:
        # Build and cache the system prompt (full timeline embedded)
        chat["system"] = build_system_prompt(user)

        # Animated session-by-session data ingestion
        _animate_data_load(user)
        st.divider()

        # Clary header
        st.markdown(
            "<p style='color:#8b949e;font-size:0.85em;margin-bottom:4px'>"
            "🧠&nbsp; INITIAL PATTERN ANALYSIS — streaming</p>",
            unsafe_allow_html=True,
        )

        # Build the first LLM turn
        initial_msg = build_initial_user_message(user)
        chat["llm_messages"] = [{"role": "user", "content": initial_msg}]

        # Stream the initial analysis
        with st.chat_message("assistant", avatar="🩺"):
            content, thinking = _do_stream(chat, is_initial=True)

        if content:
            # Render pattern cards immediately below the chat bubble
            json_str = _extract_json(content)
            if json_str:
                _render_patterns(json_str, user)
                _render_analysis_controls(user, content)
            else:
                st.markdown(content)
                st.warning(
                    "Pattern JSON not detected. The model may have responded in prose. "
                    "You can still ask follow-up questions."
                )
                _render_analysis_controls(user, content)

            # Persist to state and save to file cache
            chat["messages"].append({
                "role":     "assistant",
                "content":  content,
                "thinking": thinking,
                "type":     "analysis",
            })
            chat["llm_messages"].append({"role": "assistant", "content": content})
            chat["done"] = True
            _save_analysis_cache(uid, _user_hash, content, thinking)

        st.divider()
        st.success(
            "✅ Analysis complete. Ask Clary any follow-up question about "
            f"{user.name}'s patterns, sessions, or health history."
        )

        # Chat input available immediately — no rerun needed after initial analysis
        prompt = st.chat_input(
            f"Ask Clary about {user.name}'s health patterns…",
            key=f"chat_input_{uid}",
        )
        if prompt:
            _handle_followup(chat, prompt)

    # ── Phase 2: Ongoing chat ─────────────────────────────────────────────────
    else:
        # Replay all stored messages
        for msg in chat["messages"]:
            _display_past_message(msg, user)

        # Chat input for new questions
        prompt = st.chat_input(
            f"Ask Clary about {user.name}'s health patterns…",
            key=f"chat_input_{uid}",
        )
        if prompt:
            _handle_followup(chat, prompt)


if __name__ == "__main__":
    main()
