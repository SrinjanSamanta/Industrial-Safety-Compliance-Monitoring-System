"""
Add Streamlit web dashboard for real-time monitoring and incident analytics
SafeGuardAI - Streamlit Dashboard
Run:  streamlit run app.py
"""

import sqlite3
import time
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).parent / "safeguard.db"
REFRESH_SECONDS = 5


def load_data() -> pd.DataFrame:
    """Read all violations into a DataFrame (creates the table if missing)."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS violations (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   timestamp TEXT NOT NULL,
                   event_type TEXT NOT NULL,
                   details TEXT
               )"""
        )
        return pd.read_sql_query(
            "SELECT id, timestamp, event_type, details "
            "FROM violations ORDER BY id DESC",
            conn,
        )
    finally:
        conn.close()


# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(page_title="SafeGuardAI Dashboard", layout="wide")
st.title("🛡️ SafeGuardAI — Violation Dashboard")

# ── Controls ─────────────────────────────────────────────────────────
c1, c2 = st.columns([1, 5])
if c1.button("🔄 Refresh data"):
    st.rerun()
auto = c2.toggle(f"Auto-refresh every {REFRESH_SECONDS}s", value=False)

# ── Load data ────────────────────────────────────────────────────────
df = load_data()

if df.empty:
    st.info("No violations logged yet. Run **main.py** to start detecting.")
else:
    # ── Metrics ──────────────────────────────────────────────────────
    total = len(df)
    hazard_count = int((df["event_type"] == "HAZARD_BREACH").sum())
    fall_count = int((df["event_type"] == "FALL_EMERGENCY").sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Violations", total)
    m2.metric("Hazard Breaches", hazard_count)
    m3.metric("Fall Emergencies", fall_count)

    # ── Chart ────────────────────────────────────────────────────────
    st.subheader("Violations by Type")
    st.bar_chart(df["event_type"].value_counts())

    # ── Recent incidents ─────────────────────────────────────────────
    st.subheader("Recent Incidents")
    st.dataframe(df.head(50), hide_index=True)

# ── Auto-refresh ─────────────────────────────────────────────────────
if auto:
    time.sleep(REFRESH_SECONDS)
    st.rerun()
