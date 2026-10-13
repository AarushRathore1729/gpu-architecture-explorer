"""Optional hosted Streamlit wrapper for the standalone interactive report."""
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from analysis.report import to_html


st.set_page_config(page_title="GPU Architecture Explorer", page_icon="📊", layout="wide")
st.title("GPU Architecture Explorer")
uploaded = st.file_uploader("Load benchmark run.json", type="json")
default = Path("results/run.json")
try:
    data = json.load(uploaded) if uploaded else (json.loads(default.read_text()) if default.exists() else None)
    components.html(to_html(data), height=1250, scrolling=True)
except (ValueError, KeyError, TypeError) as exc:
    st.error(f"Invalid measurement file: {exc}")
