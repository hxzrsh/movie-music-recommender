import ast
import json
import random
import urllib.parse
import urllib.request
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Movie & Music Recommendation System", page_icon="●", layout="wide")

# --- 2. PERSISTENT THEME TOGGLE (SURVIVES RELOADS) ---
if "dark_mode_toggle" not in st.session_state:
    st.session_state.dark_mode_toggle = st.query_params.get("theme", "light") == "dark"

def sync_theme():
    st.query_params["theme"] = "dark" if st.session_state.dark_mode_toggle else "light"

top_spacer, toggle_col = st.columns([6.5, 1.2])
with toggle_col:
    dark_mode = st.toggle("Dark Mode", key="dark_mode_toggle", on_change=sync_theme)

if dark_mode:
    BG_CANVAS = "#0D0D0D"
    SURFACE_CARD = "#161616"
    INK_PRIMARY = "#F2F1ED"
    INK_SECONDARY = "#B5B4AF"
    INK_MUTED = "#7D7C78"
    BORDER_HAIRLINE = "rgba(242, 241, 237, 0.14)"
    GRID_LINE = "rgba(242, 241, 237, 0.035)"
    RING_BG = "rgba(22, 22, 22, 0.92)"
    MARQUEE_BG = "#1A1A1A"
    MARQUEE_FG = "#F2F1ED"
    BTN_BG = "#F2F1ED"
    BTN_FG = "#0D0D0D"
    DOT_INACTIVE = "#3E3E3C"
else:
    BG_CANVAS = "#E7E6E3"
    SURFACE_CARD = "#ECEBE8"
    INK_PRIMARY = "#111111"
    INK_SECONDARY = "#4A4A48"
    INK_MUTED = "#6E6E6B"
    BORDER_HAIRLINE = "rgba(17, 17, 17, 0.16)"
    GRID_LINE = "rgba(17, 17, 17, 0.035)"
    RING_BG = "rgba(236, 235, 232, 0.9)"
    MARQUEE_BG = "#111111"
    MARQUEE_FG = "#ECEBE8"
    BTN_BG = "#111111"
    BTN_FG = "#FFFFFF"
    DOT_INACTIVE = "#B5B4B0"

# --- 3. SWISS EDITORIAL CSS (DESKTOP + MOBILE RESPONSIVE) ---
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Anton&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

:root {{
    --bg-canvas: {BG_CANVAS};
    --surface-card: {SURFACE_CARD};
    --ink-primary: {INK_PRIMARY};
    --ink-secondary: {INK_SECONDARY};
    --ink-muted: {INK_MUTED};
    --signal-orange: #FF3B1D;
    --border-hairline: {BORDER_HAIRLINE};
}}

html, body {{
    overflow-x: hidden;
}}

.stApp {{
    background-color: var(--bg-canvas) !important;
    color: var(--ink-primary) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    transition: background-color 0.25s ease, color 0.25s ease;
}}

.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(to right, {GRID_LINE} 1px, transparent 1px),
        linear-gradient(to bottom, {GRID_LINE} 1px, transparent 1px);
    background-size: 80px 80px;
    pointer-events: none;
    z-index: 0;
}}

#MainMenu, footer, header {{visibility: hidden;}}
.stMarkdown h1 a, [data-testid="stHeaderActionElements"] {{
    display: none !important;
}}
.block-container {{
    padding-top: 1.2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1380px !important;
}}

/* Adaptive High-Contrast Text on All Native Widgets */
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] span,
div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] label,
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] div[role="radiogroup"] p,
div[data-testid="stSlider"] p,
div[data-testid="stSlider"] div[data-testid="stTickBarMin"],
div[data-testid="stSlider"] div[data-testid="stTickBarMax"],
div[data-testid="stThumbValue"],
div[data-testid="stToggle"] label p {{
    color: var(--ink-primary) !important;
}}

div[data-testid="stToggle"] label p {{
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}}

div[data-testid="stRadio"] div[role="radiogroup"] {{
    gap: 1.8rem;
    padding: 0.4rem 0;
}}
div[data-testid="stRadio"] label p {{
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}}

/* Selectbox & Dropdown Menu Theme Sync */
div[data-baseweb="select"] > div {{
    background-color: var(--surface-card) !important;
    border-radius: 0px !important;
    border: 1px solid var(--ink-primary) !important;
    color: var(--ink-primary) !important;
}}
div[data-baseweb="select"] span,
div[data-baseweb="select"] input,
div[data-baseweb="select"] div {{
    color: var(--ink-primary) !important;
    -webkit-text-fill-color: var(--ink-primary) !important;
}}
div[data-baseweb="select"] svg {{
    fill: var(--ink-primary) !important;
}}
div[data-baseweb="popover"],
ul[data-baseweb="menu"] {{
    background-color: var(--surface-card) !important;
    border: 1px solid var(--ink-primary) !important;
    border-radius: 0px !important;
}}
ul[data-baseweb="menu"] li {{
    color: var(--ink-primary) !important;
    background-color: var(--surface-card) !important;
}}
ul[data-baseweb="menu"] li:hover,
ul[data-baseweb="menu"] li[aria-selected="true"] {{
    background-color: #FF3B1D !important;
    color: #FFFFFF !important;
}}

/* Primary Buttons */
div.stButton > button {{
    background-color: {BTN_BG} !important;
    color: {BTN_FG} !important;
    border: 1px solid {BTN_BG} !important;
    border-radius: 0px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.76rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    padding: 0.62rem 1.2rem !important;
    transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease !important;
}}
div.stButton > button p {{
    color: {BTN_FG} !important;
}}
div.stButton > button:hover {{
    background-color: var(--signal-orange) !important;
    border-color: var(--signal-orange) !important;
    color: #FFFFFF !important;
}}
div.stButton > button:hover p {{
    color: #FFFFFF !important;
}}

/* Animations */
@keyframes fadeSlideUp {{
    0% {{ opacity: 0; transform: translateY(18px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes bounceDown {{
    0%, 100% {{ transform: translateY(0); }}
    50% {{ transform: translateY(5px); }}
}}
@keyframes marqueeScroll {{
    0% {{ transform: translateX(0%); }}
    100% {{ transform: translateX(-50%); }}
}}

/* Top Navigation Bar */
.sys-topnav {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0 1.2rem 0;
    border-bottom: 1px solid var(--border-hairline);
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink-primary);
    animation: fadeSlideUp 0.45s ease-out;
}}
.sys-topnav span.plus-mark {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    font-size: 1rem;
    letter-spacing: 0;
}}

/* Hero Container */
.sys-hero {{
    position: relative;
    background: var(--surface-card);
    border: 1px solid var(--border-hairline);
    padding: 2.6rem 3rem;
    margin-top: 1.2rem;
    display: grid;
    grid-template-columns: 1.25fr 1.05fr 0.2fr;
    align-items: center;
    gap: 2rem;
    overflow: hidden;
    box-shadow: 0 24px 50px rgba(0, 0, 0, 0.08);
    animation: fadeSlideUp 0.6s ease-out;
}}

.sys-brand {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    font-size: clamp(2.4rem, 4.3vw, 4.1rem);
    line-height: 1.02;
    letter-spacing: -0.04em;
    color: var(--ink-primary);
    margin: 0;
}}
.sys-brand .dot {{
    color: var(--signal-orange);
}}

.sys-sub-meta {{
    margin-top: 1.3rem;
    font-size: 0.95rem;
    color: var(--ink-primary);
    font-weight: 600;
    letter-spacing: -0.01em;
}}
.sys-sub-meta .sub-tag {{
    display: block;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--ink-muted);
    margin-top: 4px;
    letter-spacing: 0.08em;
}}

.sys-dots {{
    display: flex;
    gap: 12px;
    margin: 1.5rem 0;
    align-items: center;
}}
.sys-dots span {{
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: {DOT_INACTIVE};
}}
.sys-dots span.active {{
    width: 7px;
    height: 7px;
    background: var(--signal-orange);
    box-shadow: 0 0 0 4px rgba(255, 59, 29, 0.18);
}}

.sys-spec-block {{
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    line-height: 1.65;
    color: var(--ink-muted);
    text-transform: uppercase;
    margin-top: 1.4rem;
}}
.sys-spec-block strong {{
    color: var(--ink-primary);
}}

.sys-visual-stage {{
    position: relative;
    height: 290px;
    display: flex;
    align-items: center;
    justify-content: center;
}}
.orange-slab {{
    position: absolute;
    right: -3rem;
    left: 15%;
    height: 165px;
    background: var(--signal-orange);
    z-index: 1;
}}
.orange-ring {{
    position: relative;
    z-index: 2;
    width: 205px;
    height: 205px;
    border-radius: 50%;
    border: 14px solid var(--signal-orange);
    background: {RING_BG};
    backdrop-filter: blur(6px);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: 0 18px 35px rgba(255, 59, 29, 0.22);
    transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}}
.ring-letter {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    font-size: 4.2rem;
    line-height: 1;
    letter-spacing: -0.04em;
    color: var(--signal-orange);
}}
.ring-caption {{
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    color: var(--ink-primary);
    margin-top: 6px;
}}

.sys-right-rail {{
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    justify-content: space-between;
    height: 100%;
}}
.vertical-text {{
    writing-mode: vertical-rl;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--ink-primary);
    line-height: 1.5;
}}
.scroll-arrow-box {{
    position: absolute;
    right: 0;
    bottom: 0;
    width: 46px;
    height: 54px;
    background: var(--signal-orange);
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    font-weight: 700;
}}
.scroll-arrow-box span {{
    animation: bounceDown 1.6s infinite ease-in-out;
}}

/* Ticker Bar */
.marquee-wrap {{
    width: 100%;
    overflow: hidden;
    background: {MARQUEE_BG};
    color: {MARQUEE_FG};
    border-left: 1px solid var(--border-hairline);
    border-right: 1px solid var(--border-hairline);
    border-bottom: 1px solid var(--border-hairline);
    padding: 0.55rem 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}}
.marquee-track {{
    display: inline-block;
    white-space: nowrap;
    animation: marqueeScroll 28s linear infinite;
}}
.marquee-wrap:hover .marquee-track {{
    animation-play-state: paused;
}}
.marquee-accent {{
    color: var(--signal-orange);
    font-weight: 700;
}}

/* Section Header */
.section-header-row {{
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    border-bottom: 1px solid var(--ink-primary);
    padding-bottom: 0.75rem;
    margin: 2rem 0 1.4rem 0;
}}
.big-index-num {{
    font-family: 'Anton', sans-serif;
    font-size: 3.4rem;
    line-height: 0.85;
    color: var(--ink-primary);
    letter-spacing: 0.02em;
}}
.section-label-mono {{
    font-family: 'Space Mono', monospace;
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--ink-secondary);
}}

/* Recommendation Cards */
.editorial-card {{
    background: var(--surface-card);
    border: 1px solid var(--border-hairline);
    padding: 14px;
    position: relative;
    transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.3s ease, box-shadow 0.3s ease;
    animation: fadeSlideUp 0.5s ease-out backwards;
    margin-bottom: 1rem;
}}

/* Monochrome to Color Image Reveal */
.media-frame {{
    position: relative;
    width: 100%;
    overflow: hidden;
    background: #1A1A1A;
    margin-bottom: 12px;
}}
.poster-ratio {{ aspect-ratio: 2 / 3; }}
.album-ratio {{ aspect-ratio: 1 / 1; }}

.media-frame img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    filter: grayscale(100%) contrast(1.08);
    transition: filter 0.4s ease, transform 0.4s ease;
}}

@media (hover: hover) {{
    .sys-hero:hover .orange-ring {{
        transform: scale(1.04);
    }}
    .editorial-card:hover {{
        transform: translateY(-6px);
        border-color: var(--ink-primary);
        box-shadow: 7px 7px 0px var(--signal-orange);
    }}
    .editorial-card:hover .media-frame img {{
        filter: grayscale(0%) contrast(1);
        transform: scale(1.03);
    }}
    .editorial-card:hover .card-arrow-tag {{
        background: var(--signal-orange);
    }}
}}

/* Typographic Placeholder When Artwork Is Unavailable */
.fallback-art {{
    width: 100%;
    height: 100%;
    padding: 18px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    background: #181818;
    border-bottom: 3px solid var(--signal-orange);
}}
.fb-top {{
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    color: #8E8E8A;
}}
.fb-mid {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    font-size: 1.35rem;
    line-height: 1.15;
    color: #ECEBE8;
    text-transform: uppercase;
    letter-spacing: -0.02em;
}}
.fb-bot {{
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--signal-orange);
}}

.card-arrow-tag {{
    position: absolute;
    bottom: 0;
    left: 0;
    background: #111
