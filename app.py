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
    BG_CANVAS = "#0A0A0B"
    SURFACE_CARD = "#121214"
    HERO_BG = "#181819"
    SURFACE_SUBTLE = "#1E1E20"
    INK_PRIMARY = "#FFFFFF"
    INK_SECONDARY = "#A1A1AA"
    INK_MUTED = "#6E6E73"
    RECOMMENDER_INK = "#FFFFFF"
    BORDER_HAIRLINE = "rgba(255, 255, 255, 0.10)"
    GRID_LINE = "rgba(255, 255, 255, 0.025)"
    MARQUEE_BG = "#141416"
    MARQUEE_FG = "#A1A1AA"
    BTN_BG = "#FFFFFF"
    BTN_FG = "#181819"
    PILL_BG = "#FFFFFF"
    PILL_FG = "#181819"
    DOT_INACTIVE = "#323236"
    CARD_SHADOW = "0 18px 38px rgba(0, 0, 0, 0.45)"
else:
    BG_CANVAS = "#F5F5F3"
    SURFACE_CARD = "#FFFFFF"
    HERO_BG = "#FFFFFF"
    SURFACE_SUBTLE = "#F2F2F0"
    INK_PRIMARY = "#181819"
    INK_SECONDARY = "#525252"
    INK_MUTED = "#8A8A8A"
    RECOMMENDER_INK = "#181819"
    BORDER_HAIRLINE = "rgba(24, 24, 25, 0.11)"
    GRID_LINE = "rgba(24, 24, 25, 0.025)"
    MARQUEE_BG = "#FFFFFF"
    MARQUEE_FG = "#525252"
    BTN_BG = "#181819"
    BTN_FG = "#FFFFFF"
    PILL_BG = "#181819"
    PILL_FG = "#FFFFFF"
    DOT_INACTIVE = "#D4D4D0"
    CARD_SHADOW = "0 18px 38px rgba(24, 24, 25, 0.07)"

# --- 3. FIGMA SF PRO + BABY PURPLE (#D199FF) ACCENTS CSS ---
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

:root, .stApp {{
    --primary-color: #D199FF !important;
    --bg-canvas: {BG_CANVAS};
    --surface-card: {SURFACE_CARD};
    --hero-bg: {HERO_BG};
    --surface-subtle: {SURFACE_SUBTLE};
    --ink-primary: {INK_PRIMARY};
    --ink-secondary: {INK_SECONDARY};
    --ink-muted: {INK_MUTED};
    --figma-pink: #FF93C7;
    --figma-purple: #D199FF;
    --arrow-accent: #D199FF;
    --border-hairline: {BORDER_HAIRLINE};
}}

html, body {{
    overflow-x: hidden;
}}

.stApp {{
    background-color: var(--bg-canvas) !important;
    color: var(--ink-primary) !important;
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Inter', sans-serif !important;
    transition: background-color 0.25s ease, color 0.25s ease;
}}

.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(to right, {GRID_LINE} 1px, transparent 1px),
        linear-gradient(to bottom, {GRID_LINE} 1px, transparent 1px);
    background-size: 96px 96px;
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
    max-width: 1340px !important;
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
div[data-testid="stToggle"] label p {{
    color: var(--ink-primary) !important;
}}

/* Baby Purple (#D199FF) Native Controls: Toggle, Radio & Slider */
div[data-testid="stToggle"] label p {{
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}}
div[data-testid="stToggle"] input:checked + div,
div[data-testid="stToggle"] label[data-baseweb="checkbox"]:has(input:checked) > div:first-of-type {{
    background-color: #D199FF !important;
}}
div[data-testid="stToggle"] input:checked + div > div,
div[data-testid="stToggle"] label[data-baseweb="checkbox"]:has(input:checked) > div:first-of-type > div {{
    background-color: #181819 !important;
}}

div[data-testid="stRadio"] div[role="radiogroup"] {{
    gap: 1.8rem;
    padding: 0.4rem 0;
}}
div[data-testid="stRadio"] label p {{
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}}
div[data-testid="stRadio"] input:checked + div,
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:first-of-type {{
    background-color: #D199FF !important;
    border-color: #D199FF !important;
}}
div[data-testid="stRadio"] input:checked + div > div,
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:first-of-type > div {{
    background-color: #181819 !important;
}}

div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div[style*="255, 75, 75"] {{
    filter: hue-rotate(272deg) brightness(1.25) !important;
}}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div[style*="255, 75, 75"] div[role="slider"] {{
    filter: hue-rotate(88deg) brightness(0.8) !important;
}}
div[data-testid="stSlider"] div[role="slider"] {{
    background-color: #D199FF !important;
    border-color: #D199FF !important;
    box-shadow: 0 0 0 3px rgba(209, 153, 255, 0.25) !important;
}}
div[data-testid="stSlider"] div[data-testid="stThumbValue"] {{
    color: #D199FF !important;
}}

/* Selectbox & Dropdown Menu Theme Sync (Sharp Corners) */
div[data-baseweb="select"] > div {{
    background-color: var(--surface-card) !important;
    border-radius: 2px !important;
    border: 1px solid var(--border-hairline) !important;
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
    border: 1px solid var(--border-hairline) !important;
    border-radius: 2px !important;
}}
ul[data-baseweb="menu"] li {{
    color: var(--ink-primary) !important;
    background-color: var(--surface-card) !important;
}}
ul[data-baseweb="menu"] li:hover,
ul[data-baseweb="menu"] li[aria-selected="true"] {{
    background-color: #D199FF !important;
    color: #181819 !important;
}}

/* Minimalist Sharp Buttons */
div.stButton > button {{
    background-color: {BTN_BG} !important;
    color: {BTN_FG} !important;
    border: 1px solid {BTN_BG} !important;
    border-radius: 2px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.74rem !important;
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
    background-color: #D199FF !important;
    border-color: #D199FF !important;
    color: #181819 !important;
}}
div.stButton > button:hover p {{
    color: #181819 !important;
}}

/* Animations */
@keyframes fadeSlideUp {{
    0% {{ opacity: 0; transform: translateY(16px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes bounceDown {{
    0%, 100% {{ transform: translateY(0); }}
    50% {{ transform: translateY(4px); }}
}}
@keyframes marqueeScroll {{
    0% {{ transform: translateX(0%); }}
    100% {{ transform: translateX(-50%); }}
}}

/* Top Architectural Navigation Bar */
.sys-topnav {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1.4rem;
    background: var(--hero-bg);
    border: 1px solid var(--border-hairline);
    border-radius: 3px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink-primary);
    animation: fadeSlideUp 0.45s ease-out;
}}
.nav-left-group {{
    display: flex;
    align-items: center;
    gap: 12px;
}}
.geo-mark {{
    width: 22px;
    height: 22px;
    border-radius: 2px;
    border: 1px solid var(--border-hairline);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0;
}}
.nav-pill {{
    border: 1px solid var(--border-hairline);
    border-radius: 2px;
    padding: 4px 12px;
    font-size: 0.66rem;
    color: var(--ink-secondary);
}}

/* Figma Hero Stage (#181819) */
.sys-hero {{
    position: relative;
    background: var(--hero-bg);
    border: 1px solid var(--border-hairline);
    border-radius: 4px;
    padding: 2.8rem 2.6rem 2rem 2.6rem;
    margin-top: 1rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    overflow: hidden;
    box-shadow: {CARD_SHADOW};
    animation: fadeSlideUp 0.6s ease-out;
}}

.hero-top-tag {{
    font-family: 'Space Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--ink-muted);
    margin-bottom: 2.6rem;
}}

.hero-center-row {{
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.2rem;
    margin: 0.5rem 0 2.8rem 0;
}}

.hero-side-ring {{
    width: 34px;
    height: 34px;
    border-radius: 2px;
    border: 1px solid var(--border-hairline);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--ink-muted);
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    flex-shrink: 0;
}}

/* SF Pro Display Single-Line Title with Pink (#FF93C7) -> Purple (#D199FF) -> White (#FFFFFF) */
.sys-brand {{
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Inter', sans-serif;
    font-weight: 600;
    font-size: clamp(1.65rem, 3.35vw, 3.25rem);
    line-height: 1.1;
    letter-spacing: -0.02em;
    text-transform: uppercase;
    color: {RECOMMENDER_INK};
    margin: 0 auto;
    white-space: nowrap;
}}
.sys-brand .sf-gradient {{
    background: linear-gradient(90deg, #FF93C7 0%, #FF93C7 38%, #D199FF 62%, #D199FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.sys-sub-meta {{
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Inter', sans-serif;
    font-size: 0.94rem;
    color: var(--ink-secondary);
    font-weight: 400;
    letter-spacing: 0.005em;
}}

.hero-pill-bar {{
    display: inline-flex;
    align-items: center;
    gap: 12px;
    margin-top: 1.5rem;
    padding: 0.55rem 1.35rem;
    border: 1px solid var(--border-hairline);
    border-radius: 2px;
    background: var(--surface-subtle);
    font-family: 'Space Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-primary);
}}

.hero-bottom-bar {{
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 2.6rem;
    padding-top: 1.1rem;
    border-top: 1px solid var(--border-hairline);
}}

.hero-meta-left {{
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink-muted);
}}

.sys-dots {{
    display: flex;
    gap: 9px;
    align-items: center;
}}
.sys-dots span {{
    width: 5px;
    height: 5px;
    border-radius: 1px;
    background: {DOT_INACTIVE};
}}
.sys-dots span.active {{
    width: 6px;
    height: 6px;
    background: #D199FF;
}}

.scroll-arrow-box {{
    width: 34px;
    height: 34px;
    border-radius: 2px;
    border: 1px solid var(--border-hairline);
    background: var(--surface-subtle);
    color: #D199FF;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    font-weight: 700;
}}
.scroll-arrow-box span {{
    animation: bounceDown 1.6s infinite ease-in-out;
}}

/* Minimal Ticker Bar */
.marquee-wrap {{
    width: 100%;
    overflow: hidden;
    background: {MARQUEE_BG};
    color: {MARQUEE_FG};
    border: 1px solid var(--border-hairline);
    border-radius: 3px;
    padding: 0.65rem 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 0.85rem;
    margin-bottom: 2rem;
}}
.marquee-track {{
    display: inline-block;
    white-space: nowrap;
    animation: marqueeScroll 32s linear infinite;
}}
.marquee-wrap:hover .marquee-track {{
    animation-play-state: paused;
}}
.marquee-strong {{
    color: var(--ink-primary);
    font-weight: 700;
}}

/* Section Header */
.section-header-row {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 1px solid var(--border-hairline);
    padding-bottom: 0.75rem;
    margin: 1.8rem 0 1.4rem 0;
}}
.big-index-num {{
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', sans-serif;
    font-weight: 600;
    font-size: 2.2rem;
    line-height: 1;
    color: var(--ink-primary);
    letter-spacing: -0.03em;
}}
.section-label-mono {{
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--ink-secondary);
}}

/* Recommendation Cards (Sharp 90% Corners) */
.editorial-card {{
    background: var(--surface-card);
    border: 1px solid var(--border-hairline);
    border-radius: 3px;
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
    border-radius: 2px;
    background: #181819;
    margin-bottom: 12px;
}}
.poster-ratio {{ aspect-ratio: 2 / 3; }}
.album-ratio {{ aspect-ratio: 1 / 1; }}

.media-frame img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    filter: grayscale(100%) contrast(1.05);
    transition: filter 0.4s ease, transform 0.4s ease;
}}

@media (hover: hover) {{
    .editorial-card:hover {{
        transform: translateY(-5px);
        border-color: #D199FF;
        box-shadow: {CARD_SHADOW};
    }}
    .editorial-card:hover .media-frame img {{
        filter: grayscale(0%) contrast(1);
        transform: scale(1.03);
    }}
    .editorial-card:hover .card-arrow-tag {{
        background: #D199FF;
        color: #181819;
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
    background: #181819;
}}
.fb-top {{
    font-family: 'Space Mono', monospace;
    font-size: 0.64rem;
    letter-spacing: 0.14em;
    color: #71717A;
}}
.fb-mid {{
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', sans-serif;
    font-weight: 600;
    font-size: 1.25rem;
    line-height: 1.2;
    color: #FFFFFF;
    letter-spacing: -0.02em;
}}
.fb-bot {{
    font-family: 'Space Mono', monospace;
    font-size: 0.66rem;
    color: #A1A1AA;
}}

/* Signature Arrow Tag (#D199FF Baby Purple) */
.card-arrow-tag {{
    position: absolute;
    bottom: 8px;
    left: 8px;
    background: rgba(24, 24, 25, 0.85);
    backdrop-filter: blur(6px);
    color: #D199FF;
    border-radius: 2px;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    font-weight: 700;
    padding: 3px 9px;
    transition: background 0.25s ease, color 0.25s ease;
}}

.card-index-strip {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--ink-muted);
    margin-bottom: 8px;
}}
div[data-testid="stMarkdownContainer"] .match-pill-mono,
.match-pill-mono {{
    background: {PILL_BG} !important;
    color: {PILL_FG} !important;
    border-radius: 2px !important;
    padding: 2px 8px;
    font-weight: 700;
    font-size: 0.65rem;
}}
.ed-title {{
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', sans-serif;
    font-weight: 600;
    font-size: 1.02rem;
    line-height: 1.28;
    color: var(--ink-primary);
    margin-bottom: 4px;
    letter-spacing: -0.015em;
}}
.ed-meta {{
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--ink-muted);
    margin-bottom: 8px;
}}
.ed-desc {{
    font-size: 0.8rem;
    line-height: 1.48;
    color: var(--ink-secondary);
}}

/* --- MOBILE RESPONSIVE OVERRIDES --- */
@media (max-width: 820px) {{
    .block-container {{
        padding-top: 0.8rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 3rem !important;
    }}
    .sys-topnav {{
        padding: 0.65rem 1rem;
        font-size: 0.6rem;
        letter-spacing: 0.06em;
    }}
    .sys-topnav .nav-right {{
        display: none;
    }}
    .sys-hero {{
        padding: 2.2rem 1.2rem 1.5rem 1.2rem;
        border-radius: 3px;
        margin-top: 0.75rem;
    }}
    .hero-top-tag {{
        font-size: 0.6rem;
        margin-bottom: 1.4rem;
    }}
    .hero-center-row {{
        margin: 0.3rem 0 1.6rem 0;
    }}
    .hero-side-ring {{
        display: none;
    }}
    .sys-brand {{
        font-size: 1.65rem;
        line-height: 1.15;
        white-space: normal;
    }}
    .sys-sub-meta {{
        font-size: 0.86rem;
    }}
    .hero-pill-bar {{
        font-size: 0.6rem;
        padding: 0.48rem 1rem;
        margin-top: 1.2rem;
    }}
    .hero-bottom-bar {{
        margin-top: 1.8rem;
        padding-top: 0.9rem;
    }}
    .hero-meta-left {{
        font-size: 0.58rem;
        letter-spacing: 0.06em;
    }}
    .marquee-wrap {{
        font-size: 0.62rem;
        margin-bottom: 1.4rem;
    }}
    .section-header-row {{
        margin: 1.4rem 0 1rem 0;
        padding-bottom: 0.55rem;
    }}
    .big-index-num {{
        font-size: 1.8rem;
    }}
    .section-label-mono {{
        font-size: 0.64rem;
        letter-spacing: 0.08em;
    }}
    .media-frame img {{
        filter: grayscale(0%) contrast(1);
    }}
}}
</style>
""", unsafe_allow_html=True)

# --- 4. TOP NAVIGATION & FIGMA HERO ---
st.markdown("""
<div class="sys-topnav">
<div class="nav-left-group">
<span class="geo-mark">+1</span>
<span>MOVIE & MUSIC RECOMMENDATION SYSTEM</span>
</div>
<div class="nav-right">
<span class="nav-pill">MOVIES &nbsp;·&nbsp; MUSIC &nbsp;·&nbsp; 2026</span>
</div>
</div>

<div class="sys-hero">
<div class="hero-top-tag">CURATED // 001 &nbsp;·&nbsp; INDEX 2026</div>

<div class="hero-center-row">
<div class="hero-side-ring">+</div>
<h1 class="sys-brand"><span class="sf-gradient">MOVIE &amp; MUSIC</span> RECOMMENDER</h1>
<div class="hero-side-ring">+</div>
</div>

<div class="sys-sub-meta">Curated for your taste.</div>

<div class="hero-pill-bar">
<span>14,800+ FILMS & TRACKS</span>
<span>·</span>
<span>INSTANT TASTE MATCHING</span>
</div>

<div class="hero-bottom-bar">
<div class="hero-meta-left">FILMS & AUDIO</div>
<div class="sys-dots">
<span class="active"></span><span></span><span></span><span></span><span></span>
</div>
<div class="scroll-arrow-box"><span>↓</span></div>
</div>
</div>

<div class="marquee-wrap">
<div class="marquee-track">
<span class="marquee-strong">001 //</span> MOVIE & MUSIC RECOMMENDATION SYSTEM &nbsp;·&nbsp;
4,800 FILMS &nbsp;·&nbsp; 10,000 TRACKS &nbsp;·&nbsp;
<span class="marquee-strong">CURATED FOR YOUR TASTE</span> &nbsp;·&nbsp;
HOVER ANY CARD FOR COLOR &nbsp;·&nbsp;
<span class="marquee-strong">001 //</span> MOVIE & MUSIC RECOMMENDATION SYSTEM &nbsp;·&nbsp;
4,800 FILMS &nbsp;·&nbsp; 10,000 TRACKS &nbsp;·&nbsp;
<span class="marquee-strong">CURATED FOR YOUR TASTE</span> &nbsp;·&nbsp;
HOVER ANY CARD FOR COLOR &nbsp;·&nbsp;
</div>
</div>
""", unsafe_allow_html=True)

# --- 5. ARTWORK & AUDIO PREVIEW FETCHERS ---
@st.cache_data(show_spinner=False)
def fetch_movie_poster(title, year=""):
    # Primary: IMDb public suggestion catalog
    try:
        query = f"{title} {year}".strip()
        encoded = urllib.parse.quote(query)
        url = f"https://v3.sg.media-imdb.com/suggestion/x/{encoded}.json"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for item in data.get("d", []):
                img_obj = item.get("i") or {}
                img = img_obj.get("imageUrl")
                if img:
                    return img
    except Exception:
        pass

    # Secondary: iTunes Movie Search (US Storefront)
    try:
        encoded = urllib.parse.quote(title)
        url = f"https://itunes.apple.com/search?term={encoded}&media=movie&country=US&limit=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("results"):
                art = data["results"][0].get("artworkUrl100", "")
                if art:
                    return art.replace("100x100", "600x600")
    except Exception:
        pass

    return None

@st.cache_data(show_spinner=False)
def fetch_song_assets(query):
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://itunes.apple.com/search?term={encoded}&entity=song&limit=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("results"):
                item = data["results"][0]
                art = item.get("artworkUrl100", "").replace("100x100", "600x600")
                preview = item.get("previewUrl")
                return art, preview
    except Exception:
        pass
    return None, None

# --- 6. LOAD & COMPUTE SIMILARITY MATRICES ---
def parse_json_names(text):
    try:
        items = ast.literal_eval(str(text))
        return " ".join(item["name"].replace(" ", "") for item in items)
    except Exception:
        return ""

def parse_display_genres(text):
    try:
        items = ast.literal_eval(str(text))
        return " / ".join(item["name"] for item in items[:2])
    except Exception:
        return "Feature"

@st.cache_resource
def load_and_train_models():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    movies = movies.dropna(subset=["title", "overview"]).drop_duplicates(subset=["title"]).reset_index(drop=True)
    movies["display_genre"] = movies["genres"].apply(parse_display_genres)
    movies["tags"] = (
        movies["genres"].apply(parse_json_names) + " "
        + movies["keywords"].apply(parse_json_names) + " "
        + movies["overview"].fillna("")
    )
    movie_vec = TfidfVectorizer(max_features=5000, stop_words="english")
    movie_sim = cosine_similarity(movie_vec.fit_transform(movies["tags"]))

    songs = pd.read_csv("spotify_tracks.csv")
    songs = songs.dropna(subset=["track_name", "artists", "track_genre"])
    songs = (
        songs.sort_values("popularity", ascending=False)
        .drop_duplicates(subset=["track_name"])
        .head(10000)
        .reset_index(drop=True)
    )

    def build_song_tags(row):
        vibes = [str(row["track_genre"]), str(row["artists"]).replace(";", " ")]
        vibes.append("high_energy" if row["energy"] > 0.65 else "chill_vibe")
        vibes.append("happy_upbeat" if row["valence"] > 0.55 else "sad_moody")
        vibes.append("danceable" if row["danceability"] > 0.65 else "steady_beat")
        return " ".join(vibes)

    songs["tags"] = songs.apply(build_song_tags, axis=1)
    song_vec = TfidfVectorizer(max_features=5000, stop_words="english")
    song_sim = cosine_similarity(song_vec.fit_transform(songs["tags"]))

    return movies, movie_sim, songs, song_sim

def get_recommendations(df, title_col, selected_name, sim_matrix, top_n=5):
    idx = df[df[title_col] == selected_name].index[0]
    scores = sorted(list(enumerate(sim_matrix[idx])), key=lambda x: x[1], reverse=True)
    return [(df.iloc[i], round(score * 100, 1)) for i, score in scores[1 : top_n + 1]]

movies_df, movie_sim_matrix, songs_df, song_sim_matrix = load_and_train_models()

# --- 7. SECTION 001: RECOMMENDATION WORKSPACE ---
st.markdown("""
<div class="section-header-row">
<div class="big-index-num">001</div>
<div class="section-label-mono">SELECT A CATEGORY</div>
</div>
""", unsafe_allow_html=True)

mode = st.radio(
    "Category",
    ["Movies", "Music"],
    horizontal=True,
    label_visibility="collapsed",
)

# --- MODE 1: MOVIES ---
if mode == "Movies":
    if "selected_movie" not in st.session_state:
        st.session_state.selected_movie = "Inception" if "Inception" in movies_df["title"].values else movies_df["title"].iloc[0]

    c1, c2, c3, c4 = st.columns([3.5, 1.2, 1.0, 1.1], vertical_alignment="bottom")
    with c1:
        movie_list = movies_df["title"].tolist()
        default_idx = movie_list.index(st.session_state.selected_movie) if st.session_state.selected_movie in movie_list else 0
        chosen_movie = st.selectbox("Select a movie", movie_list, index=default_idx)
        st.session_state.selected_movie = chosen_movie
    with c2:
        top_n = st.slider("Results", 4, 8, 4, key="movie_results")
    with c3:
        surprise_m = st.button("Random", use_container_width=True)
    with c4:
        st.button("Recommend", use_container_width=True)

    if surprise_m:
        st.session_state.selected_movie = random.choice(movies_df["title"].tolist())
        st.rerun()

    recs = get_recommendations(movies_df, "title", st.session_state.selected_movie, movie_sim_matrix, top_n=top_n)
    st.markdown(f"<p class='section-label-mono' style='margin:1.1rem 0;'>SIMILAR TO <b>{st.session_state.selected_movie}</b></p>", unsafe_allow_html=True)

    cols = st.columns(4)
    for idx, (movie, match_pct) in enumerate(recs):
        year = str(movie.get("release_date", ""))[:4]
        poster_url = fetch_movie_poster(movie["title"], year)
        if poster_url:
            art_html = f'<img src="{poster_url}">'
        else:
            art_html = f'<div class="fallback-art"><div class="fb-top">NO. 0{idx+1} // ARCHIVE</div><div class="fb-mid">{movie["title"]}</div><div class="fb-bot">{year}</div></div>'
        with cols[idx % 4]:
            st.markdown(f"""
<div class="editorial-card">
<div class="card-index-strip">
<span>0{idx+1} // {year}</span>
<span class="match-pill-mono">{match_pct}% MATCH</span>
</div>
<div class="media-frame poster-ratio">
{art_html}
<div class="card-arrow-tag">→</div>
</div>
<div class="ed-title">{movie["title"]}</div>
<div class="ed-meta">{movie["vote_average"]}/10  ·  {movie["display_genre"]}</div>
<div class="ed-desc">{movie["overview"][:105]}...</div>
</div>
""", unsafe_allow_html=True)

# --- MODE 2: MUSIC ---
else:
    if "selected_song" not in st.session_state:
        st.session_state.selected_song = songs_df["track_name"].iloc[0]

    c1, c2, c3, c4 = st.columns([3.5, 1.2, 1.0, 1.1], vertical_alignment="bottom")
    with c1:
        song_list = songs_df["track_name"].tolist()
        default_s_idx = song_list.index(st.session_state.selected_song) if st.session_state.selected_song in song_list else 0
        chosen_song = st.selectbox("Select a track", song_list, index=default_s_idx)
        st.session_state.selected_song = chosen_song
    with c2:
        top_n_s = st.slider("Results", 4, 8, 4, key="music_results")
    with c3:
        surprise_s = st.button("Random", use_container_width=True)
    with c4:
        st.button("Recommend", use_container_width=True)

    if surprise_s:
        st.session_state.selected_song = random.choice(songs_df["track_name"].tolist())
        st.rerun()

    display_recs = get_recommendations(songs_df, "track_name", st.session_state.selected_song, song_sim_matrix, top_n=top_n_s)

    st.markdown(f"<p class='section-label-mono' style='margin:1.1rem 0;'>SIMILAR TO <b>{st.session_state.selected_song}</b></p>", unsafe_allow_html=True)

    cols = st.columns(4)
    for idx, (song, match_pct) in enumerate(display_recs):
        artist_clean = str(song["artists"]).replace(";", ", ")
        first_artist = str(song["artists"]).split(";")[0]
        cover_url, preview_url = fetch_song_assets(f"{song['track_name']} {first_artist}")
        if cover_url:
            art_html = f'<img src="{cover_url}">'
        else:
            art_html = f'<div class="fallback-art"><div class="fb-top">TRK // 0{idx+1}</div><div class="fb-mid">{song["track_name"]}</div><div class="fb-bot">{song["track_genre"].upper()}</div></div>'
        with cols[idx % 4]:
            st.markdown(f"""
<div class="editorial-card">
<div class="card-index-strip">
<span>0{idx+1} // {song["track_genre"].upper()}</span>
<span class="match-pill-mono">{match_pct}% MATCH</span>
</div>
<div class="media-frame album-ratio">
{art_html}
<div class="card-arrow-tag">→</div>
</div>
<div class="ed-title">{song["track_name"]}</div>
<div class="ed-meta">{artist_clean}</div>
<div class="ed-desc">Energy {int(song["energy"]*100)}%  ·  Valence {int(song["valence"]*100)}%</div>
</div>
""", unsafe_allow_html=True)
            if preview_url:
                st.audio(preview_url)
