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

# --- 2. THEME TOGGLE ---
top_spacer, toggle_col = st.columns([6.5, 1.2])
with toggle_col:
    dark_mode = st.toggle("Dark Mode", value=False)

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

# --- 3. SWISS EDITORIAL CSS (LIGHT & DARK THEME ENGINE) ---
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
    font-size: clamp(2.5rem, 4.3vw, 4.1rem);
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
    font-size: 0.9rem;
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
.sys-hero:hover .orange-ring {{
    transform: scale(1.04);
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
.editorial-card:hover {{
    transform: translateY(-6px);
    border-color: var(--ink-primary);
    box-shadow: 7px 7px 0px var(--signal-orange);
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
.editorial-card:hover .media-frame img {{
    filter: grayscale(0%) contrast(1);
    transform: scale(1.03);
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
    background: #111111;
    color: #FFFFFF;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    padding: 4px 9px;
    transition: background 0.25s ease;
}}
.editorial-card:hover .card-arrow-tag {{
    background: var(--signal-orange);
}}

.card-index-strip {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--ink-muted);
    margin-bottom: 6px;
}}
.match-pill-orange {{
    background: var(--signal-orange);
    color: #FFFFFF;
    padding: 2px 7px;
    font-weight: 700;
    font-size: 0.68rem;
}}
.ed-title {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    line-height: 1.25;
    color: var(--ink-primary);
    margin-bottom: 4px;
    letter-spacing: -0.01em;
}}
.ed-meta {{
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--ink-muted);
    margin-bottom: 8px;
}}
.ed-desc {{
    font-size: 0.8rem;
    line-height: 1.45;
    color: var(--ink-secondary);
}}
</style>
""", unsafe_allow_html=True)

# --- 4. TOP NAVIGATION & HERO ---
st.markdown("""
<div class="sys-topnav">
<div><span class="plus-mark">+1</span> &nbsp;&nbsp;&nbsp; MOVIE & MUSIC RECOMMENDATION SYSTEM</div>
<div>MOVIES &nbsp;&nbsp;&nbsp; MUSIC &nbsp;&nbsp;&nbsp; 2026 &nbsp;&nbsp;&nbsp; • • •</div>
</div>

<div class="sys-hero">
<div>
<h1 class="sys-brand">Movie & Music<br>Recommendation<br>System<span class="dot">.</span></h1>
<div class="sys-sub-meta">
Content-Based Discovery Engine
<span class="sub-tag">EDITION 01 // 14,800+ INDEXED TITLES</span>
</div>
<div class="sys-dots">
<span class="active"></span><span></span><span></span><span></span><span></span><span></span><span></span>
</div>
<div class="sys-spec-block">
<strong>MODEL // 001</strong><br>
---<br>
TF-IDF VECTORIZATION<br>
COSINE SIMILARITY MATRIX
</div>
</div>
<div class="sys-visual-stage">
<div class="orange-slab"></div>
<div class="orange-ring">
<div class="ring-letter">ML</div>
<div class="ring-caption">ENGINE</div>
</div>
</div>
<div class="sys-right-rail">
<div class="vertical-text">
MACHINE LEARNING<br>
RECOMMENDATION<br>
ENGINE 2026
</div>
<div class="scroll-arrow-box"><span>↓</span></div>
</div>
</div>

<div class="marquee-wrap">
<div class="marquee-track">
<span class="marquee-accent">001 //</span> MOVIE & MUSIC RECOMMENDATION SYSTEM &nbsp;—&nbsp;
4,800 MOVIES &nbsp;—&nbsp; 10,000 TRACKS &nbsp;—&nbsp;
<span class="marquee-accent">TF-IDF & COSINE SIMILARITY</span> &nbsp;—&nbsp;
HOVER ANY CARD TO REVEAL COLOR &nbsp;—&nbsp;
<span class="marquee-accent">001 //</span> MOVIE & MUSIC RECOMMENDATION SYSTEM &nbsp;—&nbsp;
4,800 MOVIES &nbsp;—&nbsp; 10,000 TRACKS &nbsp;—&nbsp;
<span class="marquee-accent">TF-IDF & COSINE SIMILARITY</span> &nbsp;—&nbsp;
HOVER ANY CARD TO REVEAL COLOR &nbsp;—&nbsp;
</div>
</div>
""", unsafe_allow_html=True)

# --- 5. ARTWORK & AUDIO PREVIEW FETCHER ---
@st.cache_data(show_spinner=False)
def fetch_media_assets(query, media_type="movie"):
    try:
        entity = "movie" if media_type == "movie" else "song"
        encoded = urllib.parse.quote(query)
        url = f"https://itunes.apple.com/search?term={encoded}&entity={entity}&limit=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
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
<div class="section-label-mono">RECOMMENDATION INDEX // SELECT CATEGORY</div>
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
        top_n = st.slider("Results", 4, 8, 4)
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
        poster_url, _ = fetch_media_assets(movie["title"], media_type="movie")
        year = str(movie.get("release_date", ""))[:4]
        if poster_url:
            art_html = f'<img src="{poster_url}">'
        else:
            art_html = f'<div class="fallback-art"><div class="fb-top">NO. 0{idx+1} // ARCHIVE</div><div class="fb-mid">{movie["title"]}</div><div class="fb-bot">{year}</div></div>'
        with cols[idx % 4]:
            st.markdown(f"""
<div class="editorial-card">
<div class="card-index-strip">
<span>0{idx+1} // {year}</span>
<span class="match-pill-orange">{match_pct}% MATCH</span>
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

    c1, c2, c3, c4 = st.columns([3.2, 1.5, 1.0, 1.1], vertical_alignment="bottom")
    with c1:
        song_list = songs_df["track_name"].tolist()
        default_s_idx = song_list.index(st.session_state.selected_song) if st.session_state.selected_song in song_list else 0
        chosen_song = st.selectbox("Select a track", song_list, index=default_s_idx)
        st.session_state.selected_song = chosen_song
    with c2:
        profile_filter = st.selectbox("Audio Profile", ["All", "High Energy", "Calm & Ambient", "Dance"])
    with c3:
        surprise_s = st.button("Random", use_container_width=True)
    with c4:
        st.button("Recommend", use_container_width=True)

    if surprise_s:
        st.session_state.selected_song = random.choice(songs_df["track_name"].tolist())
        st.rerun()

    raw_recs = get_recommendations(songs_df, "track_name", st.session_state.selected_song, song_sim_matrix, top_n=15)
    filtered_recs = []
    for song, match_pct in raw_recs:
        if profile_filter == "High Energy" and song["energy"] <= 0.65:
            continue
        if profile_filter == "Calm & Ambient" and song["energy"] > 0.65:
            continue
        if profile_filter == "Dance" and song["danceability"] <= 0.65:
            continue
        filtered_recs.append((song, match_pct))
    display_recs = filtered_recs[:4] if filtered_recs else raw_recs[:4]

    st.markdown(f"<p class='section-label-mono' style='margin:1.1rem 0;'>SIMILAR TO <b>{st.session_state.selected_song}</b></p>", unsafe_allow_html=True)

    cols = st.columns(4)
    for idx, (song, match_pct) in enumerate(display_recs):
        artist_clean = str(song["artists"]).replace(";", ", ")
        first_artist = str(song["artists"]).split(";")[0]
        cover_url, preview_url = fetch_media_assets(f"{song['track_name']} {first_artist}", media_type="song")
        if cover_url:
            art_html = f'<img src="{cover_url}">'
        else:
            art_html = f'<div class="fallback-art"><div class="fb-top">TRK // 0{idx+1}</div><div class="fb-mid">{song["track_name"]}</div><div class="fb-bot">{song["track_genre"].upper()}</div></div>'
        with cols[idx % 4]:
            st.markdown(f"""
<div class="editorial-card">
<div class="card-index-strip">
<span>0{idx+1} // {song["track_genre"].upper()}</span>
<span class="match-pill-orange">{match_pct}% MATCH</span>
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