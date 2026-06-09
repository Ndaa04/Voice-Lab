import streamlit as st
import os
import base64
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from pathlib import Path
import pandas as pd
import io
import time

st.set_page_config(
    page_title="Meme Voice Challenge",
    page_icon="🎤",
    layout="wide"
)

# ══════════════════════════════════════════════
#  DIRECTORIES
# ══════════════════════════════════════════════
MEMES_DIR      = Path("memes")
LEADERBOARD_DIR = Path("leaderboards")
MEMES_DIR.mkdir(exist_ok=True)
LEADERBOARD_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════
#  CSS — Dark audio-studio aesthetic
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@400;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Base ── */
.stApp { background:#05090F; color:#BDC8DC; font-family:'Comfortaa',sans-serif; }
.main .block-container { padding-top:1.6rem; max-width:1200px; }

/* ── Headings ── */
h1 { font-family:'Comfortaa',sans-serif !important; font-weight:700 !important;
     font-size:2.1rem !important; color:#fff !important; letter-spacing:-.5px; }
h2 { font-family:'Comfortaa',sans-serif !important; font-weight:700 !important;
     font-size:1.3rem !important; color:#EEF2FF !important;
     padding-bottom:.45rem; border-bottom:1px solid rgba(255,255,255,.06); }
h2::before { content:'// '; color:#00DC6E;
     font-family:'JetBrains Mono',monospace; font-size:.78rem; }
h3 { font-family:'Comfortaa',sans-serif !important; font-weight:600 !important;
     font-size:1rem !important; color:#8A9BB8 !important; }

/* ── Section pill ── */
.pill {
    display:inline-flex; align-items:center; gap:.35rem;
    background:rgba(0,220,110,.07); border:1px solid rgba(0,220,110,.17);
    border-radius:999px; padding:.22rem .8rem;
    font-family:'JetBrains Mono',monospace; font-size:.62rem;
    color:#00DC6E; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:.5rem;
}

/* ── Tutorial modal overlay ── */
.tutorial-overlay {
    position:fixed; inset:0; background:rgba(5,9,15,.92);
    z-index:9999; display:flex; align-items:center; justify-content:center;
    backdrop-filter:blur(6px);
}
.tutorial-box {
    background:#0C1520; border:1px solid rgba(0,220,110,.2);
    border-radius:20px; padding:2.5rem 3rem; max-width:580px; width:90%;
    box-shadow:0 0 60px rgba(0,220,110,.08);
    animation:fadeUp .4s ease;
}
@keyframes fadeUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }

/* ── Meme grid cards ── */
.meme-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:1.1rem; }
.meme-card {
    background:#0C1520; border:2px solid rgba(255,255,255,.07);
    border-radius:16px; overflow:hidden; cursor:pointer;
    transition:border-color .25s, box-shadow .25s, transform .2s;
    position:relative; aspect-ratio:3/4;
    display:flex; flex-direction:column;
}
.meme-card:hover {
    border-color:rgba(0,220,110,.45);
    box-shadow:0 0 28px rgba(0,220,110,.12);
    transform:translateY(-3px);
}
.meme-card:hover .wave-bar { animation-play-state:running; }
.meme-card:not(:hover) .wave-bar { animation-play-state:paused; }

/* Cover image / waveform area */
.meme-cover {
    flex:1; display:flex; align-items:center; justify-content:center;
    position:relative; overflow:hidden; background:#080E18;
}
.meme-cover img { width:100%; height:100%; object-fit:cover; opacity:.85; }

/* Animated waveform (when no image) */
.wave-visual {
    display:flex; align-items:center; gap:3px; height:60px;
    padding:0 1rem;
}
.wave-bar {
    width:4px; background:linear-gradient(180deg,#00DC6E,#00B4D8);
    border-radius:2px; flex-shrink:0;
    animation:waveBounce 1s ease-in-out infinite alternate;
    animation-play-state:paused;
}
@keyframes waveBounce {
    from { transform:scaleY(.2); opacity:.4; }
    to   { transform:scaleY(1);  opacity:1;  }
}

/* Overlay gradient on card */
.meme-overlay {
    position:absolute; inset:0;
    background:linear-gradient(to top, rgba(5,9,15,.9) 0%, transparent 55%);
    pointer-events:none;
}
/* Footer strip */
.meme-footer {
    padding:.65rem .8rem .7rem;
    background:#0C1520;
}
.meme-title {
    font-family:'Comfortaa',sans-serif; font-weight:700;
    font-size:.82rem; color:#D8E8F8;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.meme-tag {
    font-family:'JetBrains Mono',monospace; font-size:.58rem;
    color:#2E4060; letter-spacing:1px; text-transform:uppercase;
    margin-top:.15rem;
}

/* ── Challenge area ── */
.challenge-header {
    background:linear-gradient(135deg,#0A1820,#060D16);
    border:1px solid rgba(0,220,110,.12);
    border-radius:16px; padding:1.4rem 1.8rem; margin-bottom:1.4rem;
    position:relative; overflow:hidden;
}
.challenge-header::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:240px; height:240px;
    background:radial-gradient(circle,rgba(0,220,110,.06) 0%,transparent 70%);
    pointer-events:none;
}

/* ── Cards ── */
.card {
    background:#0C1520; border:1px solid rgba(255,255,255,.06);
    border-radius:13px; padding:1.1rem 1.3rem; height:100%;
}
.card-green  { border-left:3px solid #00DC6E; }
.card-blue   { border-left:3px solid #00B4D8; }
.card-purple { border-left:3px solid #A78BFA; }

/* ── Buttons ── */
.stButton > button {
    background:linear-gradient(135deg,#00C060,#008A40) !important;
    color:#030810 !important; border:none !important;
    border-radius:10px !important;
    font-family:'Comfortaa',sans-serif !important; font-weight:700 !important;
    font-size:.85rem !important; padding:.5rem 1.2rem !important;
    transition:all .22s !important;
    box-shadow:0 0 16px rgba(0,200,90,.15) !important;
}
.stButton > button:hover {
    background:linear-gradient(135deg,#00E676,#00C060) !important;
    box-shadow:0 0 26px rgba(0,230,118,.28) !important;
    transform:translateY(-1px);
}
.stButton > button:disabled {
    background:rgba(255,255,255,.04) !important;
    color:#283848 !important; box-shadow:none !important;
}
.cta .stButton > button {
    background:linear-gradient(135deg,#00E676,#00B84C) !important;
    font-size:.95rem !important; padding:.65rem 1.8rem !important;
    box-shadow:0 0 26px rgba(0,230,120,.2) !important;
}
.back-btn .stButton > button {
    background:rgba(255,255,255,.05) !important;
    color:#8A9BB8 !important;
    box-shadow:none !important; border:1px solid rgba(255,255,255,.08) !important;
}
.back-btn .stButton > button:hover {
    background:rgba(255,255,255,.09) !important;
    color:#C8D8EE !important; box-shadow:none !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background:#0C1520 !important; border:1px solid rgba(255,255,255,.05) !important;
    border-radius:11px !important; padding:.85rem 1rem !important;
}
[data-testid="stMetricLabel"] {
    font-family:'JetBrains Mono',monospace !important; font-size:.58rem !important;
    color:#2E3E58 !important; letter-spacing:1.5px !important; text-transform:uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family:'Comfortaa',sans-serif !important; font-weight:700 !important;
    font-size:1.7rem !important; color:#00DC6E !important;
}

/* ── Score ring ── */
.score-ring-wrap { display:flex; flex-direction:column; align-items:center; gap:.5rem; }
.score-ring {
    width:110px; height:110px; border-radius:50%;
    background:conic-gradient(var(--c) var(--pct), rgba(255,255,255,.05) 0);
    display:flex; align-items:center; justify-content:center;
    box-shadow:0 0 24px rgba(0,220,110,.15);
    position:relative;
}
.score-ring::before {
    content:''; position:absolute; inset:10px;
    background:#0C1520; border-radius:50%;
}
.score-ring-val {
    position:relative; font-family:'Comfortaa',sans-serif;
    font-weight:700; font-size:1.3rem; z-index:1;
}

/* ── Leaderboard ── */
.lb-row {
    display:flex; align-items:center; gap:.75rem;
    padding:.55rem .85rem; border-radius:9px;
    background:#0C1520; border:1px solid rgba(255,255,255,.05);
    margin-bottom:.4rem;
}
.lb-rank {
    font-family:'JetBrains Mono',monospace; font-weight:600;
    font-size:.75rem; color:#2E4060; width:22px; flex-shrink:0;
}
.lb-rank.gold   { color:#FFD700; }
.lb-rank.silver { color:#C0C0C0; }
.lb-rank.bronze { color:#CD7F32; }
.lb-name {
    font-family:'Comfortaa',sans-serif; font-weight:600;
    font-size:.82rem; color:#BDC8DC; flex:1;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.lb-score {
    font-family:'JetBrains Mono',monospace; font-weight:600;
    font-size:.78rem; color:#00DC6E; flex-shrink:0;
}
.lb-method {
    font-family:'JetBrains Mono',monospace; font-size:.58rem;
    color:#1E2E48; letter-spacing:1px; text-transform:uppercase; flex-shrink:0;
}

/* ── Progress ── */
[data-testid="stProgressBar"] > div > div {
    background:linear-gradient(90deg,#00DC6E,#00B4D8) !important; border-radius:999px !important;
}
[data-testid="stProgressBar"] > div {
    background:rgba(255,255,255,.04) !important; border-radius:999px !important;
}

/* ── Alerts ── */
.stSuccess > div { background:rgba(0,220,110,.06) !important; border-left:3px solid #00DC6E !important; border-radius:8px !important; }
.stInfo    > div { background:rgba(0,180,216,.06) !important; border-left:3px solid #00B4D8 !important; border-radius:8px !important; }
.stWarning > div { background:rgba(255,160,0,.06) !important;  border-left:3px solid #FFA000 !important; border-radius:8px !important; }
.stError   > div { background:rgba(255,60,60,.06) !important;  border-left:3px solid #FF4040 !important; border-radius:8px !important; }

/* ── Caption ── */
.stCaption, [data-testid="stCaptionContainer"] {
    font-family:'JetBrains Mono',monospace !important;
    font-size:.63rem !important; color:#1E2E48 !important;
}

/* ── Input ── */
[data-testid="stTextInput"] input {
    background:#0C1520 !important; border-color:rgba(255,255,255,.08) !important;
    color:#BDC8DC !important; font-family:'Comfortaa',sans-serif !important;
    border-radius:8px !important;
}

/* ── Divider ── */
hr { border:none !important; border-top:1px solid rgba(255,255,255,.05) !important; margin:1.5rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#05090F; }
::-webkit-scrollbar-thumb { background:rgba(0,220,110,.15); border-radius:3px; }

/* ── Radio ── */
[data-testid="stRadio"] > div { gap:.5rem !important; }
[data-testid="stRadio"] label {
    background:#0C1520; border:1px solid rgba(255,255,255,.07);
    border-radius:8px; padding:.35rem .75rem !important;
    font-family:'JetBrains Mono',monospace !important; font-size:.72rem !important;
    color:#5A7090 !important; transition:all .18s;
}
[data-testid="stRadio"] label:has(input:checked) {
    border-color:rgba(0,220,110,.35) !important;
    background:#091810 !important; color:#00DC6E !important;
}

/* Sidebar */
[data-testid="stSidebar"] { background:#070D17 !important; border-right:1px solid rgba(0,220,110,.08); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════
def audio_autoplay_html(path, volume=0.75):
    with open(str(path), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext  = Path(path).suffix.lower().lstrip('.')
    mime = {'wav':'audio/wav','mp3':'audio/mpeg','m4a':'audio/mp4','ogg':'audio/ogg'}.get(ext,'audio/wav')
    uid  = f"a{int(time.time()*1000) % 999999}"
    return f"""<audio id="{uid}" autoplay style="display:none">
      <source src="data:{mime};base64,{b64}" type="{mime}">
    </audio>
    <script>(function(){{var a=document.getElementById('{uid}');
    a.volume={volume:.2f};a.play().catch(function(){{}});}})();</script>"""

@st.cache_data(ttl=60)
def scan_memes():
    if not MEMES_DIR.exists():
        return []
    audio_ext = {'.wav','.mp3','.m4a','.ogg'}
    image_ext = {'.jpg','.jpeg','.png','.webp','.gif'}
    memes = []
    for af in sorted(MEMES_DIR.iterdir()):
        if af.suffix.lower() not in audio_ext:
            continue
        stem = af.stem
        img  = next((MEMES_DIR/f"{stem}{e}" for e in image_ext if (MEMES_DIR/f"{stem}{e}").exists()), None)
        vid  = next((MEMES_DIR/f"{stem}{e}" for e in ['.mp4','.webm','.mov'] if (MEMES_DIR/f"{stem}{e}").exists()), None)
        memes.append({"id":stem, "name":stem.replace("_"," ").title(),
                      "audio":af, "image":img, "video":vid})
    return memes

@st.cache_data
def get_waveform_png(audio_path_str: str):
    """Render static waveform → base64 PNG."""
    try:
        y, _ = librosa.load(audio_path_str, sr=16000, duration=3.0)
        fig, ax = plt.subplots(figsize=(3,.9), dpi=80)
        ax.plot(np.linspace(0,1,len(y)), y, color='#00DC6E', linewidth=.8)
        ax.set_xlim(0,1); ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    except:
        return None

@st.cache_data
def get_mfcc_seq(path_str: str):
    try:
        y, sr = librosa.load(path_str, sr=16000)
        if len(y) < sr * 0.3:
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mu = np.mean(mfcc, axis=1, keepdims=True)
        sd = np.std(mfcc,  axis=1, keepdims=True) + 1e-8
        return ((mfcc - mu) / sd).T
    except:
        return None

def compute_sim(ref, test, method="FastDTW"):
    if ref is None or test is None:
        return 0.0
    dist, _ = fastdtw(ref, test, dist=euclidean)
    avg_len  = (len(ref) + len(test)) / 2
    nd       = dist / avg_len
    return float(np.exp(-nd))

def load_lb(meme_id: str) -> pd.DataFrame:
    f = LEADERBOARD_DIR / f"{meme_id}.csv"
    if f.exists():
        try:
            return pd.read_csv(f)
        except:
            pass
    return pd.DataFrame(columns=['name','score','method','timestamp'])

def save_lb(meme_id: str, df: pd.DataFrame):
    df = df.sort_values('score', ascending=False).head(15).reset_index(drop=True)
    df.to_csv(LEADERBOARD_DIR / f"{meme_id}.csv", index=False)
    return df

def score_color(s):
    if s >= 0.7: return "#00DC6E"
    if s >= 0.4: return "#FFD060"
    return "#FF6060"

def animated_wave_bars(n=18):
    """Generate HTML for animated SVG-like wave bars."""
    heights = [20,35,50,40,60,45,30,55,42,38,58,32,48,36,52,28,44,60][:n]
    bars = ""
    for i, h in enumerate(heights):
        delay = i * 0.07
        bars += f'<div class="wave-bar" style="height:{h}px;animation-delay:{delay:.2f}s"></div>'
    return f'<div class="wave-visual">{bars}</div>'

# ══════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════
if "tutorial_done" not in st.session_state: st.session_state.tutorial_done = False
if "selected_meme" not in st.session_state: st.session_state.selected_meme = None
if "play_ref"      not in st.session_state: st.session_state.play_ref      = False
if "lb_submitted"  not in st.session_state: st.session_state.lb_submitted  = False

memes = scan_memes()

# ══════════════════════════════════════════════
#  TUTORIAL MODAL (before anything)
# ══════════════════════════════════════════════
if not st.session_state.tutorial_done:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;min-height:70vh;text-align:center">
      <div style="background:#0C1520;border:1px solid rgba(0,220,110,.2);
                  border-radius:20px;padding:2.5rem 2.8rem;max-width:560px;
                  box-shadow:0 0 60px rgba(0,220,110,.07)">
        <div style="font-size:2.8rem;margin-bottom:.8rem">🎤</div>
        <div style="font-family:'Comfortaa',sans-serif;font-weight:700;
                    font-size:1.6rem;color:#fff;margin-bottom:.5rem">Meme Voice Challenge</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;
                    color:#00DC6E;letter-spacing:2px;text-transform:uppercase;
                    margin-bottom:1.4rem">Cara Bermain</div>
        <div style="background:#080E18;border-radius:12px;padding:1.2rem 1.4rem;
                    margin-bottom:1.5rem;text-align:left">
          <div style="display:flex;flex-direction:column;gap:.75rem">
            <div style="display:flex;gap:.9rem;align-items:flex-start">
              <span style="color:#00DC6E;font-family:'JetBrains Mono',monospace;
                           font-size:.75rem;font-weight:600;flex-shrink:0">01</span>
              <span style="font-size:.88rem;color:#8A9BB8;line-height:1.5">
                <strong style="color:#BDC8DC">Pilih meme</strong> dari galeri —
                klik kartu meme yang ingin kamu tirukan.</span>
            </div>
            <div style="display:flex;gap:.9rem;align-items:flex-start">
              <span style="color:#00DC6E;font-family:'JetBrains Mono',monospace;
                           font-size:.75rem;font-weight:600;flex-shrink:0">02</span>
              <span style="font-size:.88rem;color:#8A9BB8;line-height:1.5">
                <strong style="color:#BDC8DC">Dengarkan suara referensi</strong>
                lalu tirukan sebaik mungkin — nada, ritme, dan ekspresinya.</span>
            </div>
            <div style="display:flex;gap:.9rem;align-items:flex-start">
              <span style="color:#00DC6E;font-family:'JetBrains Mono',monospace;
                           font-size:.75rem;font-weight:600;flex-shrink:0">03</span>
              <span style="font-size:.88rem;color:#8A9BB8;line-height:1.5">
                <strong style="color:#BDC8DC">Rekam suaramu</strong> dan sistem akan
                menghitung skor kemiripan menggunakan algoritma DTW.</span>
            </div>
            <div style="display:flex;gap:.9rem;align-items:flex-start">
              <span style="color:#00DC6E;font-family:'JetBrains Mono',monospace;
                           font-size:.75rem;font-weight:600;flex-shrink:0">04</span>
              <span style="font-size:.88rem;color:#8A9BB8;line-height:1.5">
                Masukkan namamu ke <strong style="color:#BDC8DC">Leaderboard</strong>
                dan tantang temanmu untuk mengalahkan skormu!</span>
            </div>
          </div>
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;
                    color:#2E4060;margin-bottom:1.4rem">
          Skor 0.0 (berbeda total) → 1.0 (identik sempurna)
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_tut, _ = st.columns([2, 4])
    with col_tut:
        st.markdown('<div class="cta">', unsafe_allow_html=True)
        if st.button("🎮  Mulai Bermain!", key="start_tutorial"):
            st.session_state.tutorial_done = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════
#  MEME GALLERY VIEW
# ══════════════════════════════════════════════
if st.session_state.selected_meme is None:

    # Header
    st.markdown('<div class="pill">🎤 Meme Voice Challenge</div>', unsafe_allow_html=True)
    st.title("Pilih Meme")
    st.markdown('<p style="color:#3C5070;font-size:.9rem;margin-bottom:1.5rem">Klik kartu untuk memulai tantangan — tirukan suara meme seakurat mungkin!</p>', unsafe_allow_html=True)

    if not memes:
        st.warning("Folder `memes/` kosong. Masukkan file audio (.wav/.mp3) ke folder `memes/` — kartu akan muncul otomatis!")
        st.info("Contoh struktur:\n```\nmemes/\n  cat_meme.mp3\n  cat_meme.jpg   ← opsional (gambar)\n  cat_meme.mp4   ← opsional (video)\n```")
        st.stop()

    # ── Render grid via HTML + st.button trick ──
    # Karena Streamlit tidak support custom click HTML, kita pakai kolom + HTML card + button overlay
    COLS_PER_ROW = 4
    rows = [memes[i:i+COLS_PER_ROW] for i in range(0, len(memes), COLS_PER_ROW)]

    for row in rows:
        cols = st.columns(len(row))
        for col, meme in zip(cols, row):
            wf_b64 = get_waveform_png(str(meme['audio']))
            with col:
                # Cover area HTML
                if meme['image'] and meme['image'].exists():
                    with open(meme['image'], 'rb') as f:
                        img_b64 = base64.b64encode(f.read()).decode()
                    ext = meme['image'].suffix.lower().lstrip('.')
                    img_mime = {'jpg':'jpeg','jpeg':'jpeg','png':'png',
                                'webp':'webp','gif':'gif'}.get(ext,'jpeg')
                    cover_html = f"""
                    <img src="data:image/{img_mime};base64,{img_b64}"
                         style="width:100%;height:180px;object-fit:cover;display:block;
                                border-radius:12px 12px 0 0">"""
                else:
                    # Waveform + animated bars (no image)
                    wf_part = (f'<img src="data:image/png;base64,{wf_b64}" '
                               f'style="width:90%;opacity:.5;margin-bottom:.4rem">'
                               if wf_b64 else '')
                    cover_html = f"""
                    <div style="height:180px;background:#080E18;border-radius:12px 12px 0 0;
                                display:flex;flex-direction:column;align-items:center;
                                justify-content:center;gap:.3rem;">
                      {wf_part}
                      {animated_wave_bars(16)}
                    </div>"""

                # Badge (has video / has image)
                badges = []
                if meme['video']: badges.append('<span style="background:rgba(167,139,250,.15);border:1px solid rgba(167,139,250,.3);color:#A78BFA;font-family:JetBrains Mono,monospace;font-size:.55rem;padding:.1rem .45rem;border-radius:4px;letter-spacing:1px">VIDEO</span>')
                if meme['image']: badges.append('<span style="background:rgba(0,180,216,.12);border:1px solid rgba(0,180,216,.25);color:#00B4D8;font-family:JetBrains Mono,monospace;font-size:.55rem;padding:.1rem .45rem;border-radius:4px;letter-spacing:1px">IMG</span>')
                badge_html = f'<div style="display:flex;gap:.3rem;margin-top:.3rem">{"".join(badges)}</div>' if badges else ''

                st.markdown(f"""
                <div style="background:#0C1520;border:2px solid rgba(255,255,255,.07);
                            border-radius:13px;overflow:hidden;margin-bottom:.1rem">
                  {cover_html}
                  <div style="padding:.6rem .8rem .7rem">
                    <div style="font-family:'Comfortaa',sans-serif;font-weight:700;
                                font-size:.84rem;color:#D8E8F8;white-space:nowrap;
                                overflow:hidden;text-overflow:ellipsis">{meme['name']}</div>
                    {badge_html}
                  </div>
                </div>""", unsafe_allow_html=True)

                # Invisible button that triggers selection
                if st.button(f"Pilih →", key=f"sel_{meme['id']}", use_container_width=True):
                    st.session_state.selected_meme = meme
                    st.session_state.play_ref      = False
                    st.session_state.lb_submitted  = False
                    st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(f'<p style="font-family:JetBrains Mono,monospace;font-size:.62rem;color:#1A2838;text-align:center">{len(memes)} meme tersedia · Tambah file audio di folder memes/ untuk meme baru</p>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  CHALLENGE VIEW
# ══════════════════════════════════════════════
else:
    meme = st.session_state.selected_meme
    # Refresh meme data (in case files changed)
    fresh = next((m for m in memes if m['id'] == meme['id']), meme)
    meme  = fresh

    # ── Back button ──
    col_back, col_title = st.columns([1, 6])
    with col_back:
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← Galeri", key="back"):
            st.session_state.selected_meme = None
            st.session_state.play_ref      = False
            st.session_state.lb_submitted  = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Challenge header ──
    st.markdown(f"""
    <div class="challenge-header">
      <div style="font-family:'JetBrains Mono',monospace;font-size:.62rem;
                  color:#00DC6E;letter-spacing:2px;text-transform:uppercase;margin-bottom:.4rem">
        🎤 Meme Challenge
      </div>
      <div style="font-family:'Comfortaa',sans-serif;font-weight:700;
                  font-size:1.5rem;color:#fff">{meme['name']}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;
                  color:#2E4060;margin-top:.3rem">
        Tirukan suara seakurat mungkin — skor mendekati 1.0 berarti sangat mirip
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Layout: 3 kolom ──
    col_ref, col_rec, col_lb = st.columns([1.1, 1.1, 1])

    # ─────── KOLOM 1: REFERENSI ───────
    with col_ref:
        st.markdown('<div class="card card-green">', unsafe_allow_html=True)
        st.markdown('<div class="pill">📢 Referensi</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-family:Comfortaa,sans-serif;font-weight:700;font-size:.95rem;color:#D8E8F8;margin-bottom:.7rem">{meme["name"]}</div>', unsafe_allow_html=True)

        # Waveform referensi
        try:
            y_ref, sr_ref = librosa.load(str(meme['audio']), sr=16000, duration=5.0)
            fig_r, ax_r = plt.subplots(figsize=(4, 1.5))
            ax_r.plot(np.linspace(0, len(y_ref)/sr_ref, len(y_ref)), y_ref,
                      color='#00DC6E', linewidth=.9, alpha=.85)
            ax_r.fill_between(np.linspace(0, len(y_ref)/sr_ref, len(y_ref)),
                              y_ref, alpha=.08, color='#00DC6E')
            ax_r.set_xlabel(""); ax_r.set_ylabel(""); ax_r.set_xticks([]); ax_r.set_yticks([])
            for sp in ax_r.spines.values(): sp.set_visible(False)
            fig_r.patch.set_alpha(0); ax_r.set_facecolor('none')
            fig_r.tight_layout(pad=.3)
            st.pyplot(fig_r, use_container_width=True)
            plt.close(fig_r)
            dur = len(y_ref)/sr_ref
            st.caption(f"Durasi: {dur:.1f}s  ·  SR: {sr_ref} Hz")
        except Exception as e:
            st.warning(f"Gagal load audio: {e}")

        # Play button
        if st.button("▶  Putar Referensi", key="play_ref_btn"):
            st.session_state.play_ref = True
        if st.session_state.play_ref:
            st.markdown(audio_autoplay_html(str(meme['audio'])), unsafe_allow_html=True)
            st.session_state.play_ref = False

        # Image / Video
        if meme.get('image') and Path(str(meme['image'])).exists():
            st.markdown('<div style="margin-top:.8rem;border-radius:10px;overflow:hidden">', unsafe_allow_html=True)
            st.image(str(meme['image']), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if meme.get('video') and Path(str(meme['video'])).exists():
            st.markdown('<div style="margin-top:.6rem">', unsafe_allow_html=True)
            if st.button("🎬  Tampilkan Video", key="show_vid"):
                st.session_state['show_video'] = not st.session_state.get('show_video', False)
            if st.session_state.get('show_video', False):
                st.video(str(meme['video']))
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ─────── KOLOM 2: REKAM ───────
    with col_rec:
        st.markdown('<div class="card card-blue">', unsafe_allow_html=True)
        st.markdown('<div class="pill">🎙️ Rekam Suaramu</div>', unsafe_allow_html=True)

        # Pilih metode
        method = st.radio(
            "Metode penghitungan:",
            ["FastDTW", "DTW (standar)"],
            horizontal=True, key="dtw_method",
            label_visibility="visible"
        )
        st.markdown('<div style="height:.4rem"></div>', unsafe_allow_html=True)

        user_audio = st.audio_input("Klik mikrofon, lalu tirukan suara meme:", key="rec_input")

        if user_audio is not None:
            st.audio(user_audio, format="audio/wav")

            # Simpan sementara
            tmp_path = f"_tmp_{meme['id']}.wav"
            with open(tmp_path, "wb") as f:
                f.write(user_audio.getbuffer())

            seq_ref  = get_mfcc_seq(str(meme['audio']))
            seq_user = get_mfcc_seq(tmp_path)

            if seq_ref is not None and seq_user is not None:
                score = compute_sim(seq_ref, seq_user, method)
                sc    = score_color(score)
                pct   = int(score * 360)
                pct_s = f"{pct}deg"

                # Score ring
                st.markdown(f"""
                <div class="score-ring-wrap" style="margin:1rem 0">
                  <div class="score-ring" style="--c:{sc};--pct:{pct_s}">
                    <div class="score-ring-val" style="color:{sc}">{score:.3f}</div>
                  </div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;
                              color:#2E4060;letter-spacing:1px">SKOR KEMIRIPAN</div>
                </div>""", unsafe_allow_html=True)

                # Feedback teks
                if score >= 0.75:
                    st.success("🏆 Luar biasa! Hampir sempurna!")
                elif score >= 0.55:
                    st.success("✅ Bagus! Cukup mirip.")
                elif score >= 0.35:
                    st.info("🔊 Lumayan — coba tirukan lebih dekat.")
                else:
                    st.warning("😅 Masih jauh — dengarkan lagi dan coba ulang!")

                # Waveform perbandingan
                try:
                    y_u, sr_u = librosa.load(tmp_path, sr=16000, duration=5.0)
                    fig_u, ax_u = plt.subplots(figsize=(4, 1.2))
                    ax_u.plot(np.linspace(0,1,len(y_u)), y_u, color='#00B4D8', linewidth=.9)
                    ax_u.fill_between(np.linspace(0,1,len(y_u)), y_u, alpha=.08, color='#00B4D8')
                    ax_u.set_xticks([]); ax_u.set_yticks([])
                    for sp in ax_u.spines.values(): sp.set_visible(False)
                    fig_u.patch.set_alpha(0); ax_u.set_facecolor('none')
                    fig_u.tight_layout(pad=.2)
                    st.caption("Waveform rekaman kamu:")
                    st.pyplot(fig_u, use_container_width=True)
                    plt.close(fig_u)
                except:
                    pass

                # ── Submit ke Leaderboard ──
                st.markdown('<div style="margin-top:.9rem;border-top:1px solid rgba(255,255,255,.05);padding-top:.9rem">', unsafe_allow_html=True)
                st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:.63rem;color:#2E4060;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:.4rem">Simpan ke Leaderboard</div>', unsafe_allow_html=True)
                player_name = st.text_input("Nama kamu:", placeholder="Masukkan nama…", key="player_name", label_visibility="collapsed")

                col_sub, col_hint = st.columns([2,3])
                with col_sub:
                    st.markdown('<div class="cta">', unsafe_allow_html=True)
                    if st.button("💾  Simpan Skor", key="submit_lb"):
                        name = player_name.strip() or "Anonymous"
                        df_lb = load_lb(meme['id'])
                        new_row = pd.DataFrame([{
                            'name': name, 'score': score,
                            'method': method,
                            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                        }])
                        df_lb = pd.concat([df_lb, new_row], ignore_index=True)
                        df_lb = save_lb(meme['id'], df_lb)
                        st.session_state.lb_submitted = True
                        st.session_state['last_score'] = score
                        st.session_state['last_name']  = name
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # Cleanup
                try: os.remove(tmp_path)
                except: pass

            else:
                st.error("Gagal memproses suara — rekam minimal 0.5 detik.")
                try: os.remove(tmp_path)
                except: pass

        # Konfirmasi submit
        if st.session_state.get('lb_submitted'):
            n = st.session_state.get('last_name','')
            s = st.session_state.get('last_score', 0)
            st.success(f"✅ Skor **{s:.3f}** atas nama **{n}** tersimpan!")
            st.session_state.lb_submitted = False

        st.markdown('</div>', unsafe_allow_html=True)

    # ─────── KOLOM 3: LEADERBOARD ───────
    with col_lb:
        st.markdown('<div class="card card-purple">', unsafe_allow_html=True)
        st.markdown('<div class="pill">🏆 Leaderboard</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-family:Comfortaa,sans-serif;font-weight:700;font-size:.88rem;color:#8A9BB8;margin-bottom:.8rem">{meme["name"]}</div>', unsafe_allow_html=True)

        df_lb = load_lb(meme['id'])

        if df_lb.empty:
            st.markdown("""
            <div style="text-align:center;padding:2rem 1rem">
              <div style="font-size:2rem;margin-bottom:.5rem">🏅</div>
              <div style="font-family:JetBrains Mono,monospace;font-size:.65rem;
                          color:#1E2E48;letter-spacing:1px">Belum ada skor.<br>Jadilah yang pertama!</div>
            </div>""", unsafe_allow_html=True)
        else:
            df_show = df_lb.sort_values('score', ascending=False).head(10).reset_index(drop=True)
            rank_icons = {0:"🥇", 1:"🥈", 2:"🥉"}
            rank_cls   = {0:"gold", 1:"silver", 2:"bronze"}
            for i, row in df_show.iterrows():
                icon = rank_icons.get(i, f"#{i+1}")
                cls  = rank_cls.get(i, "")
                sc   = score_color(row['score'])
                mth  = str(row.get('method','DTW'))[:7]
                ts   = str(row.get('timestamp',''))[:10]
                st.markdown(f"""
                <div class="lb-row">
                  <div class="lb-rank {cls}">{icon}</div>
                  <div class="lb-name">{row['name']}</div>
                  <div class="lb-score" style="color:{sc}">{row['score']:.3f}</div>
                  <div class="lb-method">{mth}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:.58rem;color:#1A2838;margin-top:.5rem">{len(df_show)} dari maks 15 skor teratas · {ts}</div>', unsafe_allow_html=True)

        # Download sticker (coming soon)
        st.markdown('<div style="margin-top:1.2rem;border-top:1px solid rgba(255,255,255,.05);padding-top:.9rem">', unsafe_allow_html=True)
        st.button("🎨  Download Sticker", key="dl_sticker", disabled=True)
        st.caption("🚧 Coming soon — stiker eksklusif untuk skor tertinggi!")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer ──
    st.markdown("---")
    st.markdown('<div style="text-align:center;padding:.5rem 0"><span style="font-family:JetBrains Mono,monospace;font-size:.58rem;color:#10181F;letter-spacing:3px;text-transform:uppercase">Meme Voice Challenge · Voice Lab</span></div>', unsafe_allow_html=True)
