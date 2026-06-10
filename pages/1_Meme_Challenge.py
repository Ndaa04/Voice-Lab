import streamlit as st
import os
import base64
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
from pathlib import Path
import pandas as pd
import io
import time
import random

st.set_page_config(
    page_title="Meme Voice Challenge",
    page_icon="🎤",
    layout="wide"
)

MEMES_DIR = Path("memes")
LEADERBOARD_DIR = Path("leaderboards")
MEMES_DIR.mkdir(exist_ok=True)
LEADERBOARD_DIR.mkdir(exist_ok=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@400;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

.stApp { background:#05090F; color:#BDC8DC; font-family:'Comfortaa',sans-serif; }
.main .block-container { padding-top:1.6rem; max-width:1200px; }

h1 { font-family:'Comfortaa',sans-serif !important; font-weight:700 !important;
     font-size:2.1rem !important; color:#fff !important; letter-spacing:-.5px; }
h2 { font-family:'Comfortaa',sans-serif !important; font-weight:700 !important;
     font-size:1.3rem !important; color:#EEF2FF !important;
     padding-bottom:.45rem; border-bottom:1px solid rgba(255,255,255,.06); }
h2::before { content:'// '; color:#00DC6E;
     font-family:'JetBrains Mono',monospace; font-size:.78rem; }
h3 { font-family:'Comfortaa',sans-serif !important; font-weight:600 !important;
     font-size:1rem !important; color:#8A9BB8 !important; }

.pill {
    display:inline-flex; align-items:center; gap:.35rem;
    background:rgba(0,220,110,.07); border:1px solid rgba(0,220,110,.17);
    border-radius:999px; padding:.22rem .8rem;
    font-family:'JetBrains Mono',monospace; font-size:.62rem;
    color:#00DC6E; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:.5rem;
}

.meme-card {
    background:#0C1520; border:2px solid rgba(255,255,255,.07);
    border-radius:16px; overflow:hidden; cursor:pointer;
    transition:border-color .25s, box-shadow .25s, transform .2s;
    display:flex; flex-direction:column;
}
.meme-card:hover {
    border-color:rgba(0,220,110,.45);
    box-shadow:0 0 28px rgba(0,220,110,.12);
    transform:translateY(-3px);
}
.meme-card:hover .wave-bar { animation-play-state:running; }
.meme-card:not(:hover) .wave-bar { animation-play-state:paused; }

.waveform-container {
    height:100px;
    background:#080E18;
    border-radius:12px 12px 0 0;
    display:flex;
    align-items:center;
    justify-content:center;
}
.meme-card img {
    width:100%;
    height:100px;
    object-fit:cover;
    display:block;
    border-radius:12px 12px 0 0;
}

.meme-footer {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 0.5rem 0.3rem;
    background: #0C1520;
}
            
.meme-title {
    font-family: 'Comfortaa', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    color: #D8E8F8;
    line-height: 1.2;
    word-break: break-word;
    white-space: normal;
    overflow-wrap: break-word;
    max-width: 100%;
    margin-bottom: 0.2rem;
    text-align: center !important;
}
            
.badge-wrapper {
    display:flex;
    gap:0.3rem;
    justify-content:center;
    margin-top:0.2rem;
}
.badge-video, .badge-img {
    display:inline-block;
    background:rgba(167,139,250,.15);
    border:1px solid rgba(167,139,250,.3);
    color:#A78BFA;
    font-family:'JetBrains Mono',monospace;
    font-size:0.45rem;
    padding:0.05rem 0.3rem;
    border-radius:4px;
    letter-spacing:1px;
}
.badge-img {
    background:rgba(0,180,216,.12);
    border-color:rgba(0,180,216,.25);
    color:#00B4D8;
}

.stButton > button {
    background:linear-gradient(135deg,#00C060,#008A40) !important;
    color:#030810 !important;
    border:none !important;
    border-radius:10px !important;
    font-family:'Comfortaa',sans-serif !important;
    font-weight:700 !important;
    font-size:0.75rem !important;
    padding:0.3rem 0.8rem !important;
    transition:all .22s !important;
    box-shadow:0 0 16px rgba(0,200,90,.15) !important;
    width:100%;
}
.stButton > button:hover {
    background:linear-gradient(135deg,#00E676,#00C060) !important;
    box-shadow:0 0 26px rgba(0,230,118,.28) !important;
    transform:translateY(-1px);
}
.cta .stButton > button {
    background:linear-gradient(135deg,#00E676,#00B84C) !important;
    font-size:.95rem !important; padding:.65rem 1.8rem !important;
    box-shadow:0 0 26px rgba(0,230,120,.2) !important;
    width:auto;
}
.back-btn .stButton > button {
    background:rgba(255,255,255,.05) !important;
    color:#8A9BB8 !important;
    box-shadow:none !important;
    border:1px solid rgba(255,255,255,.08) !important;
    width:auto;
}
.back-btn .stButton > button:hover {
    background:rgba(255,255,255,.09) !important;
    color:#C8D8EE !important;
}

.wave-visual {
    display:flex;
    align-items:center;
    justify-content:center;
    gap:3px;
    height:100%;
}
.wave-bar {
    width:4px;
    background:linear-gradient(180deg,#00DC6E,#00B4D8);
    border-radius:2px;
    flex-shrink:0;
    animation:waveBounce 1s ease-in-out infinite alternate;
    animation-play-state:paused;
}
@keyframes waveBounce {
    from { transform:scaleY(.2); opacity:.4; }
    to   { transform:scaleY(1);  opacity:1;  }
}

[data-testid="stMetric"] {
    background:#0C1520 !important;
    border:1px solid rgba(255,255,255,.05) !important;
    border-radius:11px !important;
    padding:.85rem 1rem !important;
}
[data-testid="stMetricValue"] {
    font-family:'Comfortaa',sans-serif !important;
    font-weight:700 !important;
    font-size:1.7rem !important;
    color:#00DC6E !important;
}
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

[data-testid="stRadio"] > div { gap:.5rem !important; }
[data-testid="stRadio"] label {
    background:#0C1520; border:1px solid rgba(255,255,255,.07);
    border-radius:8px; padding:.35rem .75rem !important;
    font-family:'JetBrains Mono',monospace !important; font-size:.72rem !important;
    color:#5A7090 !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    border-color:rgba(0,220,110,.35) !important;
    background:#091810 !important; color:#00DC6E !important;
}

[data-testid="stAudioInput"] > div {
    background:#0C1520 !important;
    border:1px solid rgba(255,255,255,.08) !important;
    border-radius:10px !important;
}
.stCaption, [data-testid="stCaptionContainer"] {
    font-family:'JetBrains Mono',monospace !important;
    font-size:.63rem !important; color:#1E2E48 !important;
}
hr { border:none !important; border-top:1px solid rgba(255,255,255,.05) !important; margin:1.5rem 0 !important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#05090F; }
::-webkit-scrollbar-thumb { background:rgba(0,220,110,.15); border-radius:3px; }
[data-testid="stSidebar"] { background:#070D17 !important; border-right:1px solid rgba(0,220,110,.08); }

/* Panel challenge — tanpa border wrapper ganda */
.panel-box {
    background:#0C1520;
    border-radius:13px;
    border:1px solid rgba(255,255,255,.08);
    padding:1rem 1.1rem;
    height:100%;
}
.panel-box.green  { border-left:3px solid #00DC6E; }
.panel-box.blue   { border-left:3px solid #00B4D8; }
.panel-box.purple { border-left:3px solid #A78BFA; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════
def audio_autoplay_html(path, volume=0.75):
    with open(str(path), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = Path(path).suffix.lower().lstrip('.')
    mime = {'wav':'audio/wav','mp3':'audio/mpeg','m4a':'audio/mp4','ogg':'audio/ogg'}.get(ext,'audio/wav')
    uid = f"a{int(time.time()*1000) % 999999}"
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
        img = next((MEMES_DIR/f"{stem}{e}" for e in image_ext if (MEMES_DIR/f"{stem}{e}").exists()), None)
        vid = next((MEMES_DIR/f"{stem}{e}" for e in ['.mp4','.webm','.mov'] if (MEMES_DIR/f"{stem}{e}").exists()), None)
        memes.append({"id":stem, "name":stem.replace("_"," ").title(),
                      "audio":af, "image":img, "video":vid})
    return memes

def animated_wave_bars(n=24):
    heights = [random.randint(15, 60) for _ in range(n)]
    delays = [random.uniform(0, 0.5) for _ in range(n)]
    bars = ""
    for h, d in zip(heights, delays):
        bars += f'<div class="wave-bar" style="height:{h}px;animation-delay:{d:.2f}s"></div>'
    return f'<div class="wave-visual">{bars}</div>'

@st.cache_data  # hanya untuk file referensi (path stabil)
def get_mfcc_ref(path_str: str):
    try:
        y, sr = librosa.load(path_str, sr=16000)
        if len(y) < sr * 0.3:
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mu = np.mean(mfcc, axis=1, keepdims=True)
        sd = np.std(mfcc, axis=1, keepdims=True) + 1e-8
        return ((mfcc - mu) / sd).T
    except Exception as e:
        st.error(f"MFCC error: {e}")
        return None

# TANPA cache — dipanggil setiap rekaman baru
def get_mfcc_seq(path_str: str):
    try:
        y, sr = librosa.load(path_str, sr=16000)
        if len(y) < sr * 0.3:
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mu = np.mean(mfcc, axis=1, keepdims=True)
        sd = np.std(mfcc, axis=1, keepdims=True) + 1e-8
        return ((mfcc - mu) / sd).T
    except Exception as e:
        st.error(f"MFCC error: {e}")
        return None

def compute_dtw_raw(ref, test):
    """Return raw FastDTW distance. Smaller = more similar."""
    if ref is None or test is None:
        return None
    try:
        raw_dist, _ = fastdtw(ref, test, dist=cosine)
        return float(raw_dist)
    except Exception as e:
        st.error(f"DTW error: {e}")
        return None

# Threshold untuk DTW raw cosine distance:
# 0-15  → sangat mirip (hijau)
# 15-35 → lumayan (kuning)
# >35   → jauh (merah)
def score_color(s):
    if s <= 15.0:  return "#00DC6E"
    if s <= 35.0:  return "#FFD060"
    return "#FF6060"

def score_label(s):
    if s <= 15.0:  return ("🏆 Luar biasa! Sangat mirip!", "success")
    if s <= 35.0:  return ("✅ Bagus! Cukup mirip.", "success")
    if s <= 55.0:  return ("🔊 Lumayan, coba lagi.", "info")
    return ("😅 Masih jauh, coba ulang!", "warning")

# Untuk score ring: makin kecil DTW makin penuh.
# Kita clamp di [0, 100] lalu invert: ring_pct = max(0, 1 - score/100)
def score_ring_pct(s):
    return max(0.0, 1.0 - s / 100.0)

def load_lb(meme_id: str) -> pd.DataFrame:
    f = LEADERBOARD_DIR / f"{meme_id}.csv"
    if f.exists():
        try:
            return pd.read_csv(f)
        except:
            pass
    return pd.DataFrame(columns=['name','score','method','timestamp'])

def save_lb(meme_id: str, df: pd.DataFrame):
    # ascending=True: nilai DTW terkecil = terbaik
    df = df.sort_values('score', ascending=True).head(15).reset_index(drop=True)
    df.to_csv(LEADERBOARD_DIR / f"{meme_id}.csv", index=False)
    return df

# ══════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════
if "tutorial_done" not in st.session_state: st.session_state.tutorial_done = False
if "selected_meme" not in st.session_state: st.session_state.selected_meme = None
if "play_ref" not in st.session_state: st.session_state.play_ref = False
if "lb_submitted" not in st.session_state: st.session_state.lb_submitted = False

memes = scan_memes()

# ══════════════════════════════════════════════
#  TUTORIAL
# ══════════════════════════════════════════════
if not st.session_state.tutorial_done:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;min-height:70vh;text-align:center">
      <div style="background:#0C1520;border:1px solid rgba(0,220,110,.2);
                  border-radius:20px;padding:2rem;max-width:560px;
                  box-shadow:0 0 60px rgba(0,220,110,.07)">
        <div style="font-size:2.8rem;margin-bottom:.8rem">🎤</div>
        <div style="font-family:'Comfortaa',sans-serif;font-weight:700;
                    font-size:1.6rem;color:#fff;margin-bottom:.5rem">Meme Voice Challenge</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;
                    color:#00DC6E;letter-spacing:2px;text-transform:uppercase;
                    margin-bottom:1.4rem">Cara Bermain</div>
        <div style="background:#080E18;border-radius:12px;padding:1rem;
                    margin-bottom:1.5rem;text-align:left">
          <div style="display:flex;flex-direction:column;gap:.75rem">
            <div>01. Pilih meme dari galeri</div>
            <div>02. Dengarkan suara referensi</div>
            <div>03. Rekam suaramu</div>
            <div>04. Simpan skor ke leaderboard</div>
          </div>
        </div>
        <div style="background:#0A1A10;border:1px solid rgba(0,220,110,.15);border-radius:10px;padding:.8rem;margin-bottom:1rem;font-family:'JetBrains Mono',monospace;font-size:.62rem;color:#5A9070;">
          Skor = nilai DTW mentah · Makin kecil = makin mirip 🎯
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
    st.markdown('<div class="pill">🎤 Meme Voice Challenge</div>', unsafe_allow_html=True)
    st.title("Pilih Meme")
    st.markdown('<p style="color:#3C5070;font-size:.9rem;margin-bottom:1.5rem">Klik kartu untuk memulai tantangan — tirukan suara meme seakurat mungkin!</p>', unsafe_allow_html=True)

    if not memes:
        st.warning("Folder `memes/` kosong. Masukkan file audio (.wav/.mp3) ke folder `memes/` — kartu akan muncul otomatis!")
        st.stop()

    COLS_PER_ROW = 4
    rows = [memes[i:i+COLS_PER_ROW] for i in range(0, len(memes), COLS_PER_ROW)]

    for row in rows:
        cols = st.columns(len(row))
        for col, meme in zip(cols, row):
            with col:
                if meme['image'] and meme['image'].exists():
                    with open(meme['image'], 'rb') as f:
                        img_b64 = base64.b64encode(f.read()).decode()
                    ext = meme['image'].suffix.lower().lstrip('.')
                    img_mime = {'jpg':'jpeg','jpeg':'jpeg','png':'png','webp':'webp','gif':'gif'}.get(ext,'jpeg')
                    cover_html = f'<img src="data:image/{img_mime};base64,{img_b64}" style="width:100%;height:180px;object-fit:cover;display:block;border-radius:12px 12px 0 0">'
                else:
                    cover_html = f'<div class="waveform-container">{animated_wave_bars(24)}</div>'

                badges = []
                if meme['video']:
                    badges.append('<span class="badge-video">VIDEO</span>')
                if meme['image']:
                    badges.append('<span class="badge-img">IMG</span>')
                badge_html = f'<div style="display:flex;gap:.3rem;margin-top:.3rem">{"".join(badges)}</div>' if badges else ''

                st.markdown(f"""
                <div class="meme-card">
                  {cover_html}
                  <div style="padding:.6rem .8rem .7rem; background:#0C1520;">
                    <div class="meme-title">{meme['name']}</div>
                    {badge_html}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"Pilih →", key=f"sel_{meme['id']}", use_container_width=True):
                    st.session_state.selected_meme = meme
                    st.session_state.play_ref = False
                    st.session_state.lb_submitted = False
                    st.rerun()

    st.markdown("---")
    st.markdown(f'<p style="font-family:JetBrains Mono,monospace;font-size:.62rem;color:#1A2838;text-align:center">{len(memes)} meme tersedia · Tambah file audio di folder memes/ untuk meme baru</p>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  CHALLENGE VIEW
# ══════════════════════════════════════════════
else:
    meme = st.session_state.selected_meme
    fresh = next((m for m in memes if m['id'] == meme['id']), None)
    if fresh is None:
        st.error("Meme tidak ditemukan. Kembali ke galeri.")
        st.session_state.selected_meme = None
        st.rerun()
    meme = fresh

    col_back, _ = st.columns([1, 6])
    with col_back:
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← Galeri", key="back"):
            st.session_state.selected_meme = None
            st.session_state.play_ref = False
            st.session_state.lb_submitted = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0A1820,#060D16);border:1px solid rgba(0,220,110,.12);border-radius:16px;padding:1.4rem 1.8rem;margin-bottom:1.4rem">
      <div style="font-family:'JetBrains Mono',monospace;font-size:.62rem;color:#00DC6E;letter-spacing:2px">🎤 Meme Challenge</div>
      <div style="font-family:'Comfortaa',sans-serif;font-weight:700;font-size:1.5rem;color:#fff">{meme['name']}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#2E4060">Skor = DTW raw distance · Makin kecil = makin mirip 🎯</div>
    </div>""", unsafe_allow_html=True)

    col_ref, col_rec, col_lb = st.columns([1.1, 1.1, 1])

    # ── PANEL REFERENSI ──────────────────────────────
    with col_ref:
        st.markdown('<div class="panel-box green">', unsafe_allow_html=True)
        st.markdown('<div class="pill">📢 Referensi</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-family:Comfortaa,sans-serif;font-weight:700;font-size:.95rem;color:#D8E8F8;margin-bottom:.7rem">{meme["name"]}</div>', unsafe_allow_html=True)

        try:
            y_ref, sr_ref = librosa.load(str(meme['audio']), sr=16000, duration=5.0)
            fig_r, ax_r = plt.subplots(figsize=(4, 1.5))
            fig_r.patch.set_facecolor('none')        # background figure transparan
            ax_r.set_facecolor('none')               # background axes transparan
            t = np.linspace(0, len(y_ref)/sr_ref, len(y_ref))
            ax_r.plot(t, y_ref, color='#00DC6E', linewidth=.9)
            ax_r.fill_between(t, y_ref, alpha=.12, color='#00DC6E')
            ax_r.tick_params(colors='#3C5070')
            ax_r.set_xticks([]); ax_r.set_yticks([])
            for sp in ax_r.spines.values(): sp.set_visible(False)
            st.pyplot(fig_r, use_container_width=True, transparent=True)
            plt.close(fig_r)
            st.caption(f"Durasi: {len(y_ref)/sr_ref:.1f}s  ·  SR: {sr_ref} Hz")
        except Exception as e:
            st.warning(f"Gagal load waveform: {e}")

        if st.button("▶  Putar Referensi", key="play_ref_btn"):
            st.session_state.play_ref = True
        if st.session_state.play_ref:
            st.markdown(audio_autoplay_html(str(meme['audio'])), unsafe_allow_html=True)
            st.session_state.play_ref = False

        if meme.get('image') and meme['image'].exists():
            st.image(str(meme['image']), use_container_width=True)
        if meme.get('video') and meme['video'].exists():
            if st.button("🎬  Tampilkan Video", key="show_vid"):
                st.session_state['show_video'] = not st.session_state.get('show_video', False)
            if st.session_state.get('show_video', False):
                st.video(str(meme['video']))

        st.markdown('</div>', unsafe_allow_html=True)

    # ── PANEL REKAM ──────────────────────────────────
    with col_rec:
        st.markdown('<div class="panel-box blue">', unsafe_allow_html=True)
        st.markdown('<div class="pill">🎙️ Rekam Suaramu</div>', unsafe_allow_html=True)

        user_audio = st.audio_input("Klik mikrofon, lalu tirukan suara meme:", key="rec_input")

        if user_audio is not None:
            st.audio(user_audio, format="audio/wav")
            # Nama unik tiap rekaman agar cache tidak mengganggu
            tmp_path = f"_tmp_{meme['id']}_{int(time.time()*1000)}.wav"
            with open(tmp_path, "wb") as f:
                f.write(user_audio.getbuffer())

            with st.spinner("Menghitung skor…"):
                seq_ref  = get_mfcc_ref(str(meme['audio']))
                seq_user = get_mfcc_seq(tmp_path)
                score    = compute_dtw_raw(seq_ref, seq_user)

            if score is not None:
                sc  = score_color(score)
                pct = score_ring_pct(score)
                deg = int(pct * 360)

                st.markdown(f"""
                <div class="score-ring-wrap" style="margin:1rem 0">
                  <div class="score-ring" style="--c:{sc};--pct:{deg}deg">
                    <div class="score-ring-val" style="color:{sc}">{score:.2f}</div>
                  </div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#2E4060">DTW DISTANCE (↓ makin bagus)</div>
                </div>""", unsafe_allow_html=True)

                msg, kind = score_label(score)
                if kind == "success": st.success(msg)
                elif kind == "info":  st.info(msg)
                else:                 st.warning(msg)

                st.markdown('<div style="margin-top:.9rem;border-top:1px solid rgba(255,255,255,.05);padding-top:.9rem">', unsafe_allow_html=True)
                player_name = st.text_input("Nama kamu:", placeholder="Masukkan nama…", key="player_name", label_visibility="collapsed")
                if st.button("💾  Simpan Skor", key="submit_lb"):
                    name = player_name.strip() or "Anonymous"
                    df_lb = load_lb(meme['id'])
                    new_row = pd.DataFrame([{'name': name, 'score': score, 'method': 'FastDTW', 'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}])
                    df_lb = pd.concat([df_lb, new_row], ignore_index=True)
                    save_lb(meme['id'], df_lb)
                    st.session_state.lb_submitted = True
                    st.session_state['last_score'] = score
                    st.session_state['last_name'] = name
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Gagal menghitung skor. Pastikan rekaman minimal 0.5 detik.")

            try: os.remove(tmp_path)
            except: pass

        if st.session_state.get('lb_submitted'):
            st.success(f"✅ Skor **{st.session_state.get('last_score',0):.2f}** atas nama **{st.session_state.get('last_name','')}** tersimpan!")
            st.session_state.lb_submitted = False

        st.markdown('</div>', unsafe_allow_html=True)

    # ── PANEL LEADERBOARD ────────────────────────────
    with col_lb:
        st.markdown('<div class="panel-box purple">', unsafe_allow_html=True)
        st.markdown('<div class="pill">🏆 Leaderboard</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-family:Comfortaa,sans-serif;font-weight:700;font-size:.88rem;color:#8A9BB8;margin-bottom:.8rem">{meme["name"]}</div>', unsafe_allow_html=True)

        df_lb = load_lb(meme['id'])
        if df_lb.empty:
            st.markdown("""<div style="text-align:center;padding:2rem 1rem"><div style="font-size:2rem">🏅</div><div style="font-family:JetBrains Mono,monospace;font-size:.65rem;color:#1E2E48">Belum ada skor. Jadilah yang pertama!</div></div>""", unsafe_allow_html=True)
        else:
            # Tampilkan ascending: terkecil = terbaik
            df_show = df_lb.sort_values('score', ascending=True).head(10).reset_index(drop=True)
            for i, row in df_show.iterrows():
                icon = {0:"🥇",1:"🥈",2:"🥉"}.get(i, f"#{i+1}")
                cls  = {0:"gold",1:"silver",2:"bronze"}.get(i, "")
                sc   = score_color(row['score'])
                mth  = str(row.get('method','DTW'))[:7]
                st.markdown(f"""<div class="lb-row"><div class="lb-rank {cls}">{icon}</div><div class="lb-name">{row['name']}</div><div class="lb-score" style="color:{sc}">{row['score']:.2f}</div><div class="lb-method">{mth}</div></div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="text-align:center;padding:.5rem 0"><span style="font-family:JetBrains Mono,monospace;font-size:.58rem;color:#10181F;letter-spacing:3px">Meme Voice Challenge · Voice Lab</span></div>', unsafe_allow_html=True)
