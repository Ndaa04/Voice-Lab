import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import requests
import zipfile
import io
from pathlib import Path
import base64

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# ─────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────
st.set_page_config(page_title="Sound Lab", page_icon="🎧", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

.stApp { background:#060A12; color:#C8D4E8; font-family:'Comfortaa',sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] { background:#080E18 !important; border-right:1px solid rgba(0,230,120,0.10); }
[data-testid="stSidebar"]::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg,#00E676,#00B4D8,#7C3AED);
}

/* Typography */
h1 { font-family:'Comfortaa',sans-serif !important; font-weight:800 !important;
     font-size:2.4rem !important; color:#fff !important; letter-spacing:-1px; }
h2 { font-family:'Comfortaa',sans-serif !important; font-weight:700 !important;
     font-size:1.4rem !important; color:#F0F4FF !important; padding-bottom:0.5rem;
     border-bottom:1px solid rgba(255,255,255,0.07); }
h2::before { content:'// '; color:#00E676; font-family:'JetBrains Mono','Courier New',monospace; font-size:0.8rem; }
h3 { font-family:'Comfortaa',sans-serif !important; font-weight:600 !important;
     font-size:1.05rem !important; color:#A0B0D0 !important; }

/* Main block */
.main .block-container { padding-top:1.5rem; max-width:1200px; }

/* Section pill */
.spill {
    display:inline-flex; align-items:center; gap:.4rem;
    background:rgba(0,230,120,0.07); border:1px solid rgba(0,230,120,0.18);
    border-radius:999px; padding:.25rem .85rem;
    font-family:'JetBrains Mono','Courier New',monospace; font-size:.66rem;
    color:#00E676; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:.6rem;
}

/* Cards generic */
.card {
    background:#0D1622; border:1px solid rgba(255,255,255,0.07);
    border-radius:14px; padding:1.2rem 1.4rem; margin-bottom:.8rem;
}
.card-accent { border-left:3px solid #00E676; }

/* Model comparison cards */
.model-card {
    background:#0D1622; border:1px solid rgba(255,255,255,0.08);
    border-radius:14px; padding:1.3rem 1.5rem;
}
.model-card.best { border:2px solid #00E676; box-shadow:0 0 18px rgba(0,230,120,0.12); }

/* Reduction method boxes */
.red-box {
    background:#0D1622; border:2px solid rgba(255,255,255,0.08);
    border-radius:12px; padding:1rem 1.2rem; text-align:center;
    cursor:pointer; transition:all .2s;
}
.red-box.active { border-color:#00E676; background:#0A1E14; box-shadow:0 0 16px rgba(0,230,120,0.1); }

/* Sound category button (tombol kotak kategori) */
.stButton button[data-testid="baseButton-secondary"] {
    background: #0D1622;
    border: 2px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem;
    font-family: 'Comfortaa', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    color: #E0EAF8;
    white-space: pre-wrap;
    line-height: 1.3;
    height: auto;
    min-height: 80px;
    transition: all 0.2s;
}
.stButton button[data-testid="baseButton-secondary"]:hover {
    border-color: #00E676;
    background: #0A1E14;
}

/* Buttons umum */
.stButton > button {
    background:linear-gradient(135deg,#00C060,#009A4E) !important;
    color:#050810 !important;
    border:none !important;
    border-radius:10px !important;
    font-family:'Comfortaa',sans-serif !important;
    font-weight:700 !important;
    font-size:.88rem !important;
    letter-spacing:.4px;
    padding:.55rem 1.3rem !important;
    transition:all .25s !important;
    box-shadow:0 0 18px rgba(0,200,96,.18) !important;
}
.stButton > button:hover {
    background:linear-gradient(135deg,#00E676,#00C060) !important;
    box-shadow:0 0 28px rgba(0,230,118,.32) !important;
    transform:translateY(-1px);
}
.stButton > button:disabled {
    background:rgba(255,255,255,.04) !important;
    color:#2A3A55 !important;
    box-shadow:none !important;
    cursor:default !important;
}

/* Primary CTA button */
.cta-btn .stButton > button {
    background:linear-gradient(135deg,#00E676,#00B84C) !important;
    font-size:1rem !important;
    padding:.7rem 2rem !important;
    box-shadow:0 0 28px rgba(0,230,120,.22) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background:#0D1622 !important;
    border:1px solid rgba(255,255,255,.06) !important;
    border-radius:12px !important;
    padding:.9rem 1.1rem !important;
}
[data-testid="stMetricLabel"] {
    font-family:'JetBrains Mono','Courier New',monospace !important;
    font-size:.6rem !important;
    color:#3A4A60 !important;
    letter-spacing:1.5px !important;
    text-transform:uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family:'Comfortaa',sans-serif !important;
    font-weight:800 !important;
    font-size:1.7rem !important;
    color:#00E676 !important;
}

/* Selectbox / number input */
[data-testid="stSelectbox"] > div, [data-testid="stNumberInput"] > div {
    background:#0D1622 !important;
    border:1px solid rgba(255,255,255,.08) !important;
    border-radius:8px !important;
}

/* Slider */
[data-testid="stSlider"] > div > div > div { background:#00E676 !important; }

/* Expanders */
[data-testid="stExpander"] {
    background:#0A1018 !important;
    border:1px solid rgba(255,255,255,.06) !important;
    border-radius:12px !important;
}

/* Progress */
[data-testid="stProgressBar"] > div > div {
    background:linear-gradient(90deg,#00E676,#00B4D8) !important;
    border-radius:999px !important;
}
[data-testid="stProgressBar"] > div {
    background:rgba(255,255,255,.05) !important;
    border-radius:999px !important;
}

/* Caption */
.stCaption, [data-testid="stCaptionContainer"] {
    font-family:'JetBrains Mono','Courier New',monospace !important;
    font-size:.68rem !important;
    color:#2A3A55 !important;
}

/* Alerts */
.stSuccess > div { background:rgba(0,230,120,.07) !important; border-left:3px solid #00E676 !important; border-radius:8px !important; }
.stInfo    > div { background:rgba(0,180,216,.07) !important; border-left:3px solid #00B4D8 !important; border-radius:8px !important; }
.stWarning > div { background:rgba(255,160,0,.07) !important; border-left:3px solid #FFA000 !important; border-radius:8px !important; }
.stError   > div { background:rgba(255,60,60,.07)  !important; border-left:3px solid #FF3C3C !important; border-radius:8px !important; }

/* Table / dataframe */
[data-testid="stDataFrame"] { border:1px solid rgba(255,255,255,.06) !important; border-radius:10px !important; }

/* Divider */
hr { border:none !important; border-top:1px solid rgba(255,255,255,.06) !important; margin:1.8rem 0 !important; }

/* Scrollbar */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#060A12; }
::-webkit-scrollbar-thumb { background:rgba(0,230,120,.18); border-radius:3px; }

/* Volume slider label */
.vol-label {
    font-family:'JetBrains Mono','Courier New',monospace;
    font-size:.7rem;
    color:#3A4A60;
    letter-spacing:1.5px;
    text-transform:uppercase;
    margin-bottom:.3rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB GLOBAL STYLE
# ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0A1018', 'axes.facecolor': '#0D1622',
    'axes.edgecolor': '#1E2A3A', 'axes.labelcolor': '#5A6A85',
    'axes.titlecolor': '#A0B0D0', 'axes.titlesize': 11,
    'axes.labelsize': 9, 'xtick.color': '#3A4A60', 'ytick.color': '#3A4A60',
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'grid.color': '#1A2535', 'grid.linestyle': '--', 'grid.alpha': .5,
    'text.color': '#7A8FAA', 'font.family': 'monospace', 'lines.linewidth': 1.4,
})

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def audio_to_b64_html(file_path, volume=1.0):
    """Return HTML5 audio element (autoplay, no controls, with volume JS)."""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    ext = Path(file_path).suffix.lower().replace('.', '')
    mime = 'audio/wav' if ext == 'wav' else 'audio/mpeg' if ext in ('mp3','mpeg') else 'audio/ogg'
    uid = f"aud_{random.randint(0,9999999)}"
    html = f"""
    <audio id="{uid}" autoplay style="display:none">
      <source src="data:{mime};base64,{b64}" type="{mime}">
    </audio>
    <script>
      var a = document.getElementById('{uid}');
      a.volume = {volume:.2f};
      a.play().catch(function(){{}});
    </script>
    """
    return html

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="spill">🎧 Audio ML Lab</div>', unsafe_allow_html=True)
st.title("Klasifikasi Suara: Dari Sampling ke 3D")
st.markdown('<p style="color:#5A6A85; font-size:.95rem; margin-bottom:1.5rem">Pilih kategori, atur parameter, lalu jalankan analisis — lihat MFCC, reduksi dimensi, dan perbandingan model dalam satu alur.</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 0. LOAD DATASET
# ─────────────────────────────────────────────
@st.cache_resource
def load_esc50():
    data_dir = Path("data/ESC-50-master")
    if not data_dir.exists():
        with st.spinner("Mengunduh ESC-50 (~600 MB)..."):
            url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
            r = requests.get(url, stream=True)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall("data")
    meta = pd.read_csv(data_dir / "meta" / "esc50.csv")
    audio_files = [data_dir / "audio" / f for f in meta["filename"]]
    return meta, audio_files

with st.spinner("Memuat dataset ESC-50..."):
    meta, audio_files = load_esc50()

# ─────────────────────────────────────────────
# 1. PILIH KATEGORI + TOMBOL MULAI ANALISIS DI TENGAH
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="spill">📂 Langkah 1</div>', unsafe_allow_html=True)
st.header("Pilih Kategori Suara")

all_categories = sorted(meta['category'].unique())

selected_cats = st.multiselect(
    "Pilih kategori yang akan diklasifikasikan:",
    options=all_categories,
    default=all_categories[:3],
    help="Pilih minimal 2 kategori untuk klasifikasi."
)

if not selected_cats:
    st.warning("Pilih minimal satu kategori untuk melanjutkan.")
    st.stop()

mask = meta['category'].isin(selected_cats)
filtered_meta_preview = meta[mask].reset_index(drop=True)
st.caption(f"Ditemukan {len(filtered_meta_preview)} file audio dari {len(selected_cats)} kategori.")

st.markdown("---")
# Tombol Mulai Analisis di tengah
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="cta-btn" style="text-align:center">', unsafe_allow_html=True)
    start_analysis = st.button("🚀 Mulai Analisis", key="start_btn", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

if start_analysis:
    st.session_state.analysis_started = True
    st.session_state.pop("features_cache", None)

if not st.session_state.analysis_started:
    # Pesan tanpa warna biru, menggunakan custom div
    st.markdown("""
    <div style="background:#0D1622; padding:1rem; border-radius:12px; text-align:center; border-left:3px solid #00E676; color:#5A6A85;">
        Pilih kategori di atas, lalu klik <strong style="color:#00E676">Mulai Analisis</strong> untuk melanjutkan.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# Filter & batasi data setelah analisis dimulai
# ─────────────────────────────────────────────
mask = meta['category'].isin(selected_cats)
filtered_meta = meta[mask].reset_index(drop=True)
filtered_audio_all = [audio_files[i] for i in range(len(meta)) if meta.iloc[i]['category'] in selected_cats]

n_files_raw = len(filtered_meta)
max_files = 800
if n_files_raw > max_files:
    indices = random.sample(range(n_files_raw), max_files)
    filtered_meta = filtered_meta.iloc[indices].reset_index(drop=True)
    filtered_audio = [filtered_audio_all[i] for i in indices]
    st.warning(f"Dataset besar — diambil {max_files} file secara acak.")
else:
    filtered_audio = filtered_audio_all

n_files = len(filtered_meta)

# ─────────────────────────────────────────────
# 2. CONTOH SUARA PER KATEGORI (KOTAK SEBAGAI TOMBOL)
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="spill">🔊 Langkah 2</div>', unsafe_allow_html=True)
st.header("Contoh Suara per Kategori")

st.markdown('<p class="vol-label">🔉 Volume Global</p>', unsafe_allow_html=True)
global_volume = st.slider("Volume", min_value=0.0, max_value=1.0, value=0.7, step=0.05,
                           label_visibility="collapsed")
st.caption("Klik tombol kategori untuk memutar suara acak. Klik lagi untuk berhenti.")

# State untuk menyimpan kategori yang sedang diputar dan file path-nya
if "playing_cat" not in st.session_state:
    st.session_state.playing_cat = None
if "playing_file" not in st.session_state:
    st.session_state.playing_file = None

MAX_COLS = 5
n_cats = len(selected_cats)
rows = [selected_cats[i:i+MAX_COLS] for i in range(0, n_cats, MAX_COLS)]

for row in rows:
    cols = st.columns(len(row))
    for col, cat in zip(cols, row):
        cat_files_df = meta[meta['category'] == cat]
        cat_count = len(cat_files_df)
        is_playing = (st.session_state.playing_cat == cat)

        # Tampilan tombol dengan ikon play/stop
        icon = "⏹" if is_playing else "▶"
        label = f"{icon} {cat}\n{cat_count} FILE"
        if col.button(label, key=f"cat_btn_{cat}", use_container_width=True):
            if is_playing:
                # Stop
                st.session_state.playing_cat = None
                st.session_state.playing_file = None
            else:
                # Play
                random_file = random.choice(cat_files_df['filename'].values)
                audio_path = Path("data/ESC-50-master/audio") / random_file
                st.session_state.playing_cat = cat
                st.session_state.playing_file = str(audio_path)
            st.rerun()

        # Jika kategori ini sedang bermain, injeksikan audio autoplay
        if is_playing and st.session_state.playing_file:
            fpath = st.session_state.playing_file
            if Path(fpath).exists():
                html = audio_to_b64_html(fpath, volume=global_volume)
                st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. PARAMETER PENGOLAHAN AUDIO
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="spill">⚙️ Langkah 3</div>', unsafe_allow_html=True)
st.header("Parameter Pengolahan Audio")
st.markdown('<p style="color:#5A6A85; font-size:.9rem; margin-bottom:1.2rem">Atur parameter sampling dan ekstraksi fitur. Perubahan akan dipakai saat kamu klik <strong style="color:#C8D0E0">Proses & Klasifikasi</strong>.</p>', unsafe_allow_html=True)

col_p1, col_p2, col_p3, col_p4 = st.columns(4)
with col_p1:
    st.markdown('<div style="border-left: 3px solid #00E676; padding-left: 0.6rem; margin-bottom: 0.4rem; color: #E0EAF8; font-weight: 600;">Sample Rate (Hz)</div>', unsafe_allow_html=True)
    sample_rate = st.selectbox("SR", [8000, 16000, 22050, 44100], index=1, label_visibility="collapsed")
    st.caption("Sampel per detik")

with col_p2:
    st.markdown('<div style="border-left: 3px solid #00E676; padding-left: 0.6rem; margin-bottom: 0.4rem; color: #E0EAF8; font-weight: 600;">Durasi (detik)</div>', unsafe_allow_html=True)
    duration = st.number_input("Dur", min_value=0.5, max_value=10.0, value=3.0, step=0.5, label_visibility="collapsed")
    st.caption("Potongan awal file")

with col_p3:
    st.markdown('<div style="border-left: 3px solid #00E676; padding-left: 0.6rem; margin-bottom: 0.4rem; color: #E0EAF8; font-weight: 600;">Jumlah MFCC</div>', unsafe_allow_html=True)
    n_mfcc = st.number_input("MFCC", min_value=5, max_value=40, value=20, step=1, label_visibility="collapsed")
    st.caption("Koefisien per frame")

with col_p4:
    st.markdown('<div style="border-left: 3px solid #00E676; padding-left: 0.6rem; margin-bottom: 0.4rem; color: #E0EAF8; font-weight: 600;">Maks File</div>', unsafe_allow_html=True)
    max_per_cat = st.number_input("MaxF", min_value=10, max_value=200, value=40, step=10, label_visibility="collapsed")
    st.caption("Per kategori")

# ─────────────────────────────────────────────
# PARAMETER REDUKSI UNTUK KLASIFIKASI
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="spill">🧪 Langkah 3b</div>', unsafe_allow_html=True)
st.header("Parameter Reduksi Dimensi untuk Klasifikasi")
st.markdown('<p style="color:#5A6A85; font-size:.9rem; margin-bottom:1.2rem">Pilih metode reduksi dan jumlah dimensi yang akan digunakan dalam pelatihan model. Hasil akurasi akan berubah sesuai pilihan ini.</p>', unsafe_allow_html=True)

col_red_ml1, col_red_ml2 = st.columns(2)
with col_red_ml1:
    st.markdown('<div style="border-left: 3px solid #00E676; padding-left: 0.6rem; margin-bottom: 0.4rem; color: #E0EAF8; font-weight: 600;">Metode Reduksi (Klasifikasi)</div>', unsafe_allow_html=True)
    reduction_method_ml = st.selectbox(
        "Metode reduksi (sama dengan visualisasi atau berbeda?)",
        options=["Sama dengan visualisasi", "PCA", "t-SNE", "UMAP"],
        index=0,
        label_visibility="collapsed"
    )
    st.caption("Jika 'Sama dengan visualisasi', akan mengikuti Langkah 5.")

with col_red_ml2:
    st.markdown('<div style="border-left: 3px solid #00E676; padding-left: 0.6rem; margin-bottom: 0.4rem; color: #E0EAF8; font-weight: 600;">Jumlah Dimensi</div>', unsafe_allow_html=True)
    n_components_ml = st.slider("Dimensi hasil reduksi (2–30)", min_value=2, max_value=30, value=10, step=1, label_visibility="collapsed")
    st.caption("Semakin kecil dimensi, semakin besar kompresi fitur.")

# ─────────────────────────────────────────────
# 4. STEP-BY-STEP: CONTOH FILE (Step 1 waveform)
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="spill">📈 Langkah 4</div>', unsafe_allow_html=True)
st.header("Proses Pengolahan Suara")

if "sample_idx" not in st.session_state:
    st.session_state.sample_idx = 0

st.subheader("Step 1 — Sampling & Waveform")

# 1. Ambil data file contoh terlebih dahulu
sidx = st.session_state.sample_idx % len(filtered_audio)
sample_path = filtered_audio[sidx]
sample_fname = filtered_meta.iloc[sidx]['filename']
sample_cat = filtered_meta.iloc[sidx]['category']

# 2. Buat kolom untuk menyejajarkan kotak info dan tombol
col_card, col_btn = st.columns([5, 2])

with col_card:
    st.markdown(f"""
    <div class="card" style="margin-bottom:.8rem; padding: 0.8rem 1.2rem;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:.68rem;color:#3A4A60;letter-spacing:1.5px;text-transform:uppercase">File Contoh</span><br>
        <span style="color:#C8D0E0;font-weight:700;font-size:1.1rem">{sample_fname}</span>
        <span style="margin-left:.8rem;background:rgba(0,230,120,.08);border:1px solid rgba(0,230,120,.2);
        color:#00E676;font-family:'JetBrains Mono',monospace;font-size:.62rem;padding:.15rem .5rem;border-radius:4px">{sample_cat}</span>
    </div>
    """, unsafe_allow_html=True)

with col_btn:
    # Menambahkan spasi kosong agar tombol turun ke tengah, sejajar dengan kotak info
    st.markdown("<div style='margin-top: 1.1rem;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Ganti Contoh", key="reload_sample", use_container_width=True):
        st.session_state.sample_idx = random.randint(0, len(filtered_audio) - 1)
        st.rerun()

try:
    y_sample, sr_sample = librosa.load(sample_path, sr=sample_rate, duration=duration)
    st.write(f"Sample rate: `{sr_sample}` Hz &nbsp;|&nbsp; Durasi: `{len(y_sample)/sr_sample:.2f}` detik &nbsp;|&nbsp; Sampel: `{len(y_sample)}`")
    fig_wav, ax_wav = plt.subplots(figsize=(10, 2.2))
    librosa.display.waveshow(y_sample, sr=sr_sample, ax=ax_wav, color='#00E676', alpha=0.85)
    ax_wav.fill_between(np.linspace(0, len(y_sample)/sr_sample, len(y_sample)), y_sample, alpha=0.07, color='#00E676')
    ax_wav.set_title("Waveform — Hasil Sampling")
    ax_wav.grid(True)
    fig_wav.tight_layout()
    st.pyplot(fig_wav)
except Exception as e:
    st.error(f"Gagal memuat file contoh: {e}")

st.subheader("Step 2 — FFT & Spektrogram")
try:
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y_sample)), ref=np.max)
    fig_spec, ax_spec = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(D, sr=sr_sample, x_axis='time', y_axis='hz', ax=ax_spec, cmap='magma')
    plt.colorbar(img, ax=ax_spec, format="%+2.0f dB")
    ax_spec.set_title("Spektrogram (FFT per frame)")
    ax_spec.grid(True)
    fig_spec.tight_layout()
    st.pyplot(fig_spec)
except Exception as e:
    st.error(f"Gagal membuat spektrogram: {e}")

# ─────────────────────────────────────────────
# 5. PILIH METODE REDUKSI DIMENSI (VISUALISASI)
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="spill">🌐 Langkah 5</div>', unsafe_allow_html=True)
st.header("Pilih Metode Reduksi & Visualisasi")
st.markdown('<p style="color:#5A6A85; font-size:.9rem; margin-bottom:1.2rem">Pilih metode proyeksi 3D di bawah ini. Tombol yang sedang aktif akan memiliki tanda centang (✅).</p>', unsafe_allow_html=True)

# Set nilai default jika belum ada
if "reduction_method" not in st.session_state:
    st.session_state.reduction_method = "UMAP" if UMAP_AVAILABLE else "PCA"

# Membuat 3 kolom sejajar
col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    if UMAP_AVAILABLE:
        # Jika UMAP sedang dipilih, tampilkan centang di teks tombolnya
        lbl = "✅ 📊 UMAP (Terpilih)" if st.session_state.reduction_method == "UMAP" else "📊 Pilih UMAP"
        if st.button(lbl, use_container_width=True, key="btn_umap"):
            st.session_state.reduction_method = "UMAP"
            st.rerun()
        # Keterangan diletakkan di bawah tombol agar rapi
        st.markdown('<div style="text-align:center; font-family:\'JetBrains Mono\',monospace; font-size:0.7rem; color:#5A6A85; padding-top:0.3rem; line-height:1.5">Menjaga struktur lokal + global.<br>Sangat Cepat & Akurat.</div>', unsafe_allow_html=True)
    else:
        st.button("❌ UMAP (Error)", disabled=True, use_container_width=True)
        st.markdown('<div style="text-align:center; font-family:\'JetBrains Mono\',monospace; font-size:0.7rem; color:#5A6A85; padding-top:0.3rem">Modul umap-learn tidak tersedia.</div>', unsafe_allow_html=True)

with col_m2:
    lbl = "✅ 📉 PCA (Terpilih)" if st.session_state.reduction_method == "PCA" else "📉 Pilih PCA"
    if st.button(lbl, use_container_width=True, key="btn_pca"):
        st.session_state.reduction_method = "PCA"
        st.rerun()
    st.markdown('<div style="text-align:center; font-family:\'JetBrains Mono\',monospace; font-size:0.7rem; color:#5A6A85; padding-top:0.3rem; line-height:1.5">Transformasi linear.<br>Cepat, cocok sebagai baseline.</div>', unsafe_allow_html=True)

with col_m3:
    lbl = "✅ 🌀 t-SNE (Terpilih)" if st.session_state.reduction_method == "t-SNE" else "🌀 Pilih t-SNE"
    if st.button(lbl, use_container_width=True, key="btn_tsne"):
        st.session_state.reduction_method = "t-SNE"
        st.rerun()
    st.markdown('<div style="text-align:center; font-family:\'JetBrains Mono\',monospace; font-size:0.7rem; color:#5A6A85; padding-top:0.3rem; line-height:1.5">Menjaga klaster lokal.<br>Lambat pada data besar.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 6. TOMBOL PROSES & TAMPILAN HASIL (DALAM FRAGMENT)
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="spill">🚀 Langkah 6</div>', unsafe_allow_html=True)
st.header("Proses & Hitung Akurasi")

# Membungkus proses ini dalam fragment agar loadingnya LOKAL
@st.fragment
def proses_ml_fragment():
    st.markdown(f"""
    <div class="card" style="margin-bottom:1.2rem">
      <p style="color:#5A6A85;font-size:.9rem;margin:0;line-height:1.8">
        Konfigurasi siap dieksekusi:
        <span style="color:#C8D0E0">SR={sample_rate} Hz</span> &nbsp;·&nbsp;
        <span style="color:#C8D0E0">Durasi={duration}s</span> &nbsp;·&nbsp;
        <span style="color:#C8D0E0">MFCC={n_mfcc}</span> &nbsp;·&nbsp;
        <span style="color:#C8D0E0">Metode Viz={st.session_state.reduction_method}</span>
      </p>
    </div>
    """, unsafe_allow_html=True)

    col_run, _ = st.columns([3, 5])
    with col_run:
        run_btn = st.button("⚡ Eksekusi Model & Visualisasi 3D", key="run_analysis", use_container_width=True)

    # Hanya berjalan saat tombol diklik
    if run_btn:
        with st.spinner("⏳ Memproses audio, mengekstrak MFCC, dan melatih AI..."):
            
            # 1. Batasi file sesuai input (Max File per kategori)
            sampled_frames = []
            for cat in selected_cats:
                cat_rows = filtered_meta[filtered_meta['category'] == cat]
                n_take = min(len(cat_rows), max_per_cat)
                sampled_frames.append(cat_rows.sample(n_take, random_state=42))
            limited_meta = pd.concat(sampled_frames, ignore_index=True)

            limited_audio = [Path("data/ESC-50-master/audio") / row['filename'] for _, row in limited_meta.iterrows()]
            n_proc = len(limited_meta)

            # 2. Ekstraksi Fitur MFCC dengan Progress Bar Lokal
            prog = st.progress(0, text=f"Mengekstrak fitur dari {n_proc} file...")
            features, labels = [], []
            for i, (fpath, row) in enumerate(zip(limited_audio, (r for _, r in limited_meta.iterrows()))):
                try:
                    y_f, _ = librosa.load(str(fpath), sr=sample_rate, duration=duration)
                    if len(y_f) > 0:
                        mfcc = librosa.feature.mfcc(y=y_f, sr=sample_rate, n_mfcc=n_mfcc)
                        feat = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
                        features.append(feat)
                        labels.append(row['category'])
                except Exception:
                    pass
                prog.progress((i + 1) / n_proc)
            prog.empty() # Sembunyikan progress bar jika sudah selesai

            if len(features) < 10:
                st.error("Gagal diproses. Pastikan dataset ESC-50 ada di folder yang benar.")
                return

            X = np.array(features)
            y_labels = np.array(labels)
            scaler = StandardScaler()
            X_norm = scaler.fit_transform(X)

            # 3. Reduksi Dimensi untuk Visualisasi 3D
            method_viz = st.session_state.reduction_method
            if method_viz == "UMAP" and UMAP_AVAILABLE:
                n_neighbors_viz = min(15, len(X_norm) - 2)
                reducer_viz = umap.UMAP(n_components=3, random_state=42, n_neighbors=n_neighbors_viz, min_dist=0.1)
            elif method_viz == "PCA":
                reducer_viz = PCA(n_components=3)
            else:
                perp = min(30, len(X_norm) - 1)
                reducer_viz = TSNE(n_components=3, perplexity=perp, random_state=42)
            
            X_3d = reducer_viz.fit_transform(X_norm)

            # 4. Reduksi Dimensi untuk Pelatihan AI (Machine Learning)
            method_ml = method_viz if reduction_method_ml == "Sama dengan visualisasi" else reduction_method_ml
            if method_ml == "UMAP" and UMAP_AVAILABLE:
                n_neighbors_ml = min(15, len(X_norm) - 2)
                reducer_ml = umap.UMAP(n_components=n_components_ml, random_state=42, n_neighbors=n_neighbors_ml, min_dist=0.1)
            elif method_ml == "PCA":
                reducer_ml = PCA(n_components=n_components_ml)
            elif method_ml == "t-SNE":
                perp = min(30, len(X_norm) - 1)
                reducer_ml = TSNE(n_components=min(n_components_ml, 3), perplexity=perp, random_state=42)
            else:
                reducer_ml = PCA(n_components=n_components_ml)
            
            X_ml = reducer_ml.fit_transform(X_norm)

            # 5. Latih 3 Model Machine Learning
            X_tr, X_te, y_tr, y_te = train_test_split(X_ml, y_labels, test_size=0.2, random_state=42, stratify=y_labels)
            knn = KNeighborsClassifier(n_neighbors=5); knn.fit(X_tr, y_tr)
            rf  = RandomForestClassifier(n_estimators=100, random_state=42); rf.fit(X_tr, y_tr)
            svm = SVC(kernel='rbf', random_state=42); svm.fit(X_tr, y_tr)
            acc_knn = accuracy_score(y_te, knn.predict(X_te))
            acc_rf  = accuracy_score(y_te, rf.predict(X_te))
            acc_svm = accuracy_score(y_te, svm.predict(X_te))

            # ─────────────────────────────────────────────
            # MUNCULKAN HASIL LANGSUNG DI BAWAH TOMBOL
            # ─────────────────────────────────────────────
            st.markdown("---")
            st.markdown('<div class="spill">📊 Hasil Klasifikasi</div>', unsafe_allow_html=True)
            st.header("Perbandingan Akurasi Model AI")
            
            best_acc = max(acc_knn, acc_rf, acc_svm)
            col_m1, col_m2, col_m3 = st.columns(3)
            models = [("k-NN", "k=5, 5 tetangga terdekat", acc_knn, col_m1),
                      ("Random Forest", "100 pohon keputusan", acc_rf, col_m2),
                      ("SVM (RBF)", "Support Vector, kernel radial", acc_svm, col_m3)]

            for name, desc, acc, col in models:
                is_best = abs(acc - best_acc) < 1e-9
                card_cls = "model-card best" if is_best else "model-card"
                bar_w = int(acc * 100)
                bar_color = "#00E676" if is_best else "#00B4D8"
                
                best_label = '<div style="font-family:JetBrains Mono,monospace; font-size:.6rem; color:#00E676; letter-spacing:2px; margin-bottom:.5rem">🏆 TERBAIK</div>' if is_best else ''
                
                with col:
                    st.markdown(f"""
<div class="{card_cls}">
{best_label}
<div style="font-family:'Comfortaa',sans-serif; font-weight:700; font-size:1.05rem; color:#E0EAF8; margin-bottom:.2rem">{name}</div>
<div style="font-family:'JetBrains Mono',monospace; font-size:.65rem; color:#3A4A60; margin-bottom:.8rem">{desc}</div>
<div style="font-family:'Comfortaa',sans-serif; font-weight:800; font-size:2rem; color:{bar_color}; margin-bottom:.6rem">{acc*100:.1f}%</div>
<div style="background:rgba(255,255,255,.05); border-radius:999px; height:6px; overflow:hidden">
<div style="width:{bar_w}%; height:100%; background:linear-gradient(90deg,{bar_color},{bar_color}88); border-radius:999px; transition:width .8s ease"></div>
</div>
</div>
""", unsafe_allow_html=True)
            
            best_name = [n for n,_,a,_ in models if abs(a-best_acc)<1e-9][0]
            st.success(f"🏆 Sistem merekomendasikan **{best_name}** sebagai algoritma paling cerdas dengan akurasi **{best_acc*100:.1f}%** untuk data ini.")

            st.markdown("---")
            st.markdown('<div class="spill">🌐 Visualisasi 3D</div>', unsafe_allow_html=True)
            st.header("Ruang Dimensi Klaster Suara")

            df_plot = pd.DataFrame({
                'x': X_3d[:, 0], 'y': X_3d[:, 1], 'z': X_3d[:, 2],
                'Kategori': y_labels,
                'File': [Path(f).name for f in limited_audio[:len(y_labels)]]
            })

            fig3d = px.scatter_3d(
                df_plot, x='x', y='y', z='z', color='Kategori',
                hover_data=['File'],
                title=f"Proyeksi {method_viz} — {n_proc} file audio",
                opacity=0.75, height=700,
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            fig3d.update_traces(marker=dict(size=3.5))
            fig3d.update_layout(
                paper_bgcolor='#060A12', plot_bgcolor='#060A12',
                font=dict(family='JetBrains Mono, monospace', color='#7A8FAA', size=11),
                scene=dict(
                    bgcolor='#0A1018',
                    xaxis=dict(gridcolor='#1A2535', title=f"{method_viz} 1"),
                    yaxis=dict(gridcolor='#1A2535', title=f"{method_viz} 2"),
                    zaxis=dict(gridcolor='#1A2535', title=f"{method_viz} 3"),
                ),
                legend=dict(bgcolor='rgba(10,16,24,.8)', bordercolor='rgba(255,255,255,.08)', borderwidth=1)
            )
            st.plotly_chart(fig3d, width='stretch')

# Panggil fungsi fragment di sini
proses_ml_fragment()

# ─────────────────────────────────────────────
# FOOTER HALAMAN
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:.8rem 0">
    <span style="font-family:'JetBrains Mono',monospace; font-size:.6rem; color:#3A4A60; letter-spacing:3px; text-transform:uppercase">
        Sound Lab — Audio ML Playground · ESC-50 Dataset
    </span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 7. TAMPILKAN HASIL (jika sudah diproses)
# ─────────────────────────────────────────────
if st.session_state.get("results_ready", False):
    acc_knn  = st.session_state.acc_knn
    acc_rf   = st.session_state.acc_rf
    acc_svm  = st.session_state.acc_svm
    X_3d     = st.session_state.X_3d
    y_labels = st.session_state.y_labels
    limited_meta = st.session_state.limited_meta
    limited_audio = st.session_state.limited_audio
    n_proc = st.session_state.n_proc
    method_viz = st.session_state.method_used
    method_ml = st.session_state.get("method_ml", "PCA")
    n_comp_ml = st.session_state.get("n_comp_ml", n_components_ml)

    st.markdown("---")
    st.subheader("Step 3 — Matriks Fitur MFCC")
    X_norm_disp = st.session_state.X_norm
    st.markdown(f'<p style="color:#5A6A85;font-size:.9rem">Matriks fitur: <strong style="color:#C8D0E0">{X_norm_disp.shape[0]}</strong> file × <strong style="color:#C8D0E0">{X_norm_disp.shape[1]}</strong> fitur (mean + std MFCC)</p>', unsafe_allow_html=True)
    st.caption("5 file pertama, 5 kolom pertama:")
    st.dataframe(
        pd.DataFrame(X_norm_disp[:5, :5], columns=[f"F{i+1}" for i in range(5)]).style.format("{:.3f}").background_gradient(cmap='Blues'),
        use_container_width=False
    )

    st.markdown("---")
    st.markdown('<div class="spill">📊 Hasil Klasifikasi</div>', unsafe_allow_html=True)
    st.header("Perbandingan Model Klasifikasi")
    st.markdown(f'<p style="color:#5A6A85;font-size:.88rem;margin-bottom:1.2rem">Proses: {n_proc} file, reduksi {method_ml} → {n_comp_ml} dimensi</p>', unsafe_allow_html=True)

    best_acc = max(acc_knn, acc_rf, acc_svm)
    col_m1, col_m2, col_m3 = st.columns(3)
    models = [("k-NN", "k=5, 5 tetangga terdekat", acc_knn, col_m1),
              ("Random Forest", "100 pohon keputusan", acc_rf, col_m2),
              ("SVM (RBF)", "Support Vector, kernel radial", acc_svm, col_m3)]

    for name, desc, acc, col in models:
        is_best = abs(acc - best_acc) < 1e-9
        card_cls = "model-card best" if is_best else "model-card"
        bar_w = int(acc * 100)
        bar_color = "#00E676" if is_best else "#00B4D8"
        
        best_label = '<div style="font-family:JetBrains Mono,monospace; font-size:.6rem; color:#00E676; letter-spacing:2px; margin-bottom:.5rem">🏆 TERBAIK</div>' if is_best else ''
        
        with col:
            st.markdown(f"""
<div class="{card_cls}">
{best_label}
<div style="font-family:'Comfortaa',sans-serif; font-weight:700; font-size:1.05rem; color:#E0EAF8; margin-bottom:.2rem">{name}</div>
<div style="font-family:'JetBrains Mono',monospace; font-size:.65rem; color:#3A4A60; margin-bottom:.8rem">{desc}</div>
<div style="font-family:'Comfortaa',sans-serif; font-weight:800; font-size:2rem; color:{bar_color}; margin-bottom:.6rem">{acc*100:.1f}%</div>
<div style="background:rgba(255,255,255,.05); border-radius:999px; height:6px; overflow:hidden">
<div style="width:{bar_w}%; height:100%; background:linear-gradient(90deg,{bar_color},{bar_color}88); border-radius:999px; transition:width .8s ease"></div>
</div>
</div>
""", unsafe_allow_html=True)

    best_name = [n for n,_,a,_ in models if abs(a-best_acc)<1e-9][0]
    st.success(f"🏆 Model terbaik: **{best_name}** dengan akurasi **{best_acc*100:.1f}%**")

    st.markdown("---")
    st.markdown('<div class="spill">🌐 Visualisasi 3D</div>', unsafe_allow_html=True)
    st.header("Klaster Suara dalam 3D")

    df_plot = pd.DataFrame({
        'x': X_3d[:, 0], 'y': X_3d[:, 1], 'z': X_3d[:, 2],
        'Kategori': y_labels,
        'File': [Path(f).name for f in limited_audio[:len(y_labels)]]
    })

    fig3d = px.scatter_3d(
        df_plot, x='x', y='y', z='z', color='Kategori',
        hover_data=['File'],
        title=f"Proyeksi {method_viz} — {n_proc} file audio",
        opacity=0.75, height=700,
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig3d.update_traces(marker=dict(size=3.5))
    fig3d.update_layout(
        paper_bgcolor='#060A12', plot_bgcolor='#060A12',
        font=dict(family='JetBrains Mono, monospace', color='#7A8FAA', size=11),
        scene=dict(
            bgcolor='#0A1018',
            xaxis=dict(gridcolor='#1A2535', title=f"{method_viz} 1"),
            yaxis=dict(gridcolor='#1A2535', title=f"{method_viz} 2"),
            zaxis=dict(gridcolor='#1A2535', title=f"{method_viz} 3"),
        ),
        legend=dict(bgcolor='rgba(10,16,24,.8)', bordercolor='rgba(255,255,255,.08)', borderwidth=1)
    )
    st.plotly_chart(fig3d, width='stretch')

    st.success("Analisis selesai! Ubah kategori, parameter, atau metode reduksi lalu klik **Proses & Hitung Akurasi** untuk membandingkan hasil.")

st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:.8rem 0">
    <span style="font-family:'JetBrains Mono',monospace; font-size:.6rem; color:#1A2535; letter-spacing:3px; text-transform:uppercase">
        Sound Lab — Audio ML Playground · ESC-50 Dataset
    </span>
</div>
""", unsafe_allow_html=True)
