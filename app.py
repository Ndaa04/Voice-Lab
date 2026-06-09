import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.spatial.distance import cosine as cos_dist, euclidean

# ========== KONFIGURASI HALAMAN ==========
st.set_page_config(
    page_title="Voice Lab",
    page_icon="🎤",
    layout="wide"
)

# ========== CUSTOM CSS PREMIUM AUDIO STUDIO — WHITE THEME ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;500;600;700&display=swap');

    .stApp {
        background: #FFFFFF;
        color: #1A1A2E;
        font-family: 'Comfortaa', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .stApp::after {
        content: '';
        position: fixed;
        top: -200px; left: -200px;
        width: 600px; height: 600px;
        background: radial-gradient(circle, rgba(0,150,80,0.03) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
    }
    [data-testid="stSidebar"] {
        background: #F5F7FA !important;
        border-right: 1px solid rgba(0,180,100,0.15);
    }
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00C060, #00B4D8, #7C3AED);
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Comfortaa', sans-serif;
        font-weight: 600;
        letter-spacing: -0.2px;
    }
    h2 {
        font-size: 1.55rem !important;
        color: #1A1A2E !important;
        margin-top: 2.2rem !important;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(0,0,0,0.08);
    }
    h2::before {
        content: '// ';
        color: #00C060;
        font-family: 'Comfortaa', monospace;
        font-size: 0.85rem;
        font-weight: 400;
    }
    h3 {
        font-size: 1.2rem !important;
        color: #3A4A60 !important;
        font-weight: 500;
    }
    .hero-banner {
        background: linear-gradient(135deg, #F0FFF8 0%, #E8F5FF 50%, #F5F0FF 100%);
        border: 1px solid rgba(0,200,100,0.2);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before, .hero-banner::after {
        content: '';
        position: absolute;
        width: 250px; height: 250px;
        background: radial-gradient(circle, rgba(0,200,100,0.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-banner::before { top: -80px; right: -80px; }
    .hero-banner::after  { bottom: -50px; left: 30%; }
    .hero-tagline {
        font-family: 'Comfortaa', monospace;
        font-size: 0.75rem;
        color: #00A050;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
        display: block;
    }
    .hero-title {
        font-family: 'Comfortaa', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: #1A1A2E;
        line-height: 1.15;
        margin-bottom: 0.8rem;
    }
    .hero-title .green { color: #00A050; }
    .hero-desc {
        color: #556070;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 620px;
    }
    .waveform-deco {
        display: flex;
        align-items: center;
        gap: 3px;
        margin-top: 1.5rem;
    }
    .waveform-deco span {
        display: inline-block;
        width: 4px;
        background: linear-gradient(180deg, #00C060, #00B4D8);
        border-radius: 2px;
        opacity: 0.5;
        animation: wavepulse 1.2s ease-in-out infinite alternate;
    }
    @keyframes wavepulse {
        from { transform: scaleY(0.3); }
        to   { transform: scaleY(1); }
    }
    .step-card {
        background: #F8FAFC;
        border: 1px solid rgba(0,0,0,0.07);
        border-left: 3px solid #00C060;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        height: 100%;
    }
    .step-num {
        font-family: 'Comfortaa', monospace;
        font-size: 0.65rem;
        color: #00A050;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .step-title { font-size: 1rem; font-weight: 600; color: #1A1A2E; margin-bottom: 0.3rem; }
    .step-desc  { font-size: 0.85rem; color: #556070; line-height: 1.5; }
    .section-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(0,180,80,0.08);
        border: 1px solid rgba(0,180,80,0.2);
        border-radius: 40px;
        padding: 0.2rem 0.8rem;
        font-family: 'Comfortaa', monospace;
        font-size: 0.7rem;
        color: #00A050;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .badge {
        display: inline-block;
        background: rgba(0,180,80,0.08);
        border: 1px solid rgba(0,180,80,0.22);
        color: #00A050;
        font-family: 'Comfortaa', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        margin-right: 0.4rem;
        margin-bottom: 0.5rem;
    }
    .badge.blue   { background: rgba(0,150,200,0.08);   border-color: rgba(0,150,200,0.22);   color: #0090C0; }
    .badge.purple { background: rgba(100,50,200,0.08);  border-color: rgba(100,50,200,0.22);  color: #6030B0; }
    .stButton > button {
        background: linear-gradient(135deg, #00C060, #009A4E) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Comfortaa', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1.2rem !important;
        transition: 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(0,180,80,0.2) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #00E676, #00C060) !important;
        box-shadow: 0 4px 16px rgba(0,200,80,0.25) !important;
        transform: translateY(-1px);
    }
    [data-testid="stMetric"] {
        background: #F0F4F8 !important;
        border: 1px solid rgba(0,0,0,0.07) !important;
        border-radius: 12px !important;
        padding: 0.8rem 1rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Comfortaa', monospace !important;
        font-size: 0.6rem !important;
        color: #7A8FAA !important;
        letter-spacing: 1.2px !important;
        text-transform: uppercase !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Comfortaa', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
        color: #00A050 !important;
    }
    [data-testid="stExpander"] {
        background: #F8FAFC !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }
    [data-testid="stExpander"] summary {
        font-family: 'Comfortaa', sans-serif !important;
        font-weight: 500 !important;
        color: #556070 !important;
    }
    [data-testid="stExpander"] summary:hover { color: #00A050 !important; }
    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #00C060, #00B4D8) !important;
        border-radius: 999px !important;
    }
    [data-testid="stProgressBar"] > div {
        background: rgba(0,0,0,0.07) !important;
        border-radius: 999px !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        font-family: 'Comfortaa', monospace !important;
        font-size: 0.68rem !important;
        color: #8A9AB0 !important;
    }
    .stSuccess > div {
        background: rgba(0,180,80,0.07) !important;
        border-left: 3px solid #00C060 !important;
        border-radius: 8px !important;
        color: #006030 !important;
    }
    .stInfo > div {
        background: rgba(0,150,200,0.07) !important;
        border-left: 3px solid #00A0C8 !important;
        border-radius: 8px !important;
    }
    .stWarning > div {
        background: rgba(200,130,0,0.07) !important;
        border-left: 3px solid #C08000 !important;
        border-radius: 8px !important;
    }
    .stError > div {
        background: rgba(200,50,50,0.07) !important;
        border-left: 3px solid #C03030 !important;
        border-radius: 8px !important;
    }
    .stAudioInput > div {
        background: #F0F4F8 !important;
        border: 1px solid rgba(0,180,80,0.15) !important;
        border-radius: 12px !important;
    }
    hr {
        border: none !important;
        border-top: 1px solid rgba(0,0,0,0.07) !important;
        margin: 2rem 0 !important;
    }
    [data-testid="stPageLink"] {
        background: #F0F4F8 !important;
        border: 1px solid rgba(0,180,80,0.15) !important;
        border-radius: 12px !important;
        transition: 0.2s !important;
    }
    [data-testid="stPageLink"]:hover {
        border-color: rgba(0,180,80,0.35) !important;
        box-shadow: 0 0 16px rgba(0,180,80,0.08) !important;
    }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #F5F7FA; }
    ::-webkit-scrollbar-thumb { background: rgba(0,180,80,0.2); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(0,180,80,0.4); }
</style>
""", unsafe_allow_html=True)

# ========== MATPLOTLIB GLOBAL STYLE ==========
plt.rcParams.update({
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#F8FAFC',
    'axes.edgecolor': '#CCCCCC',
    'axes.labelcolor': '#444444',
    'axes.titlecolor': '#222222',
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'xtick.color': '#666666',
    'ytick.color': '#666666',
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'grid.color': '#DDDDDD',
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'text.color': '#444444',
    'font.family': 'monospace',
    'lines.linewidth': 1.4,
})


# ========== FUNGSI EKSTRAKSI FITUR SPEKTRAL (VERSI DIPERBAIKI) ==========
@st.cache_data
def extract_spectral_features(file_path, sr_target=16000, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=sr_target)

    # --- MFCC (timbre dasar) ---
    mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std  = np.std(mfcc, axis=1)

    # --- Spectral features dasar ---
    centroid_m  = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)) / (sr / 2)
    rolloff_m   = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)) / (sr / 2)
    bandwidth_m = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)) / (sr / 2)
    flatness_m  = np.mean(librosa.feature.spectral_flatness(y=y))
    zcr_m       = np.mean(librosa.feature.zero_crossing_rate(y))

    contrast   = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_m = np.mean(contrast, axis=1)
    contrast_n = contrast_m / (np.max(np.abs(contrast_m)) + 1e-9)

    # --- Pitch / F0 — PEMBEDA SPESIES UTAMA ---
    # Kucing : 600–1500 Hz
    # Manusia: 80–300 Hz
    # Perbedaan ini bisa 5–10× lipat → harus sangat dominan di feature vector
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),   # ~65 Hz
        fmax=librosa.note_to_hz('C7'),   # ~2093 Hz
        sr=sr
    )
    f0_voiced = f0[voiced_flag & ~np.isnan(f0)] if f0 is not None else np.array([])

    if len(f0_voiced) > 0:
        f0_mean_raw  = np.mean(f0_voiced)
        f0_median    = np.median(f0_voiced)
        f0_std_raw   = np.std(f0_voiced)
        f0_range_raw = np.max(f0_voiced) - np.min(f0_voiced)
        voiced_ratio = len(f0_voiced) / (len(f0) + 1e-9)

        # Normalisasi ke [0,1] dengan skala 0–2000 Hz
        f0_mean_n   = np.clip(f0_mean_raw   / 2000.0, 0, 1)
        f0_median_n = np.clip(f0_median     / 2000.0, 0, 1)
        f0_std_n    = np.clip(f0_std_raw    / 1000.0, 0, 1)
        f0_range_n  = np.clip(f0_range_raw  / 2000.0, 0, 1)
    else:
        # Tidak ada voiced frame — suara tidak bernada
        f0_mean_n = f0_median_n = f0_std_n = f0_range_n = voiced_ratio = 0.0

    # --- Harmonic-to-Noise Ratio proxy ---
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = np.mean(y_harm ** 2)
    perc_energy = np.mean(y_perc ** 2) + 1e-9
    hnr_norm    = np.clip(harm_energy / perc_energy / 10.0, 0, 1)

    # --- Formant proxy: rasio energi pita frekuensi ---
    # Kucing: energi dominan di pita tinggi (>1000 Hz)
    # Manusia: konsentrasi energi di 200–800 Hz (formant F1–F2)
    stft_mag = np.abs(librosa.stft(y))
    freqs    = librosa.fft_frequencies(sr=sr)

    band_low  = np.mean(stft_mag[freqs <  500, :])
    band_mid  = np.mean(stft_mag[(freqs >= 500) & (freqs < 1500), :])
    band_high = np.mean(stft_mag[freqs >= 1500, :])
    total_e   = band_low + band_mid + band_high + 1e-9

    ratio_low  = band_low  / total_e
    ratio_mid  = band_mid  / total_e
    ratio_high = band_high / total_e

    # ================================================================
    # FEATURE VECTOR — bobot eksplisit:
    #
    # Pitch diberi bobot TINGGI (×6): perbedaan kucing vs manusia
    # bisa 5× lipat di dimensi ini. Tanpa pembobotan kuat, 13 dim
    # MFCC akan mendominasi dan mengaburkan sinyal pitch.
    #
    # Formant proxy (×4): distribusi energi frekuensi berbeda spesies.
    # MFCC dikurangi ke ×0.5 agar tidak mendominasi.
    # ================================================================
    PITCH_WEIGHT   = 6.0
    FORMANT_WEIGHT = 4.0
    HNR_WEIGHT     = 2.0
    MFCC_WEIGHT    = 0.5

    feature_vector = np.concatenate([
        # Pitch — paling diskriminatif antar spesies
        [f0_mean_n   * PITCH_WEIGHT],
        [f0_median_n * PITCH_WEIGHT],
        [f0_std_n    * PITCH_WEIGHT * 0.5],
        [f0_range_n  * PITCH_WEIGHT * 0.5],
        [voiced_ratio],

        # Distribusi energi frekuensi (formant proxy)
        [ratio_low   * FORMANT_WEIGHT],
        [ratio_mid   * FORMANT_WEIGHT],
        [ratio_high  * FORMANT_WEIGHT],

        # HNR
        [hnr_norm    * HNR_WEIGHT],

        # Spectral features
        [centroid_m],
        [rolloff_m],
        [bandwidth_m],
        [flatness_m],
        [zcr_m],
        contrast_n,

        # MFCC (bobot dikurangi)
        mfcc_mean * MFCC_WEIGHT,
        mfcc_std  * MFCC_WEIGHT * 0.5,
    ])

    return feature_vector, y, sr


def spectral_distance(vec_a, vec_b):
    # Cosine similarity: menangkap kesamaan "arah" vektor (shape)
    cd      = cos_dist(vec_a, vec_b)
    cos_sim = 1.0 - cd   # [0,1], 1 = identik

    # Euclidean similarity: sensitif terhadap MAGNITUDO perbedaan
    # tanh(x/3) lebih cepat mendekati 1 saat jarak besar
    euc_dist = euclidean(vec_a, vec_b)
    euc_sim  = 1.0 - np.tanh(euc_dist / 3.0)

    # Cosine 20% + Euclidean 80%
    # Euclidean lebih dominan karena kita butuh sensitivitas magnitudo
    similarity = 0.20 * cos_sim + 0.80 * euc_sim

    return float(np.clip(similarity, 0.0, 1.0)), float(cd)


def plot_spectral_comparison(file_a, file_b, label_a="Audio A", label_b="Audio B", sr_target=16000):
    y_a, sr = librosa.load(file_a, sr=sr_target)
    y_b, _  = librosa.load(file_b, sr=sr_target)

    n_mfcc = 13
    mfcc_a = np.mean(librosa.feature.mfcc(y=y_a, sr=sr, n_mfcc=n_mfcc), axis=1)
    mfcc_b = np.mean(librosa.feature.mfcc(y=y_b, sr=sr, n_mfcc=n_mfcc), axis=1)

    cent_a = librosa.feature.spectral_centroid(y=y_a, sr=sr)[0]
    cent_b = librosa.feature.spectral_centroid(y=y_b, sr=sr)[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))

    x = np.arange(n_mfcc)
    width = 0.35
    ax1.bar(x - width/2, mfcc_a, width, label=label_a, color='#00A050', alpha=0.8)
    ax1.bar(x + width/2, mfcc_b, width, label=label_b, color='#FF6B35', alpha=0.8)
    ax1.set_title("Mean MFCC — Perbandingan Spektral")
    ax1.set_xlabel("Koefisien MFCC ke-")
    ax1.set_ylabel("Nilai rata-rata")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i+1) for i in x])
    ax1.legend(fontsize=8)
    ax1.grid(True, axis='y')

    t_a = np.linspace(0, len(y_a)/sr, len(cent_a))
    t_b = np.linspace(0, len(y_b)/sr, len(cent_b))
    ax2.plot(t_a, cent_a, color='#00A050', linewidth=1.2, alpha=0.85, label=label_a)
    ax2.plot(t_b, cent_b, color='#FF6B35', linewidth=1.2, alpha=0.85, label=label_b, linestyle='--')
    ax2.set_title("Spectral Centroid — Titik Berat Frekuensi")
    ax2.set_xlabel("Waktu (detik)")
    ax2.set_ylabel("Frekuensi (Hz)")
    ax2.legend(fontsize=8)
    ax2.grid(True)

    fig.tight_layout()
    return fig


# ========== HERO HEADER ==========
st.markdown("""
<div class="hero-banner">
    <span class="hero-tagline">🎤 Interactive Audio Lab</span>
    <div class="hero-title">Voice <span class="green">Lab</span></div>
    <p class="hero-desc">
        Jelajahi dunia <strong style="color:#1A1A2E">audio processing</strong> secara interaktif —
        dari gelombang sintetik sampai pengenalan suara berbasis <strong style="color:#00A050">Spectral Distance</strong>.
        Rekam, bandingkan, dan rasakan sendiri cara komputer "mendengar".
    </p>
    <div class="waveform-deco">
        <span style="height:8px;  animation-delay:0s"></span>
        <span style="height:22px; animation-delay:0.10s"></span>
        <span style="height:36px; animation-delay:0.20s"></span>
        <span style="height:14px; animation-delay:0.30s"></span>
        <span style="height:44px; animation-delay:0.40s"></span>
        <span style="height:26px; animation-delay:0.50s"></span>
        <span style="height:38px; animation-delay:0.60s"></span>
        <span style="height:12px; animation-delay:0.70s"></span>
        <span style="height:30px; animation-delay:0.80s"></span>
        <span style="height:46px; animation-delay:0.90s"></span>
        <span style="height:18px; animation-delay:1.00s"></span>
        <span style="height:32px; animation-delay:0.05s"></span>
        <span style="height:42px; animation-delay:0.15s"></span>
        <span style="height:10px; animation-delay:0.25s"></span>
        <span style="height:28px; animation-delay:0.35s"></span>
        <span style="height:40px; animation-delay:0.45s"></span>
        <span style="height:20px; animation-delay:0.55s"></span>
        <span style="height:34px; animation-delay:0.65s"></span>
        <span style="height:16px; animation-delay:0.75s"></span>
        <span style="height:48px; animation-delay:0.85s"></span>
        <span style="height:24px; animation-delay:0.95s"></span>
        <span style="height:38px; animation-delay:0.08s"></span>
        <span style="height:12px; animation-delay:0.18s"></span>
        <span style="height:44px; animation-delay:0.28s"></span>
    </div>
</div>
""", unsafe_allow_html=True)


# ========== BAGIAN 1: APA ITU AUDIO? ==========
st.header("Apa Itu Audio?")
st.markdown("""
<p style="color:#556070; font-size:0.95rem; line-height:1.8; margin-bottom:1.5rem">
Audio adalah <strong style="color:#1A1A2E">getaran udara</strong> yang ditangkap telinga dan direpresentasikan
sebagai sinyal digital oleh komputer. Setiap suara punya <strong style="color:#00A050">bentuk gelombang (waveform)</strong>
yang unik — misalnya, suara kucing mengeong punya pola yang sama sekali berbeda dengan suara manusia.
</p>
""", unsafe_allow_html=True)

st.subheader("Contoh Waveform Sinyal Sintetik")
st.markdown('<span class="badge">Synthetic</span><span class="badge blue">400 Hz + 800 Hz</span>', unsafe_allow_html=True)
st.caption("Gelombang sintetik sangat teratur dan mudah dipelajari — penjumlahan dua frekuensi berbeda.")

sr_synth = 16000
t_synth = np.linspace(0, 0.1, int(sr_synth * 0.1))
y_synth = np.sin(2 * np.pi * 400 * t_synth) + 0.5 * np.sin(2 * np.pi * 800 * t_synth)
fig_synth, ax_synth = plt.subplots(figsize=(9, 2.5))
ax_synth.plot(t_synth, y_synth, color='#00A050', linewidth=1.2, alpha=0.9)
ax_synth.fill_between(t_synth, y_synth, alpha=0.08, color='#00A050')
ax_synth.set_title("Waveform Sinyal Sintetik (400 Hz + 800 Hz) — 0.1 detik")
ax_synth.set_xlabel("Waktu (detik)")
ax_synth.set_ylabel("Amplitudo")
ax_synth.grid(True)
fig_synth.tight_layout()
st.pyplot(fig_synth)
st.caption("Gelombang ini sangat halus dan periodik, berbeda dengan suara alami yang lebih kompleks.")

with st.expander("🐱 Contoh Asli: Waveform Suara Kucing"):
    cat_file = "Cat.mp3"
    if os.path.exists(cat_file):
        st.audio(cat_file, format="audio/mpeg")
        st.caption("▶️ Dengarkan suara kucing asli.")
        y_cat, sr_cat = librosa.load(cat_file, sr=16000)
        duration = len(y_cat) / sr_cat
        sample_limit = min(2 * sr_cat, len(y_cat))
        t_cat = np.linspace(0, sample_limit / sr_cat, sample_limit)
        fig_cat, ax_cat = plt.subplots(figsize=(9, 2.5))
        ax_cat.plot(t_cat, y_cat[:sample_limit], color='#0090C0', linewidth=0.8, alpha=0.9)
        ax_cat.fill_between(t_cat, y_cat[:sample_limit], alpha=0.07, color='#0090C0')
        ax_cat.set_title("Waveform Asli Cat.mp3 (2 detik pertama)")
        ax_cat.set_xlabel("Waktu (detik)")
        ax_cat.set_ylabel("Amplitudo")
        ax_cat.grid(True)
        fig_cat.tight_layout()
        st.pyplot(fig_cat)
        st.caption(f"Durasi total: {duration:.2f} detik. Bentuknya lebih acak dibanding sinyal sintetik.")
    else:
        st.warning(f"File {cat_file} tidak ditemukan.")
        st.info("Pastikan file Cat.mp3 berada di folder yang sama dengan aplikasi ini.")


# ========== BAGIAN 2: BAGAIMANA KOMPUTER PROSES SUARA ==========
st.header("Bagaimana Komputer Memproses Suara?")
st.markdown('<p style="color:#556070; font-size:0.93rem; margin-bottom:1.5rem">Komputer mengubah gelombang suara menjadi angka, lalu menganalisisnya melalui beberapa tahap berikut.</p>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="step-card"><div class="step-num">STEP 01</div><div class="step-title">Sampling</div><div class="step-desc">Mengubah gelombang analog menjadi angka diskrit pada interval tetap</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="step-card"><div class="step-num">STEP 02</div><div class="step-title">Waveform</div><div class="step-desc">Visualisasi deretan angka sampel sebagai gelombang digital</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="step-card"><div class="step-num">STEP 03</div><div class="step-title">Windowing</div><div class="step-desc">Memotong suara panjang menjadi frame pendek 20–50 ms</div></div>', unsafe_allow_html=True)

c4, c5, c6 = st.columns(3)
with c4:
    st.markdown('<div class="step-card"><div class="step-num">STEP 04</div><div class="step-title">FFT</div><div class="step-desc">Mengubah frame dari domain waktu ke spektrum frekuensi</div></div>', unsafe_allow_html=True)
with c5:
    st.markdown('<div class="step-card"><div class="step-num">STEP 05</div><div class="step-title">Spektrogram</div><div class="step-desc">Melihat bagaimana frekuensi berubah seiring waktu (2D)</div></div>', unsafe_allow_html=True)
with c6:
    st.markdown('<div class="step-card"><div class="step-num">STEP 06</div><div class="step-title">MFCC</div><div class="step-desc">Fitur ringkas yang meniru persepsi pendengaran manusia</div></div>', unsafe_allow_html=True)


# ========== LOAD FILE SUARA KUCING ==========
cat_file = "Cat.mp3"
if os.path.exists(cat_file):
    y_full, sr = librosa.load(cat_file, sr=16000)
    durasi_total = len(y_full) / sr
    cat_available = True
    durasi_pendek = 0.05
    n_pendek = int(durasi_pendek * sr)
    y_pendek = y_full[:n_pendek]
    t_pendek = np.linspace(0, durasi_pendek, n_pendek)
    durasi_panjang = 2.0
    n_panjang = int(durasi_panjang * sr)
    y_panjang = y_full[:n_panjang]
    t_panjang = np.linspace(0, durasi_panjang, n_panjang)
else:
    cat_available = False
    st.warning("File Cat.mp3 tidak ditemukan. Contoh asli tidak tersedia.")

COLORS = ['#00A050', '#FF6B35', '#6030B0', '#C08000', '#0090C0']


# ========== LANGKAH 1: SAMPLING ==========
st.subheader("1 — Sampling")
st.markdown('<span class="badge">Analog → Digital</span>', unsafe_allow_html=True)
st.markdown('<p style="color:#556070; font-size:0.9rem; line-height:1.8"><strong style="color:#1A1A2E">Sampling</strong> mengambil nilai amplitudo pada interval waktu tetap. <strong style="color:#00A050">Sample rate</strong> (Hz) = jumlah sampel per detik.</p>', unsafe_allow_html=True)

sr_sintetis = 200
t_cont = np.linspace(0, 0.1, 1000)
y_cont = np.sin(2 * np.pi * 50 * t_cont)
t_sample_sint = np.linspace(0, 0.1, int(sr_sintetis * 0.1))
y_sample_sint = np.sin(2 * np.pi * 50 * t_sample_sint)
fig1, ax1 = plt.subplots(figsize=(9, 3))
ax1.plot(t_cont, y_cont, color='#0090C0', alpha=0.4, linewidth=1.5, label='Gelombang analog (ilustrasi)')
ml, sl, bl = ax1.stem(t_sample_sint, y_sample_sint, linefmt='#00A050', markerfmt='o', basefmt=' ', label=f'Sampel ({sr_sintetis} Hz)')
sl.set_linewidth(1); ml.set_color('#00A050'); ml.set_markersize(5)
ax1.legend(fontsize=8); ax1.grid(True)
fig1.tight_layout()
st.pyplot(fig1)
st.caption("Titik hijau adalah hasil sampling. Komputer menyimpan angka-angka ini.")

if cat_available:
    with st.expander("🐱 Sampling pada suara kucing asli (50 ms)"):
        sr_ilustrasi = 200
        step = sr // sr_ilustrasi
        t_sample_kucing = t_pendek[::step]
        y_sample_kucing = y_pendek[::step]
        fig2, ax2 = plt.subplots(figsize=(9, 3))
        ax2.plot(t_pendek, y_pendek, color='#0090C0', alpha=0.3, linewidth=1.2, label='Gelombang asli')
        ml2, sl2, bl2 = ax2.stem(t_sample_kucing, y_sample_kucing, linefmt='#00A050', markerfmt='o', basefmt=' ', label=f'Sampel ({sr_ilustrasi} Hz)')
        sl2.set_linewidth(1); ml2.set_color('#00A050'); ml2.set_markersize(5)
        ax2.set_xlabel("Waktu (detik)"); ax2.set_ylabel("Amplitudo")
        ax2.set_title("Sampling suara kucing (50 ms)")
        ax2.legend(fontsize=8); ax2.grid(True)
        fig2.tight_layout()
        st.pyplot(fig2)
        st.caption(f"{len(y_sample_kucing)} sampel disimpan komputer.")
        st.session_state['y_sample_kucing'] = y_sample_kucing
        st.session_state['t_sample_kucing'] = t_sample_kucing
        st.session_state['sr_ilustrasi'] = sr_ilustrasi


# ========== LANGKAH 2: WAVEFORM ==========
st.subheader("2 — Representasi Digital (Waveform)")
st.markdown('<span class="badge">Angka → Visualisasi</span>', unsafe_allow_html=True)
st.markdown('<p style="color:#556070; font-size:0.9rem; line-height:1.8">Angka-angka hasil sampling diplot dan dihubungkan garis → <strong style="color:#00A050">Waveform</strong>, bentuk digital suara yang disimpan komputer.</p>', unsafe_allow_html=True)

fig3, ax3 = plt.subplots(figsize=(9, 2.5))
ax3.plot(t_sample_sint, y_sample_sint, color='#6030B0', marker='o', markersize=4, linewidth=1.2)
ax3.fill_between(t_sample_sint, y_sample_sint, alpha=0.06, color='#6030B0')
ax3.set_title("Waveform sinyal sinus (hasil sampling 200 Hz)")
ax3.set_xlabel("Waktu (detik)"); ax3.set_ylabel("Amplitudo digital"); ax3.grid(True)
fig3.tight_layout()
st.pyplot(fig3)
st.caption("Garis menghubungkan titik-titik sampling — inilah representasi digitalnya.")

if cat_available and 'y_sample_kucing' in st.session_state:
    with st.expander("🐱 Waveform dari sampling suara kucing"):
        y_samp  = st.session_state['y_sample_kucing']
        t_samp  = st.session_state['t_sample_kucing']
        sr_ilust = st.session_state['sr_ilustrasi']
        fig4, ax4 = plt.subplots(figsize=(9, 2.5))
        ax4.plot(t_samp, y_samp, color='#6030B0', marker='o', markersize=4, linewidth=1.2)
        ax4.fill_between(t_samp, y_samp, alpha=0.06, color='#6030B0')
        ax4.set_title(f"Waveform suara kucing ({len(y_samp)} sampel, {sr_ilust} Hz)")
        ax4.set_xlabel("Waktu (detik)"); ax4.set_ylabel("Amplitudo digital"); ax4.grid(True)
        fig4.tight_layout()
        st.pyplot(fig4)


# ========== LANGKAH 3: WINDOWING ==========
st.subheader("3 — Windowing")
st.markdown('<span class="badge">Segmentasi</span><span class="badge blue">Frame 25 ms</span>', unsafe_allow_html=True)
st.markdown('<p style="color:#556070; font-size:0.9rem; line-height:1.8">Suara panjang dipotong jadi <strong style="color:#00A050">frame pendek 20–50 ms</strong> yang tumpang tindih, lalu setiap frame dianalisis dengan FFT.</p>', unsafe_allow_html=True)

durasi_sint = 0.5
t_sint = np.linspace(0, durasi_sint, int(16000 * durasi_sint))
y_sint = np.sin(2 * np.pi * 440 * t_sint)
frame_durasi = 0.025
hop_durasi   = 0.010
frame_len = int(frame_durasi * 16000)
hop_len   = int(hop_durasi   * 16000)
fig5, ax5 = plt.subplots(figsize=(9, 3))
ax5.plot(t_sint, y_sint, color='#0090C0', alpha=0.5, linewidth=1)
for i in range(5):
    start = i * hop_len
    if start + frame_len < len(t_sint):
        ax5.axvspan(t_sint[start], t_sint[start + frame_len], alpha=0.14, color=COLORS[i % len(COLORS)])
ax5.set_xlabel("Waktu (detik)"); ax5.set_title("Windowing — setiap warna adalah frame 25 ms"); ax5.grid(True)
fig5.tight_layout()
st.pyplot(fig5)
st.caption("Setiap window akan dianalisis dengan FFT untuk melihat frekuensi pada saat itu.")

if cat_available:
    with st.expander("🐱 Windowing pada suara kucing (2 detik pertama)"):
        fig6, ax6 = plt.subplots(figsize=(9, 3))
        ax6.plot(t_panjang, y_panjang, color='#0090C0', alpha=0.45, linewidth=0.8)
        for i in range(8):
            start = i * hop_len
            if start + frame_len < len(t_panjang):
                ax6.axvspan(t_panjang[start], t_panjang[start + frame_len], alpha=0.12, color=COLORS[i % len(COLORS)])
        ax6.set_xlabel("Waktu (detik)"); ax6.set_title("Windowing suara kucing (25 ms, overlap 10 ms)"); ax6.grid(True)
        fig6.tight_layout()
        st.pyplot(fig6)


# ========== LANGKAH 4: FFT ==========
st.subheader("4 — FFT (Fast Fourier Transform)")
st.markdown('<span class="badge">Waktu → Frekuensi</span>', unsafe_allow_html=True)
st.markdown('<p style="color:#556070; font-size:0.9rem; line-height:1.8"><strong style="color:#00A050">FFT</strong> mengubah frame dari domain waktu menjadi <strong style="color:#1A1A2E">spektrum frekuensi</strong> — menunjukkan frekuensi apa yang dominan.</p>', unsafe_allow_html=True)

fs_demo   = 16000
frame_demo = np.linspace(0, 0.025, int(0.025 * fs_demo))
y_demo    = np.sin(2 * np.pi * 440 * frame_demo) + 0.5 * np.sin(2 * np.pi * 880 * frame_demo)
fft_vals  = np.fft.fft(y_demo)
freqs     = np.fft.fftfreq(len(y_demo), 1 / fs_demo)
magnitude = np.abs(fft_vals[:len(fft_vals) // 2])
freqs_pos = freqs[:len(freqs) // 2]
fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(9, 3))
ax7a.plot(frame_demo, y_demo, color='#0090C0', linewidth=1.2)
ax7a.fill_between(frame_demo, y_demo, alpha=0.07, color='#0090C0')
ax7a.set_title("Frame 25 ms (domain waktu)"); ax7a.set_xlabel("Waktu (detik)"); ax7a.grid(True)
ax7b.plot(freqs_pos, magnitude, color='#FF6B35', linewidth=1.2)
ax7b.fill_between(freqs_pos, magnitude, alpha=0.09, color='#FF6B35')
ax7b.set_title("Spektrum frekuensi (FFT)"); ax7b.set_xlabel("Frekuensi (Hz)"); ax7b.set_xlim(0, 2000); ax7b.grid(True)
plt.tight_layout()
st.pyplot(fig7)
st.caption("Puncak di 440 Hz dan 880 Hz terlihat jelas di spektrum.")

if cat_available:
    with st.expander("🐱 FFT pada satu frame suara kucing"):
        frame_kucing = y_panjang[:frame_len]
        fft_kucing   = np.fft.fft(frame_kucing)
        mag_kucing   = np.abs(fft_kucing[:len(fft_kucing) // 2])
        freqs_kucing = np.fft.fftfreq(len(frame_kucing), 1 / sr)[:len(frame_kucing) // 2]
        fig8, ax8 = plt.subplots(figsize=(9, 2.5))
        ax8.plot(freqs_kucing, mag_kucing, color='#FF6B35', linewidth=1)
        ax8.fill_between(freqs_kucing, mag_kucing, alpha=0.08, color='#FF6B35')
        ax8.set_xlim(0, 4000); ax8.set_xlabel("Frekuensi (Hz)"); ax8.set_ylabel("Magnitudo")
        ax8.set_title("Spektrum frekuensi satu frame suara kucing"); ax8.grid(True)
        fig8.tight_layout()
        st.pyplot(fig8)
        st.caption("Banyak puncak frekuensi — suara kucing memang kompleks.")


# ========== LANGKAH 5: SPEKTROGRAM ==========
st.subheader("5 — Spektrogram")
st.markdown('<span class="badge">Waktu + Frekuensi</span><span class="badge purple">2D Heat Map</span>', unsafe_allow_html=True)
st.markdown('<p style="color:#556070; font-size:0.9rem; line-height:1.8"><strong style="color:#00A050">Spektrogram</strong> = hasil FFT banyak frame berurutan. Sumbu X = waktu, Y = frekuensi, warna = kekuatan sinyal.</p>', unsafe_allow_html=True)

t_gliss  = np.linspace(0, 1, 16000)
f_gliss  = np.linspace(200, 1000, len(t_gliss))
y_gliss  = np.sin(2 * np.pi * f_gliss * t_gliss)
D_gliss  = librosa.amplitude_to_db(np.abs(librosa.stft(y_gliss)), ref=np.max)
fig9, ax9 = plt.subplots(figsize=(9, 3))
img = librosa.display.specshow(D_gliss, sr=16000, x_axis='time', y_axis='hz', ax=ax9, cmap='magma')
ax9.set_title("Spektrogram glissando (frekuensi naik 200 → 1000 Hz)")
plt.colorbar(img, ax=ax9, format="%+2.0f dB")
fig9.tight_layout()
st.pyplot(fig9)
st.caption("Garis miring dari frekuensi rendah ke tinggi — itulah glissando.")

if cat_available:
    with st.expander("🐱 Spektrogram suara kucing (2 detik pertama)"):
        D_cat = librosa.amplitude_to_db(np.abs(librosa.stft(y_panjang)), ref=np.max)
        fig10, ax10 = plt.subplots(figsize=(9, 3))
        img = librosa.display.specshow(D_cat, sr=sr, x_axis='time', y_axis='hz', ax=ax10, cmap='magma')
        ax10.set_title("Spektrogram Cat.mp3")
        plt.colorbar(img, ax=ax10, format="%+2.0f dB")
        fig10.tight_layout()
        st.pyplot(fig10)
        st.caption("Distribusi frekuensi berubah seiring waktu — kompleks dan dinamis.")


# ========== LANGKAH 6: MFCC ==========
st.subheader("6 — MFCC (Mel-Frequency Cepstral Coefficients)")
st.markdown('<span class="badge">Feature Extraction</span><span class="badge blue">13 Koefisien</span>', unsafe_allow_html=True)
st.markdown('<p style="color:#556070; font-size:0.9rem; line-height:1.8"><strong style="color:#00A050">MFCC</strong> adalah fitur ringkas yang meniru persepsi pendengaran manusia. 13–20 koefisien per frame — inilah yang dipakai AI untuk pengenalan suara.</p>', unsafe_allow_html=True)

mfcc_gliss = librosa.feature.mfcc(y=y_gliss, sr=16000, n_mfcc=13)
fig11, ax11 = plt.subplots(figsize=(9, 3))
img = librosa.display.specshow(mfcc_gliss, sr=16000, x_axis='time', ax=ax11, cmap='viridis')
ax11.set_title("MFCC (13 koefisien) dari glissando")
plt.colorbar(img, ax=ax11)
fig11.tight_layout()
st.pyplot(fig11)
st.caption("Setiap baris = satu koefisien MFCC yang menunjukkan aspek berbeda dari spektrum.")

if cat_available:
    with st.expander("🐱 MFCC dari suara kucing (2 detik pertama)"):
        mfcc_cat = librosa.feature.mfcc(y=y_panjang, sr=sr, n_mfcc=13)
        fig12, ax12 = plt.subplots(figsize=(9, 3))
        img = librosa.display.specshow(mfcc_cat, sr=sr, x_axis='time', ax=ax12, cmap='viridis')
        ax12.set_title("MFCC suara kucing")
        plt.colorbar(img, ax=ax12)
        fig12.tight_layout()
        st.pyplot(fig12)
        st.caption("MFCC inilah yang dibandingkan untuk mengenali atau mencocokkan suara.")


# ========== BAGIAN 3: VOICE SIMILARITY CHALLENGE ==========
st.header("Pengenalan Suara — Voice Similarity Challenge")

st.markdown("""
<div style="background:#F0FFF8; border:1px solid rgba(0,180,80,0.15); border-radius:14px; padding:1.2rem 1.6rem; margin-bottom:1.5rem">
<p style="color:#556070; margin:0; font-size:0.92rem; line-height:1.8">
Sistem menggunakan <strong style="color:#00A050">Spectral Distance</strong> untuk mengukur kemiripan suara.
Fitur yang diekstrak meliputi <strong style="color:#1A1A2E">Pitch/F0 (bobot ×6), Formant Proxy (bobot ×4),
HNR (bobot ×2), MFCC, Spectral Centroid, Contrast, Flatness, dan ZCR</strong> — digabung menjadi satu
feature vector, lalu dihitung jaraknya menggunakan
<strong style="color:#00A050">Cosine (20%) + Weighted Euclidean (80%)</strong>.
Skor: <strong style="color:#1A1A2E">0–1</strong> (1 = identik secara spektral).
</p>
</div>
""", unsafe_allow_html=True)

with st.expander("📐 Cara Kerja Spectral Distance (v2 — Species-Aware)"):
    st.markdown("""
    **Spectral Distance v2** dirancang khusus untuk membedakan spesies, bukan hanya timbre.

    **Fitur & bobot:**
    | Fitur | Bobot | Alasan |
    |---|---|---|
    | **Pitch / F0** | ×6 | Kucing 600–1500 Hz vs manusia 80–300 Hz (5–10× beda) |
    | **Formant proxy** | ×4 | Rasio energi <500 Hz / 500–1500 Hz / >1500 Hz berbeda spesies |
    | **HNR** | ×2 | Harmonicity suara berbeda antar spesies |
    | **MFCC** | ×0.5 | Dikurangi agar tidak mengaburkan sinyal pitch |
    | Spectral centroid/rolloff/flatness/ZCR | ×1 | Fitur pelengkap |

    **Metode jarak:**
    - **80% Euclidean** `tanh(d/3)`: sensitif terhadap perbedaan magnitudo besar (pitch)
    - **20% Cosine**: menangkap perbedaan "bentuk" distribusi spektral

    **Mengapa versi lama gagal?**
    Versi sebelumnya menggunakan bobot pitch ×3 dan Euclidean ×0.65 dengan `tanh(d/5)`.
    Setelah normalisasi, nilai pitch hanya ~0.04–0.08, sehingga 13 dimensi MFCC masih mendominasi
    dan membuat manusia terlihat "mirip" kucing. Versi baru memperkuat pitch 2× dan memperketat
    fungsi Euclidean agar lebih sensitif terhadap perbedaan besar.
    """)

cat1_path = "Cat.mp3"
cat2_path = "Cat2.mp3"

if not os.path.exists(cat1_path):
    st.error(f"File {cat1_path} tidak ditemukan. Letakkan Cat.mp3 di folder yang sama.")
    st.stop()

try:
    feat_ref, _, _ = extract_spectral_features(cat1_path)
except Exception as e:
    st.error(f"Gagal memproses Cat.mp3: {e}")
    st.stop()

cat2_available = os.path.exists(cat2_path)

st.markdown("---")
st.markdown("**🎵 Perbandingan Suara Kucing Asli**")

col_cat1, col_cat2 = st.columns(2)
with col_cat1:
    st.markdown("**Kucing 1 (Cat.mp3)**")
    st.audio(cat1_path, format="audio/mpeg")
    st.caption("Suara referensi / patokan.")
with col_cat2:
    if cat2_available:
        st.markdown("**Kucing 2 (Cat2.mp3)**")
        st.audio(cat2_path, format="audio/mpeg")
        st.caption("Suara kucing pembanding.")
    else:
        st.info("File Cat2.mp3 tidak tersedia.")

if cat2_available:
    st.markdown("<br>", unsafe_allow_html=True)
    try:
        feat_cat2, _, _ = extract_spectral_features(cat2_path)
        sim_cat2, dist_cat2 = spectral_distance(feat_ref, feat_cat2)

        _, col_hasil_tengah, _ = st.columns([1, 2, 1])
        with col_hasil_tengah:
            st.metric("Skor Kemiripan Spektral", f"{sim_cat2:.3f}")
            st.caption(f"Cosine Distance: {dist_cat2:.4f}")
            if sim_cat2 > 0.75:
                st.success("✅ **Sangat mirip** secara spektral — karakteristik frekuensi hampir sama.")
            elif sim_cat2 > 0.55:
                st.info("🔊 **Cukup mirip** secara spektral, namun ada perbedaan karakter frekuensi.")
            else:
                st.warning("⚠️ **Berbeda** secara spektral — distribusi frekuensi cukup jauh berbeda.")

        st.markdown("**📊 Visualisasi Perbandingan Spektral**")
        fig_cmp = plot_spectral_comparison(cat1_path, cat2_path, "Cat 1", "Cat 2")
        st.pyplot(fig_cmp)
        st.caption("Kiri: perbandingan mean MFCC per koefisien. Kanan: Spectral Centroid seiring waktu.")
    except Exception as e:
        st.error(f"Gagal menghitung spectral distance: {e}")


# ========== TANTANGAN ==========
st.subheader("🎙️ Tantangan: Siapa yang Lebih Mirip Kucing?")
st.markdown("""
<div style="background:#F0FFF8; border:1px solid rgba(0,180,80,0.12); border-radius:12px; padding:1rem 1.4rem; margin-bottom:1.2rem">
<p style="color:#556070; margin:0; font-size:0.88rem; line-height:1.7">
Rekam dua suara berbeda. Sistem akan menghitung <strong style="color:#00A050">Spectral Distance v2</strong>
antara rekaman kamu dan <strong style="color:#00A050">Cat.mp3</strong> —
semakin mirip pitch dan distribusi frekuensinya, semakin tinggi skornya!
Manusia normal biasanya mendapat skor <strong style="color:#C03030">di bawah 0.40</strong>,
sementara sesama kucing akan mendapat <strong style="color:#00A050">di atas 0.65</strong>.
</p>
</div>
""", unsafe_allow_html=True)


@st.fragment
def rekaman_challenge_section():
    if "human1_audio" not in st.session_state:    st.session_state.human1_audio = None
    if "human2_audio" not in st.session_state:    st.session_state.human2_audio = None
    if "human1_processed" not in st.session_state: st.session_state.human1_processed = False
    if "human2_processed" not in st.session_state: st.session_state.human2_processed = False

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.markdown("**👤 Manusia 1**")
        audio1 = st.audio_input("Rekam suara tiruan kucing (Manusia 1)", key="human1_input")
        if audio1 is not None:
            with st.spinner("⏳ Menghitung Spectral Distance Manusia 1..."):
                temp1 = "temp_human1.wav"
                with open(temp1, "wb") as f:
                    f.write(audio1.getbuffer())
                try:
                    feat1, _, _ = extract_spectral_features(temp1)
                    sim1, dist1 = spectral_distance(feat_ref, feat1)
                    st.session_state.human1_audio = (temp1, feat1, sim1, dist1)
                    st.session_state.human1_processed = True
                    st.metric("Skor Spektral vs Cat.mp3", f"{sim1:.3f}")
                    st.caption(f"Cosine Distance: {dist1:.4f}")
                    if sim1 < 0.40:
                        st.info("🙋 Terdeteksi sebagai **manusia** — pitch terlalu rendah untuk kucing.")
                    elif sim1 < 0.60:
                        st.warning("🐱 Lumayan! Sedikit mirip kucing, tapi masih terdengar manusiawi.")
                    else:
                        st.success("🐱 Luar biasa! Suaramu mendekati spektrum kucing!")
                except Exception as e:
                    st.error(f"Gagal memproses: {e}")
                    if os.path.exists(temp1): os.remove(temp1)
        elif st.session_state.human1_processed:
            _, _, sim1, dist1 = st.session_state.human1_audio
            st.metric("Skor Spektral vs Cat.mp3", f"{sim1:.3f}")
            st.caption(f"Cosine Distance: {dist1:.4f}")

    with col_h2:
        st.markdown("**👤 Manusia 2**")
        audio2 = st.audio_input("Rekam suara tiruan kucing (Manusia 2)", key="human2_input")
        if audio2 is not None:
            with st.spinner("⏳ Menghitung Spectral Distance Manusia 2..."):
                temp2 = "temp_human2.wav"
                with open(temp2, "wb") as f:
                    f.write(audio2.getbuffer())
                try:
                    feat2, _, _ = extract_spectral_features(temp2)
                    sim2, dist2 = spectral_distance(feat_ref, feat2)
                    st.session_state.human2_audio = (temp2, feat2, sim2, dist2)
                    st.session_state.human2_processed = True
                    st.metric("Skor Spektral vs Cat.mp3", f"{sim2:.3f}")
                    st.caption(f"Cosine Distance: {dist2:.4f}")
                    if sim2 < 0.40:
                        st.info("🙋 Terdeteksi sebagai **manusia** — pitch terlalu rendah untuk kucing.")
                    elif sim2 < 0.60:
                        st.warning("🐱 Lumayan! Sedikit mirip kucing, tapi masih terdengar manusiawi.")
                    else:
                        st.success("🐱 Luar biasa! Suaramu mendekati spektrum kucing!")
                except Exception as e:
                    st.error(f"Gagal memproses: {e}")
                    if os.path.exists(temp2): os.remove(temp2)
        elif st.session_state.human2_processed:
            _, _, sim2, dist2 = st.session_state.human2_audio
            st.metric("Skor Spektral vs Cat.mp3", f"{sim2:.3f}")
            st.caption(f"Cosine Distance: {dist2:.4f}")

    if st.session_state.human1_processed and st.session_state.human2_processed:
        st.markdown("---")
        st.markdown("### 🏆 Hasil")

        _, _, sim1, dist1 = st.session_state.human1_audio
        _, _, sim2, dist2 = st.session_state.human2_audio
        temp1 = st.session_state.human1_audio[0]
        temp2 = st.session_state.human2_audio[0]

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric("Manusia 1", f"{sim1:.3f}", delta=f"{sim1 - sim2:+.3f} vs M2")
        with col_r2:
            st.metric("Manusia 2", f"{sim2:.3f}", delta=f"{sim2 - sim1:+.3f} vs M1")
        with col_r3:
            winner = "Manusia 1 🥇" if sim1 > sim2 else ("Manusia 2 🥇" if sim2 > sim1 else "Seri 🤝")
            st.metric("Pemenang", winner)

        if sim1 > sim2:
            st.success(f"✅ **Manusia 1 menang!** Spektrum suaranya lebih dekat ke kucing (skor {sim1:.3f} vs {sim2:.3f}).")
        elif sim2 > sim1:
            st.success(f"✅ **Manusia 2 menang!** Spektrum suaranya lebih dekat ke kucing (skor {sim2:.3f} vs {sim1:.3f}).")
        else:
            st.info("🤝 **Seri!** Kedua suara memiliki jarak spektral yang sama ke Cat.mp3.")

        st.markdown("**📊 Perbandingan Spektral: Manusia 1 vs Manusia 2 vs Kucing**")
        try:
            y1,    sr1    = librosa.load(temp1,     sr=16000)
            y2,    sr2    = librosa.load(temp2,     sr=16000)
            y_ref, sr_ref = librosa.load(cat1_path, sr=16000)

            mfcc1  = np.mean(librosa.feature.mfcc(y=y1,    sr=sr1,    n_mfcc=13), axis=1)
            mfcc2  = np.mean(librosa.feature.mfcc(y=y2,    sr=sr2,    n_mfcc=13), axis=1)
            mfcc_r = np.mean(librosa.feature.mfcc(y=y_ref, sr=sr_ref, n_mfcc=13), axis=1)

            x = np.arange(13)
            fig_win, ax_win = plt.subplots(figsize=(10, 3.5))
            ax_win.plot(x, mfcc_r, color='#0090C0', linewidth=2,   marker='o', markersize=5, label='Cat.mp3 (referensi)')
            ax_win.plot(x, mfcc1,  color='#00A050', linewidth=1.5,  marker='s', markersize=4, label=f'Manusia 1 (skor {sim1:.3f})')
            ax_win.plot(x, mfcc2,  color='#FF6B35', linewidth=1.5,  marker='^', markersize=4, label=f'Manusia 2 (skor {sim2:.3f})')
            ax_win.set_title("Mean MFCC — Manusia 1 vs Manusia 2 vs Kucing")
            ax_win.set_xlabel("Koefisien MFCC ke-")
            ax_win.set_ylabel("Nilai rata-rata")
            ax_win.set_xticks(x)
            ax_win.set_xticklabels([str(i+1) for i in x])
            ax_win.legend(fontsize=8)
            ax_win.grid(True)
            fig_win.tight_layout()
            st.pyplot(fig_win)
            st.caption("Kurva yang lebih dekat ke biru (Cat.mp3) = spektrum lebih mirip kucing.")
        except Exception as e:
            st.warning(f"Visualisasi perbandingan tidak tersedia: {e}")

    if st.session_state.human1_processed or st.session_state.human2_processed:
        st.markdown("<br>", unsafe_allow_html=True)
        col_space1, col_btn, col_space3 = st.columns([1, 1, 1])
        with col_btn:
            if st.button("🔄 Reset Rekaman", key="reset_human", use_container_width=True):
                for key in ['human1_audio', 'human2_audio']:
                    if st.session_state[key]:
                        try: os.remove(st.session_state[key][0])
                        except: pass
                st.session_state.human1_audio     = None
                st.session_state.human2_audio     = None
                st.session_state.human1_processed = False
                st.session_state.human2_processed = False
                st.rerun()


rekaman_challenge_section()


# ========== EXPLORE SECTION ==========
st.markdown("---")
st.markdown('<div class="section-pill">🧪 Explore</div>', unsafe_allow_html=True)
st.header("Jelajahi Fitur Voice Lab")
st.markdown('<p style="color:#556070; font-size:0.9rem; margin-bottom:1.2rem">Setelah paham dasarnya, coba fitur-fitur seru berikut!</p>', unsafe_allow_html=True)

colA, colB, colC = st.columns(3)
with colA:
    st.page_link(
        "pages/1_Meme_Challenge.py",
        label="🤣 Meme Voice Challenge",
        help="Tonton video meme, rekam suaramu, dan bandingkan kemiripannya!",
        use_container_width=True
    )
    st.caption("✅ Sudah tersedia")
with colB:
    st.page_link(
        "pages/2_Classification.py",
        label="📊 Klasifikasi Suara 3D",
        help="Visualisasi dataset ESC-50 dalam ruang 3D, klik titik untuk mendengar suara.",
        use_container_width=True
    )
    st.caption("🆓 Eksplorasi 50 kelas suara")
with colC:
    st.button("🏎️ F1 Qualifying Voice (Segera)", disabled=True, use_container_width=True)
    st.caption("🚧 Segera hadir")

st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:1rem 0 0.5rem">
    <span style="font-family:'Space Mono',monospace; font-size:0.62rem; color:#AABBCC; letter-spacing:3px; text-transform:uppercase">
        Voice Lab — Audio Processing Playground
    </span>
</div>
""", unsafe_allow_html=True)
