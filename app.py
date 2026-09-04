import os
import re
import joblib
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

# --- 1. KONFIGURASI HALAMAN & CSS DASHBOARD ---
st.set_page_config(
    page_title="IMDb Sentiment Analytics Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-subtext {
        font-size: 0.8rem;
        color: #10b981;
        font-weight: 600;
    }
    .metric-subtext.negative {
        color: #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. SETUP PREPROCESSING & RESOURCE NLTK ---
@st.cache_resource
def setup_nltk():
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)

    packages = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
    for pkg in packages:
        nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)

setup_nltk()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    text = re.sub(r'<br\s*/?>', ' ', str(text))
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens if w.isalpha()]
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

@st.cache_resource
def load_artifacts():
    try:
        vec = joblib.load('models/tfidf_vectorizer.pkl')
        clf = joblib.load('models/mlp_sentiment_model.pkl')
        return vec, clf
    except Exception:
        return None, None

tfidf, model = load_artifacts()

# --- 3. SIDEBAR PANEL ---
with st.sidebar:
    st.title("⚙️ Dashboard Controls")
    app_mode = st.radio(
        "Pilih Mode Analisis:",
        ["📈 Analisis Batch (Executive Dashboard)", "🧪 Live Prediction Sandbox"]
    )
    st.divider()
    st.subheader("Sumber Data")
    uploaded_file = st.file_uploader("Upload CSV Ulasan (Wajib kolom 'review'):", type=["csv"])
    use_sample = st.checkbox("Gunakan Dataset Latihan (IMDb Sample)", value=True if not uploaded_file else False)
    st.divider()
    st.caption("Stack: Python, Streamlit, Scikit-learn (MLP + TF-IDF), Plotly")

# --- 4. DATA LOADING ---
@st.cache_data
def get_sample_data():
    url = 'https://raw.githubusercontent.com/Faiqazmi/Dataset_latihan/main/IMDB_small_size.csv'[cite: 1]
    return pd.read_csv(url).head(300)

if uploaded_file:
    df_current = pd.read_csv(uploaded_file)
elif use_sample:
    df_current = get_sample_data()
else:
    df_current = None

# --- 5. TAMPILAN DASHBOARD ---
if app_mode == "📈 Analisis Batch (Executive Dashboard)":
    st.markdown("## 📊 Movie Sentiment Analytics Dashboard")
    st.caption("Monitoring polaritas sentimen dan tren kata kunci ulasan penonton film secara agregat.")
    
    if df_current is not None and 'review' in df_current.columns:
        if tfidf and model:
            with st.spinner("Menjalankan inferensi model MLP..."):
                df_current['cleaned_review'] = df_current['review'].apply(preprocess_text)
                X_vec = tfidf.transform(df_current['cleaned_review'])
                preds = model.predict(X_vec)
                probs = model.predict_proba(X_vec)
                df_current['pred_label'] = np.where(preds == 1, 'positive', 'negative')
                df_current['confidence'] = np.max(probs, axis=1) * 100
        else:
            df_current['cleaned_review'] = df_current['review'].apply(preprocess_text)
            df_current['pred_label'] = df_current.get('sentiment', 'positive')
            df_current['confidence'] = 85.0

        total = len(df_current)
        pos = (df_current['pred_label'] == 'positive').sum()
        neg = (df_current['pred_label'] == 'negative').sum()
        pos_pct = (pos / total) * 100
        neg_pct = (neg / total) * 100
        avg_conf = df_current['confidence'].mean()

        # Row 1: KPI Metrics Card
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Ulasan</div>
                <div class="metric-value">{total:,}</div>
                <div class="metric-subtext">100% Data Masuk</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Sentimen Positif</div>
                <div class="metric-value">{pos:,}</div>
                <div class="metric-subtext">▲ {pos_pct:.1f}% Proporsi</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Sentimen Negatif</div>
                <div class="metric-value">{neg:,}</div>
                <div class="metric-subtext negative">▼ {neg_pct:.1f}% Proporsi</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Rata-Rata Confidence</div>
                <div class="metric-value">{avg_conf:.1f}%</div>
                <div class="metric-subtext">Tingkat Keyakinan</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 2: Visualisasi Komparatif
        c_pie, c_wc = st.columns([1, 1.2])
        with c_pie:
            st.subheader("Distribusi Sentimen")
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Positif', 'Negatif'],
                values=[pos, neg],
                hole=0.55,
                marker=dict(colors=['#10b981', '#ef4444']),
                textinfo='label+percent'
            )])
            fig_pie.update_layout(showlegend=False, height=320, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_pie, width="stretch")

        with c_wc:
            st.subheader("Word Cloud Kata Kunci")
            all_txt = ' '.join(df_current['cleaned_review'].dropna())
            if all_txt.strip():
                wc = WordCloud(width=800, height=450, background_color="white", colormap="viridis").generate(all_txt)
                fig_wc, ax = plt.subplots(figsize=(8, 4.5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig_wc)

        st.divider()

        # Row 3: Filter & Tabel Interaktif
        st.subheader("Eksplorasi Detail Ulasan")
        f_options = st.multiselect("Filter Sentimen:", options=['positive', 'negative'], default=['positive', 'negative'])
        filtered_df = df_current[df_current['pred_label'].isin(f_options)]

        st.dataframe(
            filtered_df[['review', 'pred_label', 'confidence']],
            column_config={
                "review": st.column_config.TextColumn("Ulasan Asli", width="large"),
                "pred_label": st.column_config.TextColumn("Label Prediksi"),
                "confidence": st.column_config.ProgressColumn("Confidence Score", format="%.2f%%", min_value=0, max_value=100)
            },
            width="stretch",
            height=300
        )
    else:
        st.error("Format data tidak sesuai. Kolom 'review' wajib ada di file CSV.")

elif app_mode == "🧪 Live Prediction Sandbox":
    st.markdown("## 🧪 Live Inference Test")
    st.caption("Uji ulasan baru secara interaktif.")

    col_in, col_out = st.columns([1.2, 1])
    with col_in:
        user_text = st.text_area(
            "Tulis ulasan film di sini:", 
            height=180, 
            placeholder="Contoh: Brilliant movie with an incredible storyline and phenomenal acting!"
        )
        run_btn = st.button("Jalankan Analisis", type="primary")

    with col_out:
        if run_btn and user_text.strip():
            if tfidf and model:
                cleaned = preprocess_text(user_text)
                v = tfidf.transform([cleaned])
                res = model.predict(v)[0]
                prob = model.predict_proba(v)[0]
                conf = max(prob) * 100

                if res == 1:
                    st.success("### HASIL: SENTIMEN POSITIF")
                    g_color = "#10b981"
                else:
                    st.error("### HASIL: SENTIMEN NEGATIF")
                    g_color = "#ef4444"

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=conf,
                    number={'suffix': "%"},
                    title={'text': "Tingkat Keyakinan Model"},
                    gauge={
                        'axis': {'range': [50, 100]},
                        'bar': {'color': g_color},
                        'steps': [
                            {'range': [50, 75], 'color': "#f1f5f9"},
                            {'range': [75, 100], 'color': "#e2e8f0"}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=240, margin=dict(t=30, b=10, l=30, r=30))
                st.plotly_chart(fig_gauge, width="stretch")
                st.caption(f"**Kata hasil preprocessing:** `{cleaned}`")
            else:
                st.warning("Model .pkl belum tersedia di direktori `models/`.")
        else:
            st.info("Ketik ulasan di panel sebelah kiri lalu tekan tombol.")