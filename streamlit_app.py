# =============================================================================
# STREAMLIT APP - Analisis Clustering Cacat Produk Industri Manufaktur
# Menampilkan hasil clustering (dari clustering_analysis.ipynb / .py) beserta
# interpretasi model & insight bisnisnya.
#
# Cara jalankan (dari folder project):
#   streamlit run streamlit_app.py
# =============================================================================

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Clustering Cacat Produk Manufaktur",
    page_icon="🏭",
    layout="wide",
)

# -----------------------------------------------------------------------
# Util: palet warna RGB random (konsisten dengan notebook analisis)
# -----------------------------------------------------------------------
def random_rgb_palette(n, seed=42):
    rng = np.random.RandomState(seed)
    colors = []
    for _ in range(n):
        r, g, b = rng.randint(30, 230, size=3) / 255.0
        colors.append((r, g, b))
    return colors


@st.cache_data
def load_data():
    df = pd.read_csv("outputs/defects_data_with_clusters.csv")
    df["defect_date"] = pd.to_datetime(df["defect_date"])
    return df


@st.cache_data
def load_supporting_tables():
    tables = {}
    for name in ["model_comparison", "tuning_kmeans_scores", "feature_importance"]:
        try:
            tables[name] = pd.read_csv(f"outputs/{name}.csv")
        except FileNotFoundError:
            tables[name] = None
    return tables


df = load_data()
tables = load_supporting_tables()
n_clusters = df["cluster"].nunique()
palette = random_rgb_palette(max(n_clusters, 1))

# =============================================================================
# HEADER
# =============================================================================
st.title("🏭 Analisis Clustering Cacat Produk Industri Manufaktur")
st.caption(
    "Dashboard hasil clustering (KMeans) terhadap data cacat produk manufaktur, "
    "beserta interpretasi model & insight bisnis untuk keperluan quality control."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Data Cacat", f"{len(df):,}")
col2.metric("Jumlah Cluster", n_clusters)
col3.metric("Rata-rata Repair Cost", f"${df['repair_cost'].mean():,.2f}")
col4.metric("Total Produk Unik", df["product_id"].nunique())

st.divider()

# =============================================================================
# SIDEBAR - FILTER
# =============================================================================
st.sidebar.header("🔍 Filter Data")
selected_clusters = st.sidebar.multiselect(
    "Pilih Cluster",
    options=sorted(df["cluster"].unique()),
    default=sorted(df["cluster"].unique()),
)
selected_severity = st.sidebar.multiselect(
    "Severity", options=df["severity"].unique(), default=list(df["severity"].unique())
)
selected_type = st.sidebar.multiselect(
    "Defect Type", options=df["defect_type"].unique(), default=list(df["defect_type"].unique())
)

df_filtered = df[
    df["cluster"].isin(selected_clusters)
    & df["severity"].isin(selected_severity)
    & df["defect_type"].isin(selected_type)
]

st.sidebar.markdown("---")
st.sidebar.info(
    f"Menampilkan **{len(df_filtered):,}** dari **{len(df):,}** baris data "
    "berdasarkan filter di atas."
)

if df_filtered.empty:
    st.warning("Tidak ada data yang cocok dengan filter yang dipilih. Silakan ubah filter di sidebar.")
    st.stop()

# =============================================================================
# TAB LAYOUT
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Hasil Clustering", "🧠 Interpretasi Model", "🏷️ Profil per Cluster", "📄 Data Mentah"]
)

# -----------------------------------------------------------------------
# TAB 1: HASIL CLUSTERING
# -----------------------------------------------------------------------
with tab1:
    st.subheader("Visualisasi Sebaran Cluster (PCA 2D)")
    st.markdown(
        "Data dengan fitur kategorikal (`defect_type`, `defect_location`, `severity`, "
        "`inspection_method`) diproyeksikan ke 2 dimensi (PCA) agar bisa divisualisasikan."
    )

    cat_cols = ["defect_type", "defect_location", "severity", "inspection_method"]
    dfe = pd.get_dummies(df_filtered[cat_cols])
    if len(df_filtered) >= 2 and dfe.shape[1] >= 2:
        X_scaled = StandardScaler().fit_transform(dfe)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(9, 6))
        for i, cl in enumerate(sorted(df_filtered["cluster"].unique())):
            mask = df_filtered["cluster"].values == cl
            ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                s=45, alpha=0.75, color=palette[i % len(palette)],
                edgecolor="white", linewidth=0.4, label=f"Cluster {cl}",
            )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variansi)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variansi)")
        ax.set_title("Sebaran Cluster (PCA 2D)")
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig)
    else:
        st.info("Data terfilter terlalu sedikit / kurang variatif untuk divisualisasikan PCA.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Distribusi Jumlah Data per Cluster")
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df_filtered["cluster"].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values,
               color=random_rgb_palette(len(counts), seed=7))
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Jumlah Data")
        st.pyplot(fig)

    with col_b:
        st.subheader("Rata-rata Repair Cost per Cluster")
        fig, ax = plt.subplots(figsize=(6, 4))
        avg_cost = df_filtered.groupby("cluster")["repair_cost"].mean().sort_index()
        ax.bar(avg_cost.index.astype(str), avg_cost.values,
               color=random_rgb_palette(len(avg_cost), seed=13))
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Rata-rata Repair Cost ($)")
        st.pyplot(fig)

# -----------------------------------------------------------------------
# TAB 2: INTERPRETASI MODEL
# -----------------------------------------------------------------------
with tab2:
    st.subheader("Perbandingan Algoritma Clustering")
    st.markdown(
        "Beberapa algoritma clustering dibandingkan (KMeans, Agglomerative, Gaussian "
        "Mixture, DBSCAN) dan dipilih yang memberikan **Silhouette Score** tertinggi."
    )
    if tables["model_comparison"] is not None:
        st.dataframe(tables["model_comparison"], use_container_width=True)
    else:
        st.info("File `outputs/model_comparison.csv` tidak ditemukan. Jalankan notebook analisis terlebih dahulu.")

    if tables["tuning_kmeans_scores"] is not None:
        st.subheader("Hasil Tuning Jumlah Cluster (k) - KMeans")
        st.dataframe(tables["tuning_kmeans_scores"], use_container_width=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        t = tables["tuning_kmeans_scores"]
        ax.plot(t["k"], t["silhouette"], marker="o", color=palette[0])
        ax.set_xlabel("Jumlah Cluster (k)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Score vs Jumlah Cluster")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    st.subheader("Fitur Paling Berpengaruh (SHAP / Permutation Importance)")
    st.markdown(
        "Dihitung dari *surrogate model* (Random Forest) yang dilatih untuk memprediksi "
        "label cluster dari fitur asli, lalu dijelaskan dengan SHAP (atau Permutation "
        "Importance sebagai fallback bila `shap` tidak tersedia)."
    )
    if tables["feature_importance"] is not None:
        fi = tables["feature_importance"].sort_values("importance", ascending=True).tail(12)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(fi["feature"], fi["importance"], color=random_rgb_palette(len(fi), seed=21))
        ax.set_xlabel("Rata-rata kontribusi terhadap prediksi cluster")
        st.pyplot(fig)
    else:
        st.info("File `outputs/feature_importance.csv` tidak ditemukan. Jalankan notebook analisis terlebih dahulu.")

    st.subheader("📌 Catatan Metodologi & Interpretasi Metrik")
    st.markdown(
        """
        - **Silhouette Score** yang didapat tergolong moderat (bukan mendekati 1), karena
          atribut kategorikal pada dataset ini (`defect_type`, `defect_location`, `severity`,
          `inspection_method`) memang terdistribusi hampir independen satu sama lain —
          bukan kesalahan pipeline, melainkan karakteristik data itu sendiri.
        - Fitur numerik (`repair_cost`, tanggal, agregasi produk) sengaja **tidak**
          dipakai sebagai input jarak clustering karena terbukti menambah noise
          (menurunkan silhouette), namun tetap dipakai untuk profiling & insight bisnis.
        - **Class imbalance** (mis. distribusi `severity`) tidak merusak proses clustering
          karena clustering bersifat unsupervised (tidak belajar dari label). Imbalance
          baru menjadi masalah serius jika data ini dipakai untuk **klasifikasi**
          (supervised learning), karena model bisa bias ke kelas mayoritas.
        """
    )

# -----------------------------------------------------------------------
# TAB 3: PROFIL PER CLUSTER
# -----------------------------------------------------------------------
with tab3:
    st.subheader("Profil & Insight Bisnis Tiap Cluster")

    for cl in sorted(df_filtered["cluster"].unique()):
        sub = df_filtered[df_filtered["cluster"] == cl]
        with st.expander(f"📦 Cluster {cl} — {len(sub)} baris ({len(sub)/len(df_filtered)*100:.1f}% data terfilter)", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Dominan Defect Type", sub["defect_type"].mode()[0])
            c2.metric("Dominan Lokasi", sub["defect_location"].mode()[0])
            c3.metric("Dominan Severity", sub["severity"].mode()[0])
            c4.metric("Rata-rata Repair Cost", f"${sub['repair_cost'].mean():,.2f}")

            colx, coly = st.columns(2)
            with colx:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                vc = sub["severity"].value_counts()
                ax.pie(vc.values, labels=vc.index, autopct="%1.0f%%",
                       colors=random_rgb_palette(len(vc), seed=cl + 1))
                ax.set_title("Komposisi Severity")
                st.pyplot(fig)
            with coly:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                vc = sub["inspection_method"].value_counts()
                ax.pie(vc.values, labels=vc.index, autopct="%1.0f%%",
                       colors=random_rgb_palette(len(vc), seed=cl + 50))
                ax.set_title("Komposisi Inspection Method")
                st.pyplot(fig)

            st.markdown(
                f"**Insight bisnis:** Cluster ini didominasi cacat tipe "
                f"**{sub['defect_type'].mode()[0]}** di lokasi **{sub['defect_location'].mode()[0]}**, "
                f"dengan tingkat keparahan mayoritas **{sub['severity'].mode()[0]}** dan metode "
                f"inspeksi paling umum **{sub['inspection_method'].mode()[0]}**. "
                f"Rata-rata biaya perbaikannya **${sub['repair_cost'].mean():,.2f}**, dibandingkan "
                f"rata-rata keseluruhan **${df['repair_cost'].mean():,.2f}**."
            )

# -----------------------------------------------------------------------
# TAB 4: DATA MENTAH
# -----------------------------------------------------------------------
with tab4:
    st.subheader("Data Hasil Clustering")
    st.dataframe(df_filtered, use_container_width=True)
    st.download_button(
        "⬇️ Download data terfilter (CSV)",
        data=df_filtered.to_csv(index=False).encode("utf-8"),
        file_name="defects_data_filtered.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Dibuat untuk tugas UAS Kecerdasan Buatan — Deployment & Streamlit Application. "
    "Model: KMeans (lihat clustering_analysis.ipynb untuk detail lengkap pipeline)."
)
