# Analisis Clustering Cacat Produk Industri Manufaktur

Project ini dibuat untuk tugas UAS Kecerdasan Buatan (topik: *Deployment & Streamlit
Application* — bagian analisis clustering-nya, sebelum tahap deployment ke Streamlit).

## Isi Folder
```
project/
├── clustering_analysis.ipynb   # Notebook lengkap (sudah berisi output & visualisasi)
├── clustering_analysis.py      # Versi script .py (identik isinya dengan notebook)
├── requirements.txt            # Daftar library yang dibutuhkan
├── data/
│   └── defects_data.csv        # Dataset mentah
└── outputs/                    # Hasil ekspor: dataset+label cluster, grafik PNG, tabel evaluasi
```

## Cara Menjalankan (VS Code)
1. Buat & aktifkan virtual environment (opsional tapi disarankan):
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Buka `clustering_analysis.ipynb` di VS Code (pastikan extension **Jupyter** & **Python**
   sudah terpasang), pilih kernel Python venv di atas, lalu **Run All**.
   - Notebook yang dikirim sudah berisi output hasil run sebelumnya (termasuk semua
     grafik), jadi bisa langsung dibaca tanpa run ulang. Run All hanya diperlukan
     kalau ingin regenerasi hasil / mencoba parameter lain.
   - Bagian **SHAP** butuh package `shap` (sudah ada di `requirements.txt`). Jika
     `shap` belum ter-install saat run pertama, kode otomatis fallback ke
     **Permutation Importance** (tidak akan error), tapi untuk mendapatkan SHAP
     summary plot yang lebih mendalam pastikan `pip install shap` berhasil lalu run ulang.
4. Atau jalankan versi script:
   ```bash
   cd project
   python clustering_analysis.py
   ```
   (grafik akan tersimpan ke folder `outputs/` dan juga tetap ditampilkan lewat
   `plt.show()` jika dijalankan interaktif, misalnya lewat VS Code Interactive Window).

## Ringkasan Pipeline
1. **EDA awal** — cek struktur, tipe data, statistik deskriptif.
2. **Data Cleaning**
   - Missing value: dicek per kolom; aturan >75% kosong → drop kolom, selebihnya → imputasi
     (median untuk numerik, modus untuk kategorikal). *(Pada dataset ini: 0% missing value.)*
   - Duplikasi data: dicek & di-drop bila ada. *(Pada dataset ini: tidak ada duplikat.)*
   - Outlier: metode **IQR** pada `repair_cost`, ditangani dengan winsorizing/capping bila
     ditemukan. *(Pada dataset ini: 0 outlier — data memang sudah bersih.)*
   - Noise: validasi nilai biaya ≤0, normalisasi teks kategori (spasi/kapitalisasi),
     deteksi kategori langka, dan validasi tanggal.
   - Diskusi **class imbalance**: dijelaskan di notebook mengapa imbalance kelas tidak
     merusak clustering (unsupervised) tapi berpengaruh besar pada klasifikasi (supervised).
3. **Drop kolom leakage/tidak informatif**: `defect_id` (ID unik).
4. **Feature Engineering**: fitur tanggal (`defect_month`, `defect_dayofweek`,
   `defect_is_weekend`), agregasi historis per produk (`product_defect_count`,
   `product_avg_repair_cost`), dan rasio biaya (`repair_cost_ratio_to_product_avg`).
5. **Encoding**: One-Hot Encoding untuk 4 fitur kategorikal.
6. **Scaling**: `StandardScaler`.
7. **Feature Selection untuk Clustering**: dibandingkan "semua fitur" vs "fitur
   kategorikal saja" berbasis Silhouette Score → fitur kategorikal terbukti lebih
   baik (noise dari fitur numerik acak dibuang dari input jarak clustering, tapi
   tetap dipakai untuk profiling/interpretasi tiap cluster).
8. **Model Selection + Tuning**: KMeans (tuning k=2..8), Agglomerative, Gaussian
   Mixture, DBSCAN (grid `eps`/`min_samples`, dengan pembatasan agar tidak memilih
   solusi *degenerate*/tidak bermakna) — dipilih berdasarkan Silhouette Score tertinggi.
9. **Evaluasi**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index —
   disertai catatan interpretasi jujur mengapa nilai silhouette tidak dipaksakan
   mendekati 1 (lihat penjelasan di notebook, Bagian 9).
10. **Interpretasi SHAP** (surrogate `RandomForestClassifier` yang memprediksi label
    cluster) untuk mengetahui fitur apa yang paling menentukan suatu baris masuk ke
    cluster tertentu — dengan fallback Permutation Importance jika `shap` belum ter-install.
11. **Visualisasi** (palet warna RGB random): elbow/silhouette plot, perbandingan
    algoritma, PCA 2D scatter, silhouette plot per sampel, feature importance,
    correlation heatmap, komposisi kategorikal per cluster, boxplot repair cost.
12. **Profil & insight bisnis per cluster** + ekspor hasil ke `outputs/`.

## Catatan Jujur soal Skor Evaluasi
Dataset `defects_data.csv` bersifat sintetis dengan atribut kategorikal yang hampir
independen satu sama lain, sehingga struktur cluster alami memang tidak setegas data
nyata yang punya korelasi kuat. Silhouette Score terbaik yang didapat (~0.20) sudah
merupakan hasil terbaik yang valid setelah feature selection & tuning — memaksakan
angka mendekati 1 hanya bisa dicapai dengan mereduksi fitur jadi satu kolom kategorikal
saja (yang artinya cluster = menyalin ulang kategori asli, bukan clustering yang
bermakna). Penjelasan ini juga sudah dituliskan langsung di dalam notebook supaya bisa
dipakai untuk menjawab pertanyaan dosen saat presentasi/deployment Streamlit.

## Next Step (sesuai soal UAS)
Hasil `outputs/defects_data_with_clusters.csv` dari project ini bisa langsung dipakai
sebagai input untuk membangun aplikasi **Streamlit** (load CSV → tampilkan tabel &
visualisasi cluster → tampilkan interpretasi/insight bisnis) sesuai requirement
deployment pada soal UAS.
