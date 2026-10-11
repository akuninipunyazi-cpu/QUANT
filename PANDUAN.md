# PANDUAN PENGEMBANGAN MODEL & EXPLAINABLE AI (XAI)
## Sistem Peringatan Dini Prediksi Realized Volatility IHSG 5-Hari Ke Depan (PT Investion)

Dokumen ini merupakan panduan kerja resmi (*Standard Operating Procedure* dan *Development Blueprint*) dalam merancang, melatih, mengevaluasi, hingga membedah model machine learning untuk memprediksi tingkat *realized volatility* IHSG secara transparan (*Explainable AI*).

---

## DAFTAR ISI
1. [Latar Belakang & Mandat Proyek](#1-latar-belakang--mandat-proyek)
2. [Spesifikasi & Karakteristik Data](#2-spesifikasi--karakteristik-data)
3. [Alur Kerja (Workflow Pipeline)](#3-alur-kerja-workflow-pipeline)
4. [Fase 1: Eksplorasi Data, Korelasi, & Market Regime Clustering](#4-fase-1-eksplorasi-data-korelasi--market-regime-clustering)
5. [Fase 2: Feature Engineering & Rasionalitas Finansial (Task 1)](#5-fase-2-feature-engineering--rasionalitas-finansial-task-1)
6. [Fase 3: Pemisahan Time-Series & Protokol Anti-Leakage (Task 2)](#6-fase-3-pemisahan-time-series--protokol-anti-leakage-task-2)
7. [Fase 4: Pemodelan Regresi, Tuning, & Benchmark (Task 3)](#7-fase-4-pemodelan-regresi-tuning--benchmark-task-3)
8. [Fase 5: Explainable AI (XAI) & Error Analysis (Task 3)](#8-fase-5-explainable-ai-xai--error-analysis-task-3)
9. [Fase 6: Prediksi Out-of-Sample (2025–2026) & Deliverables](#9-fase-6-prediksi-out-of-sample-20252026--deliverables)
10. [Checklist Kualitas & Verifikasi Sebelum Finalisasi](#10-checklist-kualitas--verifikasi-sebelum-finalisasi)

---

## 1. Latar Belakang & Mandat Proyek

- **Peran:** Senior Quantitative Risk & Machine Learning Analyst di PT Investion.
- **Portofolio:** Dana kelolaan triliunan rupiah yang terpapar volatilitas IHSG dan guncangan makro global.
- **Tujuan Utama:** Membangun *Early Warning System* untuk memprediksi *Realized Volatility* IHSG 5 hari ke depan (`Target_Volatilitas_Y`).
- **Batasan Krusial:** 
  - Tidak menoleransi algoritma *black-box* murni tanpa penjelasan.
  - Wajib membuktikan integritas matematis melalui **Explainable AI (XAI)** menggunakan SHAP/LIME.
  - Bebas dari kebocoran data masa depan (*anti data leakage*).

---

## 2. Spesifikasi & Karakteristik Data

Dataset: `DATASET.csv` (Total 4.266 baris, 11 kolom, rentang waktu: 4 Januari 2010 – 22 Mei 2026).

| Kolom | Tipe | Peran | Keterangan & Satuan |
| :--- | :--- | :--- | :--- |
| `Date` | Datetime | Indeks Waktu | Hari kerja Senin–Jumat (2010–2026) |
| `OIL` | Float | Fitur Makro | Crude Oil Price (USD / Barrel) |
| `GOLD` | Float | Fitur Makro | Gold Spot Price (USD / Troy Ounce) |
| `USDIDR` | Float | Fitur FX | Kurs Rupiah terhadap US Dollar (IDR / USD) |
| `SP500` | Float | Fitur Pasar Global | Indeks S&P 500 (Poin) |
| `IHSG` | Float | Fitur Pasar Domestik | Indeks Harga Saham Gabungan (Poin) |
| `VIX` | Float | Fitur Sentimen Global | CBOE Volatility Index ("Fear Index") |
| `CPI` | Float | Fitur Makro | Consumer Price Index Indonesia (Indeks Inflasi) |
| `BI_rate` | Float | Fitur Kebijakan Moneter | Suku Bunga Acuan BI (% per tahun) |
| `US_rate` | Float | Fitur Kebijakan Moneter | Federal Funds Rate (% per tahun) |
| `Target_Volatilitas_Y` | Float | **Target Prediksi** | Realized Volatility IHSG 5 hari ke depan. *(Catatan: Baris data mulai 2025 bernilai NaN sebagai Test Set).* |

---

## 3. Alur Kerja (Workflow Pipeline)

```
[DATASET.csv]
      │
      ▼
[Fase 1: Korelasi Multivariat & Market Regime Clustering]
      │
      ▼
[Fase 2: Feature Engineering (X1, X2, X3) + Rasionalitas Finansial]
      │
      ▼
[Fase 3: Strict Time-Series Split & Anti-Leakage Data Pipeline]
      │
      ▼
[Fase 4: Baseline Benchmark & Model Regresi (LightGBM) + Tuning]
      │
      ▼
[Fase 5: Explainable AI (SHAP Global, Local Waterfall, Error Analysis)]
      │
      ▼
[Fase 6: Prediksi Out-of-Sample 2025–2026 & Dokumentasi Eksekutif]
```

---

## 4. Fase 1: Eksplorasi Data, Korelasi, & Market Regime Clustering

### 4.1. Analisis Korelasi Multivariat
1. **Analisis Korelasi Linear (Pearson) & Monotonik (Spearman):**
   - Menghitung korelasi seluruh variabel terhadap `Target_Volatilitas_Y`.
   - Mengidentifikasi variabel yang memiliki korelasi positif tertinggi (misal: `VIX`, pergerakan `USDIDR`).
   - Mengidentifikasi variabel dengan korelasi terendah/negatif (faktor stabilisator atau divergensi).
2. **Lagged Cross-Correlation:**
   - Menguji apakah perubahan variabel makro pada hari $t, t-1, t-5$ memberikan sinyal awal (*leading indicator*) terhadap volatilitas IHSG di masa mendatang.

### 4.2. Market Regime / Event Clustering
1. **Tujuan:** Mengelompokkan kondisi makroekonomi menjadi beberapa rezim pasar spesifik untuk melihat apakah volatilitas IHSG terdistribusi berbeda pada setiap kondisi.
2. **Indikator Input Clustering:**
   - *Spread* suku bunga (`BI_rate - US_rate`).
   - Volatilitas/Level `VIX`.
   - Perubahan kurs Rupiah (`USDIDR` return/tekanan devaluasi).
   - Dinamika komoditas (`OIL` & `GOLD`).
3. **Metode Clustering:**
   - Standarisasi fitur dengan `StandardScaler`.
   - Algoritma: **K-Means Clustering** atau **Gaussian Mixture Models (GMM)** (3–4 klaster).
   - Analisis nama rezim:
     - **Rezim 1 (Global Panic & High Shock):** VIX tinggi, Rupiah melemah tajam $\rightarrow$ Volatilitas IHSG melonjak tinggi.
     - **Rezim 2 (Monetary Tightening / Capital Outflow):** US Rate mendekati/melampaui BI Rate, likuiditas global ketat $\rightarrow$ Volatilitas sedang ke tinggi.
     - **Rezim 3 (Commodity Boom & Stable FX):** Minyak/emas menguat, Rupiah stabil $\rightarrow$ Volatilitas rendah ke moderat.
     - **Rezim 4 (Calm Expansion / Bullish):** VIX rendah, IHSG naik stabil $\rightarrow$ Volatilitas rendah.
4. **Output:** Profil klaster dan bukti empiris bahwa klaster event tertentu secara konsisten memicu lonjakan volatilitas.

---

## 5. Fase 2: Feature Engineering & Rasionalitas Finansial (Task 1)

Wajib merumuskan **minimal 3 fitur turunan baru** berbasis domain finansial dan statistik, disertai logika ekonomi yang solid.

### 5.1. Formulasi Fitur Rekomendasi
1. **Fitur 1 ($X_1$): Interest Rate Spread (Selisih Suku Bunga Domestik vs AS)**
   - **Formula:** $X_1 = \text{BI\_rate}_t - \text{US\_rate}_t$
   - **Rasionalitas Finansial:** Mengacu pada teori *Uncovered Interest Rate Parity* (UIP). Ketika selisih bunga menyempit, arus modal asing (*capital outflow*) cenderung keluar dari pasar modal berkembang (Indonesia) kembali ke aset berdenominasi USD. Tekanan likuiditas ini memicu kepanikan dan menaikkan volatilitas IHSG.
2. **Fitur 2 ($X_2$): FX Shock & Global Contagion Index (Tekanan Kurs x Sentimen VIX)**
   - **Formula:** $X_2 = \text{Pct\_Change}(\text{USDIDR}, 5)_t \times \text{VIX}_t$
   - **Rasionalitas Finansial:** Menggabungkan pelemahan Rupiah dalam 5 hari terakhir dengan tingkat kepanikan pasar global. Lonjakan simultan pada kedua indikator ini mengonfirmasi terjadinya guncangan ganda (*twin-shock*) yang secara historis menjadi pemicu utama lonjakan volatilitas di bursa domestik.
3. **Fitur 3 ($X_3$): Historical Realized Volatility Proxy (Volatility Persistence)**
   - **Formula:** $X_3 = \text{StdDev}(\text{Daily Return IHSG}, \text{window}=5)_t$ (dihitung strictly sampai hari $t$)
   - **Rasionalitas Finansial:** Menguji fenomena *volatility clustering* (Mandelbrot & Engle): volatilitas tinggi cenderung diikuti oleh volatilitas tinggi berikutnya. Tingkat fluktuasi IHSG 5 hari ke belakang merupakan jangkar informasi dasar bagi ekspektasi risiko 5 hari ke depan.
4. **Fitur Tambahan (Opsional):**
   - *Commodity Terms of Trade Ratio:* $\text{OIL}_t / \text{GOLD}_t$.
   - *Market Regime Cluster Membership:* Jarak ke centroid klaster krisis atau label klaster (One-Hot).

### 5.2. Uji Validitas Kuantitatif (Visualisasi EDA)
- **Heatmap Korelasi:** Menunjukkan korelasi Spearman & Pearson fitur $X_1, X_2, X_3$ terhadap `Target_Volatilitas_Y`.
- **Trend & Scatter Plot:** Menampilkan sebaran data dan garis tren non-linear antara fitur turunan dan volatilitas target.

---

## 6. Fase 3: Pemisahan Time-Series & Protokol Anti-Leakage (Task 2)

### 6.1. Skema Pemotongan Data (Strict Chronological Split)
Data finansial memiliki dependensi temporal (*serial correlation*), sehingga **dilarang keras** menggunakan *random train-test split*.

- **Training Set:** 2010-01-04 s/d 2022-12-31 (~13 tahun observasi pasar penuh, mencakup krisis Eropa, Taper Tantrum, dan Covid-19).
- **Validation Set:** 2023-01-01 s/d 2024-12-31 (~2 tahun data terkini dengan target Y untuk evaluasi out-of-time & hyperparameter tuning).
- **Test Set (Out-of-Sample):** 2025-01-01 s/d 2026-05-22 (359 baris tanpa label Y, digunakan murni untuk evaluasi prediksi panitia).

### 6.2. Protokol Pencegahan Kebocoran Data (Anti-Leakage)
1. **Fit Preprocessing Hanya di Data Latih:**
   ```python
   # WAJIB:
   scaler.fit(X_train)
   X_train_scaled = scaler.transform(X_train)
   X_val_scaled   = scaler.transform(X_val)
   X_test_scaled  = scaler.transform(X_test)
   # DILARANG: scaler.fit(X_full)
   ```
2. **Integritas Waktu Fitur ($t \le \text{hari ini}$):**
   - Seluruh rolling window, return, dan diferensiasi hanya menggunakan data historis sampai hari $t$.
   - Tidak ada operasi `shift(-k)` pada fitur prediktor $X$.
3. **Verifikasi Kode Anti-Leakage:**
   - Menyiapkan unit check / assertion untuk membuktikan bahwa indeks waktu fitur tidak mendahului indeks target.

---

## 7. Fase 4: Pemodelan Regresi, Tuning, & Benchmark (Task 3)

### 7.1. Model Baseline (Benchmark Naif)
- Model *Naive Persistence*: Menggunakan rata-rata volatilitas historis 5 hari ke belakang sebagai estimasi untuk 5 hari ke depan.
- Menghitung **RMSE** dan **MAE** baseline pada Validation Set sebagai batas bawah (*hurdle rate*) yang wajib dikalahkan oleh model Machine Learning.

### 7.2. Pemilihan Model Utama: LightGBM Regressor
**Alasan Pemilihan Teknis & Finansial:**
- **Kapasitas Non-Linear Finansial:** Mampu menangkap hubungan asimetris dan non-linear antara variabel makroekonomi (misal hubungan non-linear VIX saat normal vs krisis).
- **Efisiensi & Ketahanan:** Algoritma berbasis histogram (*leaf-wise tree growth*) sangat cepat, tidak rentan terhadap multikolinearitas moderat, dan tahan terhadap outlier harga ekstrem.
- **Dukungan XAI Optimal:** Terintegrasi langsung secara matematis dengan `shap.TreeExplainer`, memungkinkan komputasi *exact Shapley values* secara efisien dan konsisten.

### 7.3. Hyperparameter Tuning
- Menggunakan `TimeSeriesSplit` (Expanding Window Cross-Validation) pada Data Latih.
- Parameter kunci yang di-tune: `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `colsample_bytree`, `subsample`.
- Evaluasi akhir model terbaik pada **Validation Set (2023–2024)** menggunakan metrik **RMSE** dan **MAE**.

---

## 8. Fase 5: Explainable AI (XAI) & Error Analysis (Task 3)

Implementasi menggunakan library `shap` (SHapley Additive exPlanations).

### 8.1. Global Explanation
- **SHAP Summary Plot (Beeswarm):**
  - Mengurutkan fitur dari pengaruh terbesar ke terkecil terhadap prediksi volatilitas.
  - Membedah arah dampak: Apakah nilai fitur tinggi menaikkan atau menurunkan estimasi volatilitas?
- **SHAP Feature Importance (Bar Plot):**
  - Mengukur kontribusi absolut rata-rata setiap variabel makro dan fitur turunan.

### 8.2. Local Explanation (Analisis Hari Ekstrem / Krisis)
- Memilih 1–2 tanggal spesifik di Validation Set saat terjadi lonjakan volatilitas ekstrem.
- Membuat visualisasi **SHAP Waterfall Plot** atau **Force Plot**.
- **Terjemahan Narasi Bisnis:** Menjelaskan secara gamblang kepada dewan direksi PT Investion mengenai kontribusi masing-masing indikator pada hari tersebut (contoh: "Pada tanggal X, prediksi volatilitas melonjak ke level 0.024 terutama didorong oleh kontribusi positif VIX sebesar +0.008 dan penyempitan spread suku bunga sebesar +0.005").

### 8.3. Analisis Kesalahan Berbasis XAI (XAI-Guided Error Analysis)
- Mengidentifikasi tanggal observasi dengan residual kesalahan terbesar ($|Y_{\text{actual}} - \hat{Y}_{\text{pred}}|$).
- Menjalankan analisis SHAP pada tanggal tersebut:
  - Mengapa model *under-estimate* atau *over-estimate*?
  - Apakah model terlalu sensitif terhadap pergerakan sesaat pada salah satu variabel (misal VIX spike yang bersifat sementara) atau lambat merespons perubahan suku bunga?
  - Menghubungkan analisis kesalahan ini dengan rekomendasi mitigasi risiko untuk tim manajemen investasi.

---

## 9. Fase 6: Prediksi Out-of-Sample (2025–2026) & Deliverables

1. **Prediksi Data Uji:**
   - Menghasilkan prediksi `Target_Volatilitas_Y` untuk baris data tahun 2025 s/d 2026.
   - Menyimpan hasil prediksi dalam format file yang rapi (CSV).
2. **Paket Deliverables Final:**
   - Script/Notebook eksekusi terstruktur end-to-end.
   - Gambar visualisasi berkualitas tinggi (Korelasi, Clustering, Baseline Comparison, SHAP Summary, SHAP Waterfall, Error Analysis).
   - Laporan ringkas eksekutif berstandar kuantitatif institusional.

---

## 10. Checklist Kualitas & Verifikasi Sebelum Finalisasi

- [ ] Matriks korelasi dan analisis klastering event selesai dijalankan dan terdokumentasi maknanya.
- [ ] Minimal 3 fitur turunan baru diformulasikan dengan landasan teori ekonomi yang jelas.
- [ ] Data split murni kronologis (Train: s/d 2022, Val: 2023–2024, Test: 2025–2026).
- [ ] Tidak ada fungsi scaler atau pengisian missing value yang di-fit di luar data training.
- [ ] Model Machine Learning terbukti secara matematis mengalahkan baseline naif (RMSE & MAE lebih rendah).
- [ ] SHAP Global Summary Plot tersedia dan dijelaskan maknanya.
- [ ] SHAP Local Waterfall Plot untuk kasus ekstrem disajikan dengan narasi bisnis kuantitatif.
- [ ] Kasus kesalahan model (residual terbesar) dibedah tuntas menggunakan XAI.
- [ ] Prediksi data uji 2025–2026 telah terisi lengkap tanpa missing values.
