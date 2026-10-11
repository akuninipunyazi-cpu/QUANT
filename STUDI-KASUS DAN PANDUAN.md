### **Predictive Volatility & Explainable AI: Mengawal Dana Triliunan PT Investion di Tengah Turbulensi Ekonomi Global**

### Pada kompetisi ini, teman-teman akan berperan sebagai **Senior Quantitative Risk & Machine Learning Analyst** di **PT Investion** yang merupakan salah satu perusahaan pengelola aset terkemuka di Indonesia. Teman-teman mendapatkan kewajiban krusial yaitu untuk mengawal portofolio dana kelolaan bernilai triliunan rupiah yang sangat terpapar pada dinamika pergerakan Indeks Harga Saham Gabungan (IHSG).

### Realitasnya, pasar modal domestik tidak pernah kebal dari gejolak perekonomian global. Mulai dari lonjakan indeks kepanikan global (VIX), fluktuasi tajam harga komoditas strategis seperti minyak dan emas, tekanan nilai tukar USD/IDR, hingga adanya perbedaan kebijakan suku bunga antara The Fed dan Bank Indonesia. Berbagai faktor eksternal ini kerap memicu efek domino atau disebut fenomena *volatility clustering*, di mana guncangan likuiditas dapat merambat seketika dan berisiko menggerus nilai portofolio PT Investion hanya dalam hitungan hari.

### Dalam kondisi tersebut, pendekatan pengelolaan risiko yang hanya mengandalkan analisis tren historis atau data masa lalu menjadi kurang memadai. Maka dari itu PT invstion membutuhkan suatu mekanisme yang tidak hanya mampu mengidentifikasi kondisi pasar yang beresiko tetapi juga dapat memberikan indikasi lebih awal terhadap potensi peningkatan volatilitas. Oleh karena itu sebagai analis kuantitatif, tugas utama teman-teman bukan sekadar memperkirakan arah tren pasar secara konvensional. Teman-teman dituntut untuk membangun sebuah **Sistem Peringatan Dini (*Early Warning System*)** ditenagai oleh *Machine Learning*, yang dirancang khusus untuk memprediksi tingkat *realized volatility* IHSG untuk 5 hari ke depan.

### Namun, jajaran direksi dan komite risiko PT Investion tidak menoleransi penggunaan algoritma *black-box* dikarenakan dapat menyulitkan pihak manajemen risiko dalam memahami alasan di balik suatu prediksi. Dalam industri finansial yang diregulasi secara ketat, setiap keputusan alokasi modal dan mitigasi risiko menuntut pertanggungjawaban logis yang mutlak. Oleh karena itu, teman-teman ditantang untuk membuktikan integritas teknis pemodelan secara menyeluruh mulai dari perumusan fitur makroekonomi yang inovatif, kedisiplinan menjaga *pipeline* dari kebocoran data masa depan (*anti-leakage*), hingga membedah logika internal model menggunakan prinsip *Explainable AI* (XAI).

### Untuk memastikan proyek ini dieksekusi secara terstruktur dan memenuhi standar, PT Investion mengharapkan teman-teman sebagai analis kuantitatif untuk membuat suatu mekanisme yang dapat mengidentifikasi kondisi pasar yang beresiko serta memberikan indikasi lebih awal atau Early Warning System .

### Mekanisme tersebut diminta agar tidak menggunakan black-box khusus untuk memprediksi tingkat volatilitas IHSG secara real untuk 5 hari ke depan. Namun, membuat sistem yang dapat memberikan integritas teknis pemodelan hingga logika internal dengan pendekatan prinsip Explainable AI (XAI). Maka dari itu manajemen dari PT Investion telah menetapkan kerangka tugas (task) yang wajib teman-teman kerjakan.  **Berikut adalah rincian tugas (task) yang harus diselesaikan oleh teman-teman:**

### **Task 1: Feature Engineering & Exploratory Data Analysis (EDA)**

**Fokus:** Pemahaman domain bisnis/finansial, eksplorasi sinyal data, dan inovasi pembuatan fitur (X) untuk memprediksi volatilitas pasar.

* **Inovasi & Formulasi Fitur (X):** Buatlah minimal 3 fitur turunan baru (X1, X2, X3) berbasis analisis keuangan, statistik, atau makroekonomi yang menurut tim kalian paling berpengaruh untuk memprediksi Volatilitas IHSG 5-Hari Ke Depan (Y).  
* **Rasionalitas Bisnis & Finansial:** Jelaskan ide di balik pembuatan 3 fitur tersebut. Mengapa variabel-variabel tersebut digabungkan/ditransformasi, dan hipotesis atau logika ekonomi apa yang ingin dibuktikan terkait dampaknya terhadap volatilitas IHSG?  
* **Uji Validitas Kuantitatif (EDA):** Tampilkan visualisasi grafik (seperti *heatmap* korelasi, *plot* tren, atau *feature importance chart*) yang membuktikan secara kuantitatif bahwa 3 fitur pilihan tim kalian memang memiliki korelasi atau kontribusi signifikan terhadap tingkat volatilitas target (Y).

### **Task 2: Pemisahan Data Time-Series & Anti-Leakage Pipeline**

**Fokus:** Mengatur pemisahan data urut waktu dan memastikan model tidak mengakses data dari masa depan.

* **Potong Data Sesuai Urutan Tahun:** Karena data ini adalah *time-series*, **DILARANG KERAS** mengacak data (*random train\_test\_split*). Dataset berisi fitur-fitur sesuai dengan yang ada pada Data Dictionary. Untuk data yang dimulai pada 2025 tidak memiliki nilai Y sebagai data uji.  
* **Jaga Kebersihan Data (Anti-Leakage):** Seluruh proses pembersihan data (seperti *StandardScaler*, *MinMaxScaler*, atau pengisian data kosong) hanya boleh dipelajari (*fit*) dari Data *Training* saja, lalu diterapkan (*transform*) ke Data Validasi dan Data Uji.  
* **Bukti Anti-Leakage:** Tunjukkan bukti pada *code* kalian sederhana bahwa fitur X pada hari $t$ tidak menyenggol atau mengambil informasi harga dari hari besok ($t+1$).

### **Task 3: Pemodelan Regresi, Evaluasi, & Explainable AI (XAI)**

**Fokus:** Memilih 1 model andalan tim kalian untuk membuktikan keunggulannya secara matematis dan mengedepankan transparansi serta akuntabilitas keputusan model menggunakan metode XAI.

* **Pilih 1 Model Terbaik Versi Tim Kalian:** Bebas pilih 1 jenis algoritma regresi utama (contoh: XGBoost Regressor, LightGBM, CatBoost, Random Forest, GARCH, atau LSTM) yang menurut tim kalian paling ampuh memprediksi volatilitas. Berikan alasan singkat mengapa algoritma tersebut dipilih.  
* **Hyperparameter Tuning:** Cari kombinasi parameter terbaik untuk model tersebut hanya menggunakan Data Latih & Validasi dari Task 2\.  
* **Uji Akurasi & Baseline (Regresi):**  
  * Tampilkan skor kesalahan prediksi pada **Data Validasi** menggunakan metrik RMSE (*Root Mean Squared Error*) atau MAE (*Mean Absolute Error*).  
  * **Benchmark:** Buktikan bahwa model *Machine Learning* Anda memiliki *error* (RMSE/MAE) yang lebih kecil dibandingkan model *Baseline* sederhana (misalnya: model naif yang hanya menggunakan rata-rata volatilitas 5 hari ke belakang sebagai prediksi 5 hari ke depan).  
* **Transparansi Model (Implementasi XAI):** Wajib menggunakan *library* XAI (seperti SHAP atau LIME) untuk membongkar black box model kalian.  
  * **Penjelasan Global (Global Explanation):** Tampilkan grafik (contohnya *SHAP Summary Plot*) yang membuktikan secara keseluruhan, fitur manakah yang memiliki bobot paling besar dalam menentukan arah prediksi model dari waktu ke waktu.  
  * **Penjelasan Lokal (Local Explanation):** Pilih tanggal spesifik di mana model memprediksi volatilitas sangat tinggi/berbahaya. Tampilkan visualisasi lokal (misalnya *Waterfall Plot* atau *Force Plot*) dan terjemahkan grafiknya ke dalam bahasa bisnis: *Fitur ekonomi apa saja yang memicu lonjakan prediksi pada hari spesifik tersebut?*  
* **Analisis Kesalahan Berbasis XAI:** Identifikasi kondisi pasar di mana model tim kalian sempat meleset jauh dari kenyataan. Alih-alih sekadar menduga, gunakan XAI untuk membedah kesalahan tersebut (contoh: "Model kami gagal memprediksi volatilitas saat lonjakan VIX mendadak karena XAI menunjukkan model terlalu mengandalkan selisih suku bunga").

