# ShopPredict — Prediksi Niat Pembelian Online (ML v2.0)

Aplikasi web berbasis Flask yang menggunakan **dual model machine learning** (Random Forest + Logistic Regression) untuk memprediksi niat pembelian pelanggan berdasarkan data interaksi pengguna di halaman web.

## Deskripsi

Aplikasi ini menganalisis perilaku pengunjung website untuk memprediksi kemungkinan mereka melakukan pembelian. Menggunakan dataset 12.330 sesi pengunjung dari [UCI ML Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).

### Upgrade v2.0

| Fitur | v1.0 | v2.0 |
|-------|------|------|
| Model | Random Forest saja | Random Forest (25 fitur) + Logistic Regression (8 fitur) |
| Prediksi Manual | 7 parameter | 6 parameter (LR-only) |
| Prediksi CSV | RF saja | RF / LR / Komparasi keduanya |
| Output | Probabilitas | Tingkat Keyakinan Model + Progress Bar |
| Test | 52 unit test | 73 unit + 16 E2E (Playwright) |
| Coverage | — | 100% (226/226 statements) |

## Fitur Utama

- **Prediksi Manual (LR)** — Input 6 parameter melalui form, hasil prediksi + tingkat keyakinan model dengan progress bar visual
- **Prediksi Batch (CSV)** — Upload CSV 17 kolom, pilih model RF / LR / komparasi keduanya
- **Tingkat Keyakinan Model** — Visualisasi confidence model berupa progress bar horizontal dengan kategori (Sangat Tinggi / Tinggi / Sedang / Rendah)
- **Download Hasil CSV** — Unduh hasil prediksi batch dengan kolom Prediksi + Kepercayaan per model
- **Validasi Form (WTForms)** — Validasi server-side untuk semua form input
- **Dark Mode UI** — Interface modern dengan Tailwind CSS, animasi fade-in/slide-up
- **100% Test Coverage** — 73 unit test + 16 E2E test (Playwright), 226 statements tercakup

## Teknologi

| Komponen | Teknologi |
|----------|-----------|
| Backend | Flask 3.1.3 |
| Form & Validasi | Flask-WTF 1.3.0, WTForms 3.2.1 |
| Machine Learning | scikit-learn 1.8.0 (Random Forest + Logistic Regression) |
| Data Processing | pandas 3.0.2, numpy 2.4.4, joblib 1.5.3 |
| Frontend | HTML5, Tailwind CSS (CDN), Font Awesome 6.5 |
| Font | Inter (Google Fonts) |
| Unit Testing | pytest 9.0.3, coverage 7.13.5 |
| E2E Testing | Playwright (Chromium headless) |
| Formatter | black 26.3.1 |

## Parameter Input

### Prediksi Manual (Logistic Regression — 6 field)

| Parameter | Tipe | Rentang | Deskripsi |
|-----------|------|---------|-----------|
| Page Values | Numerik | >= 0 | Nilai rata-rata halaman sebelum transaksi |
| Exit Rates | Numerik | 0 - 1 | Rasio halaman terakhir sebelum keluar |
| Durasi Produk | Numerik | >= 0 | Durasi di halaman produk (detik) |
| Bulan | Dropdown | Feb - Des | Bulan kunjungan |
| Browser | Dropdown | 1 - 13 | Kode browser pengguna |
| Sumber Traffic | Dropdown | 1 - 20 | Kode sumber traffic |

> 2 fitur tambahan (VisitorType, Weekend) di-set default oleh server.

### Prediksi CSV (17 kolom wajib)

```
Administrative, Administrative_Duration, Informational, Informational_Duration,
ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues,
SpecialDay, Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend
```

## Instalasi

### Prasyarat

- Python 3.10+
- pip

### Langkah Instalasi

```bash
# 1. Clone repository
git clone https://github.com/Packooo/shopper-app.git
cd shopper-app

# 2. Buat virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan aplikasi
python app.py
```

Buka browser: **http://localhost:5001**

### File Model

Aplikasi membutuhkan 3 file model di direktori `model/`:

| File | Deskripsi |
|------|-----------|
| `model/model_tuned_rf.pkl` | Random Forest Classifier (25 fitur) |
| `model/model_tuned_lre.pkl` | Logistic Regression (8 fitur) |
| `model/scaler_full.pkl` | MinMaxScaler untuk 10 kolom numerik |

## Struktur Proyek

```
shopper-app/
├── app.py                 # Aplikasi Flask utama + endpoint + helper kepercayaan
├── forms.py               # Definisi form WTForms (manual + CSV upload)
├── preprocessing.py       # Pipeline preprocessing: scaling + OHE + seleksi fitur
├── test_app.py            # Unit test (73 test, 100% coverage)
├── test_e2e.py            # E2E test Playwright (16 test, Chromium headless)
├── test_data.csv          # Data CSV untuk testing (5 baris, 17 kolom)
├── requirements.txt       # Dependencies Python (pinned versions)
├── model/
│   ├── model_tuned_rf.pkl     # Model Random Forest
│   ├── model_tuned_lre.pkl    # Model Logistic Regression
│   └── scaler_full.pkl        # MinMaxScaler terlatih
├── templates/
│   └── index.html         # Template HTML (Tailwind CSS + dark mode)
└── README.md
```

## Cara Penggunaan

### 1. Prediksi Manual

1. Buka aplikasi di browser
2. Isi form **Prediksi Manual** (3 input numerik + 3 dropdown)
3. Klik **"Prediksi Sekarang"**
4. Lihat hasil:
   - **Prediksi**: "Akan Membeli" atau "Tidak Membeli"
   - **Tingkat Keyakinan Model**: persentase + progress bar + badge kategori

### 2. Prediksi via Upload CSV

1. Siapkan file CSV dengan 17 kolom sesuai format:
   ```csv
   Administrative,Administrative_Duration,Informational,Informational_Duration,ProductRelated,ProductRelated_Duration,BounceRates,ExitRates,PageValues,SpecialDay,Month,OperatingSystems,Browser,Region,TrafficType,VisitorType,Weekend
   0,0.0,0,0.0,1,0.0,0.0,0.02,0.0,0.0,Feb,1,1,1,1,Returning_Visitor,False
   ```
2. Pilih model: **Random Forest** / **Logistic Regression** / **Komparasi Keduanya**
3. Klik **"Upload dan Prediksi"**
4. Hasil ditampilkan:
   - Ringkasan statistik (Total, Beli, Tidak Beli per model)
   - Tabel data lengkap dengan kolom Prediksi + Kepercayaan
   - Kolom Kepercayaan di-warnai: hijau (Tinggi), kuning (Sedang), merah (Rendah)
5. Klik **"Download CSV"** untuk mengunduh hasil

## Output Prediksi

### Prediksi Manual

| Output | Deskripsi |
|--------|-----------|
| Prediksi | "Akan Membeli" atau "Tidak Membeli" |
| Tingkat Keyakinan Model | Persentase seberapa yakin model (0 - 100%) |
| Progress Bar | Visualisasi horizontal keyakinan model |
| Badge Kategori | Sangat Tinggi (>=90%) / Tinggi (>=75%) / Sedang (>=60%) / Rendah (<60%) |

### Prediksi CSV

| Kolom Output | Deskripsi |
|--------------|-----------|
| RF_Prediksi | Prediksi Random Forest: "Akan Membeli" / "Tidak Membeli" |
| RF_Kepercayaan | Tingkat kepercayaan RF: "99.64% (Sangat Tinggi)" |
| LR_Prediksi | Prediksi Logistic Regression (jika dipilih) |
| LR_Kepercayaan | Tingkat kepercayaan LR (jika dipilih) |

## Testing

### Unit Test

```bash
# Jalankan semua unit test
python -m pytest test_app.py -v

# Jalankan dengan coverage report
python -m coverage run --source=app,forms,preprocessing -m pytest test_app.py
python -m coverage report -m
```

**Hasil:** 73 test passed, 100% coverage (226/226 statements: app.py 158, forms.py 16, preprocessing.py 52)

### E2E Test (Playwright)

```bash
# Install Playwright (sekali saja)
pip install playwright && playwright install chromium

# Jalankan E2E test
python -m pytest test_e2e.py -v
```

**Hasil:** 16 test passed (halaman utama 7, prediksi manual 3, upload CSV 5, error handling 1)

> E2E test menjalankan Flask server otomatis di port 5002 dan menggunakan Chromium headless.

### Jalankan Semua Test

```bash
python -m pytest test_app.py test_e2e.py -v
```

## Linting & Formatting

```bash
# Format kode dengan black
black app.py forms.py preprocessing.py test_app.py test_e2e.py

# Cek tanpa mengubah file
black --check app.py forms.py preprocessing.py test_app.py test_e2e.py
```

## Arsitektur Kode

### `app.py` — Aplikasi Flask Utama

| Fungsi / Route | Deskripsi |
|----------------|-----------|
| `muat_model()` | Memuat RF, LR, dan scaler dari `model/` |
| `buat_aplikasi()` | Factory function Flask (secret key, CSRF off, 5MB limit) |
| `hitung_kepercayaan()` | Menghitung confidence: `prob` jika beli, `1-prob` jika tidak |
| `label_kepercayaan()` | Mengkategorikan confidence: Sangat Tinggi / Tinggi / Sedang / Rendah |
| `model_siap()` | Cek apakah semua model tersedia |
| `prediksi_model()` | Prediksi via model: seleksi fitur → predict_proba → threshold |
| `GET /` | Halaman utama (form manual + form CSV) |
| `POST /predict` | Prediksi manual (LR only, 8 fitur) |
| `POST /upload` | Prediksi batch CSV (RF / LR / both) |
| `GET /download-csv` | Download file CSV hasil prediksi |

### `preprocessing.py` — Pipeline Preprocessing

| Fungsi | Deskripsi |
|--------|-----------|
| `preprocess()` | Pipeline: scaling 10 kolom numerik + OHE 6 kolom kategorikal |
| `encode_one_hot()` | One-Hot Encoding dengan drop_first (57 kolom OHE) |
| `select_features()` | Seleksi fitur: RF (25 fitur) atau LR (8 fitur) |

### `forms.py` — Definisi Form WTForms

| Class | Deskripsi |
|-------|-----------|
| `FormPrediksiManual` | 3 numerik + 3 dropdown untuk prediksi LR manual |
| `FormUploadCSV` | File upload CSV + radio button pilihan model (RF/LR/both) |

## Pipeline Preprocessing

```
CSV Input (17 kolom mentah)
    │
    ▼
┌─────────────────────────┐
│ 1. Scaling Numerik      │  MinMaxScaler pada 10 kolom
│    (Administrative,     │  (dilatih pada dataset UCI asli)
│     ExitRates, dll)     │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│ 2. One-Hot Encoding     │  6 kolom kategorikal → 57 kolom binary
│    (Month, Browser,     │  drop_first=True (referensi: Aug, OS_1,
│     TrafficType, dll)   │   Browser_1, Region_1, Traffic_1, New_Visitor)
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│ 3. Seleksi Fitur        │  RF: 25 fitur | LR: 8 fitur
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│ 4. Prediksi + Confidence│  predict_proba → threshold → kepercayaan
└─────────────────────────┘
```

## Dataset

| Atribut | Detail |
|---------|--------|
| Nama | Online Shoppers Purchasing Intention Dataset |
| Sumber | [UCI ML Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) |
| Jumlah Data | 12.330 sesi pengunjung |
| Kolom | 17 fitur + 1 label (Revenue) |
| Referensi | Sakar, C.O., Polat, S.O., Katircioglu, M. et al. (2019) |

## Lisensi

Lihat repository asli untuk informasi lisensi.
