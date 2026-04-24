"""
Modul Preprocessing Data untuk Prediksi Niat Beli Online
=========================================================
Modul ini menangani seluruh pipeline preprocessing data sebelum
data dikirim ke model machine learning untuk prediksi.

Pipeline preprocessing:
1. Scaling numerik (MinMaxScaler) pada 10 kolom numerik
2. One-Hot Encoding (OHE) pada kolom kategorikal
3. Seleksi fitur sesuai model yang dipilih (RF 25 fitur / LR 8 fitur)

Catatan penting:
- Scaler dilatih pada dataset asli UCI Online Shoppers Intention
- OHE menggunakan drop_first=True (kategori pertama dijadikan referensi)
- Kolom referensi yang di-drop: Month_Aug, OS_1, Browser_1, Region_1,
  TrafficType_1, VisitorType_New_Visitor
"""

import pandas as pd
import joblib

# ============================================================
# Muat scaler yang sudah dilatih pada dataset original
# Scaler ini adalah MinMaxScaler yang mentransformasi 10 kolom
# numerik ke rentang [0, 1] berdasarkan min/max dataset asli
# ============================================================
_scaler = joblib.load("model/scaler_full.pkl")

# ============================================================
# Daftar 10 kolom numerik yang harus di-scale
# Kolom-kolom ini merepresentasikan perilaku browsing pengguna
# dan metrik halaman web yang dikunjungi
# ============================================================
SCALE_COLS = [
    "Administrative",  # Jumlah halaman admin yang dikunjungi
    "Administrative_Duration",  # Total durasi di halaman admin (detik)
    "Informational",  # Jumlah halaman informasi yang dikunjungi
    "Informational_Duration",  # Total durasi di halaman informasi (detik)
    "ProductRelated",  # Jumlah halaman produk yang dikunjungi
    "ProductRelated_Duration",  # Total durasi di halaman produk (detik)
    "BounceRates",  # Rasio pengunjung yang langsung keluar (0-1)
    "ExitRates",  # Rasio halaman terakhir sebelum keluar (0-1)
    "PageValues",  # Nilai rata-rata halaman sebelum transaksi
    "SpecialDay",  # Kedekatan waktu kunjungan dengan hari spesial (0-1)
]

# ============================================================
# 25 fitur terpilih untuk model Random Forest
# Fitur dipilih berdasarkan feature importance dari training
# Urutan sesuai ranking importance (PageValues paling penting)
# ============================================================
RF_FEATURES = [
    "PageValues",  # Fitur #1 — nilai halaman (paling berpengaruh)
    "ExitRates",  # Fitur #2 — rasio keluar
    "Administrative",  # Fitur #3 — jumlah halaman admin
    "ProductRelated_Duration",  # Fitur #4 — durasi di halaman produk
    "Administrative_Duration",  # Fitur #5 — durasi di halaman admin
    "ProductRelated",  # Fitur #6 — jumlah halaman produk
    "BounceRates",  # Fitur #7 — rasio bounce
    "Informational",  # Fitur #8 — jumlah halaman info
    "Month_Nov",  # Fitur #9 — apakah bulan November
    "Informational_Duration",  # Fitur #10 — durasi di halaman info
    "TrafficType_2",  # Fitur #11 — sumber traffic tipe 2
    "Month_May",  # Fitur #12 — apakah bulan Mei
    "OperatingSystems_2",  # Fitur #13 — OS tipe 2
    "Browser_2",  # Fitur #14 — browser tipe 2
    "Weekend",  # Fitur #15 — kunjungan di akhir pekan
    "VisitorType_Returning_Visitor",  # Fitur #16 — pengunjung kembali
    "Region_3",  # Fitur #17 — region 3
    "OperatingSystems_3",  # Fitur #18 — OS tipe 3
    "Month_Sep",  # Fitur #19 — apakah bulan September
    "SpecialDay",  # Fitur #20 — hari spesial
    "Region_2",  # Fitur #21 — region 2
    "Month_Mar",  # Fitur #22 — apakah bulan Maret
    "TrafficType_3",  # Fitur #23 — sumber traffic tipe 3
    "Month_Dec",  # Fitur #24 — apakah bulan Desember
    "Region_4",  # Fitur #25 — region 4
]

# ============================================================
# 8 fitur terpilih untuk model Logistic Regression
# Fitur dipilih berdasarkan koefisien regresi tertinggi
# Model ini lebih sederhana dan cocok untuk input manual
# ============================================================
LR_FEATURES = [
    "PageValues",  # Koefisien positif terbesar — indikator pembelian
    "ExitRates",  # Koefisien negatif — exit tinggi = tidak beli
    "TrafficType_15",  # Sumber traffic tipe 15
    "Browser_12",  # Browser tipe 12
    "TrafficType_16",  # Sumber traffic tipe 16
    "Month_Nov",  # Bulan November (musim belanja)
    "Browser_3",  # Browser tipe 3
    "ProductRelated_Duration",  # Durasi melihat produk — semakin lama semakin tertarik
]

# ============================================================
# 17 kolom fitur mentah (raw) dari dataset original
# Ini adalah kolom yang harus ada di CSV input sebelum
# dilakukan preprocessing (scaling + encoding)
# ============================================================
RAW_FEATURE_COLS = [
    "Administrative",  # int — jumlah halaman admin
    "Administrative_Duration",  # float — durasi di halaman admin
    "Informational",  # int — jumlah halaman info
    "Informational_Duration",  # float — durasi di halaman info
    "ProductRelated",  # int — jumlah halaman produk
    "ProductRelated_Duration",  # float — durasi di halaman produk
    "BounceRates",  # float — rasio bounce (0-1)
    "ExitRates",  # float — rasio exit (0-1)
    "PageValues",  # float — nilai halaman
    "SpecialDay",  # float — kedekatan hari spesial (0-1)
    "Month",  # str — nama bulan (Feb, Mar, May, dll)
    "OperatingSystems",  # int — kode sistem operasi (1-8)
    "Browser",  # int — kode browser (1-13)
    "Region",  # int — kode region (1-9)
    "TrafficType",  # int — kode sumber traffic (1-20)
    "VisitorType",  # str — tipe pengunjung (Returning/New/Other)
    "Weekend",  # bool — kunjungan di akhir pekan
]

# ============================================================
# Semua kolom hasil One-Hot Encoding (57 kolom)
# Dibuat dari 6 kolom kategorikal dengan drop_first=True:
# - Month: 9 kolom (drop Aug sebagai referensi)
# - OperatingSystems: 7 kolom (drop OS_1)
# - Browser: 12 kolom (drop Browser_1)
# - Region: 8 kolom (drop Region_1)
# - TrafficType: 19 kolom (drop TrafficType_1)
# - VisitorType: 2 kolom (drop New_Visitor)
# ============================================================
ALL_OHE_COLUMNS = [
    # === Bulan (referensi: August) ===
    "Month_Dec",
    "Month_Feb",
    "Month_Jul",
    "Month_June",
    "Month_Mar",
    "Month_May",
    "Month_Nov",
    "Month_Oct",
    "Month_Sep",
    # === Sistem Operasi (referensi: OS 1) ===
    "OperatingSystems_2",
    "OperatingSystems_3",
    "OperatingSystems_4",
    "OperatingSystems_5",
    "OperatingSystems_6",
    "OperatingSystems_7",
    "OperatingSystems_8",
    # === Browser (referensi: Browser 1) ===
    "Browser_2",
    "Browser_3",
    "Browser_4",
    "Browser_5",
    "Browser_6",
    "Browser_7",
    "Browser_8",
    "Browser_9",
    "Browser_10",
    "Browser_11",
    "Browser_12",
    "Browser_13",
    # === Region (referensi: Region 1) ===
    "Region_2",
    "Region_3",
    "Region_4",
    "Region_5",
    "Region_6",
    "Region_7",
    "Region_8",
    "Region_9",
    # === Sumber Traffic (referensi: TrafficType 1) ===
    "TrafficType_2",
    "TrafficType_3",
    "TrafficType_4",
    "TrafficType_5",
    "TrafficType_6",
    "TrafficType_7",
    "TrafficType_8",
    "TrafficType_9",
    "TrafficType_10",
    "TrafficType_11",
    "TrafficType_12",
    "TrafficType_13",
    "TrafficType_14",
    "TrafficType_15",
    "TrafficType_16",
    "TrafficType_17",
    "TrafficType_18",
    "TrafficType_19",
    "TrafficType_20",
    # === Tipe Pengunjung (referensi: New_Visitor) ===
    "VisitorType_Other",
    "VisitorType_Returning_Visitor",
]


def encode_one_hot(df):
    """
    Melakukan One-Hot Encoding pada kolom kategorikal.

    Proses:
    1. Konversi kolom Weekend dari boolean ke integer (0/1)
    2. Inisialisasi semua 57 kolom OHE dengan nilai 0
    3. Set nilai 1 pada kolom yang sesuai untuk setiap kategori
    4. Hapus kolom kategorikal asli (sudah diganti OHE)

    Parameter:
        df (DataFrame): DataFrame dengan kolom kategorikal mentah

    Return:
        DataFrame: DataFrame dengan kolom kategorikal sudah di-encode
                   menjadi kolom binary (0/1)
    """
    result = df.copy()

    # Konversi Weekend dari True/False ke 1/0
    if "Weekend" in result.columns:
        result["Weekend"] = result["Weekend"].astype(int)

    # Inisialisasi semua kolom OHE dengan 0 (default = kategori referensi)
    for col in ALL_OHE_COLUMNS:
        result[col] = 0

    # === Encoding Bulan ===
    # Jika bulan = Nov, maka Month_Nov = 1, sisanya tetap 0
    # Bulan Aug tidak punya kolom karena jadi referensi (drop_first)
    if "Month" in result.columns:
        for month in ["Dec", "Feb", "Jul", "June", "Mar", "May", "Nov", "Oct", "Sep"]:
            result[f"Month_{month}"] = (result["Month"] == month).astype(int)

    # === Encoding Sistem Operasi ===
    # OS 1 adalah referensi (tidak punya kolom sendiri)
    # Jika OS = 2, maka OperatingSystems_2 = 1
    if "OperatingSystems" in result.columns:
        os_col = result["OperatingSystems"].astype(int)
        for val in range(2, 9):
            result[f"OperatingSystems_{val}"] = (os_col == val).astype(int)

    # === Encoding Browser ===
    # Browser 1 adalah referensi
    # Jika Browser = 12, maka Browser_12 = 1
    if "Browser" in result.columns:
        br_col = result["Browser"].astype(int)
        for val in range(2, 14):
            result[f"Browser_{val}"] = (br_col == val).astype(int)

    # === Encoding Region ===
    # Region 1 adalah referensi
    if "Region" in result.columns:
        rg_col = result["Region"].astype(int)
        for val in range(2, 10):
            result[f"Region_{val}"] = (rg_col == val).astype(int)

    # === Encoding Sumber Traffic ===
    # TrafficType 1 adalah referensi
    # Jika Traffic = 15, maka TrafficType_15 = 1
    if "TrafficType" in result.columns:
        tt_col = result["TrafficType"].astype(int)
        for val in range(2, 21):
            result[f"TrafficType_{val}"] = (tt_col == val).astype(int)

    # === Encoding Tipe Pengunjung ===
    # New_Visitor adalah referensi (tidak punya kolom sendiri)
    if "VisitorType" in result.columns:
        result["VisitorType_Other"] = (result["VisitorType"] == "Other").astype(int)
        result["VisitorType_Returning_Visitor"] = (
            result["VisitorType"] == "Returning_Visitor"
        ).astype(int)

    # Hapus kolom kategorikal asli (sudah diganti kolom OHE)
    cats = [
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
    ]
    result = result.drop(columns=[c for c in cats if c in result.columns])

    return result


def preprocess(df):
    """
    Pipeline preprocessing lengkap: scaling + encoding.

    Langkah-langkah:
    1. Konversi 10 kolom numerik ke tipe numeric (menangani string)
    2. Scaling 10 kolom numerik menggunakan MinMaxScaler terlatih
    3. One-Hot Encoding pada 6 kolom kategorikal

    Parameter:
        df (DataFrame): DataFrame mentah dengan 17 kolom RAW_FEATURE_COLS

    Return:
        DataFrame: DataFrame yang sudah siap untuk prediksi model
                   (kolom numerik ter-scale + kolom kategorikal ter-encode)
    """
    result = df.copy()

    # Langkah 1 & 2: Konversi ke numerik lalu scale dengan MinMaxScaler
    for col in SCALE_COLS:
        result[col] = pd.to_numeric(result[col])
    result[SCALE_COLS] = _scaler.transform(result[SCALE_COLS])

    # Langkah 3: One-Hot Encoding kolom kategorikal
    result = encode_one_hot(result)

    return result


def select_features(df, model_type):
    """
    Memilih kolom fitur sesuai model yang akan digunakan.

    Random Forest menggunakan 25 fitur (lebih kompleks, akurasi lebih tinggi).
    Logistic Regression menggunakan 8 fitur (lebih sederhana, interpretable).

    Parameter:
        df (DataFrame): DataFrame yang sudah di-preprocess
        model_type (str): "rf" untuk Random Forest, "lr" untuk Logistic Regression

    Return:
        DataFrame: Subset kolom sesuai fitur yang dibutuhkan model

    Raises:
        ValueError: Jika model_type bukan "rf" atau "lr"
    """
    if model_type == "rf":
        return df[RF_FEATURES]  # 25 fitur untuk Random Forest
    if model_type == "lr":
        return df[LR_FEATURES]  # 8 fitur untuk Logistic Regression
    raise ValueError(f"Tipe model tidak dikenal: {model_type}")
