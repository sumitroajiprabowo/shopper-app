"""
Modul Utama Aplikasi ShopPredict (Flask)
========================================
Aplikasi web untuk prediksi niat beli pengunjung toko online
menggunakan dataset UCI Online Shoppers Purchasing Intention.

Fitur utama:
1. Prediksi manual via form input — khusus Logistic Regression (8 fitur)
   Output: Prediksi + Tingkat Keyakinan Model (progress bar visual)
2. Prediksi batch via upload CSV — Random Forest / LR / komparasi keduanya
   Output: Tabel data + kolom Prediksi + Kepercayaan per model
3. Download hasil prediksi CSV

Arsitektur:
- Flask sebagai web framework
- WTForms untuk validasi form input
- Pandas untuk manipulasi data CSV
- scikit-learn (joblib) untuk memuat model ML yang sudah dilatih
- Hasil CSV disimpan di memori server (dict) dengan UUID sebagai kunci

Endpoint:
- GET  /              → Halaman utama (form manual + form upload CSV)
- POST /predict       → Prediksi manual (LR only)
- POST /upload        → Prediksi batch via CSV (RF/LR/both)
- GET  /download-csv  → Download file CSV hasil prediksi terakhir
"""

from flask import (
    Flask,
    render_template,
    request,
    make_response,
    redirect,
    url_for,
    session,
)
import joblib
import pandas as pd
import io
import uuid

from forms import FormPrediksiManual, FormUploadCSV
from preprocessing import (
    preprocess,
    select_features,
    RAW_FEATURE_COLS,
    SCALE_COLS,
)


def muat_model():
    """
    Memuat 3 file model ML dari direktori model/.

    File yang dimuat:
    - model/model_tuned_rf.pkl  → Random Forest Classifier (25 fitur)
    - model/model_tuned_lre.pkl → Logistic Regression (8 fitur)
    - model/scaler_full.pkl     → MinMaxScaler untuk 10 kolom numerik

    Return:
        tuple: (rf_model, lr_model, scaler) atau (None, None, None) jika gagal

    Catatan:
    - Model dimuat sekali saat aplikasi start (module-level)
    - Jika file tidak ditemukan, aplikasi tetap jalan tapi prediksi dinonaktifkan
    """
    try:
        rf = joblib.load("model/model_tuned_rf.pkl")
        lr = joblib.load("model/model_tuned_lre.pkl")
        scaler = joblib.load("model/scaler_full.pkl")
        return rf, lr, scaler
    except FileNotFoundError as e:
        print(f"PERINGATAN: Tidak dapat memuat file model: {e}")
        return None, None, None


def buat_aplikasi():
    """
    Membuat dan mengkonfigurasi instance Flask.

    Konfigurasi:
    - secret_key: untuk session cookie (menyimpan download_id CSV)
    - WTF_CSRF_ENABLED: False — CSRF dinonaktifkan karena app internal
    - MAX_CONTENT_LENGTH: 5MB — batas maksimum ukuran file upload

    Return:
        Flask: Instance aplikasi Flask yang sudah dikonfigurasi
    """
    app = Flask(__name__)
    app.secret_key = "shopper-app-secret-key"
    app.config["WTF_CSRF_ENABLED"] = False
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB
    return app


# ============================================================
# Inisialisasi aplikasi dan model (dijalankan sekali saat import)
# ============================================================
app = buat_aplikasi()
rf_model, lr_model, scaler = muat_model()

# Threshold probabilitas untuk klasifikasi biner
# Jika probabilitas >= threshold → "Akan Membeli" (1)
# Jika probabilitas <  threshold → "Tidak Membeli" (0)
THRESHOLD_RF = 0.5  # Threshold Random Forest
THRESHOLD_LR = 0.5  # Threshold Logistic Regression


def hitung_kepercayaan(probabilitas, prediksi):
    """
    Menghitung tingkat kepercayaan (confidence) model terhadap prediksinya.

    Logika:
    - Jika prediksi "Akan Membeli" (1): confidence = probabilitas
      Contoh: prob=0.88 → model 88% yakin akan membeli
    - Jika prediksi "Tidak Membeli" (0): confidence = 1 - probabilitas
      Contoh: prob=0.14 → model 86% yakin TIDAK akan membeli

    Return:
        float: Tingkat kepercayaan (0.0 - 1.0)
    """
    return probabilitas if prediksi == 1 else (1 - probabilitas)


def label_kepercayaan(confidence):
    """
    Mengkategorikan tingkat kepercayaan menjadi label deskriptif.

    Kategori:
    - >= 90%: Sangat Tinggi — model sangat yakin
    - >= 75%: Tinggi — model cukup yakin
    - >= 60%: Sedang — model agak ragu
    - <  60%: Rendah — model tidak yakin, prediksi bisa salah

    Return:
        str: Label kategori kepercayaan
    """
    if confidence >= 0.90:
        return "Sangat Tinggi"
    elif confidence >= 0.75:
        return "Tinggi"
    elif confidence >= 0.60:
        return "Sedang"
    return "Rendah"


# Penyimpanan sementara CSV hasil prediksi di memori server
# Key: UUID string, Value: string CSV
# Menggunakan dict biasa (bukan session) karena session cookie
# terlalu kecil untuk menyimpan data CSV (batas 4KB)
_csv_download_store = {}


@app.errorhandler(413)
def file_terlalu_besar(e):
    """
    Handler untuk error 413 (Request Entity Too Large).

    Dipanggil otomatis oleh Flask ketika ukuran file upload
    melebihi MAX_CONTENT_LENGTH (5MB).

    Menampilkan halaman utama dengan pesan error di bagian CSV upload.
    Menggunakan MultiDict kosong agar form tidak menampilkan data sebelumnya.
    """
    from werkzeug.datastructures import MultiDict

    kosong = MultiDict()  # Form kosong tanpa data sebelumnya
    return (
        render_template(
            "index.html",
            form_prediksi=FormPrediksiManual(formdata=kosong),
            form_upload=FormUploadCSV(formdata=kosong),
            csv_error="Error: Ukuran file melebihi batas maksimum (5MB)",
        ),
        413,
    )


def model_siap():
    """
    Mengecek apakah semua model ML berhasil dimuat.

    Return:
        bool: True jika rf_model, lr_model, dan scaler semuanya tersedia
    """
    return all([rf_model is not None, lr_model is not None, scaler is not None])


def prediksi_model(df_processed, model_type):
    """
    Menjalankan prediksi menggunakan model yang dipilih.

    Langkah:
    1. Seleksi fitur sesuai model (25 untuk RF, 8 untuk LR)
    2. Hitung probabilitas kelas positif (akan membeli)
    3. Terapkan threshold untuk menghasilkan prediksi biner

    Parameter:
        df_processed (DataFrame): Data yang sudah di-preprocess (scaled + encoded)
        model_type (str): "rf" untuk Random Forest, "lr" untuk Logistic Regression

    Return:
        tuple: (probabilitas, prediksi)
            - probabilitas: array float, probabilitas kelas positif (0.0 - 1.0)
            - prediksi: list int, 1 = akan membeli, 0 = tidak membeli
    """
    features = select_features(df_processed, model_type)  # Pilih kolom fitur
    model = rf_model if model_type == "rf" else lr_model  # Pilih model
    threshold = THRESHOLD_RF if model_type == "rf" else THRESHOLD_LR
    probabilitas = model.predict_proba(features)[:, 1]  # Ambil prob kelas 1
    prediksi = [1 if p >= threshold else 0 for p in probabilitas]
    return probabilitas, prediksi


# ============================================================
# Route: Halaman Utama
# ============================================================
@app.route("/")
def home():
    """
    Menampilkan halaman utama dengan 2 form:
    - Form prediksi manual (Logistic Regression, 6 input)
    - Form upload CSV (RF/LR/komparasi, 17 kolom)
    """
    return render_template(
        "index.html",
        form_prediksi=FormPrediksiManual(),
        form_upload=FormUploadCSV(),
    )


# ============================================================
# Route: Prediksi Manual (POST /predict)
# Khusus model Logistic Regression — 8 fitur
# ============================================================
@app.route("/predict", methods=["POST"])
def predict():
    """
    Memproses prediksi manual dari form input (Logistic Regression only).

    Alur:
    1. Validasi form (6 field: 3 numerik + 3 dropdown)
    2. Bangun DataFrame 17 kolom (isi 0 untuk kolom yang tidak diinput)
    3. Set nilai dari form untuk 6 kolom yang diinput user
    4. Set default: VisitorType = "New_Visitor", Weekend = False
    5. Jalankan preprocessing (scaling + OHE)
    6. Prediksi menggunakan model LR
    7. Tampilkan hasil: "Akan Membeli" atau "Tidak Membeli"

    Catatan:
    - Kolom yang tidak diinput user diisi 0 (Administrative, Informational, dll)
    - VisitorType di-set "New_Visitor" karena ini adalah referensi OHE
      (semua kolom VisitorType_* = 0, jadi tidak memengaruhi prediksi)
    - Hanya LR yang digunakan karena form hanya mengumpulkan 6 dari 8 fitur LR
      (2 fitur sisanya: VisitorType dan Weekend di-set default)
    """
    form_prediksi = FormPrediksiManual()
    form_upload = FormUploadCSV()

    # Cek apakah model ML sudah dimuat dengan benar
    if not model_siap():
        return render_template(
            "index.html",
            form_prediksi=form_prediksi,
            form_upload=form_upload,
            error="Error: Model tidak siap. Periksa log server.",
        )

    # Validasi form — cek semua field terisi dan valid
    if not form_prediksi.validate_on_submit():
        error_messages = []
        for field_name, errors in form_prediksi.errors.items():
            if field_name == "csrf_token":
                continue  # Abaikan error CSRF (sudah dinonaktifkan)
            for err in errors:
                error_messages.append(err)
        pesan_error = "; ".join(error_messages) if error_messages else "Validasi gagal"
        return render_template(
            "index.html",
            form_prediksi=form_prediksi,
            form_upload=form_upload,
            error=f"Error: {pesan_error}",
        )

    # Bangun DataFrame 17 kolom — semua diisi 0 sebagai default
    # Kolom yang tidak diinput user (Administrative, Informational, dll)
    # dibiarkan 0 karena tidak termasuk dalam 8 fitur LR
    raw_data = {col: [0] for col in RAW_FEATURE_COLS}

    # Isi 3 kolom numerik dari form input
    raw_data["PageValues"] = [form_prediksi.PageValues.data]
    raw_data["ExitRates"] = [form_prediksi.ExitRates.data]
    raw_data["ProductRelated_Duration"] = [form_prediksi.ProductRelated_Duration.data]

    # Isi 3 kolom kategorikal dari dropdown
    raw_data["Month"] = [form_prediksi.Month.data]  # Misal: "Nov"
    raw_data["Browser"] = [int(form_prediksi.Browser.data)]  # Misal: 12
    raw_data["TrafficType"] = [int(form_prediksi.TrafficType.data)]  # Misal: 15

    # Default untuk kolom yang tidak ada di form
    # New_Visitor = referensi OHE → semua kolom VisitorType_* = 0
    raw_data["VisitorType"] = ["New_Visitor"]
    raw_data["Weekend"] = [False]  # Asumsikan bukan akhir pekan

    # Buat DataFrame dan jalankan preprocessing (scaling + OHE)
    df = pd.DataFrame(raw_data)
    df_processed = preprocess(df)

    # Prediksi menggunakan Logistic Regression
    prob, pred = prediksi_model(df_processed, "lr")

    # Hitung tingkat kepercayaan model terhadap prediksinya
    raw_prob = float(prob[0])
    confidence = hitung_kepercayaan(raw_prob, pred[0])

    # Siapkan hasil untuk ditampilkan di template
    results = {
        "lr": {
            "hasil": "Akan Membeli" if pred[0] == 1 else "Tidak Membeli",
            "confidence": round(confidence, 4),  # Tingkat kepercayaan 4 desimal
            "confidence_label": label_kepercayaan(confidence),  # Kategori kepercayaan
            "label": "Logistic Regression (8 fitur)",
        }
    }

    return render_template(
        "index.html",
        form_prediksi=form_prediksi,
        form_upload=form_upload,
        results=results,
    )


# ============================================================
# Route: Prediksi via Upload CSV (POST /upload)
# Mendukung Random Forest, Logistic Regression, atau keduanya
# ============================================================
@app.route("/upload", methods=["POST"])
def predict_csv():
    """
    Memproses prediksi batch dari file CSV yang di-upload.

    Alur:
    1. Validasi form upload (file CSV + pilihan model)
    2. Baca CSV menjadi DataFrame
    3. Validasi: CSV tidak kosong, semua 17 kolom ada, nilai numerik valid
    4. Konversi tipe data kolom kategorikal (int untuk OS/Browser/Region/Traffic)
    5. Preprocessing (scaling + OHE)
    6. Prediksi sesuai model yang dipilih (RF/LR/both)
    7. Tambahkan kolom hasil prediksi ke DataFrame
    8. Simpan CSV hasil ke memory store untuk download
    9. Tampilkan tabel hasil + ringkasan statistik

    Validasi CSV yang dilakukan:
    - File tidak kosong (minimal 1 baris data)
    - Semua 17 kolom RAW_FEATURE_COLS ada di CSV
    - 10 kolom numerik (SCALE_COLS) tidak mengandung nilai non-numerik atau NaN
    """
    form_prediksi = FormPrediksiManual()
    form_upload = FormUploadCSV()

    # Cek model ML sudah dimuat
    if not model_siap():
        return render_template(
            "index.html",
            form_prediksi=form_prediksi,
            form_upload=form_upload,
            csv_error="Error: Model tidak siap. Periksa log server.",
        )

    # Validasi form upload (file ada + ekstensi .csv)
    if not form_upload.validate_on_submit():
        error_messages = []
        for field_name, errors in form_upload.errors.items():
            if field_name == "csrf_token":
                continue
            for err in errors:
                error_messages.append(err)
        pesan_error = "; ".join(error_messages) if error_messages else "Validasi gagal"
        return render_template(
            "index.html",
            form_prediksi=form_prediksi,
            form_upload=form_upload,
            csv_error=f"Error: {pesan_error}",
        )

    file = form_upload.file.data  # File object dari upload
    model_choice = form_upload.model_choice.data  # "rf", "lr", atau "both"

    try:
        # === Langkah 1: Baca CSV menjadi DataFrame ===
        df = pd.read_csv(file)

        # === Langkah 2: Validasi — CSV tidak boleh kosong ===
        if df.empty:
            return render_template(
                "index.html",
                form_prediksi=form_prediksi,
                form_upload=form_upload,
                csv_error="Error: File CSV kosong (tidak ada baris data)",
            )

        # === Langkah 3: Validasi — Semua 17 kolom harus ada ===
        kolom_hilang = [col for col in RAW_FEATURE_COLS if col not in df.columns]
        if kolom_hilang:
            return render_template(
                "index.html",
                form_prediksi=form_prediksi,
                form_upload=form_upload,
                csv_error=f"File CSV tidak memiliki kolom: {', '.join(kolom_hilang)}",
            )

        # === Langkah 4: Validasi — Kolom numerik harus berisi angka ===
        # Cek 10 kolom SCALE_COLS tidak mengandung string/NaN/kosong
        for col in SCALE_COLS:
            if not pd.to_numeric(df[col], errors="coerce").notna().all():
                return render_template(
                    "index.html",
                    form_prediksi=form_prediksi,
                    form_upload=form_upload,
                    csv_error=f"Error: Kolom '{col}' mengandung nilai non-numerik atau kosong",
                )

        # === Langkah 5: Konversi tipe data kolom kategorikal ===
        # Kolom ini harus integer untuk OHE encoding yang benar
        df["OperatingSystems"] = pd.to_numeric(df["OperatingSystems"]).astype(int)
        df["Browser"] = pd.to_numeric(df["Browser"]).astype(int)
        df["Region"] = pd.to_numeric(df["Region"]).astype(int)
        df["TrafficType"] = pd.to_numeric(df["TrafficType"]).astype(int)

        # Konversi Weekend dari string "True"/"False" ke boolean Python
        # CSV menyimpan boolean sebagai string, perlu di-map manual
        df["Weekend"] = df["Weekend"].map(
            {"True": True, "False": False, True: True, False: False}
        )

        # === Langkah 6: Preprocessing (scaling + OHE) ===
        df_processed = preprocess(df[RAW_FEATURE_COLS])

        # === Langkah 7: Prediksi sesuai model yang dipilih ===

        # Prediksi Random Forest (25 fitur)
        if model_choice in ("rf", "both"):
            rf_prob, rf_pred = prediksi_model(df_processed, "rf")
            df["RF_Prediksi"] = [
                "Akan Membeli" if p == 1 else "Tidak Membeli" for p in rf_pred
            ]
            # Hitung kepercayaan per baris: seberapa yakin model terhadap prediksinya
            rf_confidence = [
                hitung_kepercayaan(p, pr) for p, pr in zip(rf_prob, rf_pred)
            ]
            df["RF_Kepercayaan"] = [
                f"{c:.2%} ({label_kepercayaan(c)})" for c in rf_confidence
            ]

        # Prediksi Logistic Regression (8 fitur)
        if model_choice in ("lr", "both"):
            lr_prob, lr_pred = prediksi_model(df_processed, "lr")
            df["LR_Prediksi"] = [
                "Akan Membeli" if p == 1 else "Tidak Membeli" for p in lr_pred
            ]
            lr_confidence = [
                hitung_kepercayaan(p, pr) for p, pr in zip(lr_prob, lr_pred)
            ]
            df["LR_Kepercayaan"] = [
                f"{c:.2%} ({label_kepercayaan(c)})" for c in lr_confidence
            ]

        # === Langkah 8: Konversi ke format untuk template ===
        csv_results = df.to_dict("records")  # List of dicts untuk tabel HTML
        csv_columns = df.columns.tolist()  # Daftar nama kolom untuk header

        # === Langkah 9: Simpan CSV ke memory store untuk download ===
        # UUID unik sebagai kunci agar setiap upload punya file download sendiri
        download_id = str(uuid.uuid4())
        output_stream = io.StringIO()
        df.to_csv(output_stream, index=False, encoding="utf-8")

        # Bersihkan store lama dan simpan yang baru
        # Hanya 1 CSV yang disimpan sekaligus untuk hemat memori
        _csv_download_store.clear()
        _csv_download_store[download_id] = output_stream.getvalue()

        # Simpan download_id di session cookie agar route /download-csv
        # bisa mengambil CSV yang benar
        session["csv_download_id"] = download_id

        # === Langkah 10: Hitung ringkasan statistik prediksi ===
        ringkasan = {"total": len(df)}
        if model_choice in ("rf", "both"):
            ringkasan["rf_beli"] = sum(rf_pred)  # Jumlah akan beli
            ringkasan["rf_tidak"] = len(rf_pred) - sum(rf_pred)  # Jumlah tidak beli
        if model_choice in ("lr", "both"):
            ringkasan["lr_beli"] = sum(lr_pred)
            ringkasan["lr_tidak"] = len(lr_pred) - sum(lr_pred)

        return render_template(
            "index.html",
            form_prediksi=form_prediksi,
            form_upload=form_upload,
            csv_results=csv_results,  # Data tabel hasil prediksi
            csv_columns=csv_columns,  # Header kolom tabel
            csv_summary=ringkasan,  # Ringkasan statistik
            model_choice=model_choice,  # Model yang dipilih user
        )

    except Exception as e:
        # Tangkap semua error yang tidak terduga (format CSV salah, dll)
        return render_template(
            "index.html",
            form_prediksi=form_prediksi,
            form_upload=form_upload,
            csv_error=f"Terjadi kesalahan: {e}",
        )


# ============================================================
# Route: Download CSV Hasil Prediksi (GET /download-csv)
# ============================================================
@app.route("/download-csv")
def download_csv():
    """
    Mengirim file CSV hasil prediksi untuk di-download user.

    Alur:
    1. Ambil download_id dari session cookie
    2. Cari CSV data di _csv_download_store berdasarkan download_id
    3. Jika tidak ditemukan, redirect ke halaman utama
    4. Jika ditemukan, kirim sebagai response dengan header download

    Catatan:
    - File dikirim dengan nama "hasil_prediksi.csv"
    - Content-type: text/csv agar browser mengenali sebagai CSV
    - Data CSV disimpan sebagai string di memori, bukan file di disk
    """
    download_id = session.get("csv_download_id")
    csv_data = _csv_download_store.get(download_id) if download_id else None

    # Jika tidak ada data CSV (belum upload atau expired), kembali ke home
    if not csv_data:
        return redirect(url_for("home"))

    # Buat HTTP response dengan CSV data sebagai attachment
    response = make_response(csv_data)
    response.headers["Content-Disposition"] = "attachment; filename=hasil_prediksi.csv"
    response.headers["Content-type"] = "text/csv; charset=utf-8"
    return response


# ============================================================
# Entry Point: Jalankan server Flask
# ============================================================
def jalankan_server():
    """
    Menjalankan server Flask untuk development.

    Konfigurasi:
    - host 0.0.0.0: bisa diakses dari semua network interface
    - port 5000: port default Flask
    - debug True: auto-reload saat kode berubah (jangan di production!)
    """
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":  # pragma: no cover
    jalankan_server()
