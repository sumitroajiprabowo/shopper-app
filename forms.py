"""
Modul Definisi Form untuk Aplikasi ShopPredict
===============================================
Modul ini mendefinisikan form HTML menggunakan Flask-WTF (WTForms).
Terdapat 2 form utama:

1. FormPrediksiManual — Form input manual untuk prediksi Logistic Regression
   - 3 input numerik: PageValues, ExitRates, ProductRelated_Duration
   - 3 dropdown: Month (bulan), Browser, TrafficType (sumber traffic)
   - Hanya menggunakan model LR karena hanya 8 fitur yang dibutuhkan

2. FormUploadCSV — Form upload file CSV untuk prediksi batch
   - Input file CSV (17 kolom lengkap)
   - Pilihan model: Random Forest / Logistic Regression / Komparasi

Catatan:
- CSRF dinonaktifkan di config app (WTF_CSRF_ENABLED = False)
- InputRequired digunakan agar nilai 0 tetap dianggap valid
  (DataRequired akan menolak nilai 0 karena dianggap falsy)
"""

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import FloatField, SelectField, RadioField, SubmitField
from wtforms.validators import InputRequired, NumberRange


class FormPrediksiManual(FlaskForm):
    """
    Form prediksi manual menggunakan Logistic Regression (8 fitur).

    Dari 8 fitur LR, hanya 6 yang perlu input user:
    - PageValues (numerik) — nilai rata-rata halaman sebelum transaksi
    - ExitRates (numerik) — rasio halaman terakhir sebelum keluar
    - ProductRelated_Duration (numerik) — durasi melihat halaman produk

    - Month (dropdown) — bulan kunjungan, di-encode jadi Month_Nov
    - Browser (dropdown) — kode browser, di-encode jadi Browser_3, Browser_12
    - TrafficType (dropdown) — sumber traffic, di-encode jadi TrafficType_15, TrafficType_16

    Fitur ke-7 dan ke-8 (VisitorType, Weekend) di-set default oleh server:
    - VisitorType = "New_Visitor" (referensi OHE, semua kolom = 0)
    - Weekend = False (0)
    """

    # === Input Numerik ===

    # Nilai rata-rata halaman yang dikunjungi sebelum menyelesaikan transaksi
    # Fitur paling berpengaruh dalam model LR (koefisien positif terbesar)
    # Nilai 0 = pengunjung tidak melihat halaman bernilai tinggi
    PageValues = FloatField(
        "Page Values",
        validators=[
            InputRequired(message="Page Values wajib diisi"),
            NumberRange(min=0, message="Page Values tidak boleh negatif"),
        ],
    )

    # Rasio halaman yang menjadi halaman terakhir sebelum pengunjung keluar
    # Rentang valid: 0.0 (tidak pernah jadi halaman keluar) sampai 1.0 (selalu)
    # Koefisien negatif di LR — exit tinggi = kemungkinan beli rendah
    ExitRates = FloatField(
        "Exit Rates",
        validators=[
            InputRequired(message="Exit Rates wajib diisi"),
            NumberRange(min=0, max=1, message="Exit Rates harus antara 0 dan 1"),
        ],
    )

    # Total durasi (dalam detik) yang dihabiskan di halaman terkait produk
    # Semakin lama browsing produk, semakin tinggi kemungkinan membeli
    ProductRelated_Duration = FloatField(
        "Durasi Produk (detik)",
        validators=[
            InputRequired(message="Durasi Produk wajib diisi"),
            NumberRange(min=0, message="Durasi Produk tidak boleh negatif"),
        ],
    )

    # === Input Dropdown (Kategorikal) ===

    # Bulan kunjungan — di-encode menjadi kolom OHE (Month_Nov, dll)
    # Model LR hanya menggunakan Month_Nov, tapi semua bulan tersedia
    # agar user bisa memilih bulan sebenarnya (bulan lain = Month_Nov = 0)
    # Catatan: Januari tidak ada di dataset original UCI
    Month = SelectField(
        "Bulan",
        choices=[
            ("Feb", "Februari"),
            ("Mar", "Maret"),
            ("May", "Mei"),
            ("June", "Juni"),
            ("Jul", "Juli"),
            ("Aug", "Agustus"),
            ("Sep", "September"),
            ("Oct", "Oktober"),
            ("Nov", "November"),
            ("Dec", "Desember"),
        ],
        validators=[InputRequired()],
    )

    # Kode browser pengguna (1-13 sesuai dataset UCI)
    # Di-encode menjadi kolom OHE: Browser_2 s/d Browser_13
    # Model LR menggunakan Browser_3 dan Browser_12
    # Browser 1 = referensi (tidak punya kolom OHE sendiri)
    Browser = SelectField(
        "Browser",
        choices=[(str(i), f"Browser {i}") for i in range(1, 14)],
        validators=[InputRequired()],
    )

    # Kode sumber traffic pengunjung (1-20 sesuai dataset UCI)
    # Di-encode menjadi kolom OHE: TrafficType_2 s/d TrafficType_20
    # Model LR menggunakan TrafficType_15 dan TrafficType_16
    # TrafficType 1 = referensi (tidak punya kolom OHE sendiri)
    TrafficType = SelectField(
        "Sumber Traffic",
        choices=[(str(i), f"Traffic {i}") for i in range(1, 21)],
        validators=[InputRequired()],
    )

    # Tombol submit form prediksi manual
    submit = SubmitField("Prediksi Sekarang")


class FormUploadCSV(FlaskForm):
    """
    Form upload file CSV untuk prediksi batch (banyak baris sekaligus).

    File CSV harus memiliki 17 kolom sesuai RAW_FEATURE_COLS di preprocessing.py:
    - 10 kolom numerik (Administrative, Admin_Duration, Informational, dll)
    - 6 kolom kategorikal (Month, OperatingSystems, Browser, Region,
      TrafficType, VisitorType)
    - 1 kolom boolean (Weekend)

    User dapat memilih model prediksi:
    - RF: Random Forest dengan 25 fitur (akurasi lebih tinggi)
    - LR: Logistic Regression dengan 8 fitur (lebih sederhana)
    - Both: Komparasi kedua model dalam satu tabel hasil
    """

    # Input file CSV — hanya menerima file berekstensi .csv
    # Ukuran maksimum dibatasi oleh MAX_CONTENT_LENGTH di config app (5MB)
    file = FileField(
        "File CSV",
        validators=[
            FileRequired(message="File CSV wajib dipilih"),
            FileAllowed(["csv"], message="Hanya file CSV yang diperbolehkan"),
        ],
    )

    # Pilihan model prediksi — ditampilkan sebagai radio button di UI
    # Default: Random Forest (rf) karena akurasi lebih tinggi
    model_choice = RadioField(
        "Model Prediksi",
        choices=[
            ("rf", "Random Forest (25 fitur)"),  # 25 fitur, akurasi tinggi
            ("lr", "Logistic Regression (8 fitur)"),  # 8 fitur, interpretable
            ("both", "Komparasi Keduanya"),  # Tampilkan hasil kedua model
        ],
        default="rf",
        validators=[InputRequired()],
    )

    # Tombol submit form upload CSV
    submit = SubmitField("Upload dan Prediksi")
