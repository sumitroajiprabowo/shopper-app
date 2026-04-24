"""
Unit Test untuk Aplikasi Prediksi Niat Pembelian Online (v2.0 — LR Manual + Dual CSV)
=====================================================================================
Test suite lengkap yang mencakup seluruh fungsionalitas aplikasi ShopPredict.

Cakupan test (64 test case):
1. TestMuatModel          — Pemuatan model ML (RF + LR + scaler) dari file pkl
2. TestBuatAplikasi       — Konfigurasi instance Flask (secret key, max upload, dll)
3. TestModelSiap          — Fungsi pengecekan kesiapan model (semua harus tersedia)
4. TestPrediksiModel      — Fungsi prediksi: input DataFrame → output probabilitas + prediksi
5. TestFormPrediksiManual — Validasi WTForms: rentang, field kosong, nilai negatif
6. TestFormUploadCSV      — Validasi file upload: tipe file, file wajib
7. TestHalamanUtama       — GET / : elemen UI (form, input, dropdown, label)
8. TestPrediksiManual     — POST /predict : prediksi LR, validasi, error handling
9. TestUploadCSV          — POST /upload : RF/LR/both, validasi CSV, error handling
10. TestMemoryStore       — Penyimpanan CSV di memori server (dict + UUID)
11. TestDownloadCSV       — GET /download-csv : download file, redirect jika kosong
12. TestCSRFErrorHandling — Penanganan CSRF saat diaktifkan
13. TestEntryPoint        — Fungsi jalankan_server() memanggil app.run()

Cara menjalankan:
    python -m pytest test_app.py -v

Catatan teknis:
- Menggunakan unittest.mock.patch untuk mengisolasi test dari dependensi
- BaseTestCase menyediakan setUp/tearDown standar (test client + bersihkan store)
- CSV test data didefinisikan sebagai string constant di atas test class
- InputRequired (bukan DataRequired) digunakan agar nilai 0 tetap valid
"""

import unittest
from unittest.mock import patch
import io
import os

# Set working directory ke lokasi file ini agar model/*.pkl bisa ditemukan
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from app import (
    app,
    rf_model,
    lr_model,
    scaler,
    muat_model,
    buat_aplikasi,
    model_siap,
    prediksi_model,
    hitung_kepercayaan,
    label_kepercayaan,
    jalankan_server,
    _csv_download_store,
)
from forms import FormPrediksiManual, FormUploadCSV
from preprocessing import RAW_FEATURE_COLS

# ============================================================
# Data Test: Input form manual untuk 2 skenario prediksi
# ============================================================

# Skenario 1: Pengunjung yang kemungkinan besar akan membeli
# PageValues tinggi (50) + ExitRates rendah (0.01) = sinyal beli kuat
DATA_AKAN_BELI = {
    "PageValues": "50",  # Nilai halaman tinggi → indikator beli
    "ExitRates": "0.01",  # Exit rate rendah → tidak cepat keluar
    "ProductRelated_Duration": "200",  # Lama melihat produk → tertarik
    "Month": "Nov",  # November = musim belanja
    "Browser": "2",  # Browser tipe 2
    "TrafficType": "2",  # Sumber traffic tipe 2
}

# Skenario 2: Pengunjung yang kemungkinan besar TIDAK akan membeli
# PageValues 0 + ExitRates tinggi (0.20) = sinyal tidak beli
DATA_TIDAK_BELI = {
    "PageValues": "0",  # Tidak ada nilai halaman
    "ExitRates": "0.20",  # Exit rate tinggi → cepat keluar
    "ProductRelated_Duration": "0",  # Tidak melihat produk
    "Month": "Feb",  # Februari = bukan musim belanja
    "Browser": "1",  # Browser tipe 1 (referensi)
    "TrafficType": "1",  # Traffic tipe 1 (referensi)
}

# ============================================================
# Data Test: CSV untuk upload batch
# 17 kolom sesuai RAW_FEATURE_COLS di preprocessing.py
# ============================================================

# Header CSV — 17 kolom wajib
CSV_HEADER = (
    "Administrative,Administrative_Duration,Informational,Informational_Duration,"
    "ProductRelated,ProductRelated_Duration,BounceRates,ExitRates,PageValues,"
    "SpecialDay,Month,OperatingSystems,Browser,Region,TrafficType,VisitorType,Weekend"
)

# Baris data: pengunjung yang akan membeli (PageValues=50, ExitRates=0.01)
CSV_ROW_BELI = (
    "4,30.0,0,0.0,10,200.0,0.01,0.01,50.0,0.0,Nov,2,2,1,2,Returning_Visitor,False"
)

# Baris data: pengunjung yang tidak akan membeli (PageValues=0, ExitRates=0.20)
CSV_ROW_TIDAK = (
    "0,0.0,0,0.0,1,0.0,0.20,0.20,0.0,0.0,Feb,1,1,1,1,Returning_Visitor,False"
)

# CSV valid dengan 2 baris data (1 beli + 1 tidak beli)
CSV_VALID = f"{CSV_HEADER}\n{CSV_ROW_TIDAK}\n{CSV_ROW_BELI}\n"

# CSV dengan 100 baris untuk test performa/banyak data
CSV_BANYAK_BARIS = f"{CSV_HEADER}\n" + f"{CSV_ROW_TIDAK}\n" * 100

# CSV kosong (hanya header, tidak ada baris data) → harus ditolak
CSV_KOSONG = f"{CSV_HEADER}\n"

# CSV dengan nilai non-numerik di kolom Administrative → harus ditolak
CSV_NON_NUMERIK = f"{CSV_HEADER}\nabc,0.0,0,0.0,1,0.0,0.20,0.20,0.0,0.0,Feb,1,1,1,1,Returning_Visitor,False\n"

# CSV dengan nilai kosong/NaN di kolom Administrative → harus ditolak
CSV_DENGAN_NAN = f"{CSV_HEADER}\n,0.0,0,0.0,1,0.0,0.20,0.20,0.0,0.0,Feb,1,1,1,1,Returning_Visitor,False\n"


def buat_file_csv(konten):
    """
    Helper: Buat file-like object dari string CSV untuk simulasi upload.

    Parameter:
        konten (str): Isi file CSV sebagai string

    Return:
        tuple: (BytesIO, filename) — format yang diterima Flask test client
    """
    return (io.BytesIO(konten.encode("utf-8")), "test.csv")


class BaseTestCase(unittest.TestCase):
    """
    Base class untuk test yang membutuhkan Flask test client.

    setUp: Aktifkan mode testing + nonaktifkan CSRF + buat test client
    tearDown: Bersihkan memory store CSV agar test tidak saling memengaruhi
    """

    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False  # Nonaktifkan CSRF untuk testing
        self.client = app.test_client()

    def tearDown(self):
        _csv_download_store.clear()  # Bersihkan store agar tidak bocor antar test


# ============================================================
# Test 1: Pemuatan Model ML dari File PKL
# ============================================================
class TestMuatModel(unittest.TestCase):
    """
    Test pemuatan 3 file model dari direktori model/:
    - model_tuned_rf.pkl (Random Forest)
    - model_tuned_lre.pkl (Logistic Regression)
    - scaler_full.pkl (MinMaxScaler)
    """

    def test_rf_model_berhasil_dimuat(self):
        """Random Forest model harus berhasil dimuat saat app start."""
        self.assertIsNotNone(rf_model)

    def test_lr_model_berhasil_dimuat(self):
        """Logistic Regression model harus berhasil dimuat saat app start."""
        self.assertIsNotNone(lr_model)

    def test_scaler_berhasil_dimuat(self):
        """MinMaxScaler harus berhasil dimuat saat app start."""
        self.assertIsNotNone(scaler)

    def test_muat_model_file_tidak_ada(self):
        """Jika file model tidak ditemukan, kembalikan (None, None, None)."""
        with patch("app.joblib.load", side_effect=FileNotFoundError("not found")):
            rf, lr, sc = muat_model()
            self.assertIsNone(rf)
            self.assertIsNone(lr)
            self.assertIsNone(sc)


# ============================================================
# Test 2: Konfigurasi Instance Flask
# ============================================================
class TestBuatAplikasi(unittest.TestCase):
    """Test fungsi buat_aplikasi() menghasilkan Flask app yang benar."""

    def test_mengembalikan_instance_flask(self):
        """Harus mengembalikan instance Flask, bukan objek lain."""
        from flask import Flask

        aplikasi = buat_aplikasi()
        self.assertIsInstance(aplikasi, Flask)

    def test_secret_key_terisi(self):
        """Secret key harus terisi untuk mendukung session cookie."""
        aplikasi = buat_aplikasi()
        self.assertIsNotNone(aplikasi.secret_key)

    def test_max_content_length(self):
        """Batas upload file harus 5MB (5 * 1024 * 1024 bytes)."""
        aplikasi = buat_aplikasi()
        self.assertEqual(aplikasi.config["MAX_CONTENT_LENGTH"], 5 * 1024 * 1024)


# ============================================================
# Test 3: Fungsi Pengecekan Kesiapan Model
# ============================================================
class TestModelSiap(unittest.TestCase):
    """
    Test fungsi model_siap() — return True hanya jika
    rf_model, lr_model, DAN scaler semuanya bukan None.
    """

    def test_model_siap_saat_semua_tersedia(self):
        """Jika semua model dimuat, harus return True."""
        self.assertTrue(model_siap())

    def test_model_tidak_siap_saat_rf_none(self):
        """Jika RF model None, harus return False."""
        with patch("app.rf_model", None):
            self.assertFalse(model_siap())

    def test_model_tidak_siap_saat_lr_none(self):
        """Jika LR model None, harus return False."""
        with patch("app.lr_model", None):
            self.assertFalse(model_siap())

    def test_model_tidak_siap_saat_scaler_none(self):
        """Jika scaler None, harus return False."""
        with patch("app.scaler", None):
            self.assertFalse(model_siap())


# ============================================================
# Test 4: Fungsi Prediksi Model
# ============================================================
class TestPrediksiModel(unittest.TestCase):
    """
    Test fungsi prediksi_model(df_processed, model_type).
    Input: DataFrame yang sudah di-preprocess (scaled + encoded)
    Output: (probabilitas, prediksi) — array float + list int
    """

    def test_lr_mengembalikan_tuple(self):
        """LR harus mengembalikan tuple (probabilitas, prediksi) dengan panjang benar."""
        import pandas as pd
        from preprocessing import preprocess

        # Buat DataFrame dummy 17 kolom, semua nilai 0
        df = pd.DataFrame({col: [0] for col in RAW_FEATURE_COLS})
        df["Month"] = "Feb"
        df["VisitorType"] = "Returning_Visitor"
        df["Weekend"] = False
        df_processed = preprocess(df)
        prob, pred = prediksi_model(df_processed, "lr")
        self.assertEqual(len(prob), 1)  # 1 baris input → 1 probabilitas
        self.assertEqual(len(pred), 1)  # 1 baris input → 1 prediksi

    def test_rf_mengembalikan_tuple(self):
        """RF harus mengembalikan tuple (probabilitas, prediksi) dengan panjang benar."""
        import pandas as pd
        from preprocessing import preprocess

        df = pd.DataFrame({col: [0] for col in RAW_FEATURE_COLS})
        df["Month"] = "Feb"
        df["VisitorType"] = "Returning_Visitor"
        df["Weekend"] = False
        df_processed = preprocess(df)
        prob, pred = prediksi_model(df_processed, "rf")
        self.assertEqual(len(prob), 1)
        self.assertEqual(len(pred), 1)

    def test_model_type_tidak_dikenal_raise_error(self):
        """select_features() dengan model_type selain 'rf'/'lr' harus raise ValueError."""
        import pandas as pd
        from preprocessing import preprocess, select_features

        df = pd.DataFrame({col: [0] for col in RAW_FEATURE_COLS})
        df["Month"] = "Feb"
        df["VisitorType"] = "Returning_Visitor"
        df["Weekend"] = False
        df_processed = preprocess(df)
        with self.assertRaises(ValueError):
            select_features(df_processed, "unknown_model")

    def test_probabilitas_antara_0_dan_1(self):
        """Probabilitas harus selalu dalam rentang [0, 1]."""
        import pandas as pd
        from preprocessing import preprocess

        df = pd.DataFrame({col: [0] for col in RAW_FEATURE_COLS})
        df["Month"] = "Feb"
        df["VisitorType"] = "Returning_Visitor"
        df["Weekend"] = False
        df_processed = preprocess(df)
        prob, _ = prediksi_model(df_processed, "lr")
        self.assertGreaterEqual(prob[0], 0)
        self.assertLessEqual(prob[0], 1)


# ============================================================
# Test 5: Fungsi Hitung Kepercayaan dan Label
# ============================================================
class TestKepercayaan(unittest.TestCase):
    """
    Test fungsi hitung_kepercayaan() dan label_kepercayaan().

    hitung_kepercayaan: mengubah probabilitas menjadi confidence
    - Prediksi 1 (beli): confidence = prob (misal 0.88 → 88%)
    - Prediksi 0 (tidak): confidence = 1 - prob (misal 0.14 → 86%)

    label_kepercayaan: mengkategorikan confidence menjadi label
    - >= 90%: Sangat Tinggi
    - >= 75%: Tinggi
    - >= 60%: Sedang
    - < 60%: Rendah
    """

    def test_confidence_akan_membeli(self):
        """Prediksi beli: confidence = prob langsung."""
        self.assertAlmostEqual(hitung_kepercayaan(0.88, 1), 0.88)

    def test_confidence_tidak_membeli(self):
        """Prediksi tidak beli: confidence = 1 - prob."""
        self.assertAlmostEqual(hitung_kepercayaan(0.14, 0), 0.86)

    def test_confidence_tepat_threshold(self):
        """Prob = 0.5, prediksi beli: confidence = 0.5."""
        self.assertAlmostEqual(hitung_kepercayaan(0.5, 1), 0.5)

    def test_label_sangat_tinggi(self):
        """Confidence >= 90% → 'Sangat Tinggi'."""
        self.assertEqual(label_kepercayaan(0.95), "Sangat Tinggi")
        self.assertEqual(label_kepercayaan(0.90), "Sangat Tinggi")

    def test_label_tinggi(self):
        """Confidence 75-89% → 'Tinggi'."""
        self.assertEqual(label_kepercayaan(0.85), "Tinggi")
        self.assertEqual(label_kepercayaan(0.75), "Tinggi")

    def test_label_sedang(self):
        """Confidence 60-74% → 'Sedang'."""
        self.assertEqual(label_kepercayaan(0.70), "Sedang")
        self.assertEqual(label_kepercayaan(0.60), "Sedang")

    def test_label_rendah(self):
        """Confidence < 60% → 'Rendah'."""
        self.assertEqual(label_kepercayaan(0.55), "Rendah")
        self.assertEqual(label_kepercayaan(0.50), "Rendah")


# ============================================================
# Test 6: Validasi Form Prediksi Manual (WTForms)
# ============================================================
class TestFormPrediksiManual(BaseTestCase):
    """
    Test validasi form input manual Logistic Regression.
    Form memiliki 6 field: 3 numerik + 3 dropdown.

    Validasi yang diuji:
    - Form lengkap → valid
    - ExitRates > 1 → invalid (harus 0-1)
    - ExitRates < 0 → invalid
    - PageValues < 0 → invalid
    - ProductRelated_Duration < 0 → invalid
    - Field kosong → invalid
    """

    def test_form_valid_lengkap(self):
        """Form dengan semua field terisi dan valid harus lolos validasi."""
        with app.test_request_context("/predict", method="POST", data=DATA_AKAN_BELI):
            form = FormPrediksiManual()
            self.assertTrue(form.validate())

    def test_form_exit_rates_di_luar_rentang(self):
        """ExitRates > 1.0 harus ditolak (rentang valid: 0-1)."""
        data = DATA_AKAN_BELI.copy()
        data["ExitRates"] = "1.5"
        with app.test_request_context("/predict", method="POST", data=data):
            form = FormPrediksiManual()
            self.assertFalse(form.validate())
            self.assertIn("ExitRates", form.errors)

    def test_form_exit_rates_negatif(self):
        """ExitRates negatif harus ditolak."""
        data = DATA_AKAN_BELI.copy()
        data["ExitRates"] = "-0.5"
        with app.test_request_context("/predict", method="POST", data=data):
            form = FormPrediksiManual()
            self.assertFalse(form.validate())

    def test_form_field_kosong(self):
        """Field yang kosong (empty string) harus ditolak oleh InputRequired."""
        data = DATA_AKAN_BELI.copy()
        data["PageValues"] = ""
        with app.test_request_context("/predict", method="POST", data=data):
            form = FormPrediksiManual()
            self.assertFalse(form.validate())

    def test_form_page_values_negatif(self):
        """PageValues negatif harus ditolak (NumberRange min=0)."""
        data = DATA_AKAN_BELI.copy()
        data["PageValues"] = "-5"
        with app.test_request_context("/predict", method="POST", data=data):
            form = FormPrediksiManual()
            self.assertFalse(form.validate())
            self.assertIn("PageValues", form.errors)

    def test_form_durasi_negatif(self):
        """ProductRelated_Duration negatif harus ditolak."""
        data = DATA_AKAN_BELI.copy()
        data["ProductRelated_Duration"] = "-10"
        with app.test_request_context("/predict", method="POST", data=data):
            form = FormPrediksiManual()
            self.assertFalse(form.validate())
            self.assertIn("ProductRelated_Duration", form.errors)


# ============================================================
# Test 6: Validasi Form Upload CSV (WTForms)
# ============================================================
class TestFormUploadCSV(BaseTestCase):
    """
    Test validasi form upload CSV.
    - File wajib dipilih (FileRequired)
    - Hanya file .csv yang diperbolehkan (FileAllowed)
    """

    def test_form_tanpa_file(self):
        """Upload tanpa file harus gagal validasi."""
        with app.test_request_context(
            "/upload", method="POST", content_type="multipart/form-data"
        ):
            form = FormUploadCSV()
            self.assertFalse(form.validate())
            self.assertIn("file", form.errors)

    def test_form_file_bukan_csv(self):
        """Upload file .txt (bukan .csv) harus gagal validasi."""
        from werkzeug.datastructures import FileStorage

        fake_file = FileStorage(
            stream=io.BytesIO(b"data"), filename="test.txt", content_type="text/plain"
        )
        with app.test_request_context(
            "/upload",
            method="POST",
            content_type="multipart/form-data",
            data={"file": fake_file},
        ):
            form = FormUploadCSV()
            self.assertFalse(form.validate())


# ============================================================
# Test 7: Halaman Utama (GET /)
# ============================================================
class TestHalamanUtama(BaseTestCase):
    """
    Test halaman utama menampilkan semua elemen UI yang diperlukan:
    - Form prediksi manual + form upload CSV
    - 3 input numerik LR + 3 dropdown
    - Label "Logistic Regression"
    """

    def test_status_200(self):
        """GET / harus mengembalikan status 200 OK."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)

    def test_mengandung_form_prediksi(self):
        """Halaman harus mengandung kedua form (manual + CSV)."""
        resp = self.client.get("/")
        html = resp.data.decode()
        self.assertIn("Prediksi Manual", html)
        self.assertIn("Prediksi via CSV", html)

    def test_input_field_lr_ada(self):
        """3 input numerik LR harus ada di halaman."""
        resp = self.client.get("/")
        html = resp.data.decode()
        for col in ["PageValues", "ExitRates", "ProductRelated_Duration"]:
            self.assertIn(f'name="{col}"', html)

    def test_dropdown_field_ada(self):
        """3 dropdown (Month, Browser, TrafficType) harus ada di halaman."""
        resp = self.client.get("/")
        html = resp.data.decode()
        for col in ["Month", "Browser", "TrafficType"]:
            self.assertIn(f'name="{col}"', html)

    def test_logistic_regression_label(self):
        """Label 'Logistic Regression' harus muncul di halaman."""
        resp = self.client.get("/")
        html = resp.data.decode()
        self.assertIn("Logistic Regression", html)


# ============================================================
# Test 8: Prediksi Manual (POST /predict) — LR Only
# ============================================================
class TestPrediksiManual(BaseTestCase):
    """
    Test endpoint POST /predict yang memproses form input manual.
    Hanya menggunakan model Logistic Regression (8 fitur).

    Test case:
    - Data "akan beli" → tampilkan "Akan Membeli"
    - Data "tidak beli" → tampilkan "Tidak Membeli"
    - Selalu LR, tidak pernah RF
    - Validasi error: ExitRates > 1, field kosong, nilai negatif
    - Nilai nol semua → tetap bisa prediksi (InputRequired, bukan DataRequired)
    - Model tidak siap → tampilkan pesan error
    - GET method → 405 Method Not Allowed
    """

    def test_prediksi_lr_akan_membeli(self):
        """Data dengan PageValues=50 dan ExitRates=0.01 harus diprediksi 'Akan Membeli'."""
        resp = self.client.post("/predict", data=DATA_AKAN_BELI)
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode()
        self.assertIn("Akan Membeli", html)
        self.assertIn("Logistic Regression", html)

    def test_prediksi_tidak_membeli(self):
        """Data dengan PageValues=0 dan ExitRates=0.20 harus diprediksi 'Tidak Membeli'."""
        resp = self.client.post("/predict", data=DATA_TIDAK_BELI)
        html = resp.data.decode()
        self.assertIn("Tidak Membeli", html)

    def test_prediksi_selalu_lr(self):
        """Hasil prediksi manual harus selalu LR, tidak boleh ada 'Random Forest'."""
        resp = self.client.post("/predict", data=DATA_AKAN_BELI)
        html = resp.data.decode()
        # Ambil bagian antara "Hasil Prediksi" dan "Prediksi via CSV"
        # agar tidak terkena teks RF di section CSV
        hasil_section = html.split("Hasil Prediksi")[1].split("Prediksi via CSV")[0]
        self.assertIn("Logistic Regression", hasil_section)
        self.assertNotIn("Random Forest", hasil_section)

    def test_validasi_wtforms_exit_rates_di_luar_rentang(self):
        """ExitRates=1.5 harus menampilkan pesan 'Error'."""
        data = DATA_AKAN_BELI.copy()
        data["ExitRates"] = "1.5"
        resp = self.client.post("/predict", data=data)
        html = resp.data.decode()
        self.assertIn("Error", html)

    def test_validasi_wtforms_field_kosong(self):
        """PageValues kosong harus menampilkan pesan 'Error'."""
        data = DATA_AKAN_BELI.copy()
        data["PageValues"] = ""
        resp = self.client.post("/predict", data=data)
        html = resp.data.decode()
        self.assertIn("Error", html)

    def test_validasi_wtforms_nilai_negatif(self):
        """PageValues=-5 harus menampilkan pesan 'Error'."""
        data = DATA_AKAN_BELI.copy()
        data["PageValues"] = "-5"
        resp = self.client.post("/predict", data=data)
        html = resp.data.decode()
        self.assertIn("Error", html)

    def test_nilai_nol_semua(self):
        """Semua field bernilai 0 harus tetap bisa diprediksi (tidak error)."""
        data = DATA_TIDAK_BELI.copy()
        data["PageValues"] = "0"
        data["ExitRates"] = "0"
        data["ProductRelated_Duration"] = "0"
        resp = self.client.post("/predict", data=data)
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode()
        self.assertTrue("Akan Membeli" in html or "Tidak Membeli" in html)

    def test_model_tidak_siap(self):
        """Jika model None, harus tampilkan pesan 'Model tidak siap'."""
        with patch("app.model_siap", return_value=False):
            resp = self.client.post("/predict", data=DATA_AKAN_BELI)
            html = resp.data.decode()
            self.assertIn("Model tidak siap", html)

    def test_get_method_tidak_diizinkan(self):
        """GET /predict harus mengembalikan 405 (hanya POST yang diizinkan)."""
        resp = self.client.get("/predict")
        self.assertEqual(resp.status_code, 405)


# ============================================================
# Test 9: Upload CSV (POST /upload) — RF / LR / Both
# ============================================================
class TestUploadCSV(BaseTestCase):
    """
    Test endpoint POST /upload yang memproses file CSV untuk prediksi batch.

    Mendukung 3 model: RF (25 fitur), LR (8 fitur), atau keduanya.

    Test case:
    - CSV valid dengan pilihan RF → tampilkan RF_Prediksi
    - CSV valid dengan pilihan LR → tampilkan LR_Prediksi
    - CSV valid dengan pilihan both → tampilkan kedua kolom prediksi
    - Ringkasan statistik (total, beli, tidak beli)
    - CSV dengan kolom tidak lengkap → error
    - Upload tanpa file → error
    - File bukan CSV → error
    - CSV 1 baris → tetap bisa diprediksi
    - CSV 100 baris → performa normal
    - CSV kosong (hanya header) → error
    - CSV dengan nilai non-numerik → error
    - CSV dengan NaN → error
    - File terlalu besar (> MAX_CONTENT_LENGTH) → error 413
    - Model tidak siap → error
    - GET method → 405
    """

    def _upload(self, csv_content, model_choice="rf"):
        """Helper: Upload CSV ke endpoint /upload."""
        data = {"file": buat_file_csv(csv_content), "model_choice": model_choice}
        return self.client.post(
            "/upload", data=data, content_type="multipart/form-data"
        )

    def test_upload_csv_valid_rf(self):
        """Upload CSV valid + pilih RF → tampilkan 'Hasil Prediksi CSV' dan 'RF_Prediksi'."""
        resp = self._upload(CSV_VALID, "rf")
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode()
        self.assertIn("Hasil Prediksi CSV", html)
        self.assertIn("RF_Prediksi", html)

    def test_upload_csv_valid_lr(self):
        """Upload CSV valid + pilih LR → tampilkan 'LR_Prediksi'."""
        resp = self._upload(CSV_VALID, "lr")
        html = resp.data.decode()
        self.assertIn("LR_Prediksi", html)

    def test_upload_csv_valid_both(self):
        """Upload CSV valid + pilih both → tampilkan RF_Prediksi DAN LR_Prediksi."""
        resp = self._upload(CSV_VALID, "both")
        html = resp.data.decode()
        self.assertIn("RF_Prediksi", html)
        self.assertIn("LR_Prediksi", html)

    def test_upload_csv_menampilkan_ringkasan(self):
        """Halaman hasil harus mengandung ringkasan dengan 'Total'."""
        resp = self._upload(CSV_VALID)
        html = resp.data.decode()
        self.assertIn("Total", html)

    def test_upload_csv_kolom_tidak_lengkap(self):
        """CSV dengan kolom salah harus menampilkan pesan 'tidak memiliki kolom'."""
        csv_salah = "KolomSalah1,KolomSalah2\n1,2\n"
        resp = self._upload(csv_salah)
        html = resp.data.decode()
        self.assertIn("tidak memiliki kolom", html)

    def test_upload_tanpa_file_menampilkan_error(self):
        """Upload tanpa memilih file harus menampilkan 'Error'."""
        resp = self.client.post("/upload", data={}, content_type="multipart/form-data")
        html = resp.data.decode()
        self.assertIn("Error", html)

    def test_upload_file_bukan_csv_menampilkan_error(self):
        """Upload file .txt (bukan .csv) harus menampilkan 'Error'."""
        data = {"file": (io.BytesIO(b"konten"), "test.txt")}
        resp = self.client.post(
            "/upload", data=data, content_type="multipart/form-data"
        )
        html = resp.data.decode()
        self.assertIn("Error", html)

    def test_upload_satu_baris_csv(self):
        """CSV dengan 1 baris data harus tetap bisa diprediksi."""
        csv_satu = f"{CSV_HEADER}\n{CSV_ROW_TIDAK}\n"
        resp = self._upload(csv_satu)
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode()
        self.assertIn("Hasil Prediksi CSV", html)

    def test_upload_csv_banyak_baris(self):
        """CSV dengan 100 baris harus bisa diprediksi dan menampilkan angka 100."""
        resp = self._upload(CSV_BANYAK_BARIS)
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode()
        self.assertIn("100", html)

    def test_upload_csv_kosong_hanya_header(self):
        """CSV yang hanya berisi header (tanpa data) harus menampilkan pesan 'kosong'."""
        resp = self._upload(CSV_KOSONG)
        html = resp.data.decode()
        self.assertIn("kosong", html)

    def test_upload_csv_non_numerik(self):
        """CSV dengan nilai 'abc' di kolom numerik harus menampilkan 'non-numerik'."""
        resp = self._upload(CSV_NON_NUMERIK)
        html = resp.data.decode()
        self.assertIn("non-numerik", html)

    def test_upload_csv_dengan_nan(self):
        """CSV dengan nilai kosong di kolom numerik harus menampilkan 'non-numerik'."""
        resp = self._upload(CSV_DENGAN_NAN)
        html = resp.data.decode()
        self.assertIn("non-numerik", html)

    def test_upload_file_terlalu_besar(self):
        """
        File yang melebihi MAX_CONTENT_LENGTH harus mengembalikan 413.

        Catatan: Harus menonaktifkan TESTING dan PROPAGATE_EXCEPTIONS
        agar Flask mengembalikan 413 alih-alih melempar exception.
        """
        original_limit = app.config["MAX_CONTENT_LENGTH"]
        original_testing = app.config.get("TESTING")
        original_propagate = app.config.get("PROPAGATE_EXCEPTIONS")
        app.config["MAX_CONTENT_LENGTH"] = 100  # Set batas kecil (100 bytes)
        app.config["TESTING"] = False
        app.config["PROPAGATE_EXCEPTIONS"] = False
        try:
            client = app.test_client()
            data = {"file": buat_file_csv(CSV_VALID)}
            resp = client.post("/upload", data=data, content_type="multipart/form-data")
            self.assertEqual(resp.status_code, 413)
            html = resp.data.decode()
            self.assertIn("melebihi batas", html)
        finally:
            # Kembalikan konfigurasi semula
            app.config["MAX_CONTENT_LENGTH"] = original_limit
            app.config["TESTING"] = original_testing
            app.config["PROPAGATE_EXCEPTIONS"] = original_propagate

    def test_model_tidak_siap_saat_upload(self):
        """Jika model None, upload CSV harus menampilkan 'Model tidak siap'."""
        with patch("app.model_siap", return_value=False):
            resp = self._upload(CSV_VALID)
            html = resp.data.decode()
            self.assertIn("Model tidak siap", html)

    def test_upload_csv_exception_tak_terduga(self):
        """Exception tak terduga saat proses CSV harus menampilkan pesan error."""
        with patch("app.pd.read_csv", side_effect=RuntimeError("unexpected")):
            resp = self._upload(CSV_VALID)
            html = resp.data.decode()
            self.assertIn("Terjadi kesalahan", html)

    def test_get_method_tidak_diizinkan(self):
        """GET /upload harus mengembalikan 405 (hanya POST yang diizinkan)."""
        resp = self.client.get("/upload")
        self.assertEqual(resp.status_code, 405)


# ============================================================
# Test 10: Memory Store CSV
# ============================================================
class TestMemoryStore(BaseTestCase):
    """
    Test penyimpanan CSV hasil prediksi di memori server (_csv_download_store).

    Mekanisme:
    - Upload CSV → simpan string CSV ke dict dengan UUID sebagai key
    - UUID disimpan di session cookie (csv_download_id)
    - Store di-clear sebelum setiap upload baru (hemat memori)
    """

    def _upload(self, csv_content, model_choice="rf"):
        """Helper: Upload CSV ke endpoint /upload."""
        data = {"file": buat_file_csv(csv_content), "model_choice": model_choice}
        return self.client.post(
            "/upload", data=data, content_type="multipart/form-data"
        )

    def test_upload_menyimpan_ke_memory_store(self):
        """Setelah upload, memory store harus berisi tepat 1 entry."""
        self._upload(CSV_VALID)
        self.assertEqual(len(_csv_download_store), 1)

    def test_upload_membersihkan_data_lama(self):
        """Upload kedua harus menghapus data dari upload pertama (tetap 1 entry)."""
        self._upload(CSV_VALID)
        self.assertEqual(len(_csv_download_store), 1)
        self._upload(CSV_VALID)
        self.assertEqual(len(_csv_download_store), 1)  # Tetap 1, bukan 2

    def test_memory_store_berisi_csv_rf(self):
        """CSV di store harus mengandung kolom RF_Prediksi dan RF_Kepercayaan."""
        self._upload(CSV_VALID, "rf")
        csv_data = list(_csv_download_store.values())[0]
        self.assertIn("RF_Prediksi", csv_data)
        self.assertIn("RF_Kepercayaan", csv_data)

    def test_memory_store_berisi_csv_both(self):
        """Mode 'both' harus menyimpan kolom RF dan LR di CSV."""
        self._upload(CSV_VALID, "both")
        csv_data = list(_csv_download_store.values())[0]
        self.assertIn("RF_Prediksi", csv_data)
        self.assertIn("LR_Prediksi", csv_data)

    def test_session_hanya_simpan_uuid(self):
        """Session cookie hanya menyimpan UUID (36 karakter), bukan data CSV."""
        self._upload(CSV_VALID)
        with self.client.session_transaction() as sess:
            download_id = sess.get("csv_download_id")
            self.assertIsNotNone(download_id)
            self.assertEqual(len(download_id), 36)  # Format UUID: 8-4-4-4-12


# ============================================================
# Test 11: Download CSV (GET /download-csv)
# ============================================================
class TestDownloadCSV(BaseTestCase):
    """
    Test endpoint GET /download-csv yang mengirim file CSV hasil prediksi.

    Alur normal: upload CSV → simpan ke store → download via UUID di session.
    Edge case: tanpa session, UUID invalid, download berulang.
    """

    def _upload(self, csv_content, model_choice="rf"):
        """Helper: Upload CSV ke endpoint /upload."""
        data = {"file": buat_file_csv(csv_content), "model_choice": model_choice}
        return self.client.post(
            "/upload", data=data, content_type="multipart/form-data"
        )

    def test_download_tanpa_session_redirect(self):
        """Download tanpa upload sebelumnya harus redirect (302) ke home."""
        resp = self.client.get("/download-csv")
        self.assertEqual(resp.status_code, 302)

    def test_download_dengan_id_invalid_redirect(self):
        """Download dengan UUID yang tidak ada di store harus redirect."""
        with self.client.session_transaction() as sess:
            sess["csv_download_id"] = "id-tidak-ada-di-store"
        resp = self.client.get("/download-csv")
        self.assertEqual(resp.status_code, 302)

    def test_download_setelah_upload(self):
        """Download setelah upload harus mengembalikan file CSV dengan header benar."""
        self._upload(CSV_VALID)
        resp = self.client.get("/download-csv")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/csv", resp.content_type)  # Content-Type: text/csv
        self.assertIn("attachment", resp.headers.get("Content-Disposition", ""))
        csv_text = resp.data.decode()
        self.assertIn("RF_Prediksi", csv_text)

    def test_download_csv_banyak_baris(self):
        """Download CSV 100 baris harus menghasilkan 101 baris (1 header + 100 data)."""
        self._upload(CSV_BANYAK_BARIS)
        resp = self.client.get("/download-csv")
        self.assertEqual(resp.status_code, 200)
        baris = resp.data.decode().strip().split("\n")
        self.assertEqual(len(baris), 101)  # 1 header + 100 data

    def test_download_berulang_kali(self):
        """Download berkali-kali harus mengembalikan data yang sama (idempotent)."""
        self._upload(CSV_VALID)
        resp1 = self.client.get("/download-csv")
        resp2 = self.client.get("/download-csv")
        self.assertEqual(resp1.data, resp2.data)


# ============================================================
# Test 12: CSRF Error Handling
# ============================================================
class TestCSRFErrorHandling(unittest.TestCase):
    """
    Test penanganan error saat CSRF diaktifkan.

    Saat WTF_CSRF_ENABLED = True, semua POST request tanpa token CSRF
    yang valid harus ditolak dan menampilkan pesan 'Error'.
    """

    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = True  # Aktifkan CSRF untuk test ini
        self.client = app.test_client()

    def tearDown(self):
        app.config["WTF_CSRF_ENABLED"] = False  # Kembalikan ke default

    def test_prediksi_manual_csrf_invalid(self):
        """POST /predict tanpa CSRF token harus menampilkan 'Error'."""
        resp = self.client.post("/predict", data=DATA_AKAN_BELI)
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode()
        self.assertIn("Error", html)

    def test_upload_csv_csrf_invalid(self):
        """POST /upload tanpa CSRF token harus menampilkan 'Error'."""
        data = {"file": buat_file_csv(CSV_VALID)}
        resp = self.client.post(
            "/upload", data=data, content_type="multipart/form-data"
        )
        html = resp.data.decode()
        self.assertIn("Error", html)


# ============================================================
# Test 13: Entry Point (fungsi jalankan_server)
# ============================================================
class TestEntryPoint(unittest.TestCase):
    """Test bahwa jalankan_server() memanggil app.run() dengan parameter yang benar."""

    def test_jalankan_server_memanggil_app_run(self):
        """jalankan_server() harus memanggil app.run(host='0.0.0.0', port=5000, debug=True)."""
        with patch.object(app, "run") as mock_run:
            jalankan_server()
            mock_run.assert_called_once_with(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    unittest.main()
