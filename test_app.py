"""
Unit Test untuk Aplikasi Prediksi Niat Pembelian Online
=======================================================
Test suite lengkap yang mencakup:
- Pemuatan model machine learning
- Form WTForms (validasi manual dan CSV)
- Halaman utama (GET /)
- Prediksi manual via form (POST /predict)
- Prediksi massal via upload CSV (POST /upload)
- Download hasil CSV (GET /download-csv)
- Fungsi helper (model_siap, prediksi_dari_array)
- Edge case dan error handling

Target: 100% code coverage pada app.py dan forms.py
"""

import unittest
from unittest.mock import patch, MagicMock
import io
import os

# Pastikan working directory di root project agar file model ditemukan
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Impor komponen dari aplikasi utama
from app import (
    app,
    model,
    scaler,
    threshold,
    FEATURE_COLS,
    muat_model,
    buat_aplikasi,
    model_siap,
    prediksi_dari_array,
    jalankan_server,
)

# Impor form WTForms untuk testing validasi
from forms import FormPrediksiManual, FormUploadCSV

# ======================================
# Helper: Data form dan CSV untuk testing
# ======================================

# Data dengan probabilitas tinggi -> "Akan Membeli"
DATA_AKAN_BELI = {
    "BounceRates": "0.01",
    "Administrative_Duration": "30",
    "ProductRelated": "10",
    "ProductRelated_Duration": "200",
    "Administrative": "4",
    "ExitRates": "0.01",
    "PageValues": "50",
}

# Data dengan bounce/exit tinggi dan page values nol -> "Tidak Membeli"
DATA_TIDAK_BELI = {
    "BounceRates": "0.2",
    "Administrative_Duration": "0",
    "ProductRelated": "1",
    "ProductRelated_Duration": "0",
    "Administrative": "1",
    "ExitRates": "0.2",
    "PageValues": "0",
}

# Konten CSV valid untuk testing upload
CSV_VALID = (
    "BounceRates,Administrative_Duration,ProductRelated,"
    "ProductRelated_Duration,Administrative,ExitRates,PageValues\n"
    "0.02,80.0,1,0.0,1,0.02,0.0\n"
    "0.01,30.0,10,200.0,4,0.01,50.0\n"
)


def buat_file_csv(konten):
    """Helper: membuat tuple file CSV untuk test client multipart form."""
    return (io.BytesIO(konten.encode("utf-8")), "test.csv")


class BaseTestCase(unittest.TestCase):
    """Base class yang menyiapkan test client dan menonaktifkan CSRF."""

    def setUp(self):
        """Siapkan test client Flask dengan CSRF dinonaktifkan."""
        app.config["TESTING"] = True
        # Nonaktifkan CSRF protection untuk testing
        # (di production, CSRF aktif untuk keamanan)
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()


class TestMuatModel(unittest.TestCase):
    """Test fungsi muat_model() — memuat artefak dari file."""

    def test_model_berhasil_dimuat(self):
        """Model harus ter-load dan bukan None."""
        self.assertIsNotNone(model)

    def test_scaler_berhasil_dimuat(self):
        """Scaler harus ter-load dan bukan None."""
        self.assertIsNotNone(scaler)

    def test_threshold_berhasil_dimuat(self):
        """Threshold harus ter-load dan bertipe numerik."""
        self.assertIsNotNone(threshold)
        self.assertIsInstance(threshold, float)

    def test_muat_model_file_tidak_ada(self):
        """Jika file model tidak ada, kembalikan (None, None, None)."""
        with patch("app.joblib.load", side_effect=FileNotFoundError("not found")):
            m, s, t = muat_model()
            self.assertIsNone(m)
            self.assertIsNone(s)
            self.assertIsNone(t)


class TestBuatAplikasi(unittest.TestCase):
    """Test factory function buat_aplikasi()."""

    def test_mengembalikan_instance_flask(self):
        """Harus mengembalikan objek Flask."""
        from flask import Flask

        aplikasi = buat_aplikasi()
        self.assertIsInstance(aplikasi, Flask)

    def test_secret_key_terisi(self):
        """Secret key harus sudah dikonfigurasi."""
        aplikasi = buat_aplikasi()
        self.assertIsNotNone(aplikasi.secret_key)


class TestFiturCols(unittest.TestCase):
    """Test konstanta FEATURE_COLS."""

    def test_jumlah_kolom_fitur(self):
        """Harus ada tepat 7 kolom fitur."""
        self.assertEqual(len(FEATURE_COLS), 7)

    def test_kolom_bounce_rates_ada(self):
        """BounceRates harus ada dalam daftar fitur."""
        self.assertIn("BounceRates", FEATURE_COLS)


class TestModelSiap(unittest.TestCase):
    """Test fungsi helper model_siap()."""

    def test_model_siap_saat_semua_tersedia(self):
        """Harus True jika model, scaler, threshold semuanya ada."""
        self.assertTrue(model_siap())

    def test_model_tidak_siap_saat_model_none(self):
        """Harus False jika model adalah None."""
        with patch("app.model", None):
            self.assertFalse(model_siap())

    def test_model_tidak_siap_saat_scaler_none(self):
        """Harus False jika scaler adalah None."""
        with patch("app.scaler", None):
            self.assertFalse(model_siap())

    def test_model_tidak_siap_saat_threshold_none(self):
        """Harus False jika threshold adalah None."""
        with patch("app.threshold", None):
            self.assertFalse(model_siap())


class TestPrediksiDariArray(unittest.TestCase):
    """Test fungsi helper prediksi_dari_array()."""

    def test_mengembalikan_tuple(self):
        """Harus mengembalikan tuple (probabilitas, prediksi)."""
        import numpy as np

        fitur = np.array([[0.01, 30, 10, 200, 4, 0.01, 50]])
        prob, pred = prediksi_dari_array(fitur)
        self.assertEqual(len(prob), 1)
        self.assertEqual(len(pred), 1)

    def test_probabilitas_antara_0_dan_1(self):
        """Probabilitas harus dalam rentang 0-1."""
        import numpy as np

        fitur = np.array([[0.5, 50, 5, 100, 3, 0.05, 25]])
        prob, _ = prediksi_dari_array(fitur)
        self.assertGreaterEqual(prob[0], 0)
        self.assertLessEqual(prob[0], 1)

    def test_prediksi_bernilai_0_atau_1(self):
        """Prediksi harus berupa 0 atau 1."""
        import numpy as np

        fitur = np.array([[0.1, 20, 3, 50, 2, 0.1, 10]])
        _, pred = prediksi_dari_array(fitur)
        self.assertIn(pred[0], [0, 1])


class TestFormPrediksiManual(BaseTestCase):
    """Test WTForms FormPrediksiManual — validasi field."""

    def test_form_valid_lengkap(self):
        """Form dengan semua field valid harus lolos validasi."""
        with app.test_request_context("/predict", method="POST", data=DATA_AKAN_BELI):
            form = FormPrediksiManual()
            self.assertTrue(form.validate())

    def test_form_bounce_rates_di_luar_rentang(self):
        """Bounce Rates > 1 harus gagal validasi."""
        data = DATA_AKAN_BELI.copy()
        data["BounceRates"] = "1.5"
        with app.test_request_context("/predict", method="POST", data=data):
            form = FormPrediksiManual()
            self.assertFalse(form.validate())
            self.assertIn("BounceRates", form.errors)

    def test_form_exit_rates_negatif(self):
        """Exit Rates negatif harus gagal validasi."""
        data = DATA_AKAN_BELI.copy()
        data["ExitRates"] = "-0.5"
        with app.test_request_context("/predict", method="POST", data=data):
            form = FormPrediksiManual()
            self.assertFalse(form.validate())
            self.assertIn("ExitRates", form.errors)

    def test_form_field_kosong(self):
        """Field kosong harus gagal validasi."""
        data = DATA_AKAN_BELI.copy()
        data["PageValues"] = ""
        with app.test_request_context("/predict", method="POST", data=data):
            form = FormPrediksiManual()
            self.assertFalse(form.validate())

    def test_form_product_related_negatif(self):
        """ProductRelated negatif harus gagal validasi."""
        data = DATA_AKAN_BELI.copy()
        data["ProductRelated"] = "-5"
        with app.test_request_context("/predict", method="POST", data=data):
            form = FormPrediksiManual()
            self.assertFalse(form.validate())
            self.assertIn("ProductRelated", form.errors)

    def test_form_durasi_negatif(self):
        """Durasi negatif harus gagal validasi."""
        data = DATA_AKAN_BELI.copy()
        data["Administrative_Duration"] = "-10"
        with app.test_request_context("/predict", method="POST", data=data):
            form = FormPrediksiManual()
            self.assertFalse(form.validate())
            self.assertIn("Administrative_Duration", form.errors)


class TestFormUploadCSV(BaseTestCase):
    """Test validasi upload CSV (menggunakan request.files langsung)."""

    def test_upload_tanpa_file_menampilkan_error(self):
        """Request tanpa file harus menampilkan pesan error."""
        resp = self.client.post("/upload", data={}, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode()
        self.assertIn("wajib dipilih", html)

    def test_upload_nama_file_kosong_menampilkan_error(self):
        """File dengan nama kosong harus menampilkan pesan error."""
        data = {"file": (io.BytesIO(b""), "")}
        resp = self.client.post(
            "/upload", data=data, content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode()
        self.assertIn("wajib dipilih", html)

    def test_upload_file_bukan_csv_menampilkan_error(self):
        """File bukan CSV (.txt) harus menampilkan error validasi."""
        data = {"file": (io.BytesIO(b"konten"), "test.txt")}
        resp = self.client.post(
            "/upload", data=data, content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode()
        self.assertIn("Hanya file CSV", html)


class TestHalamanUtama(BaseTestCase):
    """Test GET / — merender halaman utama."""

    def test_status_200(self):
        """Halaman utama harus mengembalikan HTTP 200."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)

    def test_mengandung_form_prediksi(self):
        """Halaman harus memiliki section prediksi manual dan CSV."""
        resp = self.client.get("/")
        html = resp.data.decode()
        self.assertIn("Prediksi Manual", html)
        self.assertIn("Prediksi via CSV", html)

    def test_semua_input_field_ada(self):
        """Semua 7 input field fitur harus ada di form."""
        resp = self.client.get("/")
        html = resp.data.decode()
        for col in FEATURE_COLS:
            self.assertIn(f'name="{col}"', html)


class TestPrediksiManual(BaseTestCase):
    """Test POST /predict — prediksi dari form input."""

    def test_status_200(self):
        """Prediksi valid harus mengembalikan HTTP 200."""
        resp = self.client.post("/predict", data=DATA_AKAN_BELI)
        self.assertEqual(resp.status_code, 200)

    def test_prediksi_akan_membeli(self):
        """Data dengan page values tinggi harus menghasilkan 'Akan Membeli'."""
        resp = self.client.post("/predict", data=DATA_AKAN_BELI)
        html = resp.data.decode()
        self.assertIn("Akan Membeli", html)
        self.assertIn("Probabilitas", html)

    def test_prediksi_tidak_membeli(self):
        """Data dengan bounce rate tinggi harus menghasilkan 'Tidak Membeli'."""
        resp = self.client.post("/predict", data=DATA_TIDAK_BELI)
        html = resp.data.decode()
        self.assertIn("Tidak Membeli", html)

    def test_form_data_tetap_terisi(self):
        """Setelah prediksi, nilai input form harus tetap ditampilkan."""
        resp = self.client.post("/predict", data=DATA_AKAN_BELI)
        html = resp.data.decode()
        self.assertIn('value="50.0"', html)

    def test_validasi_wtforms_bounce_rates_di_luar_rentang(self):
        """WTForms harus menolak BounceRates > 1."""
        data = DATA_AKAN_BELI.copy()
        data["BounceRates"] = "1.5"
        resp = self.client.post("/predict", data=data)
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode()
        self.assertIn("Error", html)

    def test_validasi_wtforms_field_kosong(self):
        """WTForms harus menolak field yang kosong."""
        data = DATA_AKAN_BELI.copy()
        data["BounceRates"] = ""
        resp = self.client.post("/predict", data=data)
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode()
        self.assertIn("Error", html)

    def test_validasi_wtforms_nilai_negatif(self):
        """WTForms harus menolak ProductRelated negatif."""
        data = DATA_AKAN_BELI.copy()
        data["ProductRelated"] = "-5"
        resp = self.client.post("/predict", data=data)
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode()
        self.assertIn("Error", html)

    def test_nilai_nol_semua(self):
        """Semua fitur bernilai 0 harus tetap bisa diprediksi."""
        data_nol = {col: "0" for col in FEATURE_COLS}
        resp = self.client.post("/predict", data=data_nol)
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode()
        self.assertTrue("Akan Membeli" in html or "Tidak Membeli" in html)

    def test_nilai_besar(self):
        """Nilai fitur sangat besar harus tetap bisa diproses."""
        data_besar = {
            "BounceRates": "1",
            "Administrative_Duration": "99999",
            "ProductRelated": "999",
            "ProductRelated_Duration": "99999",
            "Administrative": "999",
            "ExitRates": "1",
            "PageValues": "99999",
        }
        resp = self.client.post("/predict", data=data_besar)
        self.assertEqual(resp.status_code, 200)

    def test_model_tidak_siap(self):
        """Jika model tidak siap, harus menampilkan error."""
        with patch("app.model_siap", return_value=False):
            resp = self.client.post("/predict", data=DATA_AKAN_BELI)
            self.assertEqual(resp.status_code, 200)
            html = resp.data.decode()
            self.assertIn("Model tidak siap", html)

    def test_get_method_tidak_diizinkan(self):
        """GET /predict harus mengembalikan 405 Method Not Allowed."""
        resp = self.client.get("/predict")
        self.assertEqual(resp.status_code, 405)


class TestUploadCSV(BaseTestCase):
    """Test POST /upload — prediksi massal dari file CSV."""

    def test_upload_csv_valid_status_200(self):
        """Upload CSV valid harus mengembalikan HTTP 200."""
        data = {"file": buat_file_csv(CSV_VALID)}
        resp = self.client.post(
            "/upload", data=data, content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)

    def test_upload_csv_menampilkan_tabel_hasil(self):
        """Upload CSV valid harus menampilkan tabel hasil prediksi."""
        data = {"file": buat_file_csv(CSV_VALID)}
        resp = self.client.post(
            "/upload", data=data, content_type="multipart/form-data"
        )
        html = resp.data.decode()
        self.assertIn("Hasil Prediksi CSV", html)
        self.assertIn("Total", html)

    def test_upload_csv_menampilkan_prediksi(self):
        """Hasil harus mengandung salah satu label prediksi."""
        data = {"file": buat_file_csv(CSV_VALID)}
        resp = self.client.post(
            "/upload", data=data, content_type="multipart/form-data"
        )
        html = resp.data.decode()
        self.assertTrue("Akan Membeli" in html or "Tidak Membeli" in html)

    def test_upload_csv_kolom_tidak_lengkap(self):
        """CSV dengan kolom salah harus menampilkan error kolom hilang."""
        csv_salah = "KolomSalah1,KolomSalah2\n1,2\n"
        data = {"file": buat_file_csv(csv_salah)}
        resp = self.client.post(
            "/upload", data=data, content_type="multipart/form-data"
        )
        html = resp.data.decode()
        self.assertIn("tidak memiliki kolom", html)

    def test_upload_satu_baris_csv(self):
        """CSV dengan 1 baris data harus bisa diproses."""
        csv_satu = (
            "BounceRates,Administrative_Duration,ProductRelated,"
            "ProductRelated_Duration,Administrative,ExitRates,PageValues\n"
            "0.05,60.0,3,30.0,2,0.03,10.0\n"
        )
        data = {"file": buat_file_csv(csv_satu)}
        resp = self.client.post(
            "/upload", data=data, content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode()
        self.assertIn("Hasil Prediksi CSV", html)

    def test_upload_csv_error_parsing(self):
        """CSV dengan konten corrupt harus menampilkan pesan error."""
        csv_corrupt = (
            "BounceRates,Administrative_Duration,ProductRelated,"
            "ProductRelated_Duration,Administrative,ExitRates,PageValues\n"
            "abc,def,ghi,jkl,mno,pqr,stu\n"
        )
        data = {"file": buat_file_csv(csv_corrupt)}
        resp = self.client.post(
            "/upload", data=data, content_type="multipart/form-data"
        )
        html = resp.data.decode()
        self.assertIn("Terjadi kesalahan", html)

    def test_model_tidak_siap_saat_upload(self):
        """Jika model tidak siap, upload harus menampilkan error."""
        with patch("app.model_siap", return_value=False):
            data = {"file": buat_file_csv(CSV_VALID)}
            resp = self.client.post(
                "/upload", data=data, content_type="multipart/form-data"
            )
            self.assertEqual(resp.status_code, 200)
            html = resp.data.decode()
            self.assertIn("Model tidak siap", html)

    def test_get_method_tidak_diizinkan(self):
        """GET /upload harus mengembalikan 405 Method Not Allowed."""
        resp = self.client.get("/upload")
        self.assertEqual(resp.status_code, 405)


class TestDownloadCSV(BaseTestCase):
    """Test GET /download-csv — download file hasil prediksi."""

    def test_download_tanpa_session_redirect(self):
        """Tanpa data di session, harus redirect ke halaman utama."""
        resp = self.client.get("/download-csv")
        self.assertEqual(resp.status_code, 302)

    def test_download_setelah_upload(self):
        """Setelah upload CSV, download harus mengembalikan file CSV."""
        # Langkah 1: Upload CSV untuk mengisi session
        data = {"file": buat_file_csv(CSV_VALID)}
        self.client.post("/upload", data=data, content_type="multipart/form-data")

        # Langkah 2: Download hasil
        resp = self.client.get("/download-csv")
        self.assertEqual(resp.status_code, 200)

        # Verifikasi header Content-Type dan Content-Disposition
        self.assertIn("text/csv", resp.content_type)
        self.assertIn("attachment", resp.headers.get("Content-Disposition", ""))

        # Verifikasi isi CSV mengandung kolom hasil prediksi
        csv_text = resp.data.decode()
        self.assertIn("Hasil_Prediksi", csv_text)
        self.assertIn("Probabilitas_Pembelian", csv_text)


class TestCSRFErrorHandling(unittest.TestCase):
    """Test penanganan error CSRF — memastikan csrf_token di-skip saat mengumpulkan error."""

    def setUp(self):
        """Siapkan test client dengan CSRF diaktifkan."""
        app.config["TESTING"] = True
        # Aktifkan CSRF agar error csrf_token muncul di form.errors
        app.config["WTF_CSRF_ENABLED"] = True
        self.client = app.test_client()

    def tearDown(self):
        """Kembalikan CSRF ke nonaktif untuk test lain."""
        app.config["WTF_CSRF_ENABLED"] = False

    def test_prediksi_manual_csrf_invalid(self):
        """POST /predict tanpa CSRF token harus menampilkan error validasi (bukan CSRF)."""
        resp = self.client.post("/predict", data=DATA_AKAN_BELI)
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode()
        # Harus ada Error, tapi bukan pesan tentang csrf_token
        self.assertIn("Error", html)

    def test_upload_csv_tanpa_csrf_tetap_berhasil(self):
        """POST /upload tanpa CSRF token harus tetap berhasil (upload tidak pakai WTForms)."""
        data = {"file": buat_file_csv(CSV_VALID)}
        resp = self.client.post(
            "/upload", data=data, content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode()
        self.assertIn("Hasil Prediksi CSV", html)


class TestEntryPoint(unittest.TestCase):
    """Test fungsi jalankan_server() dan blok __main__."""

    def test_jalankan_server_memanggil_app_run(self):
        """jalankan_server() harus memanggil app.run() dengan parameter benar."""
        with patch.object(app, "run") as mock_run:
            jalankan_server()
            mock_run.assert_called_once_with(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    unittest.main()
