"""
End-to-End Test untuk Aplikasi ShopPredict (v2.0 — LR Manual + Dual CSV)
========================================================================
Test menggunakan Playwright untuk mensimulasikan interaksi browser nyata.
Berbeda dengan unit test (test_app.py) yang menguji fungsi secara terisolasi,
E2E test ini menguji alur pengguna lengkap dari browser ke server dan kembali.

Alur yang diuji (16 test case):
1. TestHalamanUtama (7 test)
   - Halaman dimuat dengan judul "ShopPredict"
   - Form prediksi manual terlihat (#prediksiForm)
   - Form upload CSV terlihat (#uploadForm)
   - 3 input numerik LR ada dan terlihat
   - 3 dropdown (Month, Browser, TrafficType) ada dan terlihat
   - Label "Logistic Regression" terlihat
   - Badge "ML v2.0" terlihat

2. TestPrediksiManual (3 test)
   - Input data "akan beli" → tampil "Akan Membeli" + "Logistic Regression"
   - Input data "tidak beli" → tampil "Tidak Membeli"
   - Submit tanpa isi form → tampil "Error"

3. TestUploadCSV (5 test)
   - Upload CSV + pilih RF → tampil "RF_Prediksi"
   - Upload CSV + pilih komparasi → tampil RF_Prediksi DAN LR_Prediksi
   - Jumlah baris tabel = 5 (sesuai test_data.csv)
   - Link "Download CSV" muncul setelah upload
   - Download file CSV → nama file "hasil_prediksi.csv" + berisi "RF_Prediksi"

4. TestErrorHandling (1 test)
   - Klik upload tanpa pilih file → tampil "Error"

Prasyarat:
    pip install playwright && playwright install chromium

Cara menjalankan:
    python -m pytest test_e2e.py -v

Catatan teknis:
- Server Flask dijalankan otomatis di port 5002 (berbeda dari dev port 5000)
- Browser Chromium dijalankan headless (tanpa jendela)
- Radio button menggunakan pattern "hidden peer", sehingga klik dilakukan
  via evaluate("e => e.click()") karena elemen tidak visible secara CSS
- Selector di-scope ke form ID (#prediksiForm / #uploadForm) untuk
  menghindari ambiguitas "resolved to 2 elements"
"""

import pytest
import subprocess
import time
import signal
import os
import sys

# Import Playwright — skip seluruh file jika tidak terinstall
try:
    from playwright.sync_api import sync_playwright, expect
except ImportError:
    pytest.skip(
        "Playwright tidak terinstall. Jalankan: pip install playwright && playwright install chromium",
        allow_module_level=True,
    )

# Path ke file CSV test (5 baris data, 17 kolom)
TEST_DATA_CSV = os.path.join(os.path.dirname(__file__), "test_data.csv")

# URL server Flask untuk testing (port 5002 agar tidak konflik dengan dev)
BASE_URL = "http://localhost:5002"


# ============================================================
# Fixtures: Setup server dan browser untuk semua test
# ============================================================


@pytest.fixture(scope="module")
def server():
    """
    Jalankan Flask server di background untuk E2E testing.

    - Scope "module": server dijalankan sekali untuk semua test dalam file
    - Port 5002 agar tidak konflik dengan development server (5000/5001)
    - Polling setiap 0.5 detik sampai server siap (max 10 detik)
    - Server dihentikan (SIGTERM) setelah semua test selesai
    """
    env = os.environ.copy()
    env["FLASK_ENV"] = "testing"

    # Jalankan Flask server sebagai subprocess
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from app import app; app.run(host='0.0.0.0', port=5002, debug=False)",
        ],
        cwd=os.path.dirname(__file__),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Tunggu server siap (polling max 20 kali × 0.5 detik = 10 detik)
    for _ in range(20):
        try:
            import urllib.request

            urllib.request.urlopen(BASE_URL)
            break  # Server sudah siap menerima request
        except Exception:
            time.sleep(0.5)
    else:
        # Jika setelah 10 detik server belum siap, gagalkan test
        proc.kill()
        raise RuntimeError("Flask server gagal start di port 5002")

    yield proc  # Server berjalan selama test berlangsung

    # Bersihkan: hentikan server setelah semua test selesai
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)


@pytest.fixture(scope="module")
def browser_context(server):
    """
    Buat browser context Playwright (scope: module).

    - Menggunakan Chromium headless (tanpa jendela browser)
    - Context di-share untuk semua test dalam module
    - Browser ditutup otomatis setelah semua test selesai
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        yield context
        context.close()
        browser.close()


@pytest.fixture
def page(browser_context):
    """
    Buat page (tab) baru untuk setiap test.

    - Scope default (function): setiap test mendapat page bersih
    - Page ditutup setelah setiap test selesai
    """
    page = browser_context.new_page()
    yield page
    page.close()


# ============================================================
# Helper: Isi form prediksi manual
# ============================================================


def _fill_form_manual(page, data):
    """
    Helper: Isi form prediksi manual LR (3 numerik + 3 dropdown).

    Langkah:
    1. Navigasi ke halaman utama
    2. Scope ke form #prediksiForm (hindari ambiguitas dengan form CSV)
    3. Isi 3 input numerik: PageValues, ExitRates, ProductRelated_Duration
    4. Pilih opsi dropdown: Month, Browser, TrafficType (jika ada di data)

    Parameter:
        page: Playwright page object
        data (dict): Data form dengan key = nama field, value = string nilai
    """
    page.goto(BASE_URL)
    form = page.locator("#prediksiForm")  # Scope ke form manual saja

    # Isi 3 input numerik
    form.locator('input[name="PageValues"]').fill(data.get("PageValues", "0"))
    form.locator('input[name="ExitRates"]').fill(data.get("ExitRates", "0"))
    form.locator('input[name="ProductRelated_Duration"]').fill(
        data.get("ProductRelated_Duration", "0")
    )

    # Pilih dropdown (jika key ada di data)
    if "Month" in data:
        form.locator('select[name="Month"]').select_option(data["Month"])
    if "Browser" in data:
        form.locator('select[name="Browser"]').select_option(data["Browser"])
    if "TrafficType" in data:
        form.locator('select[name="TrafficType"]').select_option(data["TrafficType"])


# ============================================================
# Data Test: Form manual untuk E2E
# ============================================================

# Skenario: Pengunjung yang akan membeli
# PageValues=50, ExitRates rendah, durasi produk lama, bulan November
FORM_AKAN_BELI = {
    "PageValues": "50",
    "ExitRates": "0.01",
    "ProductRelated_Duration": "200",
    "Month": "Nov",
    "Browser": "2",
    "TrafficType": "2",
}

# Skenario: Pengunjung yang TIDAK akan membeli
# PageValues=0, ExitRates tinggi, tidak melihat produk, bulan Februari
FORM_TIDAK_BELI = {
    "PageValues": "0",
    "ExitRates": "0.20",
    "ProductRelated_Duration": "0",
    "Month": "Feb",
    "Browser": "1",
    "TrafficType": "1",
}


# ============================================================
# Test 1: Halaman Utama — Elemen UI Terlihat
# ============================================================
class TestHalamanUtama:
    """
    Verifikasi halaman utama menampilkan semua elemen UI yang diperlukan.
    Setiap test membuka halaman baru (fresh page) dan mengecek
    keberadaan elemen tertentu.
    """

    def test_halaman_dimuat(self, page):
        """Judul halaman harus mengandung 'ShopPredict'."""
        page.goto(BASE_URL)
        assert "ShopPredict" in page.title()

    def test_form_prediksi_manual_ada(self, page):
        """Form prediksi manual harus terlihat di halaman."""
        page.goto(BASE_URL)
        assert page.locator("text=Prediksi Manual").is_visible()
        assert page.locator("#prediksiForm").is_visible()

    def test_form_upload_csv_ada(self, page):
        """Form upload CSV harus terlihat di halaman."""
        page.goto(BASE_URL)
        assert page.locator("text=Prediksi via CSV").is_visible()
        assert page.locator("#uploadForm").is_visible()

    def test_input_field_lr_ada(self, page):
        """3 input numerik LR (PageValues, ExitRates, ProductRelated_Duration) harus terlihat."""
        page.goto(BASE_URL)
        for field in ["PageValues", "ExitRates", "ProductRelated_Duration"]:
            assert page.locator(f'input[name="{field}"]').is_visible()

    def test_dropdown_field_ada(self, page):
        """3 dropdown (Month, Browser, TrafficType) harus terlihat."""
        page.goto(BASE_URL)
        for field in ["Month", "Browser", "TrafficType"]:
            assert page.locator(f'select[name="{field}"]').is_visible()

    def test_logistic_regression_label(self, page):
        """Label 'Logistic Regression' harus muncul di halaman."""
        page.goto(BASE_URL)
        assert page.locator("text=Logistic Regression").first.is_visible()

    def test_badge_v2(self, page):
        """Badge versi 'ML v2.0' harus terlihat di navbar."""
        page.goto(BASE_URL)
        assert page.locator("text=ML v2.0").is_visible()


# ============================================================
# Test 2: Prediksi Manual — Alur LR dari Browser
# ============================================================
class TestPrediksiManual:
    """
    Test alur prediksi manual Logistic Regression via browser.

    Alur: Isi form → klik "Prediksi Sekarang" → tunggu hasil → verifikasi.
    """

    def test_prediksi_akan_membeli(self, page):
        """
        Data dengan PageValues=50 + ExitRates=0.01 harus diprediksi "Akan Membeli".
        Hasil juga harus menampilkan label "Logistic Regression".
        """
        _fill_form_manual(page, FORM_AKAN_BELI)
        page.click('button:has-text("Prediksi Sekarang")')
        page.wait_for_selector("#hasil-prediksi")  # Tunggu elemen hasil muncul
        assert page.locator("text=Akan Membeli").is_visible()
        # .nth(1) karena teks "Logistic Regression" pertama ada di deskripsi form
        assert page.locator("text=Logistic Regression").nth(1).is_visible()

    def test_prediksi_tidak_membeli(self, page):
        """Data dengan PageValues=0 + ExitRates=0.20 harus diprediksi "Tidak Membeli"."""
        _fill_form_manual(page, FORM_TIDAK_BELI)
        page.click('button:has-text("Prediksi Sekarang")')
        page.wait_for_selector("#hasil-prediksi")
        assert page.locator("text=Tidak Membeli").is_visible()

    def test_validasi_field_kosong(self, page):
        """Submit form tanpa mengisi field apapun harus menampilkan "Error"."""
        page.goto(BASE_URL)
        page.click('button:has-text("Prediksi Sekarang")')
        page.wait_for_load_state("domcontentloaded")
        assert page.locator("text=Error").first.is_visible()


# ============================================================
# Test 3: Upload CSV — Alur Batch dari Browser
# ============================================================
class TestUploadCSV:
    """
    Test alur upload CSV dan verifikasi hasil di browser.

    Catatan penting tentang radio button "hidden peer":
    - Radio button model_choice menggunakan class="hidden peer"
    - Elemen tidak visible secara CSS, jadi page.click() gagal
    - Solusi: gunakan evaluate("e => e.click()") untuk klik via JavaScript
    - Selector di-scope ke #uploadForm untuk menghindari ambiguitas
    """

    def test_upload_csv_rf(self, page):
        """
        Upload CSV + pilih Random Forest:
        - Set file input ke test_data.csv
        - Klik radio RF via evaluate() (hidden peer workaround)
        - Klik "Upload dan Prediksi"
        - Verifikasi: "Hasil Prediksi CSV" dan "RF_Prediksi" muncul
        """
        page.goto(BASE_URL)
        form = page.locator("#uploadForm")  # Scope ke form CSV
        form.locator('input[name="file"]').set_input_files(TEST_DATA_CSV)
        # Klik radio via JS karena input hidden (class="hidden peer")
        form.locator('input[name="model_choice"][value="rf"]').evaluate(
            "e => e.click()"
        )
        form.locator('button:has-text("Upload dan Prediksi")').click()
        page.wait_for_selector("#hasil-csv", timeout=10000)
        assert page.locator("text=Hasil Prediksi CSV").is_visible()
        assert page.locator("text=RF_Prediksi").is_visible()

    def test_upload_csv_komparasi(self, page):
        """
        Upload CSV + pilih komparasi (both):
        - Pilih radio "both" untuk menampilkan kedua model
        - Verifikasi: RF_Prediksi DAN LR_Prediksi ada di HTML
        """
        page.goto(BASE_URL)
        form = page.locator("#uploadForm")
        form.locator('input[name="file"]').set_input_files(TEST_DATA_CSV)
        # Klik radio "both" via JS (hidden peer)
        form.locator('input[name="model_choice"][value="both"]').evaluate(
            "e => e.click()"
        )
        form.locator('button:has-text("Upload dan Prediksi")').click()
        page.wait_for_selector("#hasil-csv", timeout=10000)
        html = page.content()
        assert "RF_Prediksi" in html
        assert "LR_Prediksi" in html

    def test_upload_csv_jumlah_baris_benar(self, page):
        """Tabel hasil harus memiliki 5 baris (sesuai test_data.csv yang berisi 5 baris)."""
        page.goto(BASE_URL)
        page.set_input_files('input[name="file"]', TEST_DATA_CSV)
        page.click('button:has-text("Upload dan Prediksi")')
        page.wait_for_selector("#hasil-csv", timeout=10000)
        rows = page.locator("table tbody tr")
        assert rows.count() == 5

    def test_upload_csv_download_link_ada(self, page):
        """Setelah upload berhasil, link "Download CSV" harus muncul."""
        page.goto(BASE_URL)
        page.set_input_files('input[name="file"]', TEST_DATA_CSV)
        page.click('button:has-text("Upload dan Prediksi")')
        page.wait_for_selector("#hasil-csv", timeout=10000)
        assert page.locator('a:has-text("Download CSV")').is_visible()

    def test_download_csv_file(self, page):
        """
        Klik link "Download CSV" harus mengunduh file dengan:
        - Nama file: "hasil_prediksi.csv"
        - Isi file mengandung "RF_Prediksi"
        """
        page.goto(BASE_URL)
        page.set_input_files('input[name="file"]', TEST_DATA_CSV)
        page.click('button:has-text("Upload dan Prediksi")')
        page.wait_for_selector("#hasil-csv", timeout=10000)

        # Tangkap event download saat klik link
        with page.expect_download() as download_info:
            page.click('a:has-text("Download CSV")')
        download = download_info.value

        # Verifikasi nama file
        assert download.suggested_filename == "hasil_prediksi.csv"

        # Verifikasi isi file
        path = download.path()
        with open(path, "r") as f:
            content = f.read()
        assert "RF_Prediksi" in content


# ============================================================
# Test 4: Error Handling — Validasi di Browser
# ============================================================
class TestErrorHandling:
    """Test bahwa error ditampilkan dengan benar di browser."""

    def test_upload_tanpa_file_error(self, page):
        """Klik 'Upload dan Prediksi' tanpa memilih file harus menampilkan 'Error'."""
        page.goto(BASE_URL)
        page.click('button:has-text("Upload dan Prediksi")')
        page.wait_for_load_state("domcontentloaded")
        assert page.locator("text=Error").first.is_visible()
