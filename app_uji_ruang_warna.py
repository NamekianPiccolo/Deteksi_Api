import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTING_DIR = os.path.join(BASE_DIR, "testing")
os.makedirs(TESTING_DIR, exist_ok=True)

def pilih_file_gambar():
    """Membuka jendela pemilihan file native Windows Explorer"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title="Pilih Foto Api atau Benda Mirip Api yang Ingin Diuji (Kelompok 6)",
        filetypes=[
            ("Berkas Gambar", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("Semua Berkas", "*.*")
        ]
    )
    root.destroy()
    return file_path

def proses_dan_tampilkan(file_path):
    if not file_path or not os.path.exists(file_path):
        print("[!] Tidak ada berkas yang dipilih. Program selesai.")
        return

    nama_file_asli = os.path.basename(file_path)
    nama_tanpa_ext = os.path.splitext(nama_file_asli)[0]

    print("=" * 70)
    print(f"[PROSES] Membaca berkas: {file_path}")
    print("=" * 70)

    # 1. Baca Citra
    img_bgr = cv2.imread(file_path)
    if img_bgr is None:
        print("[ERROR] Gagal membaca gambar! Pastikan format berkas valid.")
        return

    # Resize tampilan agar pas di layar
    h_orig, w_orig = img_bgr.shape[:2]
    scale = min(1.0, 600.0 / max(h_orig, w_orig))
    if scale < 1.0:
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    total_pixels = h * w

    # 2. Konversi Ruang Warna (Materi P04)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    H = img_hsv[:, :, 0]
    Y = img_ycrcb[:, :, 0]
    Cr = img_ycrcb[:, :, 1]
    Cb = img_ycrcb[:, :, 2]
    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]

    # 3. Masking Warna Sederhana RGB vs Aturan P04
    mask_rgb = np.uint8((R > G) & (G > B) & (R > 140)) * 255
    mask_ycrcb = (Y > Cb) & (Cr > Cb) & (Cr >= 135)

    lower_hsv = np.array([0, 50, 100], dtype=np.uint8)
    upper_hsv = np.array([35, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv) > 0

    mask_p4 = np.uint8(mask_ycrcb & mask_hsv) * 255
    hasil_ekstraksi = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_p4)

    # 4. Hitung Statistik Piksel Api
    fire_pixels = np.count_nonzero(mask_p4)
    fire_percent = (fire_pixels / total_pixels) * 100
    status_deteksi = "TERDETEKSI API" if fire_percent > 0.5 else "BUKAN API (AMAN)"

    print("\n[HASIL ANALISIS PENGOLAHAN CITRA]:")
    print(f"  * Resolusi Uji            : {w} x {h} piksel")
    print(f"  * Jumlah Piksel Api       : {fire_pixels:,} piksel")
    print(f"  * Persentase Area Api     : {fire_percent:.2f}% dari seluruh gambar")
    print(f"  * Status Keputusan Warna  : [{status_deteksi}]")

    # 5. Susun Visualisasi Interaktif
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.canvas.manager.set_window_title(f"Hasil Uji Pertemuan 4 - {nama_file_asli}")
    fig.suptitle(f"Uji Pertemuan 4: {nama_file_asli} | Status: {status_deteksi} (Area: {fire_percent:.2f}%)", fontsize=15, fontweight='bold')

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("(1) Citra Input Asli (RGB)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(H, cmap='hsv')
    axes[0, 1].set_title("(2) Kanal Hue (Sudut Warna HSV)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(Cr, cmap='hot')
    axes[0, 2].set_title("(3) Kanal Cr (Krominans Merah YCbCr)")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(mask_rgb, cmap='gray')
    axes[1, 0].set_title("(4) Masker RGB Sederhana\n(Latar belakang ikut bocor)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mask_p4, cmap='gray')
    axes[1, 1].set_title(f"(5) Masker Gabungan YCbCr + HSV\n({fire_pixels:,} piksel / {fire_percent:.2f}%)", color='blue')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(hasil_ekstraksi)
    axes[1, 2].set_title("(6) Hasil Potongan Api Akhir\n(Latar belakang dibuang jadi hitam)", color='green')
    axes[1, 2].axis('off')

    plt.tight_layout()

    # 6. Simpan Hasil ke Folder TESTING
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_png = os.path.join(TESTING_DIR, f"uji_{nama_tanpa_ext}_{timestamp}.png")
    latest_png = os.path.join(TESTING_DIR, "hasil_test_terakhir.png")
    plt.savefig(output_png, dpi=200)
    plt.savefig(latest_png, dpi=200)

    # Catat Log ke CSV di folder testing
    log_csv = os.path.join(TESTING_DIR, "log_riwayat_testing.csv")
    log_data = [{
        "Waktu": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Nama_File": nama_file_asli,
        "Resolusi": f"{w}x{h}",
        "Piksel_Api": fire_pixels,
        "Persentase_Api": round(fire_percent, 2),
        "Keputusan": status_deteksi,
        "Hasil_Gambar": os.path.basename(output_png)
    }]
    df_log = pd.DataFrame(log_data)
    if os.path.exists(log_csv):
        df_log.to_csv(log_csv, mode='a', header=False, index=False)
    else:
        df_log.to_csv(log_csv, index=False)

    print("\n" + "=" * 70)
    print(f"[+] Gambar hasil uji otomatis disimpan ke folder 'testing/':")
    print(f"    -> testing/{os.path.basename(output_png)}")
    print(f"    -> testing/{os.path.basename(latest_png)}")
    print(f"[+] Riwayat tercatat di: testing/{os.path.basename(log_csv)}")
    print("=" * 70)
    print("[*] Jendela visualisasi interaktif sedang terbuka di layar Anda.")
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        target_file = sys.argv[1]
    else:
        print("\n[INFO] Membuka jendela Windows Explorer untuk memilih file gambar...")
        target_file = pilih_file_gambar()

    if target_file:
        proses_dan_tampilkan(target_file)
    else:
        print("[!] Tidak ada file yang dipilih.")
