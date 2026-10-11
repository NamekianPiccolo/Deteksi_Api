import os
import sys
import hashlib
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "archive", "Fire-Detection")
IMG_SIZE = (128, 128)
RANDOM_SEED = 42
TEST_SIZE = 0.20

print("=" * 70)
print("[PIPELINE PENGOLAHAN CITRA DETEKSI API (KELOMPOK 6) - TAHAP 1 s/d 3]")
print("=" * 70)

# 1. Audit & Sanitasi Data
print("\n[1] Menjalankan Sanitasi Data (Cek Corrupt & Duplikat MD5)...")
seen_hashes = {}
dup_removed = 0
for root, _, files in sorted(os.walk(DATASET_DIR)):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            p = os.path.join(root, f)
            try:
                with Image.open(p) as img:
                    img.verify()
                if cv2.imread(p) is None: raise Exception()
                with open(p, 'rb') as f_obj:
                    h = hashlib.md5(f_obj.read()).hexdigest()
                if h in seen_hashes:
                    os.remove(p)
                    dup_removed += 1
                else:
                    seen_hashes[h] = p
            except Exception:
                os.remove(p)

print(f"  -> Total duplikat/rusak dibersihkan: {dup_removed}")

# Standarisasi ke RGB
for root, _, files in os.walk(DATASET_DIR):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            p = os.path.join(root, f)
            with Image.open(p) as img:
                if img.mode != 'RGB':
                    img.convert('RGB').save(p)

# Kunci Split 80:20
records = []
for root, _, files in sorted(os.walk(DATASET_DIR)):
    for f in sorted(files):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            p = os.path.join(root, f)
            label = 0 if os.path.basename(root) == '0' else 1
            records.append({
                "filepath": os.path.relpath(p, BASE_DIR),
                "filename": f,
                "label": label,
                "class_name": "Non-Fire" if label == 0 else "Fire"
            })

df_all = pd.DataFrame(records)
train_df, test_df = train_test_split(df_all, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df_all['label'])
train_df = train_df.copy(); test_df = test_df.copy()
train_df['split'] = 'train'; test_df['split'] = 'test'
df_split = pd.concat([train_df, test_df]).sort_index()
split_csv = os.path.join(BASE_DIR, "train_test_split.csv")
df_split.to_csv(split_csv, index=False)
print(f"  [+] Split disimpan di: {os.path.basename(split_csv)} (Train: {len(train_df)}, Test: {len(test_df)})")

# Ekstraksi Fitur
def segment_fire(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
    m_ycrcb = (Y > Cb) & (Cr > Cb) & (Cr >= 135)
    m_hsv = cv2.inRange(hsv, np.array([0, 50, 100]), np.array([35, 255, 255])) > 0
    comb = np.uint8(m_ycrcb & m_hsv) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(cv2.morphologyEx(comb, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

def extract_feat(rel_p):
    p = os.path.join(BASE_DIR, rel_p)
    img = cv2.imread(p)
    if img is None: return None
    r = cv2.resize(img, IMG_SIZE)
    h, w, _ = r.shape
    mask = segment_fire(r)
    f_pix = np.count_nonzero(mask)
    ratio = f_pix / (h * w)
    edges = cv2.Canny(r, 100, 200)
    e_in_f = cv2.bitwise_and(edges, edges, mask=mask)
    dens = np.count_nonzero(e_in_f) / (f_pix + 1e-6)
    hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
    h_m, s_m, v_m = np.mean(hsv, axis=(0, 1))
    h_s, s_s, v_s = np.std(hsv, axis=(0, 1))
    hh = cv2.calcHist([hsv], [0], mask if f_pix > 0 else None, [16], [0, 180]).flatten()
    hs = cv2.calcHist([hsv], [1], mask if f_pix > 0 else None, [16], [0, 256]).flatten()
    hv = cv2.calcHist([hsv], [2], mask if f_pix > 0 else None, [16], [0, 256]).flatten()
    hh = hh / (np.sum(hh) + 1e-6); hs = hs / (np.sum(hs) + 1e-6); hv = hv / (np.sum(hv) + 1e-6)
    return np.hstack([[ratio, dens], [h_m, s_m, v_m, h_s, s_s, v_s], hh, hs, hv])

print("\n[2] Mengekstrak fitur dataset...")
X_tr = np.array([extract_feat(r['filepath']) for _, r in train_df.iterrows()], dtype=np.float32)
y_tr = train_df['label'].values
X_te = np.array([extract_feat(r['filepath']) for _, r in test_df.iterrows()], dtype=np.float32)
y_te = test_df['label'].values

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

np.save(os.path.join(BASE_DIR, "X_train.npy"), X_tr_s)
np.save(os.path.join(BASE_DIR, "y_train.npy"), y_tr)
np.save(os.path.join(BASE_DIR, "X_test.npy"), X_te_s)
np.save(os.path.join(BASE_DIR, "y_test.npy"), y_te)

print("[SELESAI] Fitur tersimpan di file .npy!")
