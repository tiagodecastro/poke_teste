# debug_ocr.py
# Usa config.jsonc, resolve REL/ABS, testa Otsu vs Adaptive e mostra token+conf.

import json, re, pathlib, time, shutil
import numpy as np, cv2, pytesseract
from mss import mss
import win32gui

BASE = pathlib.Path(__file__).resolve().parent
OUT  = BASE/"stats"/"debug"; OUT.mkdir(parents=True, exist_ok=True)

def load_cfg():
    txt = (BASE/"config.jsonc").read_text(encoding="utf-8")
    txt = re.sub(r"//.*", "", txt)
    return json.loads(txt)

def client_abs(title="PROClient"):
    hwnd = win32gui.FindWindow(None, title)
    if not hwnd: raise SystemExit("Janela 'PROClient' não encontrada.")
    l,t,r,b = win32gui.GetClientRect(hwnd)
    sx, sy = win32gui.ClientToScreen(hwnd, (l, t))
    return sx, sy, r-l, b-t

def to_abs(roi, is_abs, client_xy):
    if is_abs: return tuple(roi)
    cx, cy, _, _ = client_xy
    x,y,w,h = roi
    return (cx+x, cy+y, w, h)

def detect_tesseract(cfg):
    p = shutil.which("tesseract")
    if p: return p
    p = cfg.get("tesseract_cmd")
    if p: return p
    for c in [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
        if pathlib.Path(c).exists(): return c
    return None

def pre_otsu(gray):
    up = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    up = cv2.medianBlur(up, 3)
    _, th = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

def pre_adapt(gray):
    up = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,2))
    up = clahe.apply(up)
    th = cv2.adaptiveThreshold(up,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5)
    return th

def species_from_data(th):
    data = pytesseract.image_to_data(
        th, lang="eng",
        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -c preserve_interword_spaces=1",
        output_type=pytesseract.Output.DICT
    )
    words = [re.sub(r"[^A-Za-z]","",w) for w in data["text"]]
    confs = [int(c) if c!="-1" else -1 for c in data["conf"]]
    # após "Wild"
    for i,w in enumerate(words):
        if w.lower()=="wild":
            for j in range(i+1,len(words)):
                w2=words[j].lower()
                if re.fullmatch(r"[a-z]{3,12}", w2):
                    return w2, confs[j]
            break
    # fallback: melhor token válido
    best,bc="",-1
    for w,c in zip(words,confs):
        w2=w.lower()
        if re.fullmatch(r"[a-z]{3,12}", w2) and c>bc:
            best,bc=w2,c
    return best,bc

if __name__=="__main__":
    cfg = load_cfg()
    tpath = detect_tesseract(cfg)
    if tpath: pytesseract.pytesseract.tesseract_cmd = tpath

    client = client_abs("PROClient")
    name_abs = to_abs(tuple(cfg["name_roi"]), bool(cfg.get("roi_is_absolute",False)), client)

    with mss() as sct:
        for i in range(5):
            x,y,w,h = name_abs
            bgr = np.asarray(sct.grab({"left":x,"top":y,"width":w,"height":h}))[:,:,:3]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            th1, th2 = pre_otsu(gray), pre_adapt(gray)
            s1,c1 = species_from_data(th1)
            s2,c2 = species_from_data(th2)
            sp,cf = (s1,c1) if c1>=c2 else (s2,c2)
            print(f"[{i}] Otsu=({s1},{c1})  Adapt=({s2},{c2})  => escolhido=({sp},{cf})")
            cv2.imwrite(str(OUT/f"ocr_otsu_{i}.png"), th1)
            cv2.imwrite(str(OUT/f"ocr_adapt_{i}.png"), th2)
            time.sleep(0.5)
    print("Feito. Vê stats/debug/")
