# probe_ocr.py
# Lê a imagem gerada pelo debug_roi.py (stats/debug/name.png)
# e reporta exatamente o que o Tesseract lê, com confs.
# Guarda também os pré-processamentos:
#   stats/debug/probe_otsu_psm7.png, probe_otsu_psm8.png, probe_adapt_psm7.png, probe_adapt_psm8.png

import pathlib, re, shutil, sys
import cv2, numpy as np, pytesseract
from glob import glob

BASE = pathlib.Path(__file__).resolve().parent
DBG  = BASE/"stats"/"debug"
DBG.mkdir(parents=True, exist_ok=True)

TESS_LANG = "eng"
CFG_PSM7  = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -c preserve_interword_spaces=1"
CFG_PSM8  = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -c preserve_interword_spaces=1"

def detect_tesseract():
    p = shutil.which("tesseract")
    if p: return p
    # tentativas padrão no Windows
    for c in [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
        if pathlib.Path(c).exists(): return c
    return None

def pick_name_image() -> pathlib.Path:
    p = DBG/"name.png"
    if p.exists():
        return p
    # fallback: name_*.png mais recente
    cands = sorted([pathlib.Path(x) for x in glob(str(DBG/"name_*.png"))],
                   key=lambda f: f.stat().st_mtime, reverse=True)
    if cands:
        return cands[0]
    print("ERRO: não encontrei stats/debug/name.png. Corre primeiro: python debug_roi.py")
    sys.exit(1)

def pre_otsu(gray):
    up = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
    up = cv2.medianBlur(up, 3)
    _, th = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def pre_adapt(gray):
    up = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,2))
    up = clahe.apply(up)
    th = cv2.adaptiveThreshold(up,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5)
    return th

def clean_word(w): return re.sub(r"[^A-Za-z]", "", w)
def valid_token(tok): return bool(re.fullmatch(r"[A-Za-z]{3,12}", tok))

def after_wild(data):
    words = [clean_word(w) for w in data["text"]]
    confs = [int(c) if c!="-1" else -1 for c in data["conf"]]
    for i,w in enumerate(words):
        if w.lower()=="wild":
            for j in range(i+1,len(words)):
                tok = words[j]
                if valid_token(tok): return tok, confs[j]
            return "", -1
    return "", -1

def best_token(data):
    words = [clean_word(w) for w in data["text"]]
    confs = [int(c) if c!="-1" else -1 for c in data["conf"]]
    best,bc="",-1
    for w,c in zip(words,confs):
        w2=w
        m = re.fullmatch(r"wild([A-Za-z]{3,12})", w2)
        if m: w2 = m.group(1)
        if valid_token(w2) and c>bc:
            best,bc=w2,c
    return best,bc

def run_once(gray, pre_fn, cfg, tag):
    th = pre_fn(gray)
    out = DBG/f"probe_{tag}.png"
    cv2.imwrite(str(out), th)
    text = pytesseract.image_to_string(th, lang=TESS_LANG, config=cfg).strip()
    data = pytesseract.image_to_data(th, lang=TESS_LANG, config=cfg, output_type=pytesseract.Output.DICT)
    aw_tok, aw_conf = after_wild(data)
    bt_tok, bt_conf = best_token(data)
    return {"file": str(out), "text": text, "after_wild": (aw_tok.lower(), aw_conf), "best_token": (bt_tok.lower(), bt_conf)}

def show_block(title, res):
    aw,awc = res["after_wild"]; bt,btc = res["best_token"]
    print(f"\n=== {title} ===")
    print(f"preproc_png : {res['file']}")
    print(f"text_raw    : {repr(res['text'])}")
    print(f"after_wild  : {(aw or '')}  conf={awc}")
    print(f"best_token  : {(bt or '')}  conf={btc}")

def main():
    tpath = detect_tesseract()
    if tpath: pytesseract.pytesseract.tesseract_cmd = tpath
    print(f"[i] Tesseract: {tpath or 'N/D'}")

    img_path = pick_name_image()
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"ERRO: falha a ler {img_path}"); sys.exit(1)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    print(f"[i] Fonte: {img_path}")

    r1 = run_once(gray, pre_otsu,  CFG_PSM7, "otsu_psm7")
    r2 = run_once(gray, pre_otsu,  CFG_PSM8, "otsu_psm8")
    r3 = run_once(gray, pre_adapt, CFG_PSM7, "adapt_psm7")
    r4 = run_once(gray, pre_adapt, CFG_PSM8, "adapt_psm8")

    show_block("OTSU + PSM7", r1)
    show_block("OTSU + PSM8", r2)
    show_block("ADAPT + PSM7", r3)
    show_block("ADAPT + PSM8", r4)

    # Escolha simples do melhor candidato
    choices = [
        ("OTSU+8  best", r2["best_token"]),
        ("ADAPT+8 best", r4["best_token"]),
        ("OTSU+7  wild", r1["after_wild"]),
        ("ADAPT+7 wild", r3["after_wild"]),
    ]
    pick = max(choices, key=lambda kv: kv[1][1])
    print("\n--- SUGESTÃO ---")
    print(f"Melhor leitura: {pick[0]} → token='{pick[1][0]}' conf={pick[1][1]}")
    print("Abre os PNGs em stats/debug/ para veres o que o Tesseract está a ler.")

if __name__ == "__main__":
    main()
