# debug_roi.py
# Lê ROIs do config.jsonc. Converte REL->ABS pela posição atual da janela.
# Grava capturas em stats/debug/ para validação visual.

import json, re, pathlib
import numpy as np, cv2
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
    return sx, sy, r-l, b-t  # ABS x,y,w,h

def to_abs(roi, is_abs, client_xy):
    if is_abs: return tuple(roi)
    cx, cy, _, _ = client_xy
    x,y,w,h = roi
    return (cx+x, cy+y, w, h)

if __name__ == "__main__":
    cfg = load_cfg()
    name_roi = tuple(cfg["name_roi"])
    win_roi  = tuple(cfg["window_roi"])
    is_abs   = bool(cfg.get("roi_is_absolute", False))
    client   = client_abs("PROClient")

    name_abs = to_abs(name_roi, is_abs, client)
    win_abs  = to_abs(win_roi,  is_abs, client)

    print(f"CLIENTE ABS={client}")
    print(f"NAME  usado (ABS)={name_abs}")
    print(f"WIN   usado (ABS)={win_abs}")

    with mss() as sct:
        for tag, (x,y,w,h) in [("name",name_abs), ("win",win_abs)]:
            bgr = np.asarray(sct.grab({"left":x,"top":y,"width":w,"height":h}))[:,:,:3]
            path = OUT/f"{tag}.png"
            cv2.imwrite(str(path), bgr)
            print(f"gravado: {path}")
