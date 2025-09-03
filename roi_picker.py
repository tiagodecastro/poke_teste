# roi_picker.py
# Assistente para ROIs:
#  - Marca 2 ROIs: NAME (nameplate "Wild …") e BATTLE (menu de batalha)
#  - F3: topo-esquerda | F6: baixo-direita
#  - Gera snippet JSONC para colar no config.jsonc
#  - OPCIONAL: --write-config escreve no config (perdes comentários)
#  - OPCIONAL: guarda PNG dos recortes
#
# Uso:
#   python roi_picker.py
#   python roi_picker.py --config config.jsonc --write-config

import time, ctypes, pathlib, os, sys, json, re, argparse, shutil
from ctypes import wintypes
import win32gui
from mss import mss
import numpy as np
import cv2
import keyboard

WINDOW_TITLE = "PROClient"
BASE = pathlib.Path(__file__).resolve().parent
OUT_DIR = BASE / "templates"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ---------- Clipboard seguro ----------
def copy_to_clipboard(text: str) -> bool:
    CF_UNICODETEXT = 13
    GMEM_MOVEABLE  = 0x0002
    GMEM_ZEROINIT  = 0x0040
    user32 = ctypes.windll.user32
    k32    = ctypes.windll.kernel32
    if not user32.OpenClipboard(None):
        return False
    try:
        user32.EmptyClipboard()
        data = text + "\0"
        nbytes = len(data) * ctypes.sizeof(ctypes.c_wchar)
        hglob = k32.GlobalAlloc(GMEM_MOVEABLE | GMEM_ZEROINIT, nbytes)
        if not hglob: return False
        lp = k32.GlobalLock(hglob)
        if not lp:
            k32.GlobalFree(hglob); return False
        try:
            ctypes.memmove(lp, ctypes.create_unicode_buffer(data), nbytes)
        finally:
            k32.GlobalUnlock(hglob)
        if not user32.SetClipboardData(CF_UNICODETEXT, hglob):
            return False
        return True
    finally:
        user32.CloseClipboard()

# ---------- Window / mouse ----------
pt = wintypes.POINT()
def get_cursor_pos():
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

def get_client_rect():
    hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
    if not hwnd:
        raise RuntimeError(f"Janela '{WINDOW_TITLE}' não encontrada")
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    sx, sy = win32gui.ClientToScreen(hwnd, (left, top))
    return hwnd, (sx, sy, right - left, bottom - top)  # ABS x,y,w,h

def grab(x, y, w, h):
    with mss() as sct:
        mon = {"left": x, "top": y, "width": w, "height": h}
        im = np.asarray(sct.grab(mon))[:, :, :3]
    return im

def ensure_png_path(name: str) -> pathlib.Path:
    p = pathlib.Path(name.strip())
    if not p.suffix:
        p = p.with_suffix(".png")
    if not p.is_absolute():
        if not str(p).startswith(str(OUT_DIR.name + os.sep)):
            p = OUT_DIR / p
        else:
            p = BASE / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def save_png(path: pathlib.Path, bgr: np.ndarray):
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"cv2.imwrite falhou para: {path}")
    print(f"[+] PNG salvo: {path.resolve()}")

def select_roi(label: str, client_xy):
    print(f"\n=== Selecionar ROI: {label} ===")
    print("Coloca o cursor no canto SUPERIOR-ESQUERDO e prime F3.")
    tl = None; br = None
    while tl is None:
        if keyboard.is_pressed("f3"):
            tl = get_cursor_pos(); print(f"TL={tl}"); time.sleep(0.2)
    print("Agora no canto INFERIOR-DIREITO e prime F6.")
    while br is None:
        if keyboard.is_pressed("f6"):
            br = get_cursor_pos(); print(f"BR={br}"); time.sleep(0.2)
    x1,y1 = tl; x2,y2 = br
    x,y,w,h = min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1)
    cx,cy,_,_ = client_xy
    abs_roi = (x,y,w,h)
    rel_roi = (x-cx, y-cy, w, h)
    print(f"[{label}] ABS={abs_roi} | REL={rel_roi}")
    return abs_roi, rel_roi

def load_jsonc(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    txt = path.read_text(encoding="utf-8")
    txt = re.sub(r"//.*", "", txt)
    return json.loads(txt or "{}")

def write_jsonc(path: pathlib.Path, data: dict):
    # AVISO: perde comentários
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[i] Escrito: {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.jsonc")
    ap.add_argument("--write-config", action="store_true",
                    help="Atualiza o config.jsonc (ATENÇÃO: remove comentários).")
    ap.add_argument("--save-pngs", action="store_true",
                    help="Guarda PNGs dos recortes em templates/")
    args = ap.parse_args()

    print("ROI Wizard — F3: TOP-LEFT, F6: BOTTOM-RIGHT, ESC: abortar")
    try:
        hwnd, client = get_client_rect()
        print(f"[i] Cliente '{WINDOW_TITLE}': ABS={client}")
    except Exception as e:
        print(f"[!] {e}")
        return

    # 1) NAME_ROI
    name_abs, name_rel = select_roi("NAME", client)
    # 2) BATTLE_ROI
    battle_abs, battle_rel = select_roi("BATTLE", client)

    if args.save_pngs:
        name_img = grab(*name_abs)
        battle_img = grab(*battle_abs)
        save_png(ensure_png_path("battle/name_debug.png"), name_img)
        save_png(ensure_png_path("battle/menu_debug.png"), battle_img)

    snippet_abs = (
        f'"roi_is_absolute": true,\n'
        f'"name_roi": [{name_abs[0]},{name_abs[1]},{name_abs[2]},{name_abs[3]}],\n'
        f'"window_roi": [{battle_abs[0]},{battle_abs[1]},{battle_abs[2]},{battle_abs[3]}]'
    )
    snippet_rel = (
        f'"roi_is_absolute": false,\n'
        f'"name_roi": [{name_rel[0]},{name_rel[1]},{name_rel[2]},{name_rel[3]}],\n'
        f'"window_roi": [{battle_rel[0]},{battle_rel[1]},{battle_rel[2]},{battle_rel[3]}]'
    )

    print("\n=== SNIPPET ABSOLUTO (cola no config.jsonc) ===")
    print(snippet_abs)
    print("\n=== SNIPPET RELATIVO (cola no config.jsonc) ===")
    print(snippet_rel)

    copied = copy_to_clipboard(snippet_abs + "\n\n" + snippet_rel)
    print(f"[i] Snippets {'copiados para o clipboard' if copied else 'não copiados; copia manualmente'}.")

    # Escrita opcional no config
    cfg_path = BASE / args.config
    if args.write_config:
        cfg = load_jsonc(cfg_path)
        # por defeito escrevemos ABS
        cfg["roi_is_absolute"] = True
        cfg["name_roi"]   = list(name_abs)
        cfg["window_roi"] = list(battle_abs)
        write_jsonc(cfg_path, cfg)
        print("[i] Atualizado com ABS. Se preferires REL, volta a correr com --write-config-rel (não implementado) ou edita manualmente.")

    print("\nFeito. Valida com debug_roi.py e debug_ocr.py.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAbortado.")
