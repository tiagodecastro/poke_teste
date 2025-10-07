# bot_pokemon.py
# Versão reconstruída: usa PostMessage (VK + scancode) para enviar teclas ao PROClient.
# Mantém templates, OCR (Otsu + PSM7), combos por espécie, special keywords, stats, hotkeys.

import argparse
import time
import random
import pathlib
import logging
import logging.handlers
import json
import threading
import queue
import re
import shutil
from typing import Tuple, Dict, List, Optional
from datetime import datetime
from collections import defaultdict, Counter, deque

import numpy as np
import cv2
import pytesseract
import keyboard
import win32gui
import win32con
import pywintypes
from mss import mss

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
def now_ts(): return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ---------------- Utilities for config and ROIs ----------------
def get_client_rect_by_title(title: str):
    hwnd = win32gui.FindWindow(None, title)
    if not hwnd: return None
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (left, top))
    right, bottom = win32gui.ClientToScreen(hwnd, (right, bottom))
    return (left, top, right-left, bottom-top)

def read_jsonc(path: pathlib.Path) -> Dict:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"//.*", "", text)
    return json.loads(text or "{}")

def load_config(path: pathlib.Path) -> 'Cfg':
    return Cfg(read_jsonc(path))

def _canon_species_key(k: str) -> str:
    s = k.lower().strip()
    s = re.sub(r"[^a-z]", "", s)
    return s

def roi_to_screen(roi: Tuple[int,int,int,int], title: str, absolute: bool):
    if absolute: return tuple(roi)
    rect = get_client_rect_by_title(title)
    if rect is None: return tuple(roi)
    x,y,_,_ = rect
    rx, ry, rw, rh = roi
    return (x+rx, y+ry, rw, rh)

def union_rect(a, b):
    ax,ay,aw,ah = a; bx,by,bw,bh = b
    x1 = min(ax, bx); y1 = min(ay, by)
    x2 = max(ax+aw, bx+bw); y2 = max(ay+ah, by+bh)
    return (x1, y1, x2-x1, y2-y1)

def _grab_abs(x, y, w, h):
    with mss() as sct:
        m = {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}
        im = np.asarray(sct.grab(m))[:, :, :3]
    return im

# ---------------- Config object ----------------
class Cfg:
    def __init__(self, d: Dict):
        g = d.get
        # copy keys
        for k, v in d.items():
            setattr(self, k, v)

        # defaults (mirror do ficheiro original)
        self.window_title = getattr(self, "window_title", "PROClient")
        self.roi_is_absolute = bool(getattr(self, "roi_is_absolute", True))

        self.battle_template_name = getattr(self, "battle_template_name", "default")
        self.thr_battle = float(getattr(self, "thr_battle", 0.90))
        self.battle_enter_consec = int(getattr(self, "battle_enter_consec", 2))
        self.battle_exit_consec  = int(getattr(self, "battle_exit_consec", 2))

        self.poll_interval_s = float(getattr(self, "poll_interval_s", 0.20))
        self.change_mse_thr  = float(getattr(self, "change_mse_thr", 15.0))

        # ROIs (keeps as lists/tuples)
        self.name_roi   = tuple(getattr(self, "name_roi",   [3420, 234, 360, 36]))
        self.window_roi = tuple(getattr(self, "window_roi", [3845, 650, 260, 210]))

        # Movement and timings
        self.move_base_s  = float(getattr(self, "move_base_s", 0.45))
        self.move_jitter_s = float(getattr(self, "move_jitter_s", 0.10))
        self.key_delay_s  = tuple(getattr(self, "key_delay_s", [0.45, 0.95]))

        self.encounter_delay_s     = tuple(getattr(self, "encounter_delay_s",     [1.0, 2.0]))
        self.wait_between_combos_s = tuple(getattr(self, "wait_between_combos_s", [8.0, 9.0]))
        self.post_combo2_verify_s  = tuple(getattr(self, "post_combo2_verify_s",  [1.0, 1.8]))

        # Catch guard
        self.catch_resolve_min_s   = tuple(getattr(self, "catch_resolve_min_s",   [3.2, 3.8]))
        self.retry_floor_s         = float(getattr(self, "retry_floor_s",         3.0))

        self.recover_s             = tuple(getattr(self, "recover_s",             [0.5, 1.0]))
        self.non_target_period_s   = tuple(getattr(self, "non_target_period_s",   [1.8, 2.3]))

        # directories and behavior
        self.tesseract_cmd = getattr(self, "tesseract_cmd", None)
        self.templates_dir = getattr(self, "templates_dir", "templates/battle")
        self.stats_dir = getattr(self, "stats_dir", "stats")
        self.mount_key = getattr(self, "mount_key", "null")
        self.non_target_key = getattr(self, "non_target_key", "4")
        self.patrol_keys = getattr(self, "patrol_keys", ["a","d"])
        self.dry_run = bool(getattr(self, "dry_run", False))
        self.stats_flush_s = float(getattr(self, "stats_flush_s", 5.0))

        # logging
        self.log_level = getattr(self, "log_level", "INFO")
        self.log_file = getattr(self, "log_file", "bot.log")
        self.log_max_bytes = int(getattr(self, "log_max_bytes", 1048576))
        self.log_backup_count = int(getattr(self, "log_backup_count", 3))

        # combos and targets
        self.combo_setup = getattr(self, "combo_setup", ["1","2"])
        self.combo_catch = getattr(self, "combo_catch", ["3","1"])
        self.target_combos = getattr(self, "target_combos", { "setup": ["1","2"], "catch": ["3","1"] })
        self.targets = getattr(self, "targets", {})
        self.special_keywords = getattr(self, "special_keywords", ["shiny","summer"])

# ---------------- Helper: robust tesseract detection ----------------
def detect_tesseract(cfg: Cfg) -> Optional[str]:
    p = shutil.which("tesseract")
    if p: return p
    if getattr(cfg, "tesseract_cmd", None):
        pth = pathlib.Path(cfg.tesseract_cmd)
        if pth.is_absolute():
            return str(pth)
        rel = pathlib.Path(SCRIPT_DIR) / pth
        if rel.exists(): return str(rel)
    for c in [r"Tesseract-OCR/tesseract.exe", r"Tesseract-OCR/tesseract.exe"]:
        rel = pathlib.Path(SCRIPT_DIR) / c
        if rel.exists(): return str(rel)
    for c in [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
        if pathlib.Path(c).exists(): return c
    return None

# ---------------- OCR helpers ----------------
TESS_LANG = "eng"
TESS_CFG_LINE = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -c preserve_interword_spaces=1"

def pre_otsu_line(gray: np.ndarray) -> np.ndarray:
    up = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
    up = cv2.medianBlur(up, 3)
    _, th = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def ocr_text_line(gray_name: np.ndarray) -> str:
    th = pre_otsu_line(gray_name)
    return pytesseract.image_to_string(th, lang=TESS_LANG, config=TESS_CFG_LINE)

def ocr_species(gray_name: np.ndarray) -> str:
    s = ocr_text_line(gray_name).lower().strip()
    m = re.search(r"\bwild\s*([a-z]{3,12})\b", s)
    if m: return _canon_species_key(m.group(1))
    s2 = re.sub(r"[^a-z]", "", s)
    m2 = re.fullmatch(r"wild([a-z]{3,12})", s2)
    return _canon_species_key(m2.group(1)) if m2 else ""

def normalize_for_match(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"^\s*wild(?=[a-z])", "wild ", s)
    s = re.sub(r"^\s*wild\s+", "wild ", s)
    s = re.sub(r"[^a-z\s-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def name_has_wild(text_norm: str) -> bool:
    return bool(re.search(r"\bwild\b", text_norm))

# ---------------- Template management ----------------
def save_battle_template(gray_window: np.ndarray, name: str, dir_path: pathlib.Path):
    path = dir_path / f"{name}.png"
    cv2.imwrite(str(path), gray_window)
    log.info(f"Template guardado: {path}")

def load_battle_template(name: str, dir_path: pathlib.Path) -> Optional[np.ndarray]:
    p = dir_path / f"{name}.png"
    if not p.exists(): return None
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return im

def match_template(gray_window: np.ndarray, tpl: Optional[np.ndarray]) -> Tuple[bool, float]:
    if tpl is None or gray_window is None: return False, 0.0
    res = cv2.matchTemplate(gray_window, tpl, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, _ = cv2.minMaxLoc(res)
    return maxv >= CFG.thr_battle, float(maxv)

# ---------------- Stats ----------------
def _fmt_dur(seconds: int) -> str:
    s = max(0, int(seconds)); h = s//3600; m=(s%3600)//60; sec=s%60
    return f"{h}:{m:02d}:{sec:02d}"

class SessionStats:
    def __init__(self, out_dir: pathlib.Path):
        self.start = time.time()
        self.count_total = 0
        self.caught_total = 0
        self.by_name = defaultdict(int)
        self.targets = 0
        self.special = 0
        self.non_target = 0
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._last_flush = 0.0
        self.file_json = self.out_dir / f"stats_{now_ts()}.json"
        self.file_txt  = self.out_dir / f"session_stats_{now_ts()}.txt"

    def update(self, name: str, category: Optional[str]=None, caught: Optional[bool]=None):
        self.count_total += 1
        if name: self.by_name[name] += 1
        if caught: self.caught_total += 1
        if category == "special": self.special += 1
        elif category == "target": self.targets += 1
        elif category == "non-target": self.non_target += 1
        self.write_snapshot()

    def snapshot(self) -> Dict:
        return {
            "started_at": datetime.fromtimestamp(self.start).isoformat(timespec="seconds"),
            "elapsed_s": int(time.time() - self.start),
            "encounters": self.count_total,
            "caught": self.caught_total,
            "by_name": dict(self.by_name),
            "targets": self.targets,
            "special": self.special,
            "non_target": self.non_target,
        }

    def write_snapshot(self, force: bool=False):
        if not force and time.time() - self._last_flush < CFG.stats_flush_s:
            return
        data = self.snapshot()
        self.file_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
        start_str = datetime.fromtimestamp(self.start).strftime("%Y-%m-%d %H:%M:%S")
        now_str   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        txt = (
            f"Início da sessão : {start_str}\n"
            f"Última atualização : {now_str}\n"
            f"Tempo: {_fmt_dur(data['elapsed_s'])}\n"
            f"Totais: {data['encounters']}\n"
            f"- Targets : {data['targets']}\n"
            f"- Special (shiny/summer) : {data['special']}\n"
            f"- Non-target: {data['non_target']}\n"
        )
        self.file_txt.write_text(txt, encoding="utf-8")
        self._last_flush = time.time()

# ---------------- OCR worker ----------------
class OCRWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2)
        self.last_species = deque(maxlen=3)
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            try:
                img = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            sp = ocr_species(img)
            if sp:
                self.last_species.append(sp)

    def submit(self, gray_img: np.ndarray):
        try:
            if not self.q.full():
                self.q.put_nowait(gray_img.copy())
        except queue.Full:
            pass

    def stable_species(self) -> str:
        if not self.last_species: return ""
        counts = Counter(self.last_species)
        return counts.most_common(1)[0][0]

    def reset(self):
        self.last_species.clear()
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass

# ---------------- Hotkeys ----------------
STOP_HK  = "f12"
PAUSE_HK = "f10"
SAVE_TPL_HK = "f8"
TEST_TPL_HK = "f9"

log = logging.getLogger("bot")

# ---------------- Key sending (PostMessage + scancode) ----------------
# Comprehensive mapping (VK, scancode). scancode values are typical set 1 for US layout.
# Add mappings as needed.
KEY_MAP = {
    # letters
    "a": (0x41, 0x1E),
    "b": (0x42, 0x30),
    "c": (0x43, 0x2E),
    "d": (0x44, 0x20),
    "e": (0x45, 0x12),
    "f": (0x46, 0x21),
    "g": (0x47, 0x22),
    "h": (0x48, 0x23),
    "i": (0x49, 0x17),
    "j": (0x4A, 0x24),
    "k": (0x4B, 0x25),
    "l": (0x4C, 0x26),
    "m": (0x4D, 0x32),
    "n": (0x4E, 0x31),
    "o": (0x4F, 0x18),
    "p": (0x50, 0x19),
    "q": (0x51, 0x10),
    "r": (0x52, 0x13),
    "s": (0x53, 0x1F),
    "t": (0x54, 0x14),
    "u": (0x55, 0x16),
    "v": (0x56, 0x2F),
    "w": (0x57, 0x11),
    "x": (0x58, 0x2D),
    "y": (0x59, 0x15),
    "z": (0x5A, 0x2C),

    # digits
    "0": (0x30, 0x0B),
    "1": (0x31, 0x02),
    "2": (0x32, 0x03),
    "3": (0x33, 0x04),
    "4": (0x34, 0x05),
    "5": (0x35, 0x06),
    "6": (0x36, 0x07),
    "7": (0x37, 0x08),
    "8": (0x38, 0x09),
    "9": (0x39, 0x0A),

    # arrows
    "left": (win32con.VK_LEFT, 0x4B),
    "right": (win32con.VK_RIGHT, 0x4D),
    "up": (win32con.VK_UP, 0x48),
    "down": (win32con.VK_DOWN, 0x50),

    # function keys
    "f1": (win32con.VK_F1, 0x3B),
    "f2": (win32con.VK_F2, 0x3C),
    "f3": (win32con.VK_F3, 0x3D),
    "f4": (win32con.VK_F4, 0x3E),
    "f5": (win32con.VK_F5, 0x3F),
    "f6": (win32con.VK_F6, 0x40),
    "f7": (win32con.VK_F7, 0x41),
    "f8": (win32con.VK_F8, 0x42),
    "f9": (win32con.VK_F9, 0x43),
    "f10": (win32con.VK_F10, 0x44),
    "f11": (win32con.VK_F11, 0x57),
    "f12": (win32con.VK_F12, 0x58),

    # others
    "enter": (win32con.VK_RETURN, 0x1C),
    "esc": (win32con.VK_ESCAPE, 0x01),
    "space": (win32con.VK_SPACE, 0x39),
    "tab": (win32con.VK_TAB, 0x0F),
    "shift": (win32con.VK_SHIFT, 0x2A),
    "ctrl": (win32con.VK_CONTROL, 0x1D),
    "alt": (win32con.VK_MENU, 0x38),
}

def _get_hwnd_for_cfg():
    hwnd = win32gui.FindWindow(None, CFG.window_title)
    if hwnd == 0:
        raise RuntimeError(f"Janela '{CFG.window_title}' não encontrada.")
    return hwnd

def _lparam_for_scancode(scan, extended=False, is_up=False):
    # Build lParam: repeat count (1) | scanCode<<16 | extended<<24 | context (0) | previous | transition
    l = 0x00000001 | (scan << 16)
    if extended:
        l |= (1 << 24)
    if is_up:
        l |= 0xC0000000  # previous state + transition state
    return l

def press_key_postmessage(key: str, hold: float = 0.08):
    """Send key to game window using PostMessage with scancode in lParam.
       key: like 'a', '1', 'f8', 'left', 'mount' etc (lowercase)
    """
    if CFG.dry_run:
        log.info(f"[DRY] press_key_postmessage {key} ({hold}s)")
        time.sleep(hold)
        return

    k = (key or "").lower()
    # If key is a single-char but mapping doesn't exist, try to map from CHAR
    if k not in KEY_MAP:
        log.warning(f"[KEY] '{k}' not in KEY_MAP; trying direct VK mapping...")
        # try direct VK via ord if single char
        if len(k) == 1:
            vk = ord(k.upper())
            scan = 0
            # best-effort: attempt PostMessage with VK only (may or may not be accepted)
            try:
                hwnd = _get_hwnd_for_cfg()
                win32gui.PostMessage(hwnd, win32con.WM_KEYDOWN, vk, 0)
                time.sleep(hold)
                win32gui.PostMessage(hwnd, win32con.WM_KEYUP, vk, 0)
                return
            except pywintypes.error as e:
                log.error(f"[KEY] PostMessage error (fallback VK) for '{k}': {e}")
                return
        else:
            log.error(f"[KEY] sem mapeamento e não é single char: {k}")
            return

    vk, scan = KEY_MAP[k]
    # extended? certain keys (right ctrl/alt, arrow numeric keypad) may need extended flag; keep False for most
    extended = False
    # Some keys often require extended flag (right-side modifier, numpad, arrow via keypad) — heuristics:
    if vk in (win32con.VK_RIGHT, win32con.VK_LEFT, win32con.VK_UP, win32con.VK_DOWN):
        extended = True

    try:
        hwnd = _get_hwnd_for_cfg()
        ldown = _lparam_for_scancode(scan, extended=extended, is_up=False)
        lup = _lparam_for_scancode(scan, extended=extended, is_up=True)
        win32gui.PostMessage(hwnd, win32con.WM_KEYDOWN, vk, ldown)
        time.sleep(hold)
        win32gui.PostMessage(hwnd, win32con.WM_KEYUP, vk, lup)
        # small safety sleep for game reaction
        time.sleep(0.02)
    except pywintypes.error as e:
        log.error(f"[KEY] PostMessage failed for '{key}': {e}. Executa o script como Administrador e confirma que '{CFG.window_title}' existe.")

# ---------------- Actions (replacing keyboard.* usage) ----------------
def press_with_jitter(key: str, base: float, jitter: float):
    dur = max(0.0, base + random.uniform(-jitter, jitter))
    if CFG.dry_run:
        log.info(f"[DRY] hold {key} {dur:.2f}s"); return
    press_key_postmessage(key, dur)

def press_combo(keys: List[str], key_delay: Tuple[float,float]):
    for k in keys:
        if CFG.dry_run:
            log.info(f"[DRY] tap {k}")
        else:
            press_key_postmessage(k, random.uniform(0.04, 0.12))
        time.sleep(random.uniform(*key_delay))

# ---------------- Battle detection & helpers (kept original logic) ----------------
_battle_enter_streak = 0
_battle_exit_streak  = 0
_last_active = False

def is_battle_active_from_template(gray_window_roi: np.ndarray) -> Tuple[bool, float]:
    if tpl_battle is None: return False, 0.0
    res = cv2.matchTemplate(gray_window_roi, tpl_battle, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, _ = cv2.minMaxLoc(res)
    return (maxv >= CFG.thr_battle), float(maxv)

def battle_active_instant() -> bool:
    gw, gn = capture_rois()
    active_tpl, _ = is_battle_active_from_template(gw)
    if active_tpl: return True
    text = normalize_for_match(ocr_text_line(gn))
    return name_has_wild(text)

def contains_special(text_norm: str) -> bool:
    return any(kw in text_norm for kw in SPECIAL_KEYWORDS)

def decide_target(species: str, is_special: bool):
    if is_special:
        return True, species or "", "special", TARGETS.get(species, {"setup": None, "catch": None})
    if not species:
        return False, "", "non-target", None
    if species in TARGETS:
        return True, species, "target", TARGETS[species]
    return False, "", "non-target", None

# ---------------- ROI capture functions ----------------
def compute_rects():
    cap_win = roi_to_screen(CFG.window_roi, CFG.window_title, CFG.roi_is_absolute)
    name_roi = roi_to_screen(CFG.name_roi, CFG.window_title, CFG.roi_is_absolute)
    uni = union_rect(cap_win, name_roi)
    return cap_win, name_roi, uni

def _crop_from_union(gray: np.ndarray, roi: Tuple[int,int,int,int], uni: Tuple[int,int,int,int], pad_y: int = 0):
    ux, uy, _, _ = uni
    x,y,w,h = roi
    y0 = max(0, y - uy - pad_y)
    y1 = min(gray.shape[0], y - uy + h + pad_y)
    x0 = max(0, x - ux)
    x1 = min(gray.shape[1], x - ux + w)
    return gray[y0:y1, x0:x1]

def _compute_rects_dynamic():
    cap_win = roi_to_screen(CFG.window_roi, CFG.window_title, CFG.roi_is_absolute)
    name_roi = roi_to_screen(CFG.name_roi, CFG.window_title, CFG.roi_is_absolute)
    uni = union_rect(cap_win, name_roi)
    return cap_win, name_roi, uni

def capture_rois():
    cap_win, name_roi, uni = _compute_rects_dynamic()
    frame = _grab_abs(*uni)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gw = _crop_from_union(gray, cap_win, uni, pad_y=0)
    gn = _crop_from_union(gray, name_roi, uni, pad_y=4)
    return gw, gn

# ---------------- Wait utilities ----------------
def wait_until_stable(max_wait: float, settle_ms: int = 450, poll: float = 0.10):
    deadline = time.perf_counter() + max(0.0, float(max_wait))
    last = None
    stable_for = 0.0
    while time.perf_counter() < deadline:
        gw, _ = capture_rois()
        if last is not None:
            mse = float(np.mean((last.astype(np.float32) - gw.astype(np.float32)) ** 2))
            if mse < CFG.change_mse_thr:
                stable_for += poll
                if (stable_for * 1000.0) >= settle_ms:
                    return
            else:
                stable_for = 0.0
        last = gw
        time.sleep(poll)

def wait_catch_resolution():
    """Espera após lançar a bola: mínimo + adaptativo; não spamma durante animação de lançamento."""
    min_wait = random.uniform(*CFG.catch_resolve_min_s)
    max_wait = max(CFG.wait_between_combos_s)
    t0 = time.perf_counter()
    # Espera mínima incondicional (evita spam durante animação de lançamento)
    while time.perf_counter() - t0 < min_wait:
        if not battle_active_instant():
            return
        time.sleep(0.10)
    # Depois da mínima, usa espera adaptativa até estabilizar ou bater no máximo
    remaining = max(0.0, max_wait - (time.perf_counter() - t0))
    if remaining > 0:
        wait_until_stable(remaining, settle_ms=600, poll=0.10)

# ---------------- Quick escape (verified) ----------------
def quick_escape(max_presses: int = 2, check_delay: float = 0.30):
    presses = 0
    while battle_active_instant() and presses < max_presses:
        if CFG.dry_run: log.info(f"[DRY] press {CFG.non_target_key}")
        else: press_key_postmessage(CFG.non_target_key, 0.08)
        presses += 1
        time.sleep(check_delay)
        if not battle_active_instant():
            break
    log.info(f"[NON-TARGET] Fuga {'OK' if not battle_active_instant() else 'ainda em batalha'} ({presses}x).")

# ---------------- Patrol loop (integrated original logic) ----------------
def patrol_loop():
    stop_ev = threading.Event()
    paused = False
    last_tpl_test = 0.0
    prev_in_battle = False
    sync_species_hint = ""
    last_catch_throw_t = 0.0

    def on_stop(_): log.info("Saída solicitada."); stop_ev.set()
    def on_pause(_):
        nonlocal paused
        paused = not paused
        log.info("PAUSADO" if paused else "RETOMADO")
    def on_save(_):
        global tpl_battle
        gw, _ = capture_rois()
        save_battle_template(gw, CFG.battle_template_name, TEMPLATES_DIR)
        tpl_battle = load_battle_template(CFG.battle_template_name, TEMPLATES_DIR)
    def on_test(_):
        nonlocal last_tpl_test
        if time.time()-last_tpl_test < 0.3: return
        gw, _ = capture_rois()
        ok, sc = is_battle_active_from_template(gw)
        log.info(f"[BATTLE TEST] score={sc:.3f} thr={CFG.thr_battle}")
        last_tpl_test = time.time()

    # hotkeys for control (local detection)
    keyboard.on_press_key(STOP_HK, on_stop)
    keyboard.on_press_key(PAUSE_HK, on_pause)
    keyboard.on_press_key(SAVE_TPL_HK, on_save)
    keyboard.on_press_key(TEST_TPL_HK, on_test)

    ocr = OCRWorker(); ocr.start()

    log.info(f"[Setup] Ativar mount com tecla '{CFG.mount_key}'")
    if CFG.dry_run: log.info(f"[DRY] press {CFG.mount_key}")
    else:
        # press mount key once on start
        press_key_postmessage(CFG.mount_key, 0.10)
        time.sleep(1.0)

    prev_gray_win = None
    cooldown_until = 0.0

    while not stop_ev.is_set():
        if paused:
            time.sleep(0.05); continue

        tick_t0 = time.perf_counter()
        gray_win, gray_name = capture_rois()

        active_tpl, _ = is_battle_active_from_template(gray_win)
        line_norm = normalize_for_match(ocr_text_line(gray_name)) if not active_tpl else ""
        in_battle = battle_fsm_step(active_tpl, line_norm)

        if in_battle and not prev_in_battle:
            ocr.reset()
            sync_species_hint = ""
            for _ in range(3):
                sp = ocr_species(gray_name)
                if sp:
                    sync_species_hint = sp
                    break
                time.sleep(0.12)

        if in_battle and time.time() >= cooldown_until:
            ocr.submit(gray_name)
            species_cand = ocr.stable_species() or sync_species_hint
            is_special = contains_special(line_norm)
            is_tgt, tag, category, per_spec = decide_target(species_cand, is_special)

            if is_tgt:
                label = "Alvo" if category == "target" else "SPECIAL"
                log.info(f"[ENCOUNTER] {label}: '{species_cand or '(desconhecido)'}'")
                time.sleep(random.uniform(*CFG.encounter_delay_s))

                setup_combo = (per_spec and per_spec.get("setup")) or CFG.combo_setup
                catch_combo = (per_spec and per_spec.get("catch")) or CFG.combo_catch

                # Setup -> aguardar estabilização do menu
                press_combo(setup_combo, CFG.key_delay_s)
                wait_until_stable(max(CFG.wait_between_combos_s))

                # Lançar bola (primeira tentativa)
                press_combo(catch_combo, CFG.key_delay_s)
                last_catch_throw_t = time.time()
                wait_catch_resolution()

                # Repetir apenas após floor + resolução adaptativa
                attempts = 1
                max_attempts = 6
                while battle_active_instant() and attempts < max_attempts:
                    # Floor para evitar spam durante animação
                    dt = time.time() - last_catch_throw_t
                    if dt < CFG.retry_floor_s:
                        time.sleep(CFG.retry_floor_s - dt)
                        if not battle_active_instant():
                            break
                    log.info(f"[CATCH] Batalha continua. Repetir combo de captura ({attempts}/{max_attempts-1}).")
                    press_combo(catch_combo, CFG.key_delay_s)
                    last_catch_throw_t = time.time()
                    wait_catch_resolution()
                    attempts += 1

                log.info("[RESOLVE] Batalha terminou — regressar a PATROL.")
                time.sleep(random.uniform(*CFG.recover_s))
                cooldown_until = time.time() + 1.0
                stats.update(species_cand or "", category=category)
            else:
                log.info(f"[ENCOUNTER] Não-alvo: '{species_cand or ''}'.")
                quick_escape()
                # fallback: manter a pressionar non_target_key até sair
                while battle_active_instant():
                    if CFG.dry_run: log.info(f"[DRY] press {CFG.non_target_key}")
                    else: press_key_postmessage(CFG.non_target_key, 0.08)
                    log.info("[NON-TARGET] Repetir fuga (fallback).")
                    time.sleep(random.uniform(*CFG.non_target_period_s))
                time.sleep(random.uniform(*CFG.recover_s))
                cooldown_until = time.time() + 1.0
                stats.update(species_cand or "", category="non-target")

        elif not in_battle:
            def move(k):
                press_with_jitter(k, CFG.move_base_s, CFG.move_jitter_s)
                if random.random() < 0.10:
                    time.sleep(random.uniform(0.05,0.12))
                    if CFG.dry_run: log.info(f"[DRY] tap {k}")
                    else: press_key_postmessage(k, 0.06)
            # patrol_keys default ["a","d"] but we use whatever in CFG.patrol_keys
            for k in CFG.patrol_keys:
                move(k)
            if random.random() < 0.03:
                time.sleep(random.uniform(0.02,0.05))

        if prev_gray_win is not None:
            mse = float(np.mean((prev_gray_win.astype(np.float32) - gray_win.astype(np.float32)) ** 2))
            if mse > CFG.change_mse_thr:
                _reset_fsm()
        prev_gray_win = gray_win
        prev_in_battle = in_battle

        dt = time.perf_counter() - tick_t0
        if dt < CFG.poll_interval_s:
            time.sleep(CFG.poll_interval_s - dt)

    ocr.stop_flag = True
    stats.write_snapshot(force=True)
    log.info("Loop terminado.")

# ---------------- FSM ----------------
def _reset_fsm():
    global _battle_enter_streak, _battle_exit_streak, _last_active
    _battle_enter_streak = 0; _battle_exit_streak = 0; _last_active = False

def battle_fsm_step(active_from_tpl: bool, name_text_norm: str) -> bool:
    global _battle_enter_streak, _battle_exit_streak, _last_active
    if not _last_active:
        if active_from_tpl:
            _battle_enter_streak += 1
            if _battle_enter_streak >= CFG.battle_enter_consec:
                _last_active = True; _battle_exit_streak = 0
        else:
            _battle_enter_streak = 0
    else:
        if (not active_from_tpl) and (not name_has_wild(name_text_norm)):
            _battle_exit_streak += 1
            if _battle_exit_streak >= CFG.battle_exit_consec:
                _last_active = False; _battle_enter_streak = 0
        else:
            _battle_exit_streak = 0
    return _last_active

# ---------------- Entry ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.jsonc")
    parser.add_argument("--template", type=str, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--accurate", action="store_true")
    parser.add_argument("--roi-overlay", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # load config
    CFG = load_config(pathlib.Path(args.config))

    # ensure dirs
    TEMPLATES_DIR = pathlib.Path(CFG.templates_dir); TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR = pathlib.Path(CFG.stats_dir); STATS_DIR.mkdir(parents=True, exist_ok=True)

    DRY_RUN = args.dry_run or CFG.dry_run
    CFG.dry_run = DRY_RUN

    # Logging por sessão (retenção 5)
    LOG_DIR = SCRIPT_DIR / "logs"; LOG_DIR.mkdir(parents=True, exist_ok=True)
    session_log = LOG_DIR / f"bot_{now_ts()}.log"

    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(session_log, encoding="utf-8")
    fh.setFormatter(fmt); fh.setLevel(getattr(logging, CFG.log_level.upper(), logging.INFO))
    ch = logging.StreamHandler(); ch.setFormatter(fmt); ch.setLevel(logging.INFO)
    log.setLevel(logging.DEBUG); log.addHandler(fh); log.addHandler(ch)

    # rotate old logs
    try:
        logs = sorted(LOG_DIR.glob("bot_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in logs[5:]:
            try: p.unlink()
            except Exception: pass
    except Exception:
        pass

    # tesseract path
    tpath = detect_tesseract(CFG)
    if tpath:
        pytesseract.pytesseract.tesseract_cmd = tpath
        log.info(f"Tesseract: {tpath}")
    else:
        log.warning("Tesseract não encontrado. Define 'tesseract_cmd' no config se necessário.")

    tpl_battle = load_battle_template(CFG.battle_template_name, TEMPLATES_DIR)
    if tpl_battle is None:
        log.warning("Template de batalha não encontrado. Usa F8 em batalha para gravar o primeiro.")

    SPECIAL_KEYWORDS = set(CFG.special_keywords)
    TARGETS = CFG.targets

    # stats manager
    stats = SessionStats(STATS_DIR)
    stats.write_snapshot(force=True)

    # preview compute rects once
    try:
        CAP_WIN, NAME_ROI_S, UNION_RECT = compute_rects()
        log.info(f"CAP_WIN={CAP_WIN} NAME_ROI={NAME_ROI_S} UNION={UNION_RECT}")
    except Exception as e:
        log.warning(f"Erro a computar rects iniciais: {e}")

    try:
        patrol_loop()
    except KeyboardInterrupt:
        stats.write_snapshot(force=True)
        log.info("Interrompido por teclado.")
    except Exception as e:
        log.exception(f"Erro fatal no main loop: {e}")
        stats.write_snapshot(force=True)
