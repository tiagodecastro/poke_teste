# bot_pokemon.py
# Patrulha A/D, batalha por template, OCR do nome via Otsu+PSM7.
# Suporta 'special' (shiny/summer) e combos por espécie (setup/catch).
# F8 grava template, F9 testa template, F10 pausa, F12 sai.
# Atualizações:
# - Catch guard: após lançar a bola, espera mínima + adaptativa; retry só após floor.
# - Fuga verificada rápida (2x) + fallback.
# - Log por sessão (retenção máx. 5).
# - ROIs recalculados dinamicamente (quando relativos à janela).

import argparse, time, random, pathlib, logging, json, threading, queue, re, shutil
from typing import Tuple, Dict, List, Optional
from datetime import datetime
from collections import defaultdict, Counter, deque

import numpy as np
import cv2
import pytesseract
import keyboard
import win32gui
from mss import mss

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
def now_ts(): return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

 # ---------------- Config ----------------
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
    if absolute: return roi
    rect = get_client_rect_by_title(title)
    if rect is None: return roi
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
        m = {"left": x, "top": y, "width": w, "height": h}
        im = np.asarray(sct.grab(m))[:, :, :3]
    return im

class Cfg:
    def __init__(self, d: Dict):
        g = d.get
        self.window_title = g("window_title", "PROClient")
        self.roi_is_absolute = bool(g("roi_is_absolute", True))

        self.battle_template_name = g("battle_template_name", "default")
        self.thr_battle = float(g("thr_battle", 0.90))
        self.battle_enter_consec = int(g("battle_enter_consec", 2))
        self.battle_exit_consec  = int(g("battle_exit_consec", 2))

        self.poll_interval_s = float(g("poll_interval_s", 0.20))
        self.change_mse_thr  = float(g("change_mse_thr", 15.0))

        # ROIs
        self.name_roi   = tuple(g("name_roi",   [3420, 234, 360, 36]))
        self.window_roi = tuple(g("window_roi", [3845, 650, 260, 210]))

        # Movimento
        self.move_base_s  = float(g("move_base_s", 0.45))
        self.move_jitter_s = float(g("move_jitter_s", 0.10))
        self.key_delay_s  = tuple(g("key_delay_s", [0.45, 0.95]))

        # Timings
        self.encounter_delay_s     = tuple(g("encounter_delay_s",     [1.0, 2.0]))
        self.wait_between_combos_s = tuple(g("wait_between_combos_s", [8.0, 9.0]))   # teto adaptativo
        self.post_combo2_verify_s  = tuple(g("post_combo2_verify_s",  [1.0, 1.8]))   # ainda usado noutros fluxos

        # Catch guard (NOVO)
        self.catch_resolve_min_s   = tuple(g("catch_resolve_min_s",   [3.2, 3.8]))   # mínimo após lançar bola
        self.retry_floor_s         = float(g("retry_floor_s",         3.0))          # não re-tentar antes disto

        self.recover_s             = tuple(g("recover_s",             [0.5, 1.0]))
        self.non_target_period_s   = tuple(g("non_target_period_s",   [1.8, 2.3]))


def _to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

# ---------------- OCR ----------------
TESS_LANG = "eng"
TESS_CFG_LINE = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -c preserve_interword_spaces=1"

def detect_tesseract(cfg: Cfg) -> Optional[str]:
    p = shutil.which("tesseract")
    if p: return p
    if cfg.tesseract_cmd:
        pth = pathlib.Path(cfg.tesseract_cmd)
        if pth.is_absolute():
            return str(pth)
        # Se for relativo, resolve a partir do diretório do projeto
        rel = pathlib.Path(SCRIPT_DIR) / pth
        if rel.exists(): return str(rel)
    for c in [r"Tesseract-OCR/tesseract.exe", r"Tesseract-OCR/tesseract.exe"]:
        # Tenta relativo ao projeto
        rel = pathlib.Path(SCRIPT_DIR) / c
        if rel.exists(): return str(rel)
    # Por fim, tenta nos locais padrão do Windows
    for c in [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
        if pathlib.Path(c).exists(): return c
    return None

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

# ---------------- Template ----------------
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

# ---------------- Ações ----------------
def press_with_jitter(key: str, base: float, jitter: float):
    dur = max(0.0, base + random.uniform(-jitter, jitter))
    if DRY_RUN:
        log.info(f"[DRY] hold {key} {dur:.2f}s"); return
    keyboard.press(key); time.sleep(dur); keyboard.release(key)

def press_combo(keys: List[str], key_delay: Tuple[float,float]):
    for k in keys:
        if DRY_RUN: log.info(f"[DRY] tap {k}")
        else: keyboard.press_and_release(k)
        time.sleep(random.uniform(*key_delay))

# ---------------- Estado de batalha ----------------
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

# ---------- Decisão ----------
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

# ---------------- Captura de ROIs ----------------
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
    gray = _to_gray(frame)
    gw = _crop_from_union(gray, cap_win, uni, pad_y=0)
    gn = _crop_from_union(gray, name_roi, uni, pad_y=4)
    return gw, gn

# ---------------- Esperas ----------------
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
    """Espera após lançar a bola: mínimo + adaptativo; não spamma durante animação."""
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

# ---------------- Fuga verificada ----------------
def quick_escape(max_presses: int = 2, check_delay: float = 0.30):
    presses = 0
    while battle_active_instant() and presses < max_presses:
        if DRY_RUN: log.info(f"[DRY] press {CFG.non_target_key}")
        else: keyboard.press_and_release(CFG.non_target_key)
        presses += 1
        time.sleep(check_delay)
        if not battle_active_instant():
            break
    log.info(f"[NON-TARGET] Fuga {'OK' if not battle_active_instant() else 'ainda em batalha'} ({presses}x).")

# ---------------- Loop principal ----------------
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

    keyboard.on_press_key(STOP_HK, on_stop)
    keyboard.on_press_key(PAUSE_HK, on_pause)
    keyboard.on_press_key(SAVE_TPL_HK, on_save)
    keyboard.on_press_key(TEST_TPL_HK, on_test)

    ocr = OCRWorker(); ocr.start()

    log.info(f"[Setup] Ativar mount com tecla '{CFG.mount_key}'")
    if DRY_RUN: log.info(f"[DRY] press {CFG.mount_key}")
    else: keyboard.press_and_release(CFG.mount_key); time.sleep(1.0)

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
                while battle_active_instant():
                    if DRY_RUN: log.info(f"[DRY] press {CFG.non_target_key}")
                    else: keyboard.press_and_release(CFG.non_target_key)
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
                    if DRY_RUN: log.info(f"[DRY] tap {k}")
                    else: keyboard.press_and_release(k)
            move('a'); move('d')
            if random.random() < 0.08:
                time.sleep(random.uniform(0.15,0.45))

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

    CFG = load_config(SCRIPT_DIR / args.config)

    TEMPLATES_DIR = pathlib.Path(CFG.templates_dir); TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR = pathlib.Path(CFG.stats_dir); STATS_DIR.mkdir(parents=True, exist_ok=True)

    DRY_RUN = args.dry_run or CFG.dry_run

    # Logging por sessão (retenção 5)
    LOG_DIR = SCRIPT_DIR / "logs"; LOG_DIR.mkdir(parents=True, exist_ok=True)
    session_log = LOG_DIR / f"bot_{now_ts()}.log"

    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(session_log, encoding="utf-8")
    fh.setFormatter(fmt); fh.setLevel(getattr(logging, CFG.log_level.upper(), logging.INFO))
    ch = logging.StreamHandler(); ch.setFormatter(fmt); ch.setLevel(logging.INFO)
    log.setLevel(logging.DEBUG); log.addHandler(fh); log.addHandler(ch)

    try:
        logs = sorted(LOG_DIR.glob("bot_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in logs[5:]:
            try: p.unlink()
            except Exception: pass
    except Exception:
        pass

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

    stats = SessionStats(STATS_DIR)
    stats.write_snapshot(force=True)

    CAP_WIN, NAME_ROI_S, UNION_RECT = compute_rects()
    log.info(f"CAP_WIN={CAP_WIN} NAME_ROI={NAME_ROI_S} UNION={UNION_RECT}")

    try:
        patrol_loop()
    except KeyboardInterrupt:
        stats.write_snapshot(force=True)
        log.info("Interrompido por teclado.")
