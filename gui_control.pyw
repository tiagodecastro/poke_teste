# gui_control.pyw
# GUI para controlar o bot: Start / Pause / Stop
# - Janela redimensionável com 3 painéis: Instruções, Estatísticas, Saída (tail)
# - Arranca bot sem consola (python.exe + CREATE_NO_WINDOW) e CAPTURA stdout/stderr
# - Pause/Resume via F10 e Stop via F12 (usa 'keyboard'; tenta instalar se faltar)
# - Fecha o bot ao fechar a janela


import pathlib, sys, time, subprocess, os, traceback, threading, io, re
from collections import deque

# --- Relaunch automático na venv ---
BASE = pathlib.Path(__file__).resolve().parent
VENV_PYW = BASE / ".venv" / "Scripts" / "pythonw.exe"
VENV_PY = BASE / ".venv" / "Scripts" / "python.exe"
def _is_venv():
    # Detecta se está rodando na venv do projeto
    exe = pathlib.Path(sys.executable).resolve()
    return (exe.parent.parent == BASE / ".venv")

if not _is_venv():
    # Se não está na venv, relança usando pythonw.exe da venv
    if VENV_PYW.exists():
        subprocess.Popen([str(VENV_PYW), str(__file__)], cwd=str(BASE))
        sys.exit(0)
    elif VENV_PY.exists():
        subprocess.Popen([str(VENV_PY), str(__file__)], cwd=str(BASE))
        sys.exit(0)
    else:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, "Não foi possível encontrar o Python da venv. Corre o setup/start primeiro.", "Venv não encontrada", 0x10)
        sys.exit(1)


BASE = pathlib.Path(__file__).resolve().parent
LOG  = BASE / "gui_last.log"
BOT_PY = BASE / "bot_pokemon.py"
VENV_PY  = BASE / ".venv" / "Scripts" / "python.exe"      # usamos python.exe p/ capturar stdout
VENV_PYW = BASE / ".venv" / "Scripts" / "pythonw.exe"
CREATE_NO_WINDOW = 0x08000000

def _log_write(level: str, msg: str):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {msg}\n")
    except Exception:
        pass

def log_err(msg: str):  _log_write("ERR", msg)
def log_info(msg: str): _log_write("INFO", msg)

def msgbox_error(title, text):
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, str(text), str(title), 0x10)  # MB_ICONERROR
    except Exception:
        pass

# --- Tk / ttk ---
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception:
    log_err("Falha a importar tkinter:\n" + traceback.format_exc())
    msgbox_error("Tkinter em falta", "Reinstala o Python com suporte tcl/tk (Tkinter).")
    raise SystemExit(1)

# --- keyboard (opcional, tentamos instalar) ---
def ensure_keyboard():
    try:
        import keyboard  # noqa
        return True
    except Exception:
        pass
    exe = sys.executable  # pythonw no contexto .pyw, mas serve para pip
    try:
        subprocess.check_call(
            [exe, "-m", "pip", "install", "--disable-pip-version-check", "keyboard"],
            creationflags=CREATE_NO_WINDOW
        )
        import keyboard  # noqa
        return True
    except Exception:
        log_err("Falha a instalar keyboard:\n" + traceback.format_exc())
        return False

HAS_KBD = ensure_keyboard()
if HAS_KBD:
    import keyboard  # type: ignore

INSTRUCTIONS = (
    "Hotkeys do bot:\n"
    " • F8  – Guardar template de batalha (prime dentro de uma batalha)\n"
    " • F9  – Testar template (mostra score no log)\n"
    " • F10 – Pausar/Retomar o bot\n"
    " • F12 – Sair do bot\n\n"
    "Passos recomendados:\n"
    " 1) Arranca o bot.\n"
    " 2) Na 1.ª batalha da zona, prime F8 para guardar o template.\n"
    " 3) (Opcional) F9 para ver o score.\n"
    " 4) Estatísticas por sessão ficam em 'stats/'."
)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pokemon Bot — Controlador")
        self.minsize(640, 420)
        self.geometry("860x540")
        self.resizable(True, True)

        self.proc = None
        self.tail_buf = deque(maxlen=400)   # guardamos ~400 linhas, exibimos últimas N
        self.tail_lines_var = tk.IntVar(value=8)  # quantas linhas mostrar
        self.stats_file = None
        self.cmd_line = ""

        self._make_ui()
        self._layout_weights()
        self.after(1000, self._tick_status)
        self.after(1200, self._tick_stats)
        self.after(700,  self._tick_tail)
        log_info("GUI iniciado")

    # ---------- UI ----------
    def _make_ui(self):
        # Top bar: estado + controlos
        top = ttk.Frame(self, padding=8)
        top.grid(row=0, column=0, sticky="nsew")

        self.state_var = tk.StringVar(value="Parado")
        ttk.Label(top, text="Estado:").grid(row=0, column=0, sticky="w")
        ttk.Label(top, textvariable=self.state_var, font=("Segoe UI", 10, "bold")).grid(row=0, column=1, sticky="w", padx=(6,18))

        self.btn_start = ttk.Button(top, text="Start", width=12, command=self.start_bot)
        self.btn_pause = ttk.Button(top, text="Pause/Resume", width=14, command=self.toggle_pause,
                                    state=("normal" if HAS_KBD else "disabled"))
        self.btn_stop  = ttk.Button(top, text="Stop", width=12, command=self.stop_bot, state="disabled")
        self.btn_start.grid(row=0, column=2, padx=(0,6))
        self.btn_pause.grid(row=0, column=3, padx=(0,6))
        self.btn_stop.grid(row=0,  column=4)

        ttk.Separator(self).grid(row=1, column=0, sticky="ew")

        # Main Paned: esquerda (instruções + stats) | direita (tail)
        main = ttk.PanedWindow(self, orient="horizontal")
        main.grid(row=2, column=0, sticky="nsew", padx=6, pady=6)


        # Botões F8/F9 (usam o módulo 'keyboard' já garantido por HAS_KBD)
        self.btn_save = ttk.Button(
            top, text="Gravar Template (F8)", width=20,
            command=(lambda: keyboard.send('f8')),
            state=("normal" if HAS_KBD else "disabled")
        )
        self.btn_test = ttk.Button(
            top, text="Testar Template (F9)", width=18,
            command=(lambda: keyboard.send('f9')),
            state=("normal" if HAS_KBD else "disabled")
        )

        self.btn_save.grid(row=0, column=5, padx=(6,6))
        self.btn_test.grid(row=0, column=6)


        # Esquerda: tabs com Instruções e Estatísticas
        left = ttk.Notebook(main)
        main.add(left, weight=1)

        # Tab Instruções
        tab_instr = ttk.Frame(left)
        left.add(tab_instr, text="Instruções")
        txt_instr = tk.Text(tab_instr, wrap="word")
        txt_instr.insert("1.0", INSTRUCTIONS)
        txt_instr.configure(state="disabled")
        txt_instr.pack(fill="both", expand=True, padx=6, pady=6)

        # Tab Estatísticas
        tab_stats = ttk.Frame(left)
        left.add(tab_stats, text="Estatísticas")

        sf = ttk.Frame(tab_stats, padding=6)
        sf.pack(fill="both", expand=True)

        # labels de stats
        self.lbl_stats_path = ttk.Label(sf, text="Ficheiro: (por detetar)", foreground="#666")
        self.lbl_stats_path.grid(row=0, column=0, sticky="w", columnspan=2, pady=(0,6))

        self.stats_vars = {
            "inicio": tk.StringVar(value="—"),
            "ultima": tk.StringVar(value="—"),
            "tempo":  tk.StringVar(value="—"),
            "total":  tk.StringVar(value="0"),
            "targets": tk.StringVar(value="0"),
            "special": tk.StringVar(value="0"),
            "non":     tk.StringVar(value="0"),
        }

        row = 1
        for label, key in [("Início da sessão", "inicio"),
                           ("Última atualização", "ultima"),
                           ("Tempo", "tempo"),
                           ("Total", "total"),
                           ("Targets", "targets"),
                           ("Special (shiny/summer)", "special"),
                           ("Non-target", "non")]:
            ttk.Label(sf, text=f"{label}:").grid(row=row, column=0, sticky="w", pady=2)
            ttk.Label(sf, textvariable=self.stats_vars[key], font=("Segoe UI", 10, "bold")).grid(row=row, column=1, sticky="w", padx=(6,0))
            row += 1

        # Direita: Tail do bot (últimas N linhas) + comando
        right = ttk.Frame(main)
        main.add(right, weight=2)

        cmdf = ttk.Frame(right)
        cmdf.pack(fill="x", padx=4, pady=(2,0))
        ttk.Label(cmdf, text="Comando:").pack(side="left")
        self.cmd_var = tk.StringVar(value="")
        self.cmd_label = ttk.Label(cmdf, textvariable=self.cmd_var, foreground="#666")
        self.cmd_label.pack(side="left", padx=(6,0))

        top_tail = ttk.Frame(right)
        top_tail.pack(fill="x", padx=4, pady=(4,0))
        ttk.Label(top_tail, text="Saída do bot (últimas linhas):").pack(side="left")
        ttk.Spinbox(top_tail, from_=3, to=50, textvariable=self.tail_lines_var, width=4).pack(side="left", padx=(6,0))

        self.txt_tail = tk.Text(right, height=12, wrap="none", font=("Consolas", 9))
        self.txt_tail.pack(fill="both", expand=True, padx=4, pady=4)
        self.txt_tail.configure(state="disabled")

        # Footer
        ttk.Separator(self).grid(row=3, column=0, sticky="ew")
        foot = ttk.Frame(self, padding=6)
        foot.grid(row=4, column=0, sticky="ew")
        ttk.Label(foot, text="Fecha esta janela para terminar o bot.").pack(side="left")

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _layout_weights(self):
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

    # ---------- Bot control ----------
    def start_bot(self):
        if self.proc and self.proc.poll() is None:
            return
        if not BOT_PY.exists():
            messagebox.showerror("Erro", f"Não encontro {BOT_PY.name}.")
            return

        # Prioridade: pythonw.exe da venv > python.exe da venv > sys.executable
        exe = None
        venv_pyw = BASE / ".venv" / "Scripts" / "pythonw.exe"
        venv_py = BASE / ".venv" / "Scripts" / "python.exe"
        if venv_pyw.exists():
            exe = venv_pyw
        elif venv_py.exists():
            exe = venv_py
        elif sys.executable:
            exe = sys.executable
        else:
            messagebox.showerror("Python não encontrado", "Não foi possível encontrar o executável do Python. Certifique-se que o Python está instalado.")
            return

        flags = CREATE_NO_WINDOW if str(exe).lower().endswith(".exe") else 0

        # comando (podes acrescentar args: --fast / --accurate / --config)
        cmd = [str(exe), str(BOT_PY)]
        self.cmd_line = " ".join(cmd)
        self.cmd_var.set(self.cmd_line)
        log_info(f"Start: {self.cmd_line}")

        try:
            # capturar stdout/err para mostrar no tail
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(BASE),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",      # <- 2) robusto a acentos
                errors="replace",      # <- 2) evita crash por bytes inválidos
                bufsize=1,             # line-buffered
                creationflags=flags
            )
        except Exception as e:
            log_err("Falha ao arrancar bot:\n" + traceback.format_exc())
            messagebox.showerror("Falha ao arrancar", str(e))
            self.proc = None
            return

        # reader thread do tail
        self.tail_buf.clear()
        t = threading.Thread(target=self._reader_thread, daemon=True)
        t.start()

        # pós-arranque
        self.after(1000, self._post_start_check)

    def _reader_thread(self):
        try:
            for line in self.proc.stdout:
                self.tail_buf.append(line.rstrip("\n"))
        except Exception:
            log_err("Reader thread falhou:\n" + traceback.format_exc())

    def _post_start_check(self):
        if self.proc and self.proc.poll() is not None:
            rc = self.proc.returncode
            log_info(f"Bot terminou imediatamente (RC={rc})")
            messagebox.showerror("Bot terminou",
                                 f"O bot encerrou imediatamente (RC={rc}).\n"
                                 f"Usa o start.cmd para ver logs.")
            self.proc = None
            self._set_state("Parado")
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
        else:
            self._set_state("A correr")
            self.btn_start.configure(state="disabled")
            self.btn_stop.configure(state="normal")
            log_info("Bot em execução")

    def toggle_pause(self):
        if not HAS_KBD:
            return
        if self.proc and self.proc.poll() is None:
            try:
                keyboard.send('f10')
                log_info("F10 enviado (Pause/Resume)")
            except Exception:
                log_err("Falha ao enviar F10:\n" + traceback.format_exc())

    def stop_bot(self):
        if not self.proc:
            return
        if self.proc.poll() is None:
            # saída limpa primeiro (F12)
            if HAS_KBD:
                try:
                    keyboard.send('f12')
                    log_info("F12 enviado (Stop)")
                except Exception:
                    log_err("Falha ao enviar F12:\n" + traceback.format_exc())
                for _ in range(30):
                    if self.proc.poll() is not None:
                        break
                    time.sleep(0.1)
            # força término se ainda vivo (mantém teu comportamento)
            if self.proc.poll() is None:
                try:
                    self.proc.terminate()
                    log_info("terminate() chamado")
                except Exception:
                    log_err("Falha terminate():\n" + traceback.format_exc())
                for _ in range(30):
                    if self.proc.poll() is not None:
                        break
                    time.sleep(0.1)
            if self.proc.poll() is None:
                try:
                    self.proc.kill()
                    log_info("kill() chamado")
                except Exception:
                    log_err("Falha kill():\n" + traceback.format_exc())
        self.proc = None
        self._set_state("Parado")
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        log_info("Bot parado")

    # ---------- Ticks periódicos ----------
    def _tick_status(self):
        self._set_state("A correr" if (self.proc and self.proc.poll() is None) else "Parado")
        self.after(1000, self._tick_status)

    def _tick_tail(self):
        # mostra últimas N linhas do tail
        n = max(1, int(self.tail_lines_var.get()))
        lines = list(self.tail_buf)[-n:]
        self.txt_tail.configure(state="normal")
        self.txt_tail.delete("1.0", "end")
        if lines:
            self.txt_tail.insert("1.0", "\n".join(lines))
        self.txt_tail.configure(state="disabled")
        self.after(500, self._tick_tail)

    def _tick_stats(self):
        # encontra o ficheiro de stats mais recente
        stats_dir = BASE / "stats"
        latest = None
        try:
            files = sorted(stats_dir.glob("session_stats_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
            latest = files[0] if files else None
        except Exception:
            latest = None

        if latest and latest != self.stats_file:
            self.stats_file = latest
            self.lbl_stats_path.configure(text=f"Ficheiro: {latest.name}")

        if self.stats_file and self.stats_file.exists():
            try:
                txt = self.stats_file.read_text(encoding="utf-8", errors="ignore")
                # parse simples das linhas-chave
                def extract(pat, default="—"):
                    m = re.search(pat, txt, re.MULTILINE)
                    return m.group(1).strip() if m else default
                self.stats_vars["inicio"].set(extract(r"Início da sessão\s*:\s*(.*)"))
                self.stats_vars["ultima"].set(extract(r"Última atualização\s*:\s*(.*)"))
                self.stats_vars["tempo"].set(extract(r"Tempo:\s*(.*)"))
                self.stats_vars["total"].set(extract(r"Totais:\s*(\d+)","0"))
                self.stats_vars["targets"].set(extract(r"- Targets\s*:\s*(\d+)","0"))
                self.stats_vars["special"].set(extract(r"- Special.*:\s*(\d+)","0"))
                self.stats_vars["non"].set(extract(r"- Non-target:\s*(\d+)","0"))
            except Exception:
                pass

        self.after(1000, self._tick_stats)

    def _set_state(self, txt):
        self.state_var.set(txt)

    def on_close(self):
        log_info("GUI a fechar")
        self.stop_bot()
        self.destroy()

if __name__ == "__main__":
    try:
        App().mainloop()
    except Exception as e:
        log_err("Exceção na mainloop:\n" + traceback.format_exc())
        msgbox_error("Erro no GUI", f"{e}\n\nVê o log: {LOG}")
        raise
