# gui_control.pyw
# GUI de controlo + edição de config.jsonc (Pokémons)

import pathlib, sys, time, subprocess, os, traceback, threading, re, json
from collections import deque

BASE = pathlib.Path(__file__).resolve().parent
CONFIG_FILE = BASE / "config.jsonc"
BOT_PY = BASE / "bot_pokemon.py"
LOG  = BASE / "gui_last.log"

VENV_PY  = BASE / ".venv" / "Scripts" / "python.exe"
VENV_PYW = BASE / ".venv" / "Scripts" / "pythonw.exe"
CREATE_NO_WINDOW = 0x08000000

def strip_jsonc(text: str) -> str:
    return re.sub(r'//.*', '', text)

def load_config():
    if not CONFIG_FILE.exists():
        return {}
    try:
        txt = CONFIG_FILE.read_text(encoding="utf-8")
        data = json.loads(strip_jsonc(txt))
        return data
    except Exception as e:
        print("Erro a carregar config:", e)
        return {}

def save_config(data: dict):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print("Erro a gravar config:", e)
        return False

# --- Tkinter
import tkinter as tk
from tkinter import ttk, messagebox

# --- Keyboard
def ensure_keyboard():
    try:
        import keyboard
        return True
    except Exception:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "keyboard"],
                                  creationflags=CREATE_NO_WINDOW)
            import keyboard
            return True
        except Exception:
            return False

HAS_KBD = ensure_keyboard()
if HAS_KBD:
    import keyboard

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pokemon Bot — Controlador")
        self.geometry("900x600")

        self.proc = None
        self.tail_buf = deque(maxlen=400)
        self.config_data = load_config()

        self._make_ui()
        self.after(700, self._tick_tail)

    # ----------------- UI -----------------
    def _make_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        # TAB CONTROLO
        tab_control = ttk.Frame(nb)
        nb.add(tab_control, text="Controlo")

        self.state_var = tk.StringVar(value="Parado")
        ttk.Label(tab_control, text="Estado:").pack(anchor="w", padx=8, pady=4)
        ttk.Label(tab_control, textvariable=self.state_var, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=8)

        btns = ttk.Frame(tab_control)
        btns.pack(pady=8)
        ttk.Button(btns, text="Start", command=self.start_bot).pack(side="left", padx=4)
        ttk.Button(btns, text="Pause/Resume", command=lambda: keyboard.send("f10") if HAS_KBD else None).pack(side="left", padx=4)
        ttk.Button(btns, text="Stop", command=self.stop_bot).pack(side="left", padx=4)
        ttk.Button(btns, text="Gravar Template (F8)", command=lambda: keyboard.send("f8") if HAS_KBD else None).pack(side="left", padx=4)
        ttk.Button(btns, text="Testar Template (F9)", command=lambda: keyboard.send("f9") if HAS_KBD else None).pack(side="left", padx=4)

        self.txt_tail = tk.Text(tab_control, height=15, wrap="none", font=("Consolas", 9))
        self.txt_tail.pack(fill="both", expand=True, padx=6, pady=6)
        self.txt_tail.configure(state="disabled")

        # TAB POKEMONS
        tab_poke = ttk.Frame(nb)
        nb.add(tab_poke, text="Pokémons")

        frm = ttk.Frame(tab_poke, padding=8)
        frm.pack(fill="both", expand=True)

        # Lista
        self.poke_list = tk.Listbox(frm, height=12)
        self.poke_list.grid(row=0, column=0, rowspan=6, sticky="nswe", padx=(0,8))
        frm.grid_columnconfigure(0, weight=1)
        frm.grid_rowconfigure(5, weight=1)

        # Campos
        ttk.Label(frm, text="Nome:").grid(row=0, column=1, sticky="w")
        self.name_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.name_var, width=20).grid(row=0, column=2, sticky="w")

        ttk.Label(frm, text="Tipo:").grid(row=1, column=1, sticky="w")
        self.type_var = tk.StringVar()
        ttk.Combobox(frm, textvariable=self.type_var, values=["setup", "catch", "setup+catch"], width=15).grid(row=1, column=2, sticky="w")

        ttk.Label(frm, text="Sequência (ex: 1,2,3):").grid(row=2, column=1, sticky="w")
        self.seq_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.seq_var, width=20).grid(row=2, column=2, sticky="w")

        # Botões
        ttk.Button(frm, text="Adicionar", command=self.add_pokemon).grid(row=3, column=1, pady=4)
        ttk.Button(frm, text="Editar", command=self.edit_pokemon).grid(row=3, column=2, pady=4)
        ttk.Button(frm, text="Remover", command=self.remove_pokemon).grid(row=4, column=1, pady=4)
        ttk.Button(frm, text="Guardar", command=self.save_pokemons).grid(row=4, column=2, pady=4)

        self.refresh_list()

    # ----------------- Bot Control -----------------
    def start_bot(self):
        if self.proc and self.proc.poll() is None:
            return
        exe = VENV_PY if VENV_PY.exists() else sys.executable
        try:
            self.proc = subprocess.Popen(
                [str(exe), str(BOT_PY)],
                cwd=str(BASE),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace"
            )
            self.state_var.set("A correr")
            threading.Thread(target=self._reader_thread, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def stop_bot(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except: pass
        self.proc = None
        self.state_var.set("Parado")

    def _reader_thread(self):
        try:
            for line in self.proc.stdout:
                self.tail_buf.append(line.rstrip("\n"))
        except: pass

    def _tick_tail(self):
        self.txt_tail.configure(state="normal")
        self.txt_tail.delete("1.0", "end")
        if self.tail_buf:
            self.txt_tail.insert("1.0", "\n".join(self.tail_buf))
        self.txt_tail.configure(state="disabled")
        self.after(700, self._tick_tail)

    # ----------------- Pokémons -----------------
    def refresh_list(self):
        self.poke_list.delete(0, "end")
        targets = self.config_data.get("targets", {})
        for name, actions in targets.items():
            if "setup" in actions and "catch" in actions:
                tipo = "setup+catch"
            elif "setup" in actions:
                tipo = "setup"
            elif "catch" in actions:
                tipo = "catch"
            else:
                tipo = "—"
            seq = actions.get("setup") or actions.get("catch")
            seq_str = ",".join(seq) if seq else ""
            self.poke_list.insert("end", f"{name} [{tipo}] → {seq_str}")

    def add_pokemon(self):
        name = self.name_var.get().strip().lower()
        tipo = self.type_var.get().strip()
        seq = [s.strip() for s in self.seq_var.get().split(",") if s.strip()]
        if not name or not seq or not tipo:
            messagebox.showerror("Erro", "Preenche todos os campos.")
            return
        if "targets" not in self.config_data:
            self.config_data["targets"] = {}
        if tipo == "setup+catch":
            self.config_data["targets"][name] = {"setup": seq, "catch": seq}
        else:
            self.config_data["targets"][name] = {tipo: seq}
        self.refresh_list()

    def edit_pokemon(self):
        idx = self.poke_list.curselection()
        if not idx: return
        sel = list(self.config_data.get("targets", {}).keys())[idx[0]]
        self.add_pokemon()
        if sel in self.config_data["targets"] and sel != self.name_var.get().strip().lower():
            del self.config_data["targets"][sel]
        self.refresh_list()

    def remove_pokemon(self):
        idx = self.poke_list.curselection()
        if not idx: return
        sel = list(self.config_data.get("targets", {}).keys())[idx[0]]
        del self.config_data["targets"][sel]
        self.refresh_list()

    def save_pokemons(self):
        if save_config(self.config_data):
            messagebox.showinfo("Guardado", "Pokémons gravados com sucesso!")
        else:
            messagebox.showerror("Erro", "Falha ao gravar config.")

if __name__ == "__main__":
    App().mainloop()
