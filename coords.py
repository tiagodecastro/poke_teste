# coords.py
import time, pyautogui as pag
print("Move o rato; Ctrl+C para sair")
try:
    while True:
        x, y = pag.position()
        print(f"x={x:4d}, y={y:4d}", end="\r", flush=True)
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\nbye")
