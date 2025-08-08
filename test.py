from pynput import keyboard, mouse
import threading
import time

clicking = False
click_thread = None

def click_mouse():
    m = mouse.Controller()
    while clicking:
        m.click(mouse.Button.left)
        time.sleep(0.1)  # 點擊間隔，可以依需求調整

def on_press(key):
    global clicking, click_thread

    if hasattr(key, 'char') and key.char == 'a':
        clicking = not clicking
        if clicking:
            # 啟動點擊執行緒
            click_thread = threading.Thread(target=click_mouse)
            click_thread.start()
        else:
            # 停止點擊
            if click_thread:
                click_thread.join()
            print("已停止點擊")

def on_release(key):
    pass

if __name__ == "__main__":
    print("按下 a 鍵啟動/停止滑鼠點擊")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
