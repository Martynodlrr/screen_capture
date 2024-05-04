from mss import mss
import cv2
import numpy as np
from threading import Thread
import time
import pynput.mouse
from pynput.keyboard import Listener as KeyboardListener, KeyCode
from your_yolo_module import YOLOv8, preprocess_image

class FrameProcessor:
    def __init__(self, model_path, monitor_index=1, region_size=(500, 500)):
        self.yolo_model = YOLOv8(model_path)
        self.monitor = mss().monitors[monitor_index]
        self.region = {
            'left': self.monitor['left'] + (self.monitor['width'] - region_size[0]) // 2,
            'top': self.monitor['top'] + (self.monitor['height'] - region_size[1]) // 2,
            'width': region_size[0],
            'height': region_size[1]
        }
        self.stop_thread = False
        self.mouse_listener = pynput.mouse.Listener(on_click=self.on_click)
        self.keyboard_listener = KeyboardListener(on_press=self.on_key_press)
        self.quit_keys_pressed = False

    def on_click(self, x, y, button, pressed):
        if button == pynput.mouse.Button.right and pressed:
            self.mouse_clicked = True

    def on_key_press(self, key):
        if key == (pynput.keyboard.Key.ctrl, pynput.keyboard.Key.alt, pynput.keyboard.KeyCode.from_char('f1')):
            self.quit_keys_pressed = True

    def start(self):
        self.mouse_clicked = False
        self.mouse_listener.start()
        self.keyboard_listener.start()
        self.thread = Thread(target=self.capture_and_process_frames)
        self.thread.start()

    def capture_and_process_frames(self):
        frame_count = 0
        last_obj_count = 0
        total_clicks = 0
        accurate_clicks = 0

        while not self.stop_thread and not self.quit_keys_pressed:
            with mss() as sct:
                sct_img = sct.grab(self.region)
                frame_count += 1
                frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
                detections = self.yolo_model.detect(frame)

            current_obj_count = len(detections)
            print(f"Detected {current_obj_count} objects in frame {frame_count}")

            if self.mouse_clicked:
                total_clicks += 1
                if current_obj_count < last_obj_count:
                    accurate_clicks += 1
                self.mouse_clicked = False

            last_obj_count = current_obj_count

        accuracy = accurate_clicks / total_clicks if total_clicks > 0 else 0
        print(f"Accuracy: {accuracy}")

    def stop(self):
        self.stop_thread = True
        self.thread.join()
        self.mouse_listener.stop()
        self.keyboard_listener.stop()

if __name__ == "__main__":
    frame_processor = FrameProcessor("path_to_yolov8_model_weights")
    frame_processor.start()

    # Wait for the key combination Ctrl + Alt + F1 to stop the program
    while not frame_processor.quit_keys_pressed:
        time.sleep(0.1)

    frame_processor.stop()
