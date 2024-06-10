from mss import mss
import cv2
import numpy as np
from threading import Thread
import time
import pynput.mouse
from pynput.keyboard import Listener as KeyboardListener
from inference_sdk import InferenceHTTPClient

load_dotenv()
API_KEY = os.getenv("API_KEY")

class FrameProcessor:
    def __init__(self, api_url, api_key, model_id):
        self.client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        self.model_id = model_id
        self.monitor = mss().monitors[1]
        self.stop_thread = False
        self.mouse_listener = pynput.mouse.Listener(on_click=self.on_click)
        self.keyboard_listener = KeyboardListener(on_press=self.on_key_press)
        self.quit_keys_pressed = False

    def on_click(self, x, y, button, pressed):
        if button == pynput.mouse.Button.left and pressed:
            self.mouse_clicked = True

    def on_key_press(self, key):
        if key == pynput.keyboard.Key.esc:
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
                sct_img = sct.grab(self.monitor)
                frame_count += 1
                frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
                cv2.imwrite("current_frame.jpg", frame)

                # Run the AI model from Roboflow
                result = self.client.infer("current_frame.jpg", model_id=self.model_id)
                detections = result['predictions']
                current_obj_count = len(detections)

            print(f"Detected {current_obj_count} objects in frame {frame_count}")

            if self.mouse_clicked:
                total_clicks += 1
                if current_obj_count < last_obj_count:
                    accurate_clicks += 1
                self.mouse_clicked = False

            last_obj_count = current_obj_count

            # Display accuracy on the screen
            accuracy = accurate_clicks / total_clicks if total_clicks > 0 else 0
            self.display_accuracy(frame, accuracy)

        accuracy = accurate_clicks / total_clicks if total_clicks > 0 else 0
        print(f"Final Accuracy: {accuracy}")

    def display_accuracy(self, frame, accuracy):
        accuracy_text = f"Accuracy: {accuracy:.2%}"
        cv2.putText(frame, accuracy_text, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Screen", frame)
        cv2.waitKey(1)

    def stop(self):
        self.stop_thread = True
        self.thread.join()
        self.mouse_listener.stop()
        self.keyboard_listener.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    api_url = "https://detect.roboflow.com"
    model_id = "valoaccuracy/5"
    frame_processor = FrameProcessor(api_url, API_KEY, model_id)
    frame_processor.start()

    while not frame_processor.quit_keys_pressed:
        time.sleep(0.1)

    frame_processor.stop()
