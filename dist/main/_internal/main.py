import os
import pygame
import ctypes
from mss import mss
import cv2
import numpy as np
from threading import Thread
import time
from pynput import mouse, keyboard
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import logging
from io import BytesIO
from PIL import Image

# Set up logging to a file
logging.basicConfig(filename="app.log", level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the API key from .env file located in the same directory as the executable
env_path = os.path.join(os.path.dirname(__file__), '.env')
logging.info("Loading environment variables from: %s", env_path)
load_dotenv(env_path)
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    logging.error("API_KEY not found in .env file.")
    raise ValueError("API_KEY not found in .env file.")
else:
    logging.info("API_KEY loaded successfully")

# Initialize pygame and set up the overlay window
pygame.init()
info = pygame.display.Info()
screen_width, screen_height = info.current_w, info.current_h
overlay = pygame.display.set_mode((screen_width, screen_height), pygame.NOFRAME | pygame.SRCALPHA)
pygame.display.set_caption("Accuracy Overlay")

# Set window to be transparent
hwnd = pygame.display.get_wm_info()["window"]
ctypes.windll.user32.SetWindowLongW(hwnd, -20, ctypes.windll.user32.GetWindowLongW(hwnd, -20) | 0x80000 | 0x20)
ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0x000000, 0, 0x2)

logging.info("Overlay window created successfully")

class FrameProcessor:
    def __init__(self, api_url, api_key, model_id):
        self.client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        self.model_id = model_id
        self.monitor = mss().monitors[1]  # Index 1 is the main screen
        self.stop_thread = False
        self.mouse_listener = mouse.Listener(on_click=self.on_click)
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release)
        self.quit_keys_pressed = False
        self.ctrl_pressed = False
        self.alt_pressed = False
        self.mouse_clicked = False
        self.total_clicks = 0
        self.accurate_clicks = 0

    def on_click(self, x, y, button, pressed):
        if button == mouse.Button.left and pressed:
            self.mouse_clicked = True
            logging.info("Mouse clicked at (%d, %d)", x, y)

    def on_key_press(self, key):
        if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            self.ctrl_pressed = True
        if key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
            self.alt_pressed = True
        if key == keyboard.Key.f12 and self.ctrl_pressed and self.alt_pressed:
            self.quit_keys_pressed = True
            logging.info("Quit key combination pressed")

    def on_key_release(self, key):
        if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            self.ctrl_pressed = False
        if key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
            self.alt_pressed = False

    def start(self):
        self.mouse_listener.start()
        self.keyboard_listener.start()
        self.thread = Thread(target=self.capture_and_process_frames)
        self.thread.start()
        logging.info("FrameProcessor started")

    def capture_and_process_frames(self):
        last_obj_count = 0

        while not self.stop_thread and not self.quit_keys_pressed:
            with mss() as sct:
                sct_img = sct.grab(self.monitor)
                frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

                # Convert the frame to a PIL image
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Run the AI model from Roboflow
                try:
                    result = self.client.infer(pil_img, model_id=self.model_id)
                    detections = result['predictions']
                    current_obj_count = len(detections)
                    logging.debug("Current object count: %d", current_obj_count)
                except Exception as e:
                    logging.error("Error running inference: %s", e)
                    current_obj_count = 0

            if self.mouse_clicked:
                self.total_clicks += 1
                if current_obj_count < last_obj_count:
                    self.accurate_clicks += 1
                self.mouse_clicked = False

            last_obj_count = current_obj_count

            # Calculate and display accuracy on the screen
            accuracy = (self.accurate_clicks / self.total_clicks) * 100 if self.total_clicks > 0 else 100
            self.display_accuracy(accuracy)

            time.sleep(0.1)  # Small delay to avoid high CPU usage

        accuracy = (self.accurate_clicks / self.total_clicks) * 100 if self.total_clicks > 0 else 100
        self.display_accuracy(accuracy)
        logging.info("Final Accuracy: %.2f%%", accuracy)

    def display_accuracy(self, accuracy):
        overlay.fill((0, 0, 0, 0))  # Clear the overlay
        accuracy_text = f"Accuracy: {accuracy:.2f}%"
        font = pygame.font.SysFont("Arial", 30)
        text_surface = font.render(accuracy_text, True, (0, 255, 0))
        overlay.blit(text_surface, (10, screen_height - 40))
        pygame.display.update()
        logging.info("Displayed accuracy: %s", accuracy_text)

    def stop(self):
        self.stop_thread = True
        self.thread.join()
        self.mouse_listener.stop()
        self.keyboard_listener.stop()
        pygame.quit()
        logging.info("FrameProcessor stopped")

if __name__ == "__main__":
    api_url = "https://detect.roboflow.com"
    model_id = "valoaccuracy/5"
    frame_processor = FrameProcessor(api_url, API_KEY, model_id)
    frame_processor.start()

    while not frame_processor.quit_keys_pressed:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                frame_processor.quit_keys_pressed = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    frame_processor.quit_keys_pressed = True
        time.sleep(0.01)

    frame_processor.stop()
