from mss import mss
import time
import cv2
from pynput.mouse import Listener as MouseListener
from threading import Thread

# Assuming the YOLO model and necessary preprocessing functions are already defined
from your_yolo_module import YOLOv8, preprocess_image  # You'll need to define or adjust these imports

# Initialize the object detection model
yolo_model = YOLOv8("path_to_yolov8_model_weights")

def capture_and_process_frames():
    """
    Continuously capture and process frames.
    """
    with mss() as sct:
        monitor = sct.monitors[1]
        region = {
            'left': monitor['left'] + (monitor['width'] - 500) // 2,
            'top': monitor['top'] + (monitor['height'] - 400) // 2,
            'width': 500,
            'height': 500
        }

        frame_count = 0
        last_obj_count = 0
        total_clicks = 0
        accurate_clicks = 0

        while not stop_thread:
            sct_img = sct.grab(region)
            frame_count += 1

            # Convert captured image to a format suitable for object detection
            frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
            detections = yolo_model.detect(frame)

            # Count detected objects
            current_obj_count = len(detections)
            print(f"Detected {current_obj_count} objects in frame {frame_count}")

            # Check mouse events and adjust scores
            if mouse_clicked:
                total_clicks += 1
                if current_obj_count < last_obj_count:
                    accurate_clicks += 1
                mouse_clicked = False

            last_obj_count = current_obj_count

    accuracy = accurate_clicks / total_clicks if total_clicks > 0 else 0
    print(f"Accuracy: {accuracy}")
    return accuracy

# Mouse event handling
mouse_clicked = False
def on_click(x, y, button, pressed):
    global mouse_clicked
    if button == pynput.mouse.Button.right and pressed:
        mouse_clicked = True

# Start mouse listener
listener = MouseListener(on_click=on_click)
listener.start()

# Start capturing frames
stop_thread = False
thread = Thread(target=capture_and_process_frames)
thread.start()

# Example control to stop the process (e.g., after 30 seconds)
time.sleep(30)
stop_thread = True
thread.join()
listener.stop()
