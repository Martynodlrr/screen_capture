from mss import mss
import time

def capture_and_process_frames(duration=10):
    """
    Continuously capture and process frames for a specified duration in seconds.
    """
    start_time = time.time()

    with mss() as sct:
        # Define the region to capture (center of the screen)
        monitor = sct.monitors[1]  # Using the first monitor
        region = {
            'left': monitor['left'] + (monitor['width'] - 500) // 2,
            'top': monitor['top'] + (monitor['height'] - 400) // 2,
            'width': 500,
            'height': 500
        }

        frame_count = 0
        while True:
            # Capture the screen
            sct_img = sct.grab(region)
            frame_count += 1

            # Process the captured frame here
            # For example, analyze the image, perform OCR, etc.
            # This example just prints the frame count
            print(f"Captured frame {frame_count}")

            # Check if the duration has been exceeded
            if time.time() - start_time > duration:
                break

    print(f"Finished capturing {frame_count} frames in {duration} seconds.")

# Example usage: capture and process for 5 seconds
capture_and_process_frames(5)
