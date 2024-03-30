from mss import mss

# Define the region to capture
# X, Y, Width, Height
region = {'top': 500, 'left': 500, 'width': 500, 'height': 400}

with mss() as sct:
    # Capture the defined region
    sct_img = sct.grab(region)

    # Save the captured image
    # Note: The quality parameter is not applicable here, as MSS saves in PNG format.
    output = 'screenshot.png'
    sct.save(mon=sct_img, output=output)
    print(f"Screenshot saved as {output}")
