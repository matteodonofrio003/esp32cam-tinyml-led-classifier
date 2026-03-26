import serial
import numpy as np
import cv2
import os
import time

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'  # Change this to your port (e.g., /dev/ttyUSB0 on Linux)
BAUD_RATE = 921600
IMG_WIDTH = 96
IMG_HEIGHT = 96
CHANNELS = 3  # RGB888
FRAME_SIZE = IMG_WIDTH * IMG_HEIGHT * CHANNELS

HEADER = b'\xaa\xbb'
FOOTER = b'\xcc\xdd'

# Directory to save dataset
DATASET_DIR = "dataset"
categories = ["red", "green", "blue", "off"]

for cat in categories:
    os.makedirs(os.path.join(DATASET_DIR, cat), exist_ok=True)

def main():
    try:
        # Initialize serial connection
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        print(f"Connected to {SERIAL_PORT} at {BAUD_RATE}")
        time.sleep(2) # Wait for ESP32 reset
        ser.flushInput()

        while True:
            # 1. Search for Header
            if ser.read(1) == b'\xaa':
                if ser.read(1) == b'\xbb':
                    
                    # 2. Read Payload Size (4 bytes, Big Endian)
                    size_bytes = ser.read(4)
                    if len(size_bytes) < 4: continue
                    payload_size = int.from_bytes(size_bytes, byteorder='big')

                    if payload_size != FRAME_SIZE:
                        print(f"Warning: Unexpected frame size: {payload_size}")
                        continue

                    # 3. Read Frame Data
                    # We use a loop to ensure we get exactly 'payload_size' bytes
                    frame_data = b''
                    while len(frame_data) < payload_size:
                        chunk = ser.read(payload_size - len(frame_data))
                        if not chunk: break
                        frame_data += chunk

                    # 4. Check for Footer
                    footer_bytes = ser.read(2)
                    if footer_bytes != FOOTER:
                        print("Error: Footer mismatch! Frame corrupted.")
                        continue

                    # 5. Process Image
                    # Convert bytes to numpy array
                    img_np = np.frombuffer(frame_data, dtype=np.uint8)
                    img_np = img_np.reshape((IMG_HEIGHT, IMG_WIDTH, CHANNELS))

                    # Convert RGB to BGR (OpenCV uses BGR)
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    # 6. Display and Interaction
                    display_img = cv2.resize(img_bgr, (400, 400), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("ESP32-CAM Streamer", display_img)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'): # Quit
                        break
                    elif key == ord('r'): # Save Red
                        save_frame(img_bgr, "red")
                    elif key == ord('g'): # Save Green
                        save_frame(img_bgr, "green")
                    elif key == ord('b'): # Save Blue
                        save_frame(img_bgr, "blue")
                    elif key == ord('o'): # Save Off
                        save_frame(img_bgr, "off")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'ser' in locals():
            ser.close()
        cv2.destroyAllWindows()

def save_frame(img, category):
    timestamp = int(time.time() * 1000)
    filename = os.path.join(DATASET_DIR, category, f"{category}_{timestamp}.jpg")
    cv2.imwrite(filename, img)
    print(f"Saved: {filename}")

if __name__ == "__main__":
    main()