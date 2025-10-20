import cv2
import os
from datetime import datetime

ESP32_STREAM = "http://192.168.110.74/stream"
SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"[INFO] Saving images to: {os.path.abspath(SAVE_DIR)}")
cap = cv2.VideoCapture(ESP32_STREAM)

if not cap.isOpened():
    print("[ERR] Cannot open stream.")
    exit()

fps_counter = 0
last_time = datetime.now()
print("[INFO] Press 'q' to stop capturing.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Failed to read frame.")
        continue

    # === Tính FPS ===
    now = datetime.now()
    fps_counter += 1
    if (now - last_time).total_seconds() >= 1.0:
        print(f"[INFO] FPS: {fps_counter}")
        fps_counter = 0
        last_time = now

    # === Lưu ảnh theo thời gian thực ===
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
    cv2.imwrite(os.path.join(SAVE_DIR, filename), frame)

    # Hiển thị ảnh
    cv2.imshow("ESP32-CAM Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
