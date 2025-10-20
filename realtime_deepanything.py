import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

# --- 1. Cấu hình model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vits.pth', map_location=device))
model.to(device)
model.eval()

# --- 2. Cấu hình ESP32-CAM stream ---
ESP32_STREAM = "http://192.168.110.74/stream"  # đổi theo IP của bạn
cap = cv2.VideoCapture(ESP32_STREAM)
if not cap.isOpened():
    print("[ERR] Cannot open ESP32-CAM stream.")
    exit()

print("[INFO] Press 'q' to stop streaming.")

fps_counter = 0
from datetime import datetime
last_time = datetime.now()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Failed to read frame.")
        continue

    # --- 3. Infer depth real-time ---
    depth = model.infer_image(frame)  # HxW numpy array
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)

    # --- 4. Tính FPS ---
    now = datetime.now()
    fps_counter += 1
    if (now - last_time).total_seconds() >= 1.0:
        print(f"[INFO] FPS: {fps_counter}")
        fps_counter = 0
        last_time = now

    # --- 5. Hiển thị real-time ---
    cv2.imshow("ESP32-CAM Stream", frame)
    cv2.imshow("Depth Map", depth_color)

    # --- 6. Dừng bằng phím 'q' ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Streaming stopped.")
