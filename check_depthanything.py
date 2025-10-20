import cv2
import torch
import numpy as np
import os
from depth_anything_v2.dpt import DepthAnythingV2

# --- 1. Load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vits.pth', map_location=device))
model.to(device)
model.eval()

# --- 2. Thư mục ảnh input và output ---
input_dir = "test_imgs"
output_dir = "output_imgs"
os.makedirs(output_dir, exist_ok=True)

# --- 3. Lặp qua tất cả ảnh trong thư mục ---
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        img_path = os.path.join(input_dir, filename)
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            print(f"Không đọc được ảnh: {img_path}, bỏ qua.")
            continue

        # --- 4. Infer depth ---
        depth = model.infer_image(raw_img)

        # --- 5. Chuẩn hóa depth sang 0-255 ---
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
        depth_uint8 = (depth_norm * 255).astype(np.uint8)

        # --- 6. Colormap để trực quan ---
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)

        # --- 7. Hiển thị ảnh (tùy chọn, comment nếu nhiều ảnh) ---
        cv2.imshow("Original Image", raw_img)
        cv2.imshow("Depth Map", depth_color)
        cv2.waitKey(500)  # hiển thị 0.5 giây mỗi ảnh

        # --- 8. Lưu depth map dựa trên tên file gốc ---
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_depth.png")
        cv2.imwrite(output_path, depth_color)
        print(f"Đã lưu depth map: {output_path}")

cv2.destroyAllWindows()
print("Xử lý xong tất cả ảnh.")
