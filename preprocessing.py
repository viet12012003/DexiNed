import os
import cv2
import numpy as np
from tqdm import tqdm

def augment_images(input_folder, output_folder):
    # Tạo thư mục lưu trữ nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Lấy danh sách các file ảnh từ thư mục đầu vào
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]

    # Duyệt qua từng ảnh trong thư mục đầu vào
    for img_file in tqdm(image_files, desc="Processing images"):
        # Đọc ảnh
        img_path = os.path.join(input_folder, img_file)
        img = cv2.imread(img_path)

        # Kiểm tra nếu ảnh bị lỗi không đọc được
        if img is None:
            print(f"Could not read {img_file}. Skipping.")
            continue

        # Lấy chiều cao và chiều rộng của ảnh
        h, w, _ = img.shape

        # 1. Cắt ảnh thành 2 phần theo chiều cao
        half_1 = img[:h//2, :, :]
        half_2 = img[h//2:, :, :]
        halves = [(half_1, "h1"), (half_2, "h2")]

        # Biến đếm để gán số thứ tự từ 1 đến 120
        counter = 1

        # Duyệt qua từng phần đã cắt
        for half_img, half_name in halves:
            h_half, w_half, _ = half_img.shape

            # 2. Xoay mỗi phần với 15 góc độ
            for i in range(15):
                angle = i * 15
                M = cv2.getRotationMatrix2D((w_half//2, h_half//2), angle, 1)
                rotated = cv2.warpAffine(half_img, M, (w_half, h_half))

                # 3. Lật ngang mỗi ảnh đã xoay
                for flip_type, flip_name in [(1, "f_h"), (-1, "f_both")]:
                    flipped = cv2.flip(rotated, flip_type)

                    # 4. Áp dụng gamma correction cho mỗi ảnh đã lật
                    for gamma in [0.3030, 0.6060]:
                        inv_gamma = 1.0 / gamma
                        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                        gamma_corrected = cv2.LUT(flipped, table)

                        # Lưu ảnh với tên phù hợp, thêm số thứ tự
                        output_filename = f"{img_file.split('.')[0]}_{counter}.jpg"
                        cv2.imwrite(os.path.join(output_folder, output_filename), gamma_corrected)

                        # Tăng biến đếm
                        counter += 1

# Đường dẫn thư mục đầu vào và thư mục lưu kết quả
input_folder = r"C:\Codes\DexiNed\datasets\BIPED\edges\edge_maps\train\rgbr\real"  # Thư mục chứa ảnh gốc
output_folder = r"C:\Codes\DexiNed\datasets_label"  # Thư mục để lưu kết quả

# Gọi hàm augment_images
augment_images(input_folder, output_folder)