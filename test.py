import torch
import cv2, os
import numpy as np
from model import DexiNed
from utils import image_normalization
import matplotlib.pyplot as plt

def test_one_img():
    # Cấu hình
    checkpoint_path = 'checkpoints/BIPED/10/10_model.pth'  # Đường dẫn checkpoint
    image_path = r"C:\Codes\DexiNed\datasets\BSDS\test\aug_data\0.0_1_1\24004.jpg"  # Ảnh bạn muốn test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = DexiNed().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Load ảnh
    img = cv2.imread(image_path)
    # Resize ảnh theo yêu cầu model (đa số phải chia hết cho 16)
    img = cv2.resize(img, (512, 512))  # ví dụ resize 512x512
    img_input = image_normalization(img)

    # Chuyển ảnh sang tensor
    img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # Dự đoán
    with torch.no_grad():
        preds = model(img_tensor)
        pred_edge = torch.sigmoid(preds[-1])  # Lấy output cuối cùng
        pred_edge = pred_edge.squeeze().cpu().numpy()

    pred_edge = 1.0 - pred_edge

    cv2.imwrite(r"C:\Codes\DexiNed\predict_image.png", (pred_edge * 255).astype(np.uint8))

    # # Hiển thị kết quả
    # plt.figure(figsize=(12,6))
    # plt.subplot(1,2,1)
    # plt.title('Input Image')
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    #
    # plt.subplot(1,2,2)
    # plt.title('Predicted Edge')
    # plt.imshow(pred_edge, cmap='gray')
    # plt.axis('off')
    #
    # plt.show()

test_one_img()