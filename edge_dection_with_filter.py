import cv2
import os
import numpy as np

def create_output_directory(base_dir, method_name):
    """
    Create a directory to store edge-detected images for a specific method.
    """
    output_dir = os.path.join(base_dir, method_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def apply_sobel(input_dir, output_dir):
    """
    Apply Sobel edge detection to all images in the input directory and save the results.
    """
    output_dir = create_output_directory(output_dir, "Sobel")
    for img_file in os.listdir(input_dir):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            sobel = cv2.magnitude(sobelx, sobely)
            sobel = np.uint8(255 * sobel / np.max(sobel))  # Normalize to 0-255
            sobel = 1 - sobel
            cv2.imwrite(os.path.join(output_dir, img_file), sobel)

def apply_prewitt(input_dir, output_dir):
    """
    Áp dụng bộ lọc Prewitt để phát hiện biên và lưu kết quả.
    """
    output_dir = create_output_directory(output_dir, "Prewitt")
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    for img_file in os.listdir(input_dir):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            prewittx = cv2.filter2D(img, cv2.CV_32F, kernelx)  # Đảm bảo kiểu float32
            prewitty = cv2.filter2D(img, cv2.CV_32F, kernely)  # Đảm bảo kiểu float32
            prewitt = cv2.magnitude(prewittx, prewitty)
            prewitt = np.uint8(255 * prewitt / np.max(prewitt))  # Chuẩn hóa về 0-255
            prewitt = 1 - prewitt
            cv2.imwrite(os.path.join(output_dir, img_file), prewitt)


def apply_roberts(input_dir, output_dir):
    """
    Áp dụng bộ lọc Roberts để phát hiện biên và lưu kết quả.
    """
    output_dir = create_output_directory(output_dir, "Roberts")
    kernelx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernely = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    for img_file in os.listdir(input_dir):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            robertsx = cv2.filter2D(img, cv2.CV_32F, kernelx)  # Đảm bảo float32
            robertsy = cv2.filter2D(img, cv2.CV_32F, kernely)  # Đảm bảo float32
            roberts = cv2.magnitude(robertsx, robertsy)
            roberts = np.uint8(255 * roberts / np.max(roberts))  # Chuẩn hóa về 0-255
            roberts = 1 - roberts
            cv2.imwrite(os.path.join(output_dir, img_file), roberts)

def apply_canny(input_dir, output_dir, threshold1=100, threshold2=200):
    """
    Apply Canny edge detection to all images in the input directory and save the results.
    """
    output_dir = create_output_directory(output_dir, "Canny")
    for img_file in os.listdir(input_dir):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            canny = cv2.Canny(img, threshold1, threshold2)
            canny = 255 - canny
            cv2.imwrite(os.path.join(output_dir, img_file), canny)

def process_edge_detection(input_dir, output_dir):
    """
    Apply all edge detection filters (Sobel, Prewitt, Roberts, Canny) to images in input_dir.
    """
    apply_sobel(input_dir, output_dir)
    apply_prewitt(input_dir, output_dir)
    apply_roberts(input_dir, output_dir)
    apply_canny(input_dir, output_dir)

# input_dir = r'C:\Codes\DexiNed\datasets\BIPED\edges\imgs\test\rgbr'
# output_dir = r'C:\Codes\DexiNed\result\BIPED_filtered'

input_dir = r'C:\Codes\DexiNed\datasets\BSDS\test_for_filter'
output_dir = r'C:\Codes\DexiNed\result\BSDS_filtered'

process_edge_detection(input_dir, output_dir)