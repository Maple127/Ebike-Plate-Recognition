import cv2
import torch
import numpy as np
import re
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

try:
    from cnocr import CnOcr

    HAS_CNOCR = True
except ImportError:
    HAS_CNOCR = False
    print("警告: 未检测到 CnOCR 库，识别功能将不可用。")


# ================= 图像处理与算法函数 =================

def apply_affine_transform(image, angle, shear_x=0.0):
    """
    对图像应用仿射变换：包含旋转和水平错切 (Shear)
    """
    h, w = image.shape[:2]

    # 1. 计算旋转矩阵
    center = (w // 2, h // 2)
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 2. 计算旋转后的新边界尺寸
    cos = np.abs(M_rot[0, 0])
    sin = np.abs(M_rot[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 调整旋转矩阵平移量
    M_rot[0, 2] += (new_w / 2) - center[0]
    M_rot[1, 2] += (new_h / 2) - center[1]

    # 执行旋转
    rotated = cv2.warpAffine(image, M_rot, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=255)

    # 3. 如果需要错切，基于旋转后的图像继续处理
    if shear_x != 0:
        h_r, w_r = rotated.shape[:2]
        # 错切矩阵: [[1, shear, 0], [0, 1, 0]]
        M_shear = np.float32([[1, shear_x, 0], [0, 1, 0]])

        # 计算错切后的新宽度 (高度不变)
        new_w_s = w_r + int(abs(shear_x) * h_r)

        # 调整平移，防止图像移出画布
        if shear_x < 0:
            M_shear[0, 2] += abs(shear_x) * h_r

        transformed = cv2.warpAffine(rotated, M_shear, (new_w_s, h_r), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        return transformed

    return rotated


def apply_gamma_correction(img, gamma=1.0):
    """
    Gamma 校正
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def enhance_plate_image(image):
    """
    图像增强流程：放大 -> Gamma校正 -> 自适应二值化 -> 形态学膨胀
    """
    # 放大图像
    target_h = 160
    scale = target_h / image.shape[0]
    target_w = int(image.shape[1] * scale)
    image_resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    # 转灰度
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Gamma 校正
    gray = apply_gamma_correction(gray, gamma=0.8)

    # 自适应二值化
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 19, 10
    )

    # 形态学膨胀 (连接断裂字符)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # 增加边框
    border_size = 30
    binary_padded = cv2.copyMakeBorder(
        binary, border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, value=255
    )

    return binary_padded


def clean_plate_text(raw_text):
    """
    正则过滤：仅保留大写字母和数字
    """
    return re.sub(r'[^A-Z0-9]', '', raw_text.upper())


def calculate_score(res):
    """
    计算 OCR 置信度
    """
    full_text = ""
    total_score = 0.0
    count = 0

    for line in res:
        if line['score'] > 0.25:
            full_text += line['text']
            total_score += line['score']
            count += 1

    clean_text = clean_plate_text(full_text)

    avg_score = 0
    if count > 0:
        avg_score = total_score / count

    # 长度校验与惩罚
    if not (6 <= len(clean_text) <= 8):
        avg_score *= 0.5

    return clean_text, avg_score


def compute_iou(box1, box2):
    """
    计算 IoU
    """
    xx1 = max(box1[0], box2[0])
    yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2])
    yy2 = min(box1[3], box2[3])
    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    inter = w * h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    if union == 0: return 0
    return inter / union


def recognize_one_plate(plate_img, ocr):
    """
    单车牌识别逻辑：快速识别 -> 深度搜索(旋转+错切)
    """
    if not HAS_CNOCR: return "", 0.0, None

    try:
        # 1. 图像预处理
        processed_img = enhance_plate_image(plate_img)

        # 2. 快速通道：直接识别
        res_fast = ocr.ocr(processed_img)
        text_fast, score_fast = calculate_score(res_fast)

        # 如果快速识别结果可信，直接返回
        if score_fast > 0.6 and (6 <= len(text_fast) <= 8):
            return text_fast, score_fast, processed_img

        # 3. 深度搜索通道：组合尝试旋转与错切
        best_text = text_fast
        best_score = score_fast
        best_img_display = processed_img

        # 定义搜索空间
        # 角度：覆盖左右倾斜
        angles = [0, -5, 5, -10, 10, -20, 20, -30, 30]
        # 错切因子：解决透视变形 (负值向左倾，正值向右倾)
        shears = [0, -0.2, 0.2, -0.4, 0.4]

        for angle in angles:
            for shear in shears:
                # 跳过已经尝试过的原图组合
                if angle == 0 and shear == 0: continue

                # 应用变换
                trans_img = apply_affine_transform(processed_img, angle, shear)

                # 识别
                res = ocr.ocr(trans_img)
                text, score = calculate_score(res)

                # 更新最佳结果
                if score > best_score:
                    best_score = score
                    best_text = text
                    best_img_display = trans_img

                    # 提前终止条件
                    if score > 0.85 and (6 <= len(text) <= 7):
                        return best_text, best_score, best_img_display

        # 返回搜索到的最佳结果
        if 6 <= len(best_text) <= 8:
            return best_text, best_score, best_img_display
        else:
            return "", 0.0, best_img_display

    except Exception as e:
        print(f"识别异常: {e}")
        return "", 0.0, None


# ================= GUI 界面类 =================

class PlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("电动车牌识别系统")
        self.root.geometry("1200x800")

        self.yolo_weights = 'weights/best.pt'
        self.device = select_device('')
        self.model = None
        self.ocr = None

        self.current_img_path = None
        self.current_cv_image = None

        self.setup_ui()
        self.load_models()

    def setup_ui(self):
        # 顶部控制栏
        control_frame = tk.Frame(self.root, pady=10)
        control_frame.pack(fill=tk.X)

        btn_select = tk.Button(control_frame, text="📂 选择图片", font=("微软雅黑", 12), command=self.select_image)
        btn_select.pack(side=tk.LEFT, padx=20)

        self.lbl_path = tk.Label(control_frame, text="未选择图片", font=("微软雅黑", 10), fg="gray")
        self.lbl_path.pack(side=tk.LEFT)

        btn_run = tk.Button(control_frame, text="🚀 开始识别", font=("微软雅黑", 12, "bold"), bg="#4CAF50", fg="white",
                            command=self.run_recognition)
        btn_run.pack(side=tk.RIGHT, padx=20)

        # 图像显示区域
        img_frame = tk.Frame(self.root)
        img_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        img_frame.columnconfigure(0, weight=1)
        img_frame.columnconfigure(1, weight=1)
        img_frame.columnconfigure(2, weight=1)

        self.panel_origin = self.create_img_panel(img_frame, "原图", 0)
        self.panel_binary = self.create_img_panel(img_frame, "处理过程 (二值化)", 1)
        self.panel_result = self.create_img_panel(img_frame, "最终结果", 2)

        # 底部结果区域
        result_frame = tk.Frame(self.root, pady=10, bg="#f0f0f0")
        result_frame.pack(fill=tk.X, side=tk.BOTTOM)

        tk.Label(result_frame, text="识别结果列表:", font=("微软雅黑", 12, "bold"), bg="#f0f0f0").pack(anchor=tk.W,
                                                                                                       padx=20)

        self.txt_result = tk.Text(result_frame, height=5, font=("Consolas", 14), fg="#333")
        self.txt_result.pack(fill=tk.X, padx=20, pady=5)

    def create_img_panel(self, parent, title, col):
        frame = tk.Frame(parent, bd=2, relief=tk.GROOVE)
        frame.grid(row=0, column=col, sticky="nsew", padx=5)

        tk.Label(frame, text=title, font=("微软雅黑", 11, "bold")).pack(pady=5)

        lbl_img = tk.Label(frame, text="等待输入...", bg="white", font=("微软雅黑", 10))
        lbl_img.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        return lbl_img

    def load_models(self):
        try:
            print("正在加载 YOLO 检测模型...")
            self.model = attempt_load(self.yolo_weights, map_location=self.device)
            if HAS_CNOCR:
                print("正在加载 OCR 识别模型...")
                self.ocr = CnOcr()
            print("模型加载完成。")
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败: {e}\n请确认 weights/best.pt 文件是否存在。")

    def select_image(self):
        initial_dir = os.path.join(os.getcwd(), 'test_images')
        if not os.path.exists(initial_dir):
            initial_dir = os.getcwd()

        file_path = filedialog.askopenfilename(initialdir=initial_dir, title="选择图片",
                                               filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if file_path:
            self.current_img_path = file_path
            self.lbl_path.config(text=file_path)

            self.current_cv_image = cv2.imread(file_path)
            if self.current_cv_image is not None:
                self.show_image_in_panel(self.current_cv_image, self.panel_origin)
                self.panel_binary.config(image='', text="等待识别...")
                self.panel_result.config(image='', text="等待识别...")
                self.txt_result.delete(1.0, tk.END)

    def show_image_in_panel(self, cv_img, panel):
        if len(cv_img.shape) == 3:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        h, w = cv_img.shape[:2]
        display_h = 400
        scale = display_h / h
        display_w = int(w * scale)

        img_resized = cv2.resize(cv_img, (display_w, display_h))
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_pil)

        panel.config(image=img_tk, text="")
        panel.image = img_tk

    def run_recognition(self):
        if self.current_cv_image is None:
            messagebox.showwarning("提示", "请先选择一张图片。")
            return

        if self.model is None:
            messagebox.showerror("错误", "模型未加载。")
            return

        img0 = self.current_cv_image.copy()
        img_size = 640
        img = cv2.resize(img0, (img_size, img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3: img = img.unsqueeze(0)

        # YOLO 推理
        pred = self.model(img)[0]
        pred = non_max_suppression(pred, 0.15, 0.45, agnostic=True)

        if len(pred[0]) == 0:
            self.txt_result.insert(tk.END, "❌ 未检测到车牌。\n")
            self.show_image_in_panel(img0, self.panel_result)
            return

        det = pred[0]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        processed_boxes = []
        result_str_list = []
        best_binary_display = None

        for i, (*xyxy, conf, cls) in enumerate(det):
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            # 加大检测框外扩 (Padding) 至 15 像素，防止倾斜车牌被截断
            h_img, w_img = img0.shape[:2]
            pad = 15
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w_img, x2 + pad)
            y2 = min(h_img, y2 + pad)

            current_box = [x1, y1, x2, y2]

            # IoU 去重
            is_duplicate = False
            for old_box in processed_boxes:
                iou = compute_iou(current_box, old_box)
                if iou > 0.5:
                    is_duplicate = True
                    break
            if is_duplicate: continue
            processed_boxes.append(current_box)

            plate_img = img0[y1:y2, x1:x2]
            if plate_img.shape[0] < 10 or plate_img.shape[1] < 10: continue

            final_text, ocr_conf, binary_img = recognize_one_plate(plate_img, self.ocr)

            if final_text and (6 <= len(final_text) <= 8):
                res_str = f"[{len(processed_boxes)}] 号码: {final_text}  (置信度: {ocr_conf:.2f})"
                result_str_list.append(res_str)

                best_binary_display = binary_img

                # 绘制结果
                cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)

                font_scale = 1.5
                font_thick = 2
                (text_w, text_h), _ = cv2.getTextSize(final_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
                cv2.rectangle(img0, (x1, y1 - text_h - 20), (x1 + text_w + 10, y1), (0, 255, 0), -1)
                cv2.putText(img0, final_text, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                            font_thick)

        if best_binary_display is not None:
            self.show_image_in_panel(best_binary_display, self.panel_binary)
        else:
            self.panel_binary.config(image='', text="未提取到有效特征")

        self.show_image_in_panel(img0, self.panel_result)

        self.txt_result.delete(1.0, tk.END)
        if result_str_list:
            for s in result_str_list:
                self.txt_result.insert(tk.END, s + "\n")
        else:
            self.txt_result.insert(tk.END, "⚠️ 检测到目标，但未能识别出有效车牌号。")


if __name__ == '__main__':
    root = tk.Tk()
    app = PlateApp(root)
    root.mainloop()