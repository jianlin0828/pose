#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import math
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation
import joblib  # 引入 joblib 來載入機器學習模型

# ==========================================
# 1. 系統設定與標註對照
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 新版 GT 標註表 (僅用於寫入 GT 欄位供參考，不影響 AI 判斷)
PROMPT_TO_GT = {
    "turns her head back over her shoulder": "Back_Over_Shoulder",
    "turns her head over her right shoulder": "Back_Over_Shoulder",
    "turns her head left": "Head_Turn_Left",
    "looks sideways toward the left": "Head_Turn_Left",
    "turns his head right": "Head_Turn_Right",
    "looks to his right": "Head_Turn_Right",
    "turns his head slightly to the right": "Head_Slight_Right",
    "tilts his head left": "Head_Tilt_Left",
    "head tilted right": "Head_Tilt_Right",
    "leans his head toward his right shoulder": "Head_Tilt_Right",
    "looks straight": "Frontal",
    "tilts her head downward": "Frontal",
    "faces downward": "Frontal",
    "faces slightly downward": "Frontal",
    "looks down to her left": "Head_Slight_Left",
    "looks upward, head tilted back": "Frontal",
    "looks upward": "Frontal",
    "tilts her head backward": "Frontal",
    "looks up and to his left": "Head_Slight_Left",
    "turns his face upward to the left": "Head_Slight_Left",
}

HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        HAS_MEDIAPIPE = True
except ImportError:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IDX_L_SHOULDER = 11
IDX_R_SHOULDER = 12
FACE_LANDMARKS_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SUPPORT_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class SOTAConfig:
    def __init__(self):
        self.num_classes = 9 

# ==========================================
# 2. 輔助函數
# ==========================================
def normalize_angle(angle):
    if angle is None: return 0.0 # ML 模型需要數值，None 補 0
    angle = float(angle)
    while angle > 180: angle -= 360
    while angle < -180: angle += 360
    return angle

def limit_angle(angle):
    while angle < -180: angle += 360
    while angle > 180: angle -= 360
    return angle

def load_head_model(checkpoint_path):
    print(f"📂 正在解析 SemiUHPE 權重檔: {checkpoint_path}")
    try:
        from src.networks import get_EfficientNet_V2
        from src.fisher.fisher_utils import batch_torch_A_to_R  # 確保這行不會報錯，若沒有請自行調整 import
        
        config = SOTAConfig()
        model = get_EfficientNet_V2(config, model_name="S")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # 處理 key 的前綴
        state_dict = checkpoint.get('model_state_dict_ema', checkpoint.get('model_state_dict', checkpoint))
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        model.load_state_dict(new_state_dict, strict=True)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ SemiUHPE 模型載入失敗: {e}")
        return None

def compute_pose_output(output_tensor):
    # 這裡簡化處理，假設已有 batch_torch_A_to_R
    try:
        from src.fisher.fisher_utils import batch_torch_A_to_R
        with torch.no_grad():
            rot_mat = batch_torch_A_to_R(output_tensor).cpu().numpy()[0]
            rot_mat_2 = np.transpose(rot_mat)
            r = Rotation.from_matrix(rot_mat_2)
            angles = r.as_euler("xyz", degrees=True)
            return limit_angle(angles[1]), limit_angle(angles[0] - 180), limit_angle(angles[2])
    except:
        return 0.0, 0.0, 0.0

def get_face_box_from_pose(landmarks, w, h):
    x_coords = [landmarks[i].x * w for i in FACE_LANDMARKS_INDICES]
    y_coords = [landmarks[i].y * h for i in FACE_LANDMARKS_INDICES]
    if not x_coords: return None
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    box_size = max(max_x - min_x, max_y - min_y) * 1.5
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    return [int(cx - box_size/2), int(cy - box_size/2), int(cx + box_size/2), int(cy + box_size/2)]

def calc_body_yaw(landmarks):
    l_sh = landmarks[IDX_L_SHOULDER]
    r_sh = landmarks[IDX_R_SHOULDER]
    if l_sh.visibility < 0.5 or r_sh.visibility < 0.5: return None
    dx, dz = r_sh.x - l_sh.x, r_sh.z - l_sh.z
    return -math.degrees(math.atan2(dz, dx)) * 2.0 

def calc_body_roll(landmarks, width, height):
    l_sh = landmarks[IDX_L_SHOULDER]
    r_sh = landmarks[IDX_R_SHOULDER]
    if l_sh.visibility < 0.5 or r_sh.visibility < 0.5: return 0.0
    lx, ly = l_sh.x * width, l_sh.y * height
    rx, ry = r_sh.x * width, r_sh.y * height
    return math.degrees(math.atan2(ly - ry, lx - rx))

# ==========================================
# 3. 主程式
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", required=True, help="圖片資料夾")
    parser.add_argument("--out-dir", required=True, help="輸出結果資料夾")
    parser.add_argument("--checkpoint", required=True, help="SemiUHPE 模型權重 (.pth)")
    parser.add_argument("--ml-model", required=True, help="機器學習決策模型 (.pkl)")
    parser.add_argument("--prompts-file", required=False, help="CSV檔案，包含 filename 與 prompt")
    args = parser.parse_args()

    if not HAS_MEDIAPIPE:
        print("❌ 錯誤: 未安裝 MediaPipe (pip install mediapipe)")
        return

    # 1. 載入 Prompt 對照表
    prompt_dict = {}
    if args.prompts_file and os.path.exists(args.prompts_file):
        print(f"📖 讀取 Prompt 檔: {args.prompts_file}")
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) 
            for row in reader:
                if len(row) >= 2:
                    prompt_dict[row[0].strip()] = row[1].strip()

    # 2. 載入 ML 分類模型 (Decision Tree / Random Forest)
    print(f"🤖 載入決策模型: {args.ml_model}")
    if os.path.exists(args.ml_model):
        pose_classifier = joblib.load(args.ml_model)
    else:
        print("❌ 錯誤: 找不到 .pkl 模型檔案！")
        return

    # 3. 初始化工具
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "pose_classification_ml.csv"

    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
    head_model = load_head_model(args.checkpoint)
    
    files = sorted([p for p in img_dir.rglob('*') if p.suffix.lower() in SUPPORT_EXT])
    print(f"🔍 找到 {len(files)} 張圖片")

    # 4. 開始處理
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 標題新增 Confidence (如果模型支援 predict_proba)
        writer.writerow(["Filename", "Prompt", "GT_Pose", "AI_Prediction", "Raw_Angles(BY/BP/BR/HY/HP/HR)"])

        for idx, p in enumerate(files):
            try:
                img_pil = Image.open(p).convert("RGB")
                W, H = img_pil.size
                img_arr = np.array(img_pil)
                
                # A. MediaPipe 提取身體特徵
                results = pose_detector.process(img_arr)
                
                raw_body_yaw = None
                raw_body_roll = 0.0
                h_yaw, h_pitch, h_roll = 0.0, 0.0, 0.0
                norm_body_yaw = 0.0
                
                bbox = None

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    raw_body_yaw = calc_body_yaw(lm)
                    raw_body_roll = calc_body_roll(lm, W, H)
                    norm_body_yaw = normalize_angle(raw_body_yaw)
                    bbox = get_face_box_from_pose(lm, W, H)

                # B. SemiUHPE 提取頭部特徵
                if bbox and head_model:
                    x1, y1, x2, y2 = bbox
                    crop = img_pil.crop((max(0, x1), max(0, y1), min(W, x2), min(H, y2)))
                    if crop.size[0] > 5 and crop.size[1] > 5:
                        tf = transforms.Compose([
                            transforms.Resize((224, 224)), transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        input_t = tf(crop).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            out = head_model(input_t)
                            h_yaw, h_pitch, h_roll = compute_pose_output(out)
                
                # C. 準備特徵向量 (Feature Vector)
                # 訓練時的特徵順序: [BodyYaw, BodyPitch, BodyRoll, HeadYaw, HeadPitch, HeadRoll]
                # 注意: MediaPipe 2D 沒有 Body Pitch，故補 0.0
                
                prediction = "Unknown"
                
                if raw_body_yaw is None:
                    prediction = "No_Body_Detected"
                else:
                    # 構建 1x6 的特徵陣列
                    features = np.array([[
                        norm_body_yaw,   # Body Yaw
                        0.0,             # Body Pitch (MP 2D 不支援，補 0)
                        raw_body_roll,   # Body Roll
                        h_yaw,           # Head Yaw
                        h_pitch,         # Head Pitch
                        h_roll           # Head Roll
                    ]])
                    
                    # D. AI 模型預測
                    prediction = pose_classifier.predict(features)[0]

                # E. 取得 Prompt 對照 (僅供參考)
                prompt_text = prompt_dict.get(p.name, "")
                gt_pose = "Unknown"
                for key_prompt, val_gt in PROMPT_TO_GT.items():
                    if key_prompt.lower() in prompt_text.lower():
                        gt_pose = val_gt
                        break

                # F. 寫入結果
                angle_str = f"{norm_body_yaw:.1f}/0.0/{raw_body_roll:.1f}/{h_yaw:.1f}/{h_pitch:.1f}/{h_roll:.1f}"
                
                writer.writerow([
                    p.name, 
                    prompt_text, 
                    gt_pose, 
                    prediction,
                    angle_str
                ])

                if idx % 50 == 0: print(f"處理中: {idx}/{len(files)} -> {prediction}")

            except Exception as e:
                print(f"Error processing {p.name}: {e}")

    print(f"\n✅ 完成！結果已存至: {csv_path}")

if __name__ == "__main__":
    main()