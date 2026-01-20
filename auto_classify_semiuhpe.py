#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_classify_semiuhpe_v7_final.py
功能：
1. [MediaPipe] 修正 Body Roll 計算 (179.7度問題)。
2. [最佳邏輯整合]:
   - Level 1: 嚴格回眸 (Back Over Shoulder 需看鏡頭)。
   - Level 2: 複合視線 (角落) 與 傾斜。
   - Level 3: 矩陣式側向分類 (Head Turn vs Body Turn vs Side View)。
"""

import os
import sys
import shutil
import argparse
import math
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation

# ---- 1. 環境設定 ----
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from src.networks import get_EfficientNet_V2
    from src.fisher.fisher_utils import batch_torch_A_to_R
    HAS_REPO_UTILS = True
except ImportError:
    HAS_REPO_UTILS = False
    print("❌ 找不到 src.networks")

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("❌ 請安裝 MediaPipe")

IDX_L_SHOULDER = 11
IDX_R_SHOULDER = 12
FACE_LANDMARKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
SUPPORT_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class SOTAConfig:
    def __init__(self):
        self.num_classes = 9 

# ==========================================
# 2. 計算函式
# ==========================================
def load_model_correctly(checkpoint_path):
    print(f"📂 載入權重: {checkpoint_path}")
    try:
        config = SOTAConfig()
        model = get_EfficientNet_V2(config, model_name="S")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        if 'model_state_dict_ema' in checkpoint:
            state_dict = checkpoint['model_state_dict_ema']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return None

def normalize_angle_180(angle):
    if angle is None: return None
    while angle > 180: angle -= 360
    while angle < -180: angle += 360
    return angle

def compute_pose_output(output_tensor):
    if HAS_REPO_UTILS:
        with torch.no_grad():
            rot_mat = batch_torch_A_to_R(output_tensor).cpu().numpy()[0]
    else:
        # SVD Fallback
        A = output_tensor.view(-1, 3, 3)
        U, S, V = torch.linalg.svd(A)
        R = torch.matmul(U, V.transpose(1, 2))
        if torch.det(R) < 0:
            V_fixed = V.clone()
            V_fixed[:, :, 2] *= -1
            R = torch.matmul(U, V_fixed.transpose(1, 2))
        rot_mat = R.cpu().numpy()[0]

    rot_mat_2 = np.transpose(rot_mat)
    try:
        r = Rotation.from_matrix(rot_mat_2)
        angles = r.as_euler("xyz", degrees=True)
        pred_pitch = normalize_angle_180(angles[0] - 180)
        pred_yaw = normalize_angle_180(angles[1])
        pred_roll = normalize_angle_180(angles[2])
        return pred_yaw, pred_pitch, pred_roll
    except:
        return 0.0, 0.0, 0.0

def calc_body_yaw_mp(landmarks):
    l = landmarks[IDX_L_SHOULDER]
    r = landmarks[IDX_R_SHOULDER]
    if l.visibility < 0.5 or r.visibility < 0.5: return None
    dx = r.x - l.x
    dz = r.z - l.z
    raw_angle = -math.degrees(math.atan2(dz, dx)) * 2.0
    return normalize_angle_180(raw_angle)

def calc_body_roll_mp(landmarks, w, h):
    l = landmarks[IDX_L_SHOULDER]
    r = landmarks[IDX_R_SHOULDER]
    if l.visibility < 0.5 or r.visibility < 0.5: return 0.0
    lx, ly = l.x * w, l.y * h
    rx, ry = r.x * w, r.y * h
    # 修正: 使用 abs(dx) 避免 180 度翻轉問題
    dx = abs(rx - lx) 
    dy = ry - ly 
    angle = math.degrees(math.atan2(dy, dx))
    return normalize_angle_180(angle)

def get_face_box(landmarks, w, h):
    xs = [landmarks[i].x * w for i in FACE_LANDMARKS]
    ys = [landmarks[i].y * h for i in FACE_LANDMARKS]
    if not xs: return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    size = max(x2-x1, y2-y1) * 1.5
    return [int(cx - size/2), int(cy - size/2), int(cx + size/2), int(cy + size/2)]

# ==========================================
# 3. [最佳邏輯] V7 分類矩陣
# ==========================================
def classify_action(b_yaw, b_roll, h_yaw, h_pitch, h_roll, delta):
    if b_yaw is None or h_yaw is None: return "Unknown_Fail"

    # 是否扭脖子 (用於判斷是否真的在轉頭)
    is_neck_twisted = abs(delta) > 25

    # --- Level 1: 極端姿態 (背對與回眸 - V7修正版) ---
    if abs(b_yaw) > 110: # 身體背對
        
        # 情況 A: 純背影 (頭身角度差很小 -> 沒轉頭)
        if abs(delta) < 40: 
            return "Back_View_Straight"
        
        # 情況 B: 回眸 (Back Over Shoulder)
        # 條件: 脖子扭了(Delta大) AND 臉轉回來了(Head < 90)
        elif abs(h_yaw) < 90:
            return "Back_Over_Shoulder"
            
        # 情況 C: 側背影 (Back View Side)
        # 條件: 脖子扭了，但臉還是朝旁邊 (例如看牆壁)
        else:
            return "Back_View_Side_Looking_Away"

    # 身體傾斜
    if b_roll > 15: return "Body_Lean_Right"
    if b_roll < -15: return "Body_Lean_Left"

    # --- Level 2: 複合視線 (角落) ---
    # 必須 "看向上下" 且 "脖子有扭動" (避免身體歪著看被算進來)
    if h_pitch > 15:
        if h_yaw > 15 and is_neck_twisted: return "Look_Up_Right"
        if h_yaw < -15 and is_neck_twisted: return "Look_Up_Left"
    if h_pitch < -15:
        if h_yaw > 15 and is_neck_twisted: return "Look_Down_Right"
        if h_yaw < -15 and is_neck_twisted: return "Look_Down_Left"
    
    # 純垂直
    if h_pitch > 20: return "Look_Up"
    if h_pitch < -20: return "Look_Down"
    
    # 歪頭
    if h_roll > 20: return "Head_Tilt_Left"
    if h_roll < -20: return "Head_Tilt_Right"

    # --- Level 3: 側向動作細分 (V6 矩陣邏輯) ---
    
    # 定義狀態
    is_body_side = abs(b_yaw) > 45        # 身體是否側轉
    is_head_side = abs(h_yaw) > 30        # 頭是否看旁邊
    
    if is_body_side:
        # [情境 1] 身體側轉
        if not is_head_side:
            # 頭看中間 -> "身體轉 (Face Front)" (Counter-pose)
            if b_yaw > 0: return "Body_Turn_Face_Front_Right"
            else: return "Body_Turn_Face_Front_Left"
        else:
            # 頭也看旁邊
            # 檢查是否同方向 (同號)
            if (b_yaw * h_yaw) > 0:
                # 同方向 -> "側身照 (Side View)" (自然側拍)
                if b_yaw > 0: return "Side_View_Right"
                else: return "Side_View_Left"
            else:
                # 反方向 (身體右，頭左) -> "頭轉" (視線為主)
                if h_yaw > 0: return "Head_Turn_Right"
                else: return "Head_Turn_Left"

    else:
        # [情境 2] 身體正面
        if is_head_side:
            # 頭看旁邊 -> "頭轉 (Head Turn)"
            if h_yaw > 0: return "Head_Turn_Right"
            else: return "Head_Turn_Left"
        
    # --- Level 4: 微調與正面 ---
    if h_yaw > 15: return "Head_Slight_Right"
    if h_yaw < -15: return "Head_Slight_Left"
    
    return "Frontal"

# ==========================================
# 4. 主程式
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    if not HAS_REPO_UTILS or not HAS_MEDIAPIPE: return

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_debug = out_dir / "debug_images"
    out_csv = out_dir / "classification_report.csv"
    out_debug.mkdir(parents=True, exist_ok=True)

    print("Loading Models...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
    
    head_model = load_model_correctly(args.checkpoint)
    if not head_model: return

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    files = sorted([p for p in img_dir.rglob('*') if p.suffix.lower() in SUPPORT_EXT])
    print(f"🔍 處理 {len(files)} 張圖片...")

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Action_Class", "Body_Yaw", "Body_Roll", "Head_Yaw", "Head_Pitch", "Head_Roll", "Delta"])

        for idx, p in enumerate(files):
            try:
                pil_img = Image.open(p).convert("RGB")
                w, h = pil_img.size
                draw = ImageDraw.Draw(pil_img)
                img_arr = np.array(pil_img)

                # --- Body ---
                results = pose.process(img_arr)
                b_yaw = None
                b_roll = 0.0
                bbox = None
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    b_yaw = calc_body_yaw_mp(lm)
                    b_roll = calc_body_roll_mp(lm, w, h)
                    bbox = get_face_box(lm, w, h)
                    l, r = lm[IDX_L_SHOULDER], lm[IDX_R_SHOULDER]
                    if l.visibility > 0.5 and r.visibility > 0.5:
                        lx, ly = int(l.x*w), int(l.y*h)
                        rx, ry = int(r.x*w), int(r.y*h)
                        draw.line([(lx, ly), (rx, ry)], fill="yellow", width=3)

                # --- Head ---
                h_yaw, h_pitch, h_roll = 0.0, 0.0, 0.0
                if bbox:
                    x1, y1, x2, y2 = bbox
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    crop = pil_img.crop((x1, y1, x2, y2))
                    if crop.size[0] > 10 and crop.size[1] > 10:
                        input_tensor = data_transform(crop).unsqueeze(0).to(DEVICE)
                        output = head_model(input_tensor)
                        h_yaw, h_pitch, h_roll = compute_pose_output(output)
                        draw.rectangle(bbox, outline="#00FF00", width=2)

                # --- 分類 ---
                delta = 0
                if b_yaw is not None:
                    delta = abs(h_yaw - b_yaw)
                    if delta > 180: delta = 360 - delta
                
                action_class = classify_action(b_yaw, b_roll, h_yaw, h_pitch, h_roll, delta)

                writer.writerow([
                    p.name, action_class,
                    f"{b_yaw:.1f}" if b_yaw is not None else "N/A",
                    f"{b_roll:.1f}", 
                    f"{h_yaw:.1f}", f"{h_pitch:.1f}", f"{h_roll:.1f}", 
                    f"{delta:.1f}" if b_yaw is not None else "N/A"
                ])

                # Debug Info
                is_twist = abs(delta) > 25
                info = f"{action_class}\nTwist:{is_twist}\nB:{b_yaw:.0f} H:{h_yaw:.0f}"
                draw.text((10, 10), info, fill="cyan")
                pil_img.save(out_debug / p.name)

                class_dir = out_dir / action_class
                class_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, class_dir / p.name)

                if idx % 10 == 0: print(f"[{idx}/{len(files)}] {p.name} -> {action_class}")

            except Exception as e:
                print(f"Error {p.name}: {e}")

    print(f"\n✅ 完成！分類結果已存至: {out_dir}")

if __name__ == "__main__":
    main()