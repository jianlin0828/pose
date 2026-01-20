#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import math
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation  # 轉 Euler Angle

# 1. 設定路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 2. 嘗試匯入 Repo 內部的工具 (如果有的話)
HAS_REPO_UTILS = False
try:
    from src.networks import get_EfficientNet_V2
    # 嘗試匯入官方的矩陣轉換工具 (比較穩)
    from src.fisher.fisher_utils import batch_torch_A_to_R
    HAS_REPO_UTILS = True
    print("✅ 成功載入 src.networks 及 fisher_utils")
except ImportError:
    print("⚠️ 未找到 fisher_utils，將使用內建 SVD 算法")

# 3. MediaPipe
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

def load_model_correctly(checkpoint_path):
    print(f"📂 正在解析權重檔: {checkpoint_path}")
    try:
        # 1. 建立架構
        config = SOTAConfig()
        # 注意：必須使用 src.networks 裡的工廠函式
        from src.networks import get_EfficientNet_V2
        model = get_EfficientNet_V2(config, model_name="S")
        
        # 2. 讀取 Checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # [關鍵修正] 檢查權重存放的位置
        if 'model_state_dict_ema' in checkpoint:
            print("INFO: 偵測到 EMA 權重 (最佳效果)")
            state_dict = checkpoint['model_state_dict_ema']
        elif 'model_state_dict' in checkpoint:
            print("INFO: 偵測到一般權重")
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint # 可能是單純的 state_dict

        # 3. 移除 module. 前綴
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
            
        # 4. 載入
        model.load_state_dict(new_state_dict, strict=True)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ 模型載入嚴重失敗: {e}")
        return None

def limit_angle(angle):
    # 來自 image.py 的角度限制邏輯
    while angle < -180:
        angle += 360
    while angle > 180:
        angle -= 360
    return angle

def compute_pose_output(output_tensor):
    """
    將 9D 輸出轉為 Euler Angles (Yaw, Pitch)
    """
    # 方法 A: 使用官方工具 (如果匯入成功)
    if HAS_REPO_UTILS:
        with torch.no_grad():
            # 這會處理 SVD 並確保 det=1
            rot_mat = batch_torch_A_to_R(output_tensor) 
            rot_mat = rot_mat.cpu().numpy()[0] # (3, 3)
    else:
        # 方法 B: 手刻 SVD (備案)
        A = output_tensor.view(-1, 3, 3)
        U, S, V = torch.linalg.svd(A)
        R = torch.matmul(U, V.transpose(1, 2))
        det = torch.det(R)
        # 修正行列式為負的情況 (翻轉矩陣)
        if det < 0:
            # 修正 V 的最後一列 (對應最小奇異值)
            V_fixed = V.clone()
            V_fixed[:, :, 2] *= -1
            R = torch.matmul(U, V_fixed.transpose(1, 2))
        rot_mat = R.cpu().numpy()[0]

    # [關鍵修正] 參照 image.py 的後處理邏輯
    # image.py: rot_mat_2 = np.transpose(rot_mat)
    rot_mat_2 = np.transpose(rot_mat)
    
    try:
        r = Rotation.from_matrix(rot_mat_2)
        angles = r.as_euler("xyz", degrees=True)
        
        # image.py 邏輯: 
        # roll  = angle[2]
        # pitch = angle[0] - 180
        # yaw   = angle[1]
        
        raw_pitch = angles[0]
        raw_yaw = angles[1]
        raw_roll = angles[2]
        
        pred_pitch = limit_angle(raw_pitch - 180)
        pred_yaw = limit_angle(raw_yaw)
        pred_roll = limit_angle(raw_roll)
        
        return pred_yaw, pred_pitch, pred_roll
    except Exception as e:
        print(f"矩陣轉換失敗: {e}")
        return 0.0, 0.0, 0.0

def get_face_box_from_pose(landmarks, w, h):
    x_coords = [landmarks[i].x * w for i in FACE_LANDMARKS_INDICES]
    y_coords = [landmarks[i].y * h for i in FACE_LANDMARKS_INDICES]
    if not x_coords or not y_coords: return None

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    # 放大 1.5 倍
    box_size = max(max_x - min_x, max_y - min_y) * 1.5
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    x1 = int(center_x - box_size / 2)
    y1 = int(center_y - box_size / 2)
    x2 = int(center_x + box_size / 2)
    y2 = int(center_y + box_size / 2)
    return [x1, y1, x2, y2]

def calc_body_yaw(landmarks):
    l_sh = landmarks[IDX_L_SHOULDER]
    r_sh = landmarks[IDX_R_SHOULDER]
    if l_sh.visibility < 0.5 or r_sh.visibility < 0.5: return None
    dx = r_sh.x - l_sh.x
    dz = r_sh.z - l_sh.z
    return -math.degrees(math.atan2(dz, dx)) * 2.0 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    if not HAS_MEDIAPIPE: return

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_debug = out_dir / "debug_sota"
    out_debug.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "analysis_wildhead.csv"

    print("=== 初始化 MediaPipe Pose ===")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)

    print("=== 初始化 SemiUHPE (SOTA Fixed) ===")
    head_model = load_model_correctly(args.checkpoint)
    
    files = sorted([p for p in img_dir.rglob('*') if p.suffix.lower() in SUPPORT_EXT])
    print(f"找到 {len(files)} 張圖片")

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Body_Yaw", "Head_Yaw", "Head_Pitch", "Head_Roll", "Delta_Yaw", "Status"])

        for idx, p in enumerate(files):
            try:
                img_pil = Image.open(p).convert("RGB")
                W, H = img_pil.size
                draw = ImageDraw.Draw(img_pil)
                img_arr = np.array(img_pil)

                results = pose.process(img_arr)
                
                body_yaw = None
                head_yaw, head_pitch = 0.0, 0.0
                delta = 0.0
                status = "MISSING"

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    body_yaw = calc_body_yaw(lm)
                    
                    l_sh, r_sh = lm[IDX_L_SHOULDER], lm[IDX_R_SHOULDER]
                    if l_sh.visibility > 0.5 and r_sh.visibility > 0.5:
                        lx, ly = int(l_sh.x * W), int(l_sh.y * H)
                        rx, ry = int(r_sh.x * W), int(r_sh.y * H)
                        # 畫線連接兩肩 (黃色)
                        draw.line([(lx, ly), (rx, ry)], fill="yellow", width=2)

                        # 🔴 右肩 (Right Shoulder) -> 畫藍色圓點
                        r_rad = 5 # 半徑
                        draw.ellipse((rx - r_rad, ry - r_rad, rx + r_rad, ry + r_rad), fill="blue", outline="white")

                        # 🔵 左肩 (Left Shoulder) -> 畫紅色圓點
                        l_rad = 5 # 半徑
                        draw.ellipse((lx - l_rad, ly - l_rad, lx + l_rad, ly + l_rad), fill="red", outline="white")

                    bbox = get_face_box_from_pose(lm, W, H)
                    if bbox:
                        draw.rectangle(bbox, outline="#00FF00", width=3)
                        
                        if head_model:
                            x1, y1, x2, y2 = bbox
                            # 邊界保護
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(W, x2), min(H, y2)
                            
                            if (x2-x1) > 5 and (y2-y1) > 5:
                                crop = img_pil.crop((x1, y1, x2, y2))
                                tf = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
                                input_t = tf(crop).unsqueeze(0).to(DEVICE)
                                
                                with torch.no_grad():
                                    out = head_model(input_t) # (1, 9)
                                    head_yaw, head_pitch, head_roll= compute_pose_output(out)

                        if body_yaw:
                            delta = abs(head_yaw - body_yaw)
                            if delta > 180: delta = 360 - delta
                        
                        txt = f"H:{head_yaw:.0f} P:{head_pitch:.0f}"
                        draw.text((bbox[0], bbox[1]-20), txt, fill="red")
                        status = "OK"
                    else:
                        status = "NO_FACE_BOX"
                
                writer.writerow([p.name, 
                                 f"{body_yaw:.2f}" if body_yaw else "N/A",
                                 f"{head_yaw:.2f}", f"{head_pitch:.2f}",f"{head_roll:.2f}",
                                 f"{delta:.2f}" if body_yaw else "N/A", status])
                
                img_pil.save(out_debug / p.name)
                if idx % 10 == 0: print(f"處理中: {idx}/{len(files)}")

            except Exception as e:
                print(f"Error {p.name}: {e}")

    print(f"完成! 結果已存至: {csv_path}")

if __name__ == "__main__":
    main()