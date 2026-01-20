# Usage Example:
#python eval_pose_v2.py --img-dir "圖片資料夾路徑" --out-dir "輸出路徑" --checkpoint "模型權重路徑" --prompts-file "你的prompt.csv"


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

# ==========================================
# 1. 新版 GT 標註表
# ==========================================
PROMPT_TO_GT = {
    # --- Back View 類 ---
    "turns her head back over her shoulder": "Back_Over_Shoulder",
    "turns her head over her right shoulder": "Back_Over_Shoulder",

    # --- Head Turn 類 ---
    "turns her head left": "Head_Turn_Left",
    "looks sideways toward the left": "Head_Turn_Left",
    "turns his head right": "Head_Turn_Right",
    "looks to his right": "Head_Turn_Right",
    "turns his head slightly to the right": "Head_Slight_Right",

    # --- Tilt / Lean 類 ---
    "tilts his head left": "Head_Tilt_Left",
    "head tilted right": "Head_Tilt_Right",
    "leans his head toward his right shoulder": "Head_Tilt_Right",
    
    # --- Slight / Frontal 類 ---
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

# ==========================================
# 2. 系統設定
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from src.networks import get_EfficientNet_V2
    from src.fisher.fisher_utils import batch_torch_A_to_R
    HAS_REPO_UTILS = True
except ImportError:
    HAS_REPO_UTILS = False

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
# 3. 核心運算與 V-Final-Plus-Plus 分類邏輯
# ==========================================
def normalize_angle(angle):
    if angle is None: return None
    angle = float(angle)
    while angle > 180: angle -= 360
    while angle < -180: angle += 360
    return angle

def limit_angle(angle):
    while angle < -180: angle += 360
    while angle > 180: angle -= 360
    return angle

def classify_pose_v_final(b_yaw, b_roll, h_yaw, h_pitch, h_roll, delta):
    """
    V-Final-Plus-Plus 邏輯 (最高準確率版本)
    包含：早期傾斜保護、背對門檻拉扯、智慧型符號校正、非對稱主導權
    """
    abs_b_yaw = abs(b_yaw)
    abs_h_yaw = abs(h_yaw)
    
    # --- 閾值設定 ---
    if b_yaw > 0:
        THRES_BODY_SIDE_START = 35 
    else:
        THRES_BODY_SIDE_START = 20

    THRES_BODY_BACK = 89       # 背對門檻微調至 89
    THRES_HEAD_FRONT_LIMIT = 30
    THRES_HEAD_PURE_TURN = 22 
    THRES_LEAN = 5 
    THRES_TILT = 8             # 歪頭門檻下修至 8

    # -----------------------------------------------

    # --- Priority 1: 早期傾斜保護 (Early Lean) ---
    # 擴大守備範圍：Yaw < 40，防止漏到 Frontal
    if abs_b_yaw < 40 and abs(b_roll) > THRES_LEAN:
        if b_roll > 0: return "Body_Lean_Right" # 注意: MP Roll > 0 通常是向左傾(逆時針)，但需視 GT 定義。此處沿用舊邏輯方向。
        else: return "Body_Lean_Left"

    # --- Priority 2: 背對類 (Back View) ---
    if abs_b_yaw > THRES_BODY_BACK:
        if abs(delta) < 40:
            return "Back_View_Straight"
        elif abs_h_yaw < 60: 
            return "Back_Over_Shoulder"
        else:
            return "Back_View_Side_Looking_Away"

    # --- Priority 3: 強制頭轉 (限制型) ---
    if abs_h_yaw > 55 and abs_b_yaw < 60:
         return "Head_Turn_Right" if h_yaw > 0 else "Head_Turn_Left"

    # --- Priority 4: 側向動作矩陣 (Side Matrix) ---
    is_body_side = (abs_b_yaw > THRES_BODY_SIDE_START) and (abs_b_yaw <= THRES_BODY_BACK)
    
    if is_body_side:
        # [核心] 智慧型符號校正 2.0
        final_yaw_direction_sign = 1 if b_yaw > 0 else -1
        
        # 判斷衝突且頭轉明確 (>40)
        if (b_yaw * h_yaw) < 0 and abs_h_yaw > 40:
            final_yaw_direction_sign = 1 if h_yaw > 0 else -1
            
        suffix = "Right" if final_yaw_direction_sign > 0 else "Left"
        
        is_head_side = abs_h_yaw > THRES_HEAD_FRONT_LIMIT
        
        if not is_head_side:
            return f"Body_Turn_{suffix}_Face_Front"
        else:
            # 校正後的同向判斷
            corrected_b_yaw = abs_b_yaw * final_yaw_direction_sign
            
            if (corrected_b_yaw * h_yaw) > 0: 
                # 同向 - 非對稱主導權 (左6 / 右20)
                diff = abs_h_yaw - abs_b_yaw
                dominance_gap = 20 if h_yaw > 0 else 6
                
                if diff > dominance_gap:
                    return f"Head_Turn_{suffix}"
                else:
                    return f"Side_View_{suffix}"
            else: 
                return f"Head_Turn_{suffix}"

    # --- Priority 5: 純頭轉 ---
    if abs_h_yaw > THRES_HEAD_PURE_TURN:
        return "Head_Turn_Right" if h_yaw > 0 else "Head_Turn_Left"

    # --- Priority 6: 殘餘歪頭類 (Head Tilt) ---
    if h_roll > THRES_TILT: return "Head_Tilt_Left"  # Roll > 0 通常是 Left
    if h_roll < -THRES_TILT: return "Head_Tilt_Right"

    # --- Priority 7: 殘餘傾斜類 (Body Lean) ---
    if b_roll > THRES_LEAN: return "Body_Lean_Right"
    if b_roll < -THRES_LEAN: return "Body_Lean_Left"

    # --- Priority 8: 正面類 ---
    if h_yaw > 15: return "Head_Slight_Right"
    if h_yaw < -15: return "Head_Slight_Left"
    
    return "Frontal"

# ==========================================
# 4. 模型載入與計算工具
# ==========================================
def load_model_correctly(checkpoint_path):
    print(f"📂 正在解析權重檔: {checkpoint_path}")
    try:
        config = SOTAConfig()
        from src.networks import get_EfficientNet_V2
        model = get_EfficientNet_V2(config, model_name="S")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        state_dict = checkpoint.get('model_state_dict_ema', checkpoint.get('model_state_dict', checkpoint))
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return None

def compute_pose_output(output_tensor):
    if HAS_REPO_UTILS:
        with torch.no_grad():
            rot_mat = batch_torch_A_to_R(output_tensor).cpu().numpy()[0]
    else:
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
# 5. 主程式
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", required=True, help="圖片資料夾")
    parser.add_argument("--out-dir", required=True, help="輸出結果資料夾")
    parser.add_argument("--checkpoint", required=True, help="模型權重路徑")
    parser.add_argument("--prompts-file", required=False, help="CSV檔案，包含 filename 與 prompt")
    args = parser.parse_args()

    if not HAS_MEDIAPIPE:
        print("❌ 錯誤: 未安裝 MediaPipe")
        return

    # 1. 載入 Prompts
    prompt_dict = {}
    if args.prompts_file and os.path.exists(args.prompts_file):
        print(f"📖 讀取 Prompt 檔: {args.prompts_file}")
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) 
            for row in reader:
                if len(row) >= 2:
                    prompt_dict[row[0].strip()] = row[1].strip()

    # 2. 初始化
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 移除 Debug 資料夾建立邏輯
    
    csv_path = out_dir / "pose_classification_v_final.csv"

    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
    head_model = load_model_correctly(args.checkpoint)
    
    files = sorted([p for p in img_dir.rglob('*') if p.suffix.lower() in SUPPORT_EXT])
    print(f"🔍 找到 {len(files)} 張圖片")

    # 3. 開始處理
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Prompt", "gt_pose", "prediction", "Raw_Angles(BY/BR/HY/HP/HR)"])

        for idx, p in enumerate(files):
            try:
                img_pil = Image.open(p).convert("RGB")
                W, H = img_pil.size
                img_arr = np.array(img_pil)
                results = pose_detector.process(img_arr)
                
                raw_body_yaw = None
                raw_body_roll = 0.0
                h_yaw, h_pitch, h_roll = 0.0, 0.0, 0.0
                norm_body = None
                
                # --- 計算角度 ---
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    raw_body_yaw = calc_body_yaw(lm)
                    raw_body_roll = calc_body_roll(lm, W, H)
                    
                    bbox = get_face_box_from_pose(lm, W, H)
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

                # --- 數據正規化 ---
                norm_body = normalize_angle(raw_body_yaw)
                norm_h_yaw = normalize_angle(h_yaw)
                
                delta = 0.0
                if norm_body is not None:
                    delta = abs(norm_h_yaw - norm_body)
                    if delta > 180: delta = 360 - delta

                # --- 取得 Prompt 與 GT ---
                prompt_text = prompt_dict.get(p.name, "")
                
                gt_pose = "Unknown"
                # 模糊比對
                for key_prompt, val_gt in PROMPT_TO_GT.items():
                    if key_prompt.lower() in prompt_text.lower():
                        gt_pose = val_gt
                        break
                
                # --- 進行預測 (使用 V-Final-Plus-Plus 邏輯) ---
                if norm_body is None:
                    prediction = "No_Body_Detected"
                else:
                    prediction = classify_pose_v_final(norm_body, raw_body_roll, norm_h_yaw, h_pitch, h_roll, delta)

                # --- 寫入 CSV (移除 Debug 圖片繪製) ---
                str_by = f"{norm_body:.1f}" if norm_body is not None else "N/A"
                str_br = f"{raw_body_roll:.1f}" if raw_body_roll is not None else "0.0"
                str_hy = f"{norm_h_yaw:.1f}" if norm_h_yaw is not None else "0.0"
                str_hp = f"{h_pitch:.1f}" if h_pitch is not None else "0.0"
                str_hr = f"{h_roll:.1f}" if h_roll is not None else "0.0"

                angle_str = f"BY:{str_by}/BR:{str_br}/HY:{str_hy}/HP:{str_hp}/HR:{str_hr}"
                
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