#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import math
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation  # 轉 Euler Angle

# ==========================================
# 1. 動作邏輯特徵庫 (POSE_REGISTRY)
# ==========================================
POSE_REGISTRY = [
    {"keywords": ["tilts her head downward"], "desc": "F01: Tilt Down", "check": lambda r, p, roll, d: p < -15},
    {"keywords": ["turns her head over her right shoulder"], "desc": "F02: Turn Right Back", "check": lambda r, p, roll, d: r > 25 and d > 80},
    {"keywords": ["looks down to her left"], "desc": "F03: Look Down Left", "check": lambda r, p, roll, d: p < -5 and r < -10},
    {"keywords": ["head tilted right"], "desc": "F04: Tilt Right", "check": lambda r, p, roll, d: roll < -5},
    {"keywords": ["turns her head left"], "desc": "F05: Turn Left", "check": lambda r, p, roll, d: r < -10},
    {"keywords": ["turns her head back over her shoulder"], "desc": "F06: Look Back", "check": lambda r, p, roll, d: d > 90},
    {"keywords": ["looks upward", "head tilted back"], "desc": "F07: Look Up", "check": lambda r, p, roll, d: p > 10},
    {"keywords": ["faces downward"], "desc": "F08: Face Down", "check": lambda r, p, roll, d: p < -5},
    {"keywords": ["looks sideways toward the left"], "desc": "F09: Look Side Left", "check": lambda r, p, roll, d: r < -15},
    {"keywords": ["tilts her head backward"], "desc": "F10: Tilt Back", "check": lambda r, p, roll, d: p > 15},
    {"keywords": ["turns his head slightly to the right"], "desc": "M01: Turn Right (Slight)", "check": lambda r, p, roll, d: r > 5},
    {"keywords": ["looks upward"], "desc": "M02: Look Up", "check": lambda r, p, roll, d: p > 10},
    {"keywords": ["looks up and to his left"], "desc": "M03: Up Left", "check": lambda r, p, roll, d: p > 5 and r < -5},
    {"keywords": ["looks straight"], "desc": "M04: Look Straight", "check": lambda r, p, roll, d: abs(r) < 15 and abs(p) < 10},
    {"keywords": ["faces slightly downward"], "desc": "M05: Face Down (Slight)", "check": lambda r, p, roll, d: p < 5},
    {"keywords": ["tilts his head left"], "desc": "M06: Tilt Left", "check": lambda r, p, roll, d: roll > 5},
    {"keywords": ["turns his head right"], "desc": "M07: Turn Right", "check": lambda r, p, roll, d: r > 15},
    {"keywords": ["leans his head toward his right shoulder"], "desc": "M08: Lean Right", "check": lambda r, p, roll, d: roll < -5},
    {"keywords": ["looks to his right"], "desc": "M09: Look Right", "check": lambda r, p, roll, d: r > 15},
    {"keywords": ["turns his face upward to the left"], "desc": "M10: Up Left", "check": lambda r, p, roll, d: p > 5 and r < -5}
]

# ==========================================
# 2. 系統設定與工具匯入
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 嘗試匯入 Repo 內部的工具
try:
    from src.networks import get_EfficientNet_V2
    from src.fisher.fisher_utils import batch_torch_A_to_R
    HAS_REPO_UTILS = True
    print("✅ 成功載入 src.networks 及 fisher_utils")
except ImportError:
    HAS_REPO_UTILS = False
    print("⚠️ 未找到 fisher_utils，將使用內建 SVD 算法")

# MediaPipe
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
# 3. 核心運算函式 (Math Helper)
# ==========================================
def normalize_angle(angle):
    """將角度轉為 -180 ~ 180 (0為正前方)"""
    if angle is None: return None
    angle = float(angle)
    while angle > 180: angle -= 360
    while angle < -180: angle += 360
    return angle

def get_relative_yaw(body, head):
    """計算頭相對於身體的轉向"""
    if body is None or head is None: return 0
    diff = head - body
    while diff > 180: diff -= 360
    while diff < -180: diff += 360
    return diff

def limit_angle(angle):
    while angle < -180: angle += 360
    while angle > 180: angle -= 360
    return angle

def find_rule_by_prompt(prompt_text):
    """根據 Prompt 文字匹配規則"""
    if not prompt_text: return None
    text_lower = prompt_text.lower()
    for rule in POSE_REGISTRY:
        if rule["keywords"][0].lower() in text_lower:
            return rule
    return None

# ==========================================
# 4. 模型與視覺函式
# ==========================================
def load_model_correctly(checkpoint_path):
    print(f"📂 正在解析權重檔: {checkpoint_path}")
    try:
        config = SOTAConfig()
        from src.networks import get_EfficientNet_V2
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
        
        # 取得 Yaw, Pitch, Roll (並修正偏移)
        pred_pitch = limit_angle(angles[0] - 180)
        pred_yaw = limit_angle(angles[1])
        pred_roll = limit_angle(angles[2])
        
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
    box_size = max(max_x - min_x, max_y - min_y) * 1.5
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    return [int(cx - box_size/2), int(cy - box_size/2), int(cx + box_size/2), int(cy + box_size/2)]

def calc_body_yaw(landmarks):
    l_sh = landmarks[IDX_L_SHOULDER]
    r_sh = landmarks[IDX_R_SHOULDER]
    if l_sh.visibility < 0.5 or r_sh.visibility < 0.5: return None
    dx = r_sh.x - l_sh.x
    dz = r_sh.z - l_sh.z
    # Body Yaw 計算 (基於 Z 深度)
    return -math.degrees(math.atan2(dz, dx)) * 2.0 

def calc_body_roll(landmarks, width, height):
    """
    計算肩膀連線相對於水平線的傾斜角度 (Body Roll)
    利用右肩到左肩的向量計算
    正值 (+): 身體向左傾斜 (逆時針)
    負值 (-): 身體向右傾斜 (順時針)
    """
    l_sh = landmarks[IDX_L_SHOULDER]
    r_sh = landmarks[IDX_R_SHOULDER]
    if l_sh.visibility < 0.5 or r_sh.visibility < 0.5: return 0.0
    
    # 轉換為像素座標
    lx, ly = l_sh.x * width, l_sh.y * height
    rx, ry = r_sh.x * width, r_sh.y * height
    
    # 計算右肩指向左肩的向量 (R -> L)
    # 在正面圖像中，右肩在左側(x小)，左肩在右側(x大)，dx 通常為正
    dx = lx - rx
    dy = ly - ry
    
    # atan2(dy, dx): 
    # 若左肩高(y小)、右肩低(y大) -> dy < 0 -> 角度為負 (向右歪/順時針)
    # 若左肩低(y大)、右肩高(y小) -> dy > 0 -> 角度為正 (向左歪/逆時針)
    return math.degrees(math.atan2(dy, dx))

# ==========================================
# 5. 主程式
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", required=True, help="圖片資料夾")
    parser.add_argument("--out-dir", required=True, help="輸出結果資料夾")
    parser.add_argument("--checkpoint", required=True, help="模型權重路徑")
    parser.add_argument("--prompts-file", required=False, help="CSV檔案，包含 filename 與 prompt 兩欄")
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
    else:
        print("⚠️ 未提供 Prompt 檔，將只輸出角度，不進行評分。")

    # 2. 初始化
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_debug = out_dir / "debug_eval"
    out_debug.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "evaluation_result.csv"

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
    head_model = load_model_correctly(args.checkpoint)
    
    files = sorted([p for p in img_dir.rglob('*') if p.suffix.lower() in SUPPORT_EXT])
    print(f"🔍 找到 {len(files)} 張圖片")

    # 3. 開始處理
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # CSV Header: 新增 Body_Roll 與 Rel_Roll
        writer.writerow([
            "Filename", "Prompt_Desc", "Check_Result", 
            "Rel_Yaw", "Delta_Yaw", 
            "Body_Yaw", "Body_Roll", 
            "Head_Yaw", "Head_Pitch", "Head_Roll", "Rel_Roll",
            "Raw_Prompt"
        ])

        pass_count = 0
        total_checked = 0

        for idx, p in enumerate(files):
            try:
                img_pil = Image.open(p).convert("RGB")
                W, H = img_pil.size
                draw = ImageDraw.Draw(img_pil)
                img_arr = np.array(img_pil)

                # --- 步驟 A: 視覺推論 ---
                results = pose.process(img_arr)
                
                raw_body_yaw = None
                raw_body_roll = 0.0 # 預設為 0
                h_yaw, h_pitch, h_roll = 0.0, 0.0, 0.0
                delta = 0.0
                status_color = "gray"
                
                bbox = None
                
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    
                    # 1. 計算 Body Yaw
                    raw_body_yaw = calc_body_yaw(lm)
                    
                    # 2. 計算 Body Roll [新增]
                    raw_body_roll = calc_body_roll(lm, W, H)
                    
                    # 繪製肩膀
                    l_sh, r_sh = lm[IDX_L_SHOULDER], lm[IDX_R_SHOULDER]
                    if l_sh.visibility > 0.5 and r_sh.visibility > 0.5:
                        lx, ly = int(l_sh.x * W), int(l_sh.y * H)
                        rx, ry = int(r_sh.x * W), int(r_sh.y * H)
                        draw.line([(lx, ly), (rx, ry)], fill="yellow", width=2)
                        draw.ellipse((rx-5, ry-5, rx+5, ry+5), fill="red", outline="white") 
                        draw.ellipse((lx-5, ly-5, lx+5, ly+5), fill="blue", outline="white") 

                    bbox = get_face_box_from_pose(lm, W, H)
                    if bbox and head_model:
                        draw.rectangle(bbox, outline="#00FF00", width=3)
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

                # --- 步驟 B: 數據標準化與相對角度計算 ---
                norm_body = normalize_angle(raw_body_yaw)
                norm_head = normalize_angle(h_yaw)
                
                # 計算 Delta Yaw
                if norm_body is not None:
                    delta = abs(norm_head - norm_body)
                    if delta > 180: delta = 360 - delta
                
                # 計算 Relative Yaw
                rel_yaw = get_relative_yaw(norm_body, norm_head)
                
                # 計算 Relative Roll [新增]
                # 相對歪頭 = 頭的歪斜 - 身體的歪斜
                rel_roll = h_roll - raw_body_roll
                rel_roll = limit_angle(rel_roll)

                # --- 步驟 C: 邏輯判別 ---
                prompt_text = prompt_dict.get(p.name, "")
                rule = find_rule_by_prompt(prompt_text)
                
                check_result = "N/A"
                rule_desc = "No Rule"
                
                if rule:
                    rule_desc = rule['desc']
                    
                    if norm_body is None and "Turn" in rule_desc:
                        is_pass = False 
                        check_result = "FAIL (No Body)"
                        status_color = "red"
                    else:
                        # 關鍵：這裡將第三個參數從 h_roll 改為 rel_roll
                        is_pass = rule['check'](rel_yaw, h_pitch, rel_roll, delta)
                        check_result = "PASS" if is_pass else "FAIL"
                        status_color = "green" if is_pass else "red"
                        if is_pass: pass_count += 1
                        total_checked += 1
                elif prompt_text:
                    check_result = "No Rule Match"

                # --- 步驟 D: 輸出與繪圖 ---
                writer.writerow([
                    p.name, rule_desc, check_result,
                    f"{rel_yaw:.1f}", f"{delta:.1f}",
                    f"{norm_body:.1f}" if norm_body is not None else "N/A",
                    f"{raw_body_roll:.1f}", # 新增
                    f"{h_yaw:.1f}", f"{h_pitch:.1f}", f"{h_roll:.1f}", 
                    f"{rel_roll:.1f}",      # 新增
                    prompt_text
                ])

                if bbox:
                    # 顯示資訊更新為使用 Rel_Roll
                    info_txt = f"R_Yaw:{rel_yaw:.0f} P:{h_pitch:.0f} R_Roll:{rel_roll:.0f}"
                    draw.text((bbox[0], bbox[1]-20), info_txt, fill="yellow")
                    
                    if rule:
                        res_txt = f"{rule_desc}: {check_result}"
                        draw.text((bbox[0], bbox[1]-40), res_txt, fill=status_color)

                img_pil.save(out_debug / p.name)
                
                if idx % 10 == 0: print(f"處理中: {idx}/{len(files)} | {check_result}")

            except Exception as e:
                print(f"Error {p.name}: {e}")

    print(f"\n✅ 完成！")
    print(f"📊 總檢測數: {total_checked}, 通過數: {pass_count}")
    print(f"📁 結果已存至: {csv_path}")

if __name__ == "__main__":
    main()