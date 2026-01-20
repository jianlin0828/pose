import gradio as gr
import os
import csv
import math
import torch
import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation
import sys
import mediapipe as mp

# --- 1. 設定與模型載入 ---

# 請修改為您的權重路徑
CHECKPOINT_PATH = "checkpoints/DAD-WildHead-EffNetV2-S-best.pth" 
# 如果 src 在當前目錄
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 環境設定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IDX_L_SHOULDER = 11
IDX_R_SHOULDER = 12
FACE_LANDMARKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 嘗試載入依賴
try:
    from src.networks import get_EfficientNet_V2
    from src.fisher.fisher_utils import batch_torch_A_to_R
    HAS_DEPS = True
    mp_pose = mp.solutions.pose
except ImportError:
    HAS_DEPS = False
    print("❌ 缺少必要套件 (src, mediapipe)，請檢查環境。")

class SOTAConfig:
    def __init__(self):
        self.num_classes = 9

# --- 模型載入函數 ---
def load_models(checkpoint_path):
    if not HAS_DEPS: return None, None
    print(f"📂 載入模型權重: {checkpoint_path}")
    try:
        # Load Head Model
        config = SOTAConfig()
        head_model = get_EfficientNet_V2(config, model_name="S")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        state_dict = checkpoint.get('model_state_dict_ema', checkpoint.get('model_state_dict', checkpoint))
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        head_model.load_state_dict(new_state_dict, strict=True)
        head_model.to(DEVICE)
        head_model.eval()

        # Load MediaPipe
        pose_estimator = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
        
        return head_model, pose_estimator
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return None, None

# --- 核心計算邏輯 ---
def normalize_angle_180(angle):
    if angle is None: return None
    while angle > 180: angle -= 360
    while angle < -180: angle += 360
    return angle

def get_face_box(landmarks, w, h):
    xs = [landmarks[i].x * w for i in FACE_LANDMARKS]
    ys = [landmarks[i].y * h for i in FACE_LANDMARKS]
    if not xs: return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    size = max(x2-x1, y2-y1) * 1.5
    return [int(cx - size/2), int(cy - size/2), int(cx + size/2), int(cy + size/2)]

def calc_angles(lm, w, h):
    l = lm[IDX_L_SHOULDER]
    r = lm[IDX_R_SHOULDER]
    b_yaw = None
    b_roll = 0.0
    if l.visibility > 0.5 and r.visibility > 0.5:
        # 修正後的 Body Yaw 計算 (l - r)
        dx = l.x - r.x
        dz = l.z - r.z
        b_yaw = normalize_angle_180(-math.degrees(math.atan2(dz, dx)) )
        
        # Body Roll
        lx_px, ly_px = l.x * w, l.y * h
        rx_px, ry_px = r.x * w, r.y * h
        dx_roll = abs(rx_px - lx_px)
        dy_roll = ry_px - ly_px
        b_roll = normalize_angle_180(math.degrees(math.atan2(dy_roll, dx_roll)))
        
    return b_yaw, b_roll

# --- V-Final-Optimized 邏輯 ---
def classify_custom_priority(b_yaw, b_roll, h_yaw, h_pitch, h_roll, delta):
    if b_yaw is None or h_yaw is None: return "Unknown_Fail"
    
    abs_b_yaw = abs(b_yaw)
    abs_h_yaw = abs(h_yaw)
    
    # --- 1. 閾值設定 ---
    
    if b_yaw > 0:
        THRES_BODY_SIDE_START = 35 
    else:
        THRES_BODY_SIDE_START = 20

    THRES_BODY_BACK = 89
    THRES_HEAD_FRONT_LIMIT = 30
    THRES_HEAD_PURE_TURN = 22 
    THRES_LEAN = 5 
    THRES_TILT = 8 

    # -----------------------------------------------

    # --- Priority 1: 早期傾斜保護 (Early Lean) ---
    # [修正] 擴大守備範圍：Yaw < 40 (原30)
    # 只要在 40 度以內且有歪，都算傾斜，防止漏到 Frontal
    if abs_b_yaw < 40 and abs(b_roll) > THRES_LEAN:
        if b_roll > 0: return "Body_Lean_Right"
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
        # [核心修正] 智慧型符號校正 2.0 (Smart Sign Correction)
        # 目標：解決 Side_View 和 Body_Turn 的左右鏡像問題
        
        final_yaw_direction_sign = 1 if b_yaw > 0 else -1
        
        # 判斷是否發生方向衝突 (Body 與 Head 符號相反)
        is_conflict = (b_yaw * h_yaw) < 0
        
        # 條件：如果發生衝突，且 頭部轉動非常明確 (>40)
        # 我們假設這不是 Counter-pose，而是 MP 的 Z 軸判斷錯誤 -> 相信頭部
        if is_conflict and abs_h_yaw > 40:
            final_yaw_direction_sign = 1 if h_yaw > 0 else -1
            
        # 備註：如果頭轉 < 40 且衝突，我們保留 MP 原判 (視為 Counter-pose)，避免誤殺
            
        suffix = "Right" if final_yaw_direction_sign > 0 else "Left"
        
        # 開始分類
        is_head_side = abs_h_yaw > THRES_HEAD_FRONT_LIMIT
        
        if not is_head_side:
            # 身體側，頭正
            return f"Body_Turn_{suffix}_Face_Front"
        else:
            # 身體側，頭也側
            corrected_b_yaw = abs_b_yaw * final_yaw_direction_sign
            
            if (corrected_b_yaw * h_yaw) > 0: 
                # 同向
                diff = abs_h_yaw - abs_b_yaw
                
                # [核心修正] 非對稱主導權
                # 左側 gap 降至 6 (原10)，右側維持 20
                dominance_gap = 20 if h_yaw > 0 else 6
                
                if diff > dominance_gap:
                    return f"Head_Turn_{suffix}"
                else:
                    return f"Side_View_{suffix}"
            else: 
                # 反向
                return f"Head_Turn_{suffix}"

    # --- Priority 5: 純頭轉 ---
    if abs_h_yaw > THRES_HEAD_PURE_TURN:
        return "Head_Turn_Right" if h_yaw > 0 else "Head_Turn_Left"

    # --- Priority 6: 殘餘歪頭類 (Head Tilt) ---
    if h_roll > THRES_TILT: return "Head_Tilt_Left"
    if h_roll < -THRES_TILT: return "Head_Tilt_Right"

    # --- Priority 7: 殘餘傾斜類 (Body Lean) ---
    if b_roll > THRES_LEAN: return "Body_Lean_Right"
    if b_roll < -THRES_LEAN: return "Body_Lean_Left"

    # --- Priority 8: 正面類 ---
    if h_yaw > 15: return "Head_Slight_Right"
    if h_yaw < -15: return "Head_Slight_Left"
    
    return "Frontal"

# --- 推論與資料處理 ---
def run_inference(img_path, head_model, pose_estimator):
    """回傳: 預測類別, 乾淨的圖, 詳細數據字串"""
    if not os.path.exists(img_path):
        return "File Not Found", None, "無數據"

    pil_img = Image.open(img_path).convert("RGB")
    w, h = pil_img.size
    img_arr = np.array(pil_img)
    draw = ImageDraw.Draw(pil_img)

    # 1. MediaPipe
    results = pose_estimator.process(img_arr)
    b_yaw, b_roll = None, 0.0
    bbox = None
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        b_yaw, b_roll = calc_angles(lm, w, h)
        bbox = get_face_box(lm, w, h)
        
        # 畫骨架
        l = lm[IDX_L_SHOULDER]
        r = lm[IDX_R_SHOULDER]
        if l.visibility > 0.5 and r.visibility > 0.5:
            lx, ly = int(l.x*w), int(l.y*h)
            rx, ry = int(r.x*w), int(r.y*h)
            draw.line([(lx, ly), (rx, ry)], fill="yellow", width=3)
            draw.ellipse((lx-5, ly-5, lx+5, ly+5), fill="red")
            draw.ellipse((rx-5, ry-5, rx+5, ry+5), fill="blue")

    # 2. SemiUHPE
    h_yaw, h_pitch, h_roll = 0.0, 0.0, 0.0
    if bbox and head_model:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = pil_img.crop((x1, y1, x2, y2))
        
        if crop.size[0] > 10 and crop.size[1] > 10:
            tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            with torch.no_grad():
                input_t = tf(crop).unsqueeze(0).to(DEVICE)
                out = head_model(input_t)
                rot_mat = batch_torch_A_to_R(out).cpu().numpy()[0]
                r = Rotation.from_matrix(np.transpose(rot_mat))
                angles = r.as_euler("xyz", degrees=True)
                h_pitch = normalize_angle_180(angles[0] - 180)
                h_yaw = normalize_angle_180(angles[1])
                h_roll = normalize_angle_180(angles[2])
            
            draw.rectangle(bbox, outline="#00FF00", width=2)

    # 3. Classify
    delta = 0
    if b_yaw is not None:
        delta = abs(h_yaw - b_yaw)
        if delta > 180: delta = 360 - delta

    pred_class = classify_custom_priority(b_yaw, b_roll, h_yaw, h_pitch, h_roll, delta)

    data_text = (
        f"預測類別: {pred_class}\n"
        f"---------------------------\n"
        f"Body Yaw  : {b_yaw:.1f}" if b_yaw is not None else "Body Yaw  : N/A"
    )
    data_text += f"\nBody Roll : {b_roll:.1f}"
    data_text += f"\n---------------------------"
    data_text += f"\nHead Yaw  : {h_yaw:.1f}"
    data_text += f"\nHead Pitch: {h_pitch:.1f}"
    data_text += f"\nHead Roll : {h_roll:.1f}"
    data_text += f"\n---------------------------"
    data_text += f"\nDelta     : {delta:.1f}" if b_yaw is not None else "\nDelta     : N/A"
    
    return pred_class, pil_img, data_text

# --- 評估邏輯 (含誤判分析) ---
def evaluate_dataset(csv_path, checkpoint_path, progress=gr.Progress()):
    if not os.path.exists(csv_path):
        return None, "找不到 CSV 檔案", None, None

    head_model, pose_estimator = load_models(checkpoint_path)
    if not head_model:
        return None, "模型載入失敗", None, None

    df = pd.read_csv(csv_path)
    mismatches = []
    
    total_count = 0
    correct_count = 0
    class_stats = {} 
    
    # 記錄誤判流向
    error_stats = {} 

    for idx, row in progress.tqdm(df.iterrows(), total=len(df), desc="Analyzing..."):
        img_path = row['Path']
        gt_pose = row['Pose']
        gt_sight = row.get('Sight', 'N/A')

        pred_pose, _, _ = run_inference(img_path, head_model, pose_estimator)

        if gt_pose not in class_stats:
            class_stats[gt_pose] = [0, 0] # [total, correct]
        class_stats[gt_pose][0] += 1
        total_count += 1

        if pred_pose == gt_pose:
            correct_count += 1
            class_stats[gt_pose][1] += 1
        else:
            mismatches.append({
                "idx": idx,
                "path": img_path,
                "gt_pose": gt_pose,
                "gt_sight": gt_sight,
                "pred_pose": pred_pose
            })
            
            if gt_pose not in error_stats:
                error_stats[gt_pose] = {}
            if pred_pose not in error_stats[gt_pose]:
                error_stats[gt_pose][pred_pose] = 0
            error_stats[gt_pose][pred_pose] += 1

    # --- 1. 產生主要統計表 ---
    stat_data = []
    for cls, vals in class_stats.items():
        acc = (vals[1] / vals[0]) * 100 if vals[0] > 0 else 0
        stat_data.append([cls, vals[0], vals[1], f"{acc:.1f}%"])
    
    total_acc = (correct_count / total_count) * 100 if total_count > 0 else 0
    stat_df = pd.DataFrame(stat_data, columns=["類別 (GT)", "樣本數", "正確數", "正確率"])
    stat_df = stat_df.sort_values(by="樣本數", ascending=False)

    # --- 2. 產生誤判詳細分析表 (優化版) ---
    error_data = []
    for gt, preds in error_stats.items():
        gt_total = class_stats[gt][0]
        # 計算該類別的總誤判數，方便排序
        total_errors_for_class = sum(preds.values())
        
        for pred, count in preds.items():
            rate = (count / gt_total) * 100
            # 我們加入一個隱藏權重(total_errors_for_class)來排序，讓錯誤最多的類別排前面
            error_data.append({
                "GT": gt,
                "Pred": pred,
                "Count": count,
                "Rate": f"{rate:.1f}%",
                "_sort_key": total_errors_for_class # 輔助排序用
            })
    
    # 轉成 DataFrame 並排序：先看哪個類別錯誤總數最多，再看該類別內哪個誤判最多
    error_df_raw = pd.DataFrame(error_data)
    if not error_df_raw.empty:
        error_df_raw = error_df_raw.sort_values(by=["_sort_key", "Count"], ascending=[False, False])
        # 移除輔助排序的 key
        error_df = error_df_raw[["GT", "Pred", "Count", "Rate"]]
        error_df.columns = ["真實類別 (GT)", "被誤判為 (Predicted)", "誤判數量", "誤判佔比 (佔該GT總數)"]
    else:
        error_df = pd.DataFrame(columns=["真實類別 (GT)", "被誤判為 (Predicted)", "誤判數量", "誤判佔比 (佔該GT總數)"])

    summary_text = f"總樣本數: {total_count} | 總正確率: {total_acc:.2f}% | 錯誤案例: {len(mismatches)} 張"
    
    return mismatches, summary_text, stat_df, error_df

# --- Gradio UI ---
def on_load_click(csv_file, ckpt_file):
    # 更新回傳值，多接收一個 error_df
    mismatches, summary, stat_df, error_df = evaluate_dataset(csv_file, ckpt_file)
    
    if not mismatches:
        return gr.update(visible=True), summary, stat_df, error_df, None, None, None, None, [], 0
    
    first_case = mismatches[0]
    _, img_l, info_l, img_r, info_r = load_mismatch_case(first_case, ckpt_file)
    return gr.update(visible=True), summary, stat_df, error_df, img_l, info_l, img_r, info_r, mismatches, 0

def load_mismatch_case(case_data, ckpt_path):
    global Global_Head_Model, Global_Pose_Estimator
    if 'Global_Head_Model' not in globals() or Global_Head_Model is None:
         Global_Head_Model, Global_Pose_Estimator = load_models(ckpt_path)
    
    path = case_data['path']
    gt_pose = case_data['gt_pose']
    gt_sight = case_data['gt_sight']
    
    img_left = Image.open(path).convert("RGB") if os.path.exists(path) else None
    info_left = f"File: {os.path.basename(path)}\n\n[GT]\nPose: {gt_pose}\nSight: {gt_sight}"
    pred_pose, img_right, debug_text = run_inference(path, Global_Head_Model, Global_Pose_Estimator)
    
    return 0, img_left, info_left, img_right, debug_text

def nav_click(direction, mismatch_list, current_idx, ckpt_path):
    if not mismatch_list: return None, None, None, None, current_idx
    new_idx = max(0, min(current_idx + direction, len(mismatch_list) - 1))
    case = mismatch_list[new_idx]
    _, img_l, info_l, img_r, info_r = load_mismatch_case(case, ckpt_path)
    return img_l, info_l, img_r, info_r, new_idx

with gr.Blocks(title="標註 vs 工具 差異分析器 (V-Final-Optimized)") as demo:
    state_mismatches = gr.State([])
    state_idx = gr.State(0)

    gr.Markdown("## 🔍 GT vs Tool Analysis (V-Final-Optimized Logic)")
    with gr.Row():
        inp_csv = gr.Textbox(label="CSV 路徑", value="/media/will/新增磁碟區/dataset/DeepFashion-MultiModal/1_1_final/labels.csv")
        inp_ckpt = gr.Textbox(label="權重路徑", value=CHECKPOINT_PATH)
        btn_start = gr.Button("🚀 開始", variant="primary")

    lbl_summary = gr.Textbox(label="總結", interactive=False)
    
    with gr.Column(visible=False) as result_area:
        
        # --- 上半部：左右分欄 (統計 vs 互動看圖) ---
        with gr.Row():
            # 左側：主要統計 (只放總表，比較短)
            with gr.Column(scale=1):
                gr.Markdown("### 📊 整體統計")
                df_stats = gr.Dataframe(label="正確率統計", interactive=False)
            
            # 右側：圖片互動區
            with gr.Column(scale=2):
                with gr.Row():
                    btn_prev = gr.Button("⬅️ Prev")
                    lbl_idx = gr.Label(value="0", show_label=False)
                    btn_next = gr.Button("Next ➡️")
                with gr.Row():
                    with gr.Column():
                        img_gt = gr.Image(label="GT Image", type="pil", height=500)
                        txt_gt = gr.Textbox(label="GT Info", lines=6)
                    with gr.Column():
                        img_tool = gr.Image(label="Tool Debug (Clean)", type="pil", height=500)
                        txt_tool = gr.Textbox(label="Tool Data", lines=10)

        # --- 下半部：全寬誤判分析 (移到這裡！) ---
        gr.Markdown("### 📉 誤判詳細分析")
        with gr.Accordion("點擊展開詳細誤判列表", open=True): # 預設展開或收起皆可
            # 這裡使用全寬，欄位不會再被壓縮
            df_errors = gr.Dataframe(
                label="誤判流向矩陣 (GT -> Pred)", 
                headers=["真實類別 (GT)", "被誤判為 (Predicted)", "誤判數量", "誤判佔比 (佔該GT總數)"],
                interactive=False,
                wrap=True # 讓文字自動換行，防止過長
            )

    # 事件綁定 (維持不變)
    btn_start.click(on_load_click, [inp_csv, inp_ckpt], [result_area, lbl_summary, df_stats, df_errors, img_gt, txt_gt, img_tool, txt_tool, state_mismatches, state_idx])
    btn_prev.click(lambda m, i, c: nav_click(-1, m, i, c), [state_mismatches, state_idx, inp_ckpt], [img_gt, txt_gt, img_tool, txt_tool, state_idx])
    btn_next.click(lambda m, i, c: nav_click(1, m, i, c), [state_mismatches, state_idx, inp_ckpt], [img_gt, txt_gt, img_tool, txt_tool, state_idx])
    state_idx.change(lambda i, m: f"Case {i+1} / {len(m)}", [state_idx, state_mismatches], lbl_idx)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7861)