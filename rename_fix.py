#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import csv
import argparse
from pathlib import Path

# 定義要修改的名稱映射 (舊名稱 -> 新名稱)
RENAME_MAP = {
    "Body_Turn_Face_Front_Right": "Body_Turn_Right_Face_Front",
    "Body_Turn_Face_Front_Left":  "Body_Turn_Left_Face_Front"
}

def main():
    parser = argparse.ArgumentParser(description="快速修正分類名稱與 CSV")
    parser.add_argument("--out-dir", required=True, help="包含分類資料夾與 CSV 的輸出目錄")
    args = parser.parse_args()

    base_dir = Path(args.out_dir)
    csv_path = base_dir / "classification_report.csv"

    if not base_dir.exists():
        print(f"❌ 找不到目錄: {base_dir}")
        return

    print("🚀 開始執行名稱修正...")

    # --- 步驟 1: 修改資料夾名稱 ---
    print("\n[1/2] 正在重命名資料夾...")
    for old_name, new_name in RENAME_MAP.items():
        old_path = base_dir / old_name
        new_path = base_dir / new_name

        if old_path.exists():
            if not new_path.exists():
                try:
                    os.rename(old_path, new_path)
                    print(f"  ✅ 資料夾已更名: {old_name} -> {new_name}")
                except OSError as e:
                    print(f"  ❌ 資料夾更名失敗 {old_name}: {e}")
            else:
                print(f"  ⚠️ 目標資料夾已存在，跳過更名: {new_name}")
        else:
            print(f"  ℹ️ 找不到資料夾 (可能該類別沒有圖片): {old_name}")

    # --- 步驟 2: 修改 CSV 內容 ---
    print("\n[2/2] 正在更新 CSV 報表...")
    if not csv_path.exists():
        print(f"  ❌ 找不到 CSV 檔案: {csv_path}")
        return

    updated_rows = []
    change_count = 0
    
    try:
        # 讀取 CSV
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            for row in reader:
                # 檢查 Action_Class 是否在映射表中
                if row['Action_Class'] in RENAME_MAP:
                    row['Action_Class'] = RENAME_MAP[row['Action_Class']]
                    change_count += 1
                updated_rows.append(row)

        # 寫回 CSV
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)
            
        print(f"  ✅ CSV 更新完成！共修正了 {change_count} 筆資料。")

    except Exception as e:
        print(f"  ❌ CSV 處理發生錯誤: {e}")

    print("\n🎉 全部完成！")

if __name__ == "__main__":
    main()