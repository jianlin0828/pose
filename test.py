import os

print("🔍 正在全域搜索 EfficientNet / ResNet 模型定義...")
print(f"當前目錄: {os.getcwd()}")

target_keywords = ["class EfficientNet", "class ResNet", "class EffNet"]
found_files = []

# 遍歷所有資料夾與檔案
for root, dirs, files in os.walk("."):
    # 忽略虛擬環境與隱藏檔
    if "env" in root or ".git" in root or "__pycache__" in root:
        continue
        
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    # 檢查檔案內容是否包含模型定義
                    for kw in target_keywords:
                        if kw in content:
                            print(f"✅ 找到嫌疑檔案: {path} (包含 '{kw}')")
                            found_files.append(path)
                            break
            except Exception as e:
                pass

print("-" * 30)
if found_files:
    print("💡 建議：請根據上面的路徑，修改 run_final.py 中的 import 路徑。")
    print("例如，如果找到 'src/model.py'，則 import 應改為 'src.model'")
else:
    print("❌ 找不到任何 EfficientNet/ResNet 的類別定義！")
    print("請檢查是否完整下載了 Repo，或查看 'src' 資料夾的內容。")

# 順便列出 src 資料夾內容 (如果有)
if os.path.exists("src"):
    print("\n📂 src 資料夾內容:")
    print(os.listdir("src"))