
# Pose Classification Evaluation Tool

A robust pose classification evaluation framework based on MediaPipe Pose and EfficientNet (Head Model). This project is designed to address common failures in monocular 2D pose estimation, such as Z-axis collapse and mirror ambiguity, by implementing a hierarchical logic correction strategy named "V-Final-Plus-Plus".

## Workflow

The inference pipeline consists of three main stages:

1.  **Ground Truth Generation**: 
    The system takes an image and its corresponding text prompt as input. The text prompt (e.g., *"looks to his right"*) is mapped to a standard pose class label (e.g., `Head_Turn_Right`) using a predefined dictionary. This rule-based mapping serves as the Ground Truth (GT) for evaluation.

2.  **Pose Estimation**:
    - **Body**: MediaPipe Pose extracts body landmarks (Yaw/Roll).
    - **Head**: EfficientNet-V2 predicts head angles (Yaw/Pitch/Roll).

3.  **Logic Correction & Classification**:
    The raw angles are processed through the "V-Final-Plus-Plus" algorithm. This stage applies heuristic rules (e.g., Smart Sign Correction, Back-View Thresholding) to correct geometric inconsistencies before outputting the final prediction.

## Installation

This project uses Conda for environment management.

1. Clone the repository:
   ```bash
   git clone [https://github.com/jianlin0828/pose.git](https://github.com/jianlin0828/pose.git)
   cd pose

```

2. Create and activate the environment:
```bash
conda env create -f environment.yml
conda activate <env_name>

```



## Model Zoo

Due to file size limitations, the pretrained head model is hosted externally.

* **DAD-WildHead-EffNetV2-S**: [Download Link](https://huggingface.co/HoyerChou/SemiUHPE/tree/main)

Please download the `.pth` file and place it in the `checkpoints/` directory:

```text
pose/
└── checkpoints/
    └── DAD-WildHead-EffNetV2-S-best.pth

```

## Usage

To run the evaluation script on your dataset:

```bash
python eval_pose_v2.py \
    --img-dir "data/test_images" \
    --out-dir "output/result" \
    --checkpoint "checkpoints/DAD-WildHead-EffNetV2-S-best.pth" \
    --prompts-file "data/prompts.csv"

```

### Arguments

* `--img-dir`: Path to the directory containing input images.
* `--out-dir`: Path to save the output CSV results.
* `--checkpoint`: Path to the pretrained EfficientNet weights.
* `--prompts-file`: (Optional) Path to the CSV file containing filename and prompts for accuracy calculation.

## Methodology

The core logic implements several strategies to overcome 2D estimation limitations:

* **Z-axis Logic Correction**: Uses the head orientation as an anchor to correct body mirror errors (sign flipping) during 90° side views.
* **Hierarchical Defense**: Prioritizes "Early Lean" detection and strict "Back View" thresholds (89°) to prevent class overlap.
* **Asymmetric Dominance**: Applies different thresholds for left (6°) vs. right (20°) head turns to compensate for data compression artifacts in MediaPipe.

## Output

The script generates a CSV file (`pose_classification_v_final.csv`) with the following fields:

* `Filename`: Name of the input image.
* `Prompt`: Input text prompt.
* `gt_pose`: Ground Truth class derived from the prompt.
* `prediction`: Final predicted class.
* `Raw_Angles`: Detailed angle outputs (Body/Head Yaw, Pitch, Roll).

```