# Part 2 – Computer Vision Problem Formulation and CNN Prototype

**Dataset:** Synthetic Manufacturing Defect Image Dataset  
**Dataset Source:** [Google Drive Folder](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing)

---

## Table of Contents
1. [Problem Identification](#1-problem-identification)
2. [Dataset Exploration](#2-dataset-exploration)
3. [Image Preprocessing](#3-image-preprocessing)
4. [CNN Model Architecture](#4-cnn-model-architecture)
5. [Training & Evaluation](#5-training--evaluation)
6. [CNN Concept Explanation](#6-cnn-concept-explanation)
7. [Business Use Case](#7-business-use-case)
8. [Repository Structure](#8-repository-structure)
9. [Setup & Usage](#9-setup--usage)

---

## 1. Problem Identification

**Problem Type: Image Classification**

The dataset contains product surface images organised into four labelled subfolders. Each image belongs to exactly one defect class, making this a multi-class **Image Classification** problem.

| CV Problem Type | Applies? | Reason |
|---|---|---|
| **Image Classification** | ✅ Yes | Each image has a single class label |
| Object Detection | ❌ No | No bounding box annotations |
| Semantic Segmentation | ❌ No | No pixel-level masks |
| Instance Segmentation | ❌ No | No per-object instance masks |

**Classes (4 total, 120 images each → 480 total):**
- `normal` – defect-free surface
- `scratch` – linear scratch-like marks
- `dent` – circular dent-like marks
- `stain` – colored stain-like marks

---

## 2. Dataset Exploration

| Stat | Value |
|---|---|
| Total images | 480 |
| Number of classes | 4 |
| Images per class | 120 each (perfectly balanced) |
| Class imbalance | None (1.0× ratio) |

Dataset is **perfectly balanced** — no oversampling or class weighting needed.

---

## 3. Image Preprocessing

| Step | Detail |
|---|---|
| **Resize** | 128 × 128 pixels |
| **Normalize** | Pixel values scaled `[0, 255]` → `[0, 1]` |
| **Split** | 80% training / 20% validation |
| **Augmentation (train only)** | Rotation ±20°, shift 10%, zoom 10%, horizontal flip, brightness 0.85–1.15 |
| **Validation** | Rescale only — no augmentation |

---

## 4. CNN Model Architecture

```
Input (128 × 128 × 3)
│
├─ Block 1: Conv2D(32)  → BatchNorm → ReLU → MaxPool(2×2)
├─ Block 2: Conv2D(64)  → BatchNorm → ReLU → MaxPool(2×2)
├─ Block 3: Conv2D(128) → BatchNorm → ReLU → MaxPool(2×2)
├─ Block 4: Conv2D(256) → BatchNorm → ReLU → MaxPool(2×2)
│
├─ Flatten
├─ Dense(256) → BatchNorm → ReLU → Dropout(0.5)
├─ Dense(128) → ReLU → Dropout(0.3)
└─ Dense(4, softmax)   ← Output layer
```

| Layer | Role |
|---|---|
| Conv2D | Learns spatial patterns — edges, textures, defect shapes |
| BatchNormalization | Stabilises training, speeds convergence |
| ReLU | Non-linearity without vanishing gradients |
| MaxPooling | Reduces spatial size; translation invariance |
| Flatten | Converts feature tensor to 1-D vector |
| Dense | High-level class relationships |
| Dropout | Prevents overfitting |
| Dense(softmax) | Probability over 4 classes |

---

## 5. Training & Evaluation

- **Optimizer:** Adam (lr=1e-3, ReduceLROnPlateau factor=0.5)
- **Loss:** Categorical Cross-Entropy
- **Callbacks:** EarlyStopping (patience=8), ReduceLROnPlateau (patience=4), ModelCheckpoint

Results saved in `results/` and `sample_predictions/`.

---

## 6. CNN Concept Explanation

### What is Convolution?
A small **filter (kernel)** slides across the input image computing dot products at every position, producing a **feature map** that highlights where specific patterns occur (edges, scratches, stain blobs). Multiple filters detect multiple patterns simultaneously. Deeper layers combine these into complex shape representations.

### Why is Pooling Used?
**MaxPooling** reduces feature map dimensions by keeping only the maximum value in each local window (2×2). Benefits:
- Reduces computation in deeper layers
- Provides **translation invariance** — a shifted defect still produces the same output
- Acts as regularisation by discarding noise

### Why is ReLU Commonly Used?
**ReLU** = max(0, x):
- Introduces **non-linearity** enabling complex decision boundaries
- Computationally trivial (threshold at zero)
- Avoids the **vanishing gradient** problem of sigmoid/tanh
- Sparse activations naturally regularise the model

### Why CNNs Over Regular Neural Networks for Images?

| Property | MLP | CNN |
|---|---|---|
| Parameters | Millions (all pixels connected) | Thousands (shared filters) |
| Spatial structure | Destroyed (flattened) | Preserved (2-D) |
| Translation invariance | No | Yes (pooling) |
| Feature hierarchy | No | Edges → textures → shapes |
| Scalability | Poor | Efficient |

---

## 7. Business Use Case

### Domain: Manufacturing 🏭

**Use Case: Automated Visual Quality Inspection**

This CNN directly models a real production-line problem — detecting surface defects on manufactured parts before they reach customers.

**Workflow:**
1. Camera captures product images as they move along a conveyor.
2. CNN classifies each image: **Normal / Scratch / Dent / Stain** in real time.
3. Defective products are automatically rejected before packaging.

**Impact:**
- 24/7 consistent inspection vs. error-prone manual checking
- Reduced defect escape rate and warranty claims
- Root-cause data for process improvement

**Examples:** BMW (body panel inspection), Samsung (display QC), Foxconn (PCB defect detection).

---

## 8. Repository Structure

```
part-2-cnn-computer-vision/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
├── sample_predictions/
│   └── prediction_outputs.png
└── results/
    ├── accuracy_loss_curves.png
    ├── confusion_matrix.png
    ├── class_distribution.png
    ├── sample_images.png
    └── augmented_samples.png
```

> **Note:** Dataset files are NOT uploaded. Download from the [Google Drive link](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing) and place as `part_2_cnn_computer_vision/` in the project root.

---

## 9. Setup & Usage

```bash
# 1. Clone repo
git clone <your-repo-url>
cd part-2-cnn-computer-vision

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place dataset at:
#    part_2_cnn_computer_vision/
#      ├── images/
#      │   ├── normal/
#      │   ├── scratch/
#      │   ├── dent/
#      │   └── stain/
#      └── labels.csv

# 4. Run notebook
jupyter notebook notebook.ipynb
```

Run all cells top-to-bottom. All plots save automatically to `results/` and `sample_predictions/`.
