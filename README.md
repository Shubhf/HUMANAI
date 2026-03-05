# ArtExtract: Neural Frameworks for Fine Art Analysis
This repository contains the evaluation tasks for the **GSoC 2025 ArtExtract project** under the **HumanAI Umbrella Organization**. The project utilizes hybrid deep learning architectures to classify and retrieve fine art from the WikiArt and National Gallery of Art (NGA) collections.

## 🚀 Hardware & Environment
* **GPU**: NVIDIA RTX A5500 (24GB VRAM)
* **Compute**: Accelerated inference for high-dimensional embedding extraction
* **Framework**: PyTorch 2.7.1
* **Key Libraries**: `torch`, `torchvision`, `openai-clip`, `pandas`, `seaborn`

---

## 🎨 Task 1: Multi-Task CRNN Classification
**File**: `TASK1.ipynb`

### Methodology
Implemented a **Multi-Task Convolutional-Recurrent Neural Network (CRNN)** using a **ResNet-50** backbone and a **Bidirectional LSTM**. This hybrid approach captures both localized textures and global compositional "flow".

### Key Results
* **Artist Macro-F1**: 0.8522
* **Top-5 Accuracy**: 0.9365
* **Outlier Detection**: Identified "misfit" paintings using **Shannon Entropy** (Threshold > 2.2), flagging works where the model identified conflicting stylistic signatures.

---

## 🔍 Task 2: Semantic Similarity & Retrieval
**File**: `TASK_2.ipynb`

### Methodology
Developed a similarity engine using **CLIP (ViT-B/32)** metric learning. Images are projected into a 512-dimensional latent space to identify structural poses and facial orientations across different artistic mediums.

### Key Results
* **Metric**: **Cosine Similarity** for angular distance ranking.
* **Statistical Outliers**: Flagged images with mean similarity < 0.40. Qualitative review confirmed these as non-standard entries like sculptures and abstract sketches.

---

## 📂 Repository Structure
* `task_1.ipynb`: CRNN implementation and WikiArt outlier analysis.
* `task_2.ipynb`: CLIP similarity engine and NGA outlier detection.
* `Art Report_SHUBH_GARG.pdf`: Comprehensive research report with architectural flowcharts.

## 📧 Contact
**Candidate**: Shubh Garg (Thapar Institute of Engineering and Technology)  
**Submission**: Evaluation Test: ArtExtract
