# Chest X-Ray Classification: Pneumonia Detection using Deep Learning 🏛️

This project aims to classify chest X-ray images into two categories: `NORMAL` or `PNEUMONIA`, using deep learning techniques with **DenseNet161**, implemented in **PyTorch**. The goal is to assist in early diagnosis by classifying X-ray images automatically, which can be useful in environments where fast and accurate diagnostics are needed.

## 🦠 About Pneumonia
Pneumonia is an infection that inflames the air sacs in one or both lungs, often filling them with fluid or pus. Detecting pneumonia from chest X-rays is a common diagnostic method, but manual interpretation can be time-consuming and prone to human error. This project automates this process using computer vision and deep learning.

## 🧠 Technologies Used
- **PyTorch** – For building and training deep learning models
- **DenseNet161** – A powerful pre-trained convolutional neural network
- **Computer Vision** – Using transforms and data loaders for image processing
- **CNN (Convolutional Neural Network)** – Core architecture for image classification
- **sklearn** – For stratified splitting and performance metrics

## 📦 Dataset Overview
The dataset contains chest X-ray images categorized as either:
- `NORMAL`
- `PNEUMONIA`

Original source: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) on Kaggle.

### 🔍 Dataset Structure
```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

We perform a **stratified train/test split** (no validation set in this version), ensuring balanced class distribution during training.

## ⚙️ Features Implemented
- ✅ Stratified train/test split for class balance
- ✅ Image preprocessing and augmentation
- ✅ Weighted loss and sampling to handle class imbalance
- ✅ Pretrained DenseNet161 fine-tuned for binary classification
- ✅ Model evaluation with accuracy, F1-score, and confusion matrix
- ✅ Saving model weights and dataset for reuse

## 🧪 Final Evaluation Results

| Metric | Score |
|-------|--------|
| **Test Accuracy** | 96.02% |
| **Macro-Averaged F1-Score** | 95.05% |
| **F1-Score (NORMAL)** | 92.87% |
| **F1-Score (PNEUMONIA)** | 97.24% |

### 📋 Classification Report
```
              precision    recall  f1-score   support

      NORMAL       0.90      0.96      0.93       238
   PNEUMONIA       0.98      0.96      0.97       641

    accuracy                           0.96       879
   macro avg       0.94      0.96      0.95       879
weighted avg       0.96      0.96      0.96       879
```

## 📦 File Structure
```
.
├── README.md
├── train_model.py            # Main training script
├── prepare_dataset.py        # Dataset preparation and stratified split
├── models/
│   └── densenet161_full.pth  # Saved full model
├── data/
│   └── chest_xray_stratified/ # Processed dataset
└── notebooks/
    └── pneumonia_classifier.ipynb # Jupyter notebook version
```

## ▶️ How to Run

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/pneumonia-chest-xray-classifier.git  
cd pneumonia-chest-xray-classifier
```

### 2. Install dependencies
```bash
pip install torch torchvision matplotlib numpy pandas scikit-learn
```

> If running in **Kaggle**, make sure to upload the dataset to the `/kaggle/input/` directory.

### 3. Prepare dataset (optional if not using Kaggle)
```bash
python prepare_dataset.py
```

### 4. Train and evaluate the model
Run the Jupyter Notebook or Python script:
```bash
jupyter notebook notebooks/pneumonia_classifier.ipynb
```

Or run the training script directly:
```bash
python train_model.py
```

## 💾 Model Saving
The best model is saved in `.pth` format for later deployment or retraining:
```python
torch.save(best_model, 'densenet161_full.pth')
```

You can also save only the state dictionary:
```python
torch.save(best_model.state_dict(), 'densenet161_weights.pth')
```

## 📦 Bonus: Archive Dataset
To download the processed dataset folder as a ZIP archive:
```python
from IPython.display import FileLink
import shutil

shutil.make_archive('chest_xray_stratified', 'zip', './chest_xray_stratified')
FileLink('chest_xray_stratified.zip')
```

## 🚀 Future Improvements
- Add **K-Fold Cross Validation** for more robust training
- Implement **TensorBoard** logging for training visualization
- Use **Grad-CAM** for visualizing activation regions in X-rays
- Deploy model as a **web app** using Flask or Streamlit
- Explore other architectures like ResNet, EfficientNet, etc.

## 📬 Contact
Have questions or suggestions? Feel free to open an issue or contact me at [https://www.linkedin.com/in/djebrilchennoufi/]

---
