# 🖼️ Neural Style Transfer (NST)

This project implements **Neural Style Transfer (NST)** — a deep learning technique that merges the content of one image with the artistic style of another. It leverages **PyTorch** and a pre-trained **VGG-19** model to extract and combine features from both images using a custom loss function. The project includes everything: data preprocessing, dataset creation, training, evaluation, and a slick **Gradio** interface for live demos.

---

## 🚀 Features

* **Data Preprocessing**
  Automatically organizes and splits images into training, validation, and test sets (70/15/15).

* **Custom Dataset**
  Pairs content-style images with dynamic style intensity thresholds.

* **VGG-19 Backbone**
  Extracts features from:

  * `conv4_2` for content
  * Multiple layers for style
  * Optional fine-tuning

* **Loss Functions**

  * Content loss: MSE on feature maps
  * Style loss: MSE on Gram matrices
  * Total variation loss: for smoothness
  * L2 regularization: for stability

* **Training & Evaluation**
  Stylizes each content-style pair with optimization and saves the results.

* **Gradio Interface**
  Upload images, control style intensity & optimization steps — and boom, art.

* **Reproducibility**
  Fixed random seed = consistent results across runs.

---

## 🧠 Prerequisites

* Python 3.8+
* PyTorch 1.9+ (with CUDA for GPU acceleration)
* torchvision
* numpy
* pillow (PIL)
* scikit-learn
* matplotlib
* gradio
* **A GPU is highly recommended.**

---

## 🛠️ Installation

Clone the repo:

```bash
git clone https://github.com/your-username/neural-style-transfer.git
cd neural-style-transfer
```

(Optional) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install torch torchvision numpy pillow scikit-learn matplotlib gradio
```

### 🖼️ Prepare the Dataset

* Place content images in:
  `Content/images/images/`
  *(with subfolders like architecture, art, etc.)*

* Place style images in:
  `style/`

* Supported formats: `.jpg`, `.jpeg`, `.png`

---

## 💻 Usage

### 1. Data Preparation

```bash
python main.py
```

* Samples \~7,000 content and \~3,000 style images
* Splits them into:
  `Content/train`, `Content/validation`, `Content/test`,
  `style/train`, `style/validation`, `style/test`

> If splits already exist, it reuses them.

---

### 2. Training

```bash
python main.py
```

* Trains on content-style pairs for 1,000 iterations per batch
* Stylized images saved to `stylized_outputs/`
* Tune `max_train_samples`, `batch_size`, and `num_iterations` in `main.py`

---

### 3. Evaluation

```bash
python main.py
```

* Evaluates for 500 iterations per batch
* Outputs go to `test_outputs/`
* Saves:

  * Content
  * Style
  * Stylized
  * Composite images
* Logs average losses (total, content, style, total variation)

---

### 4. Gradio Interface

```bash
python main.py
```

* Upload a content & style image
* Adjust:

  * Style intensity (0.0 = content-heavy, 1.0 = style-heavy)
  * Optimization steps (50–300)
* Stylized image previewed live

---

## 🗂️ Project Structure

```
neural-style-transfer/
├── Content/
│   ├── images/images/      # Raw content images (with subfolders)
│   ├── train/              # Training content images
│   ├── validation/         # Validation content images
│   ├── test/               # Test content images
├── style/
│   ├── train/              # Training style images
│   ├── validation/         # Validation style images
│   ├── test/               # Test style images
├── stylized_outputs/       # Stylized training outputs
├── eval_outputs/           # Stylized evaluation outputs
├── test_outputs/           # Stylized test outputs
├── examples/               # Example images
├── main.py                 # Main script
├── README.md               # This file
```

---

## 🎨 Example (Gradio)

```bash
python main.py
```

* Content: `examples/content1.jpg`
* Style: `examples/style1.jpg`
* Style Intensity: `0.5`
* Steps: `100`

Upload your images, adjust controls, and see the result live.

---

## 📝 Notes

* **Performance**: Use a GPU. Seriously.
* **Dataset Size**: Customize `max_train_samples`, `max_val_samples`, `max_test_samples`
* **Resolution**:

  * Training: 512×512
  * Gradio: 1024×1024
* **Robustness**: Handles invalid images and NaNs like a champ

---

## 🔮 Future Improvements

* Real-time NST using **AdaIN**
* Multi-style blending in one pass
* Lightweight model support (e.g., MobileNet)
* Batch mode for Gradio

---
