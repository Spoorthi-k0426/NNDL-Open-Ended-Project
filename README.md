# Pneumonia Detection from Chest X-Ray

A clean, beginner-friendly web application built with **Streamlit** and **TensorFlow / Keras** for detecting pneumonia from chest X-ray images using a fine-tuned **ResNet50V2** deep learning model.

---

## 📌 Project Overview

This project classifies chest X-ray images into two categories:
- **NORMAL**: Healthy lungs
- **PNEUMONIA**: Evidence of pneumonia infection

### Model & Decision Logic
- **Architecture**: Fine-tuned **ResNet50V2** transfer learning model
- **Input Dimensions**: `224 × 224` pixels (RGB, 3 channels)
- **Output**: Sigmoid activation yielding a pneumonia probability between `0.0` and `1.0`
- **Optimal Decision Threshold (`OPTIMAL_THRESHOLD`)**: `0.20` (selected via validation F1-score optimization in the training notebook)
- **Classification Rule**:
  $$\text{Predicted Class} = \begin{cases} \text{PNEUMONIA} & \text{if } P(\text{Pneumonia}) \ge 0.20 \\ \text{NORMAL} & \text{if } P(\text{Pneumonia}) < 0.20 \end{cases}$$
- **Confidence Metric**:
  $$\text{Confidence} = \begin{cases} P(\text{Pneumonia}) & \text{if PNEUMONIA} \\ 1 - P(\text{Pneumonia}) & \text{if NORMAL} \end{cases}$$

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **TensorFlow / Keras**: Deep learning model loading & inference
- **Streamlit**: Interactive web user interface
- **NumPy (< 2.0)**: Matrix & array computations compatible with TensorFlow
- **Pillow (PIL)**: Image loading and resizing

---

## 📂 Project Structure

```text
open_ended/
│
├── NNDL_Pneumonia_Prediction.ipynb   # Model training & evaluation notebook
├── pneumonia_resnet50v2_final.keras  # Saved ResNet50V2 trained model
├── app.py                            # Streamlit web application
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🚀 Getting Started

### 1. Install Dependencies

Open your terminal in the project directory and install the required packages:

```bash
pip install -r requirements.txt
```

> **Note**: TensorFlow and Scikit-Learn require NumPy 1.x (`numpy<2.0.0`). The `requirements.txt` file has already configured this constraint.

### 2. Export / Place the Model File

The web application expects the trained model file:
```text
pneumonia_resnet50v2_final.keras
```
to be placed directly in the project folder beside `app.py`.

#### If you trained the model in Google Colab:
1. Run **Cell 25** of `NNDL_Pneumonia_Prediction.ipynb`:
   ```python
   FINAL_MODEL_PATH = "pneumonia_resnet50v2_final.keras"
   resnet_model.save(FINAL_MODEL_PATH)
   ```
2. Download the model file to your computer using:
   ```python
   from google.colab import files
   files.download("pneumonia_resnet50v2_final.keras")
   ```
3. Move the downloaded `pneumonia_resnet50v2_final.keras` into this project directory.

#### If you trained the model locally:
Execute Cell 25 in your local Jupyter Notebook so `pneumonia_resnet50v2_final.keras` is saved directly into the folder.

---

## ▶️ Running the Web Application

Launch the Streamlit app by running:

```bash
python -m streamlit run app.py
```
*(or `streamlit run app.py`)*

Once started, open your browser at the displayed local URL (typically `http://localhost:8501`).

### How to use:
1. Click **Browse files** and upload a chest X-ray image (`.jpg`, `.jpeg`, or `.png`).
2. Verify the preview of the uploaded image.
3. Click the **Predict** button.
4. View the predicted class (**NORMAL** or **PNEUMONIA**), confidence percentage, and pneumonia risk probability.

---

## ⚠️ Medical Disclaimer

> **This project is for educational and academic demonstration purposes only and is not a certified medical diagnostic system.** It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
