# Pneumonia Detection from Chest X-Rays

An interactive web tool built with Streamlit and TensorFlow/Keras to screen chest radiographs for pneumonia using a fine-tuned ResNet50V2 model.

### What this project does

Automating preliminary chest X-ray screening can help flag potential lung infections faster, especially in high-volume or resource-constrained settings. This tool lets you drop in an anterior-posterior (AP) or postero-anterior (PA) chest scan and get an immediate classification:

* **NORMAL**: Clear lung fields without signs of acute infection.
* **PNEUMONIA**: Evidence of focal consolidation or diffuse infiltrates.


### How the model works

* **Backbone**: A pre-trained ResNet50V2 feature extractor fine-tuned on paediatric chest X-ray scans.
* **Input**: Images are loaded, converted to RGB, and resized to $224 \times 224$ pixels.
* **Output**: A single sigmoid neuron predicting the probability of pneumonia ($0.0$ to $1.0$).
* **Why the decision threshold is set to `0.20**`:
In clinical screening, a **false negative** (missing an infected patient) is far more dangerous than a false positive (flagging a healthy scan for physician review). By tuning the decision boundary on validation F1-scores, a cutoff of **0.20** gave the most dependable sensitivity without flooding the pipeline with false alarms:
* **Pneumonia**: $P(\text{Pneumonia}) \ge 0.20$
* **Normal**: $P(\text{Pneumonia}) < 0.20$

### Tech stack

* **Python 3.10+**
* **TensorFlow / Keras** for model loading and forward passes
* **Streamlit** for the frontend interface
* **Pillow (PIL)** for image ingestion and resizing
* **NumPy (< 2.0)** to maintain compatibility with TensorFlow C-extensions

### Repository layout

open_ended/
│
├── NNDL_Pneumonia_Prediction.ipynb   # Full training, ablation, and evaluation notebook
├── pneumonia_resnet50v2_final.keras  # Exported ResNet50V2 model checkpoint
├── app.py                            # Streamlit frontend and inference handler
├── requirements.txt                  # Python runtime dependencies
└── README.md                         # Documentation


### Setting up locally

#### 1. Install dependencies

Clone or download this repo, open a terminal inside the project directory, and install the environment:


pip install -r requirements.txt

*(Note: TensorFlow builds require NumPy `1.x`. Keeping `numpy<2.0.0` prevents binary ABI incompatibilities.)*

#### 2. Model checkpoint

The web app looks for `pneumonia_resnet50v2_final.keras` in the root folder alongside `app.py`.

* **If you trained via Google Colab**:
Save and download your model from your notebook:

resnet_model.save("pneumonia_resnet50v2_final.keras")
from google.colab import files
files.download("pneumonia_resnet50v2_final.keras")


Move the downloaded file directly into your `open_ended/` folder.
* **If you trained locally**:
The `.keras` file will already be saved in your directory once the final training cell finishes.


### Running the app

Launch the local server:


streamlit run app.py

Open your browser to `http://localhost:8501`.

1. Upload any chest X-ray (`.jpeg`, `.jpg`, or `.png`).
2. Verify the scan preview.
3. Hit **Predict** to view the diagnosis, calculated confidence, and risk score.


### Disclaimer

This software is developed strictly for coursework and academic demonstration. It has not been clinically validated, FDA/CE cleared, or audited for clinical decision-making. Always rely on a board-certified radiologist for medical diagnosis and patient management.