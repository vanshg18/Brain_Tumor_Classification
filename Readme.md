#  Brain Tumor MRI Image Classification

This project classifies brain MRI scans into tumor types using deep learning. It includes training with a custom CNN and MobileNetV2 (transfer learning), and a deployed Streamlit app for real-time predictions.

##  Contents

- `app.py` — Streamlit app for image upload and classification
- `Brain_Tumor.ipynb` — Google Colab notebook for model training & evaluation
- `mobilenetv2_best_model.h5` — Trained model weights (MobileNetV2)
- `requirements.txt` — Python dependencies

##  Run the App Locally

1. **Clone the repository**

```bash
git clone https://github.com/vanshg18/brain-tumor-classification.git
cd brain-tumor-classification
```
2. **Install Dependencies**
pip install -r requirements.txt

3. **Run the Streamlit App**
streamlit run app.py
