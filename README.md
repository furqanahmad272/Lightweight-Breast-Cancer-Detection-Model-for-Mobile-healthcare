# Lightweight Breast Cancer Detection Model for Mobile Healthcare

A lightweight multi-model AI system for breast cancer detection using thermal imaging data, optimized for mobile healthcare applications and edge devices.

## 🎯 Overview

This project implements a two-stage classification pipeline using lightweight deep learning models to detect breast cancer from thermal images:

1. **Stage 1 (RGB Model)**: Classifies thermal images as "Normal" or "Sick"
2. **Stage 2 (Grayscale Model)**: Further classifies "Sick" cases as "Benign" or "Malignant"

The models are optimized for deployment on resource-constrained devices, making them suitable for wearable devices and mobile healthcare applications.

## ✨ Features

- **Lightweight Architecture**: Uses ResNet18 and MobileNetV2 for efficient inference
- **Two-Stage Pipeline**: Hierarchical classification for improved accuracy
- **Mobile-Ready**: Optimized for edge deployment (TensorFlow Lite, ONNX support)
- **Flask API**: RESTful API for easy integration
- **Modular Design**: Easy to extend and customize

## 🏗️ Architecture

```
Input Thermal Image
        ↓
   Stage 1 Model (RGB)
   ├── Normal → End
   └── Sick → Stage 2 Model (Grayscale)
              ├── Benign
              └── Malignant
```

## 📋 Requirements

- Python 3.8+
- PyTorch 1.10+
- OpenCV
- NumPy
- Flask
- scikit-learn
- Pillow

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/furqanahmad272/Lightweight-Breast-Cancer-Detection-Model-for-Mobile-healthcare.git
cd Lightweight-Breast-Cancer-Detection-Model-for-Mobile-healthcare

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset

This project uses thermal breast imaging datasets. You can use:

1. **DMR-IR Dataset**: [Database for Mastology Research](http://visual.ic.uff.br/dmi/)
2. **Mendeley Thermal Dataset**: [Breast Thermal Images](https://data.mendeley.com/datasets/wmy8sh2pjj/1)

Download and place the dataset in the `data/` directory with the following structure:

```
data/
├── train/
│   ├── normal/
│   ├── benign/
│   └── malignant/
└── test/
    ├── normal/
    ├── benign/
    └── malignant/
```

## 🎓 Training

### Train Stage 1 Model (Normal vs Sick)

```bash
python src/training/train.py --stage 1 --model resnet18 --epochs 50 --batch-size 32
```

### Train Stage 2 Model (Benign vs Malignant)

```bash
python src/training/train.py --stage 2 --model resnet18 --epochs 50 --batch-size 32
```

## 🔮 Inference

### Single Image Prediction

```bash
python src/inference/predict.py --image path/to/thermal_image.jpg --model-path models/
```

### Batch Prediction

```bash
python src/inference/predict.py --image-dir path/to/images/ --model-path models/ --output results.csv
```

## 🌐 Flask API

Start the Flask server:

```bash
python app/flask_app.py
```

The API will be available at `http://localhost:5000`

### API Endpoints

#### Predict
```bash
curl -X POST -F "file=@thermal_image.jpg" http://localhost:5000/predict
```

Response:
```json
{
  "stage1_prediction": "Sick",
  "stage1_confidence": 0.92,
  "stage2_prediction": "Benign",
  "stage2_confidence": 0.85,
  "final_result": "Benign"
}
```

## 📱 Mobile Deployment

### Export to TensorFlow Lite

```bash
python src/models/export_tflite.py --model-path models/stage1_resnet18.pth --output models/stage1_model.tflite
```

### Export to ONNX

```bash
python src/models/export_onnx.py --model-path models/stage1_resnet18.pth --output models/stage1_model.onnx
```

## 📈 Performance

| Model | Stage | Accuracy | Size | Inference Time (CPU) |
|-------|-------|----------|------|---------------------|
| ResNet18 | 1 | 94.2% | 44 MB | 45 ms |
| ResNet18 | 2 | 91.7% | 44 MB | 45 ms |
| MobileNetV2 | 1 | 92.8% | 14 MB | 28 ms |
| MobileNetV2 | 2 | 89.5% | 14 MB | 28 ms |

*Note: Metrics based on validation data. Results may vary with different datasets.*

## 🗂️ Project Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── models/              # Model architectures
│   ├── preprocessing/       # Data preprocessing scripts
│   ├── training/           # Training scripts
│   └── inference/          # Inference scripts
├── app/
│   ├── flask_app.py        # Flask API
│   └── templates/          # HTML templates
├── notebooks/              # Jupyter notebooks for exploration
├── data/                   # Dataset directory
├── models/                 # Saved model weights
└── tests/                  # Unit tests
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by [Multi-Model Breast Cancer Detection Using Thermal Imaging](https://github.com/Deeksha1054/Multi-Model-Breast-Cancer-Detection-Using-Thermal-Imaging)
- Thermal imaging datasets from DMR and Mendeley
- PyTorch and OpenCV communities

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## ⚠️ Disclaimer

This project is for research and educational purposes only. It is not intended for clinical diagnosis. Always consult healthcare professionals for medical advice.
