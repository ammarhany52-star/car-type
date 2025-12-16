# Car Type Classification (Stanford Cars)

This project focuses on fine-grained car type classification using deep learning
and transfer learning techniques.

## Dataset
Stanford Cars Dataset:
https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset

## Models Used
- ResNet50
- EfficientNetB0
- MobileNetV2

## Training
- Transfer Learning with ImageNet pretrained models
- Data augmentation (rotation, zoom, horizontal flip)
- Train / Validation split

## Evaluation
- Accuracy
- Confusion Matrix
- Model comparison
- Grad-CAM explainability

## GUI
Gradio-based GUI with:
- Image upload
- Random image testing
- Model selection
- Top-3 predictions with confidence
- Grad-CAM visualization

## How to Run
```bash
pip install tensorflow gradio opencv-python
python gui/app.py
