# Pneumonia-Prediction-using-CNN-and-Hybrid-models

## Pneumonia Detection from Chest X-Ray Images

A deep learning project comparing multiple architectures for detecting pneumonia in chest X-ray images using the [ChestXRay2017 dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

## Dataset
- **Source**: Kaggle ChestXRay2017 dataset
- **Structure**:
  - Train: 5,232 images (1,341 Normal, 3,891 Pneumonia)
  - Test: 624 images (234 Normal, 390 Pneumonia)
- **Preprocessing**:
  - Grayscale conversion (1-channel) for basic CNN models
  - 3-channel conversion for AlexNet/ResNet/Transformer
  - Resized to 224×224 pixels
  - Normalized using mean=0.5, std=0.5

## Implemented Models

### 1. Simple CNN Architecture
**Structure**:
Conv2d(1→32) → ReLU → MaxPool → Conv2d(32→64) → ReLU → MaxPool → FC(645656→128) → ReLU → FC(128→2)

- Designed for single-channel input
- Achieved 99.5% training accuracy
- **Test Accuracy**: 77.56%

### 2. CNN-LSTM Hybrid
**Unique Features**:
- Combines convolutional layers with LSTM temporal processing
- Adaptive average pooling for sequence generation
- Struggled with convergence (68.59% test accuracy)

### 3. AlexNet
**Modifications**:
- Final FC layer modified for binary classification
- Trained on 3-channel images
- **Test Accuracy**: 78.85%

### 4. ResNet18
**Implementation**:
- Pretrained weights excluded
- Final FC layer adapted for 2 classes
- **Top Performer**: 79.97% test accuracy

### 5. Vision Transformer (ViT-B/16)
**Configuration**:
- Patch size: 16×16
- Pretrained weights not used
- **Test Accuracy**: 63.46%

## Performance Comparison
| Model          | Train Accuracy | Test Accuracy |
|----------------|----------------|---------------|
| Simple CNN     | 99.54%         | 77.56%        |
| CNN-LSTM       | 83.62%         | 68.59%        |
| AlexNet        | 97.36%         | 78.85%        |
| ResNet18       | 98.41%         | 79.97%        |
| ViT (Base/16)  | 74.10%         | 63.46%        |

## Key Findings
1. **Architecture Performance**:
   - ResNet18 showed best generalization (79.97% test accuracy)
   - Vision Transformer underperformed without pretraining
   - Simple CNN demonstrated significant overfitting (99.5% train vs 77.56% test)

2. **Training Characteristics**:
   - Basic CNN achieved near-perfect training accuracy in 15 epochs
   - CNN-LSTM required more epochs (100+) with slower convergence
   - AlexNet showed stable training with moderate regularization

## Hardware Configuration
- **GPUs**: 8× NVIDIA A40 (40GB VRAM each)
- **Multi-GPU Training**: DataParallel across 4 GPUs
- **CUDA Version**: 12.7

## Requirements

pip install torch torchvision timm


## Usage
1. Clone repository
2. Download dataset from Kaggle
3. Update data paths in notebooks
4. Run cells sequentially (data prep → model training → evaluation)

## Conclusion
While all models achieved reasonable performance, ResNet18 demonstrated the best balance between training efficiency and generalization capability. The project highlights the effectiveness of residual networks for medical image analysis compared to both simpler architectures and newer transformer-based approaches.
