# Vision Transformer for AI-Generated Face Detection

**Students**: Hanson Wu, Rayan Hassan
**Course**: CS 7150 - Deep Learning  
**Project**: Interpreting AI-Generated Face Detection Models

## Project Overview

In this project, we compare two deep learning approaches for classifying face images as either real or AI-generated:

- ResNet: a convolutional neural network (CNN)
- ViT (Vision Transformer): a transformer-based image model

The goal is not only to compare which model performs better, but also to understand how they make their decisions. To do that, we use Grad-CAM visualizations to inspect which parts of the face image each model focuses on when making a prediction. Grad-CAM is a gradient-based localization method that highlights the image regions most relevant to the predicted class.

We use PyTorch for both models. For the CNN, we use ResNet-18, a standard residual network available in TorchVision. For the transformer model, we use ViT-B/16, which is also provided in TorchVision.

**Key Hypothesis**: Vision Transformers emphasize **global facial structure and geometry**, while CNNs tend to focus on **local texture artifacts**.

## Project Structure

```
final_project/
├── vision_transformer_face_detection.ipynb   # Main notebook for ViT (ENTRY POINT)
├── ResNet.ipynb                              # Main notebook for ResNet (ENTRY POINT)
├── data_utils.py                             # Data loading & preprocessing
├── resnet_helpers/
│   ├── resnet_model.py                       # ResNet model & training logic
│   └── resnet_gradcam.py                     # Grad-CAM visualization for ResNet
├── vit_helpers/
│   ├── vit_model.py                          # ViT model & training logic
│   └── vit_gradcam.py                        # Grad-CAM visualization for ViT
├── resnet_model_outputs/                     # ResNet model visualizations & metrics
├── resnet_gradcam_outputs/                   # ResNet Grad-CAM visualizations & metrics
├── vit_model_outputs/                        # ViT model visualizations & metrics
├── vit_gradcam_outputs/                      # ViT Grad-CAM visualizations & metrics
├── requirements.txt                          # Python dependencies
├── README.md                                 # This file
└── data/                                     # Face dataset
    ├── dataset/
    │   ├── train/
    │   │   ├── 0/
    │   │   └── 1/
    │   ├── test/
    │   │   ├── 0/
    │   │   └── 1/
    │   └── validate/
    │       ├── 0/
    │       └── 1/
    └── source/
        ├── real/
        └── fake/
            ├── GAN/
            ├── Diffusion/
            └── FaceSwap/
```


## Dataset Structure and Reorganization

The original dataset obtained from Kaggle contains multiple nested directories and mixed sources of real and AI-generated images. Specifically, the dataset is organized into two main components:

1. A `dataset/dataset/` directory containing the data splits:
   - `train/`
   - `test/`
   - `validate/`

2. A `data_source/data_source/` directory containing the raw image sources:
   - `ffhq/` (real images)
   - `fake/`, which further includes:
     - `thispersondoesnotexist/`
     - `sfhq/`
     - `stable_diffusion/`
     - `faceswap/`

While this structure is functional, it is not ideal for analysis or interpretation. Therefore, we reorganized the dataset into a cleaner and more meaningful format.

---

### Reorganized Structure

We transformed the dataset into the following structure:

- `data/dataset/`  
  Contains the original training, validation, and test splits used for model training.

- `data/source/real/`  
  Contains real images from the FFHQ dataset (Flickr-Faces-HQ), a high-quality dataset of real human faces collected from Flickr.  
  Source: https://github.com/nvlabs/ffhq-dataset

- `data/source/fake/`  
  Contains AI-generated and manipulated images, grouped by generation type:
  - `GAN/` → includes images from `thispersondoesnotexist` and `sfhq`  
    (GAN-based generation, primarily StyleGAN variants)  
    SFHQ source: https://github.com/SelfishGene/SFHQ-dataset

  - `Diffusion/` → includes images from `stable_diffusion`  
    (diffusion-based generation)

  - `FaceSwap/` → includes images from `faceswap`  
    (face manipulation / deepfake techniques)

---

### Motivation

This reorganization allows us to clearly separate:
- real vs synthetic data, and  
- different types of synthetic generation methods.

This is particularly important for our analysis using Grad-CAM, as it enables us to compare how models focus on different facial regions depending on whether the image is GAN-generated, diffusion-generated, or manipulated.
