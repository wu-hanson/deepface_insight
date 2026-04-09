# Vision Transformer for AI-Generated Face Detection

**Student**: Hanson Wu  
**Course**: CS 7150 - Deep Learning  
**Project**: Interpreting AI-Generated Face Detection Models

## Project Overview

This project implements a **Vision Transformer (ViT)** for detecting AI-generated face images and uses **Grad-CAM** to interpret model predictions. The focus is on understanding what visual features ViTs rely on compared to CNNs.

**Key Hypothesis**: Vision Transformers emphasize **global facial structure and geometry**, while CNNs tend to focus on **local texture artifacts**.

## Project Structure

```
final_project/
├── vision_transformer_face_detection.ipynb  # Main notebook (ENTRY POINT)
├── vit_data_utils.py                        # Data loading & preprocessing
├── vit_model.py                             # ViT model & training logic
├── vit_gradcam.py                           # Grad-CAM visualization
├── requirements.txt                         # Python dependencies
├── README.md                                # This file
├── checkpoints/                             # Saved model weights
├── results/                                 # Output visualizations & metrics
└── data/                                    # Face dataset (user-provided)
    ├── train/
    │   ├── real/
    │   └── fake/
    ├── val/
    │   ├── real/
    │   └── fake/
    └── test/
        ├── real/
        └── fake/
```

## Getting Started

### 1. Install Dependencies

Using the course virtualenv:
```bash
source /Users/hansonwu/Document/Github/cs7150_code/cs7150_venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download one of these datasets from Kaggle:
- **Real vs AI-Generated Faces Dataset** (~7K images): https://www.kaggle.com/datasets/philosopher0808/real-vs-ai-generated-faces-dataset
- **140K Real and Fake Faces** (~140K images): https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

Extract into `./data/` directory with structure:
```
data/
  train/
    real/      (images)
    fake/      (images)
  val/
    real/      (images)
    fake/      (images)
  test/
    real/      (images)
    fake/      (images)
```

If you only have one split, the code will handle it gracefully.

### 3. Run the Notebook

```bash
cd /Users/hansonwu/Document/Github/cs7150_code/final_project
jupyter notebook vision_transformer_face_detection.ipynb
```

Execute cells in order:
1. **Import Libraries** - Load dependencies
2. **Load Dataset** - Verify data is present
3. **Build Model** - Initialize ViT
4. **Train** - Run training loop (30-50 epochs)
5. **Evaluate** - Compute metrics on test set
6. **Grad-CAM** - Generate attention visualizations
7. **Analyze** - Compare patterns between real/fake

## Module Documentation

### `vit_data_utils.py`
- `FaceClassificationDataset`: PyTorch dataset class
- `get_data_transforms()`: Standard preprocessing pipeline
- `create_dataloaders()`: Creates train/val/test loaders

### `vit_model.py`
- `ViTFaceDetector`: ViT-based binary classifier
- `ViTTrainer`: Training and evaluation logic
  - `train_epoch()`: Single epoch training
  - `validate()`: Validation loop
  - `evaluate()`: Test set evaluation
  - `save_checkpoint()` / `load_checkpoint()`: Model persistence

### `vit_gradcam.py`
- `ViTGradCAM`: Grad-CAM implementation for ViT
  - `generate_cam()`: Single image CAM
  - `generate_batch_cams()`: Batch processing
- `AttentionVisualization`: Utility functions
  - `create_heatmap()`: Overlay CAM on image
  - `compare_real_vs_fake_attention()`: Side-by-side comparison
  - `visualize_patch_importance()`: Highlight important patches

## Expected Results

### Classification Performance
- **Accuracy**: ~85-95% (depends on dataset)
- **Precision/Recall**: Balanced for both classes

### Key Insights from Grad-CAM
1. **Real faces**: Attention spreads across facial features
2. **AI-generated**: Attention concentrates on specific artifacts or boundaries
3. **ViT advantage**: Captures long-range dependencies in facial geometry

## Output Files

After running, check:
- `results/training_history.png` - Loss and accuracy curves
- `results/confusion_matrix.png` - Classification breakdown
- `results/gradcam_visualizations.png` - Sample attention maps
- `results/attention_comparison.png` - Real vs AI-generated pattern differences
- `results/test_metrics.json` - Numerical performance metrics
- `checkpoints/best_model.pth` - Best trained model

## Configuration

Key hyperparameters (modify in notebook cell 4):
```python
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 10
```

## Troubleshooting

**Error: "No images found in data/"**
- Check dataset directory structure
- Ensure image files have .jpg, .jpeg, .png extensions

**CUDA out of memory:**
- Reduce `BATCH_SIZE` to 16 or 8
- Use CPU: Set `device = torch.device("cpu")`

**Model training is very slow:**
- Reduce `NUM_WORKERS` in dataloader
- Reduce dataset size for initial testing

## References

[1] Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021

[2] Geirhos et al., "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness", ICLR 2019

[3] Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017

[4] Rössler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images", ICCV 2019

## Contact

Hanson Wu  
CS 7150 - Deep Learning  
Northeastern University

---

For the CNN implementation, see Rayan Hassan's work in the project repository.
