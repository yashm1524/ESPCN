# ESPCN Setup & Usage Guide

Complete guide to set up and use ESPCN for 4x super-resolution.

---

## 📋 Table of Contents

1. [Installation](#installation)
2. [Dataset Setup](#dataset-setup)
3. [Quick Start - Inference](#quick-start---inference)
4. [Training from Scratch](#training-from-scratch)
5. [Model Export](#model-export)
6. [Troubleshooting](#troubleshooting)

---

## 1. Installation

### Prerequisites

- **Python**: 3.7 or higher
- **CUDA** (for GPU training): 11.0 or higher
- **GPU**: NVIDIA GPU with 4GB+ VRAM (for training)

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/ESPCN-GOPRO.git
cd ESPCN-GOPRO
```

### Step 2: Install Dependencies

**Option A: Using pip**
```bash
pip install -r requirements.txt
```

**Option B: Using conda**
```bash
conda create -n espcn python=3.8
conda activate espcn
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

**Expected output:**
```
PyTorch version: 2.x.x
CUDA available: True  # or False if no GPU
```

---

## 2. Dataset Setup

### Option A: Download Pre-prepared GOPRO Dataset

*(If available)*

```bash
# Download from Google Drive or release page
# Extract to project root
unzip GOPRO_SR.zip
```

Expected structure:
```
GOPRO_SR/
├── train/
│   ├── LR_x4/  (720 images)
│   └── HR/     (720 images)
└── test/
    ├── LR_x4/  (309 images)
    └── HR/     (309 images)
```

### Option B: Prepare Your Own Dataset

1. **Download original GOPRO dataset:**
   - Visit: https://seungjunnah.github.io/Datasets/gopro.html
   - Download and extract

2. **Create dataset structure:**

```bash
mkdir -p GOPRO_SR/train/{LR_x4,HR}
mkdir -p GOPRO_SR/test/{LR_x4,HR}

# Copy your HR images to HR folders
# Then generate LR images:
```

3. **Generate LR images (Python script):**

```python
from PIL import Image
import os
from tqdm import tqdm

def create_lr_images(hr_dir, lr_dir, scale=4):
    """
    Create LR images by downsampling HR images
    
    Args:
        hr_dir: Path to HR images
        lr_dir: Path to save LR images
        scale: Downsampling factor (default: 4)
    """
    os.makedirs(lr_dir, exist_ok=True)
    
    hr_images = [f for f in os.listdir(hr_dir) if f.endswith('.png')]
    
    for img_name in tqdm(hr_images, desc=f"Creating LR images"):
        hr_path = os.path.join(hr_dir, img_name)
        lr_path = os.path.join(lr_dir, img_name)
        
        # Load HR image
        hr_img = Image.open(hr_path).convert('RGB')
        
        # Downsample to create LR image
        lr_width = hr_img.width // scale
        lr_height = hr_img.height // scale
        lr_img = hr_img.resize((lr_width, lr_height), resample=Image.BICUBIC)
        
        # Save LR image
        lr_img.save(lr_path)
    
    print(f"Created {len(hr_images)} LR images in {lr_dir}")

# Run for train and test sets
create_lr_images('GOPRO_SR/train/HR', 'GOPRO_SR/train/LR_x4', scale=4)
create_lr_images('GOPRO_SR/test/HR', 'GOPRO_SR/test/LR_x4', scale=4)
```

Save as `prepare_dataset.py` and run:
```bash
python prepare_dataset.py
```

---

## 3. Quick Start - Inference

### Download Pre-trained Model

1. Download `best.pth` from [Releases](https://github.com/YOUR_USERNAME/ESPCN-GOPRO/releases)
2. Place in `./assets/models/` directory

```bash
mkdir -p assets/models
# Place best.pth here
```

### Run Inference on Single Image

**Basic usage:**
```bash
python infer.py \
    -w ./assets/models/best.pth \
    --fpath_lr_image ./dataset/test/YOUR_IMAGE.png \
    -o ./output \
    --scaling_factor 4
```

**With ground truth (calculates PSNR):**
```bash
python infer.py \
    -w ./assets/models/best.pth \
    --fpath_lr_image ./GOPRO_SR/test/LR_x4/000001.png \
    --fpath_hr_image ./GOPRO_SR/test/HR/000001.png \
    -o ./output \
    --scaling_factor 4 \
    --show_plot
```

**Output:**
- `output/000001_espcn_x4.png` - Super-resolved image
- `output/000001_bicubic_x4.png` - Bicubic upsampled (for comparison)
- `output/comparison_result.png` - Side-by-side comparison
- Console: PSNR value

### Batch Inference (Multiple Images)

Create `batch_infer.py`:

```python
import os
from glob import glob
import subprocess

lr_dir = './GOPRO_SR/test/LR_x4'
hr_dir = './GOPRO_SR/test/HR'
output_dir = './batch_output'
model_path = './assets/models/best.pth'

os.makedirs(output_dir, exist_ok=True)

lr_images = sorted(glob(os.path.join(lr_dir, '*.png')))

for lr_path in lr_images:
    img_name = os.path.basename(lr_path)
    hr_path = os.path.join(hr_dir, img_name)
    
    cmd = f"""python infer.py \
        -w {model_path} \
        --fpath_lr_image {lr_path} \
        --fpath_hr_image {hr_path} \
        -o {output_dir} \
        --scaling_factor 4"""
    
    print(f"Processing: {img_name}")
    subprocess.run(cmd, shell=True)
    
print("Batch inference complete!")
```

Run:
```bash
python batch_infer.py
```

---

## 4. Training from Scratch

### Baseline Model (200 epochs, ~30.36 dB)

```bash
python train_gopro.py \
    --dirpath_train_lr ./GOPRO_SR/train/LR_x4 \
    --dirpath_train_hr ./GOPRO_SR/train/HR \
    --dirpath_val_lr ./GOPRO_SR/test/LR_x4 \
    --dirpath_val_hr ./GOPRO_SR/test/HR \
    -o ./models \
    --scaling_factor 4 \
    --epochs 200 \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --save_interval 25
```

**Training time:**
- **GPU (T4)**: ~12-15 hours
- **CPU**: ~40-60 hours (not recommended)

**Monitor training:**
```bash
# In another terminal
watch -n 10 nvidia-smi

# View last 20 lines of output
tail -20 training.log
```

### Improved Model (300 epochs, ~31.0 dB)

With data augmentation and improved architecture:

```bash
python train_gopro_improved.py \
    --dirpath_train_lr ./GOPRO_SR/train/LR_x4 \
    --dirpath_train_hr ./GOPRO_SR/train/HR \
    --dirpath_val_lr ./GOPRO_SR/test/LR_x4 \
    --dirpath_val_hr ./GOPRO_SR/test/HR \
    -o ./models_improved \
    --scaling_factor 4 \
    --epochs 300 \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --save_interval 25
```

**Training time:**
- **GPU (T4)**: ~18-24 hours

### Resume Training from Checkpoint

Modify `train_gopro.py` to add checkpoint loading:

```python
# Add after model initialization
if os.path.exists('./models/epoch_100.pth'):
    model.load_state_dict(torch.load('./models/epoch_100.pth'))
    print("Resumed from epoch 100")
```

---

## 5. Model Export

### Export to ONNX (for production deployment)

```bash
python export.py \
    -i ./models/best.pth \
    -o ./exported_models \
    -f ONNX \
    --scaling_factor 4
```

**Output:** `exported_models/espcn_model_x4.onnx`

**Use ONNX model:**
```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession('exported_models/espcn_model_x4.onnx')

# Prepare input (example)
input_data = np.random.randn(1, 1, 224, 224).astype(np.float32)

# Run inference
outputs = session.run(None, {'input': input_data})
print("Output shape:", outputs[0].shape)  # (1, 1, 896, 896)
```

### Export to TensorFlow Lite (for mobile)

```bash
python export.py \
    -i ./models/best.pth \
    -o ./exported_models \
    -f TFLite \
    --scaling_factor 4
```

**Output:** `exported_models/espcn_saved_model_x4.tflite`

### Export to CoreML (for iOS)

```bash
# Requires macOS or coremltools
python export.py \
    -i ./models/best.pth \
    -o ./exported_models \
    -f CoreML \
    --scaling_factor 4
```

**Output:** `exported_models/espcn_model_x4.mlmodel`

---

## 6. Troubleshooting

### Issue 1: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Reduce batch size
python train_gopro.py ... --batch_size 8  # instead of 16
```

### Issue 2: Model Trained on GPU, Running on CPU

**Error:**
```
RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False
```

**Solution:**
Already fixed in `infer.py` with `map_location=device`. Make sure you're using the latest version.

### Issue 3: Slow Training on CPU

**Symptom:** Training taking 40+ hours

**Solution:**
- Use GPU (T4, V100, A100)
- Or use Google Colab (free GPU)
- Or use cloud platforms (GCP, AWS)

### Issue 4: Low PSNR Results

**Symptom:** PSNR stuck at ~28-29 dB

**Possible causes:**
1. Dataset not properly prepared (LR/HR mismatch)
2. Training not converged (increase epochs)
3. Learning rate too high/low

**Solution:**
```bash
# Check dataset
python -c "from dataloader_gopro import get_data_loader; \
           train_loader, val_loader = get_data_loader(...); \
           print('Dataset loaded successfully')"

# Train longer
python train_gopro.py ... --epochs 300
```

### Issue 5: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'cv2'
```

**Solution:**
```bash
pip install opencv-python
```

---

## Advanced Usage

### Custom Dataset

To use your own dataset:

1. Prepare dataset in GOPRO structure
2. Modify `dataloader_gopro.py` if needed
3. Train with your dataset paths

### Hyperparameter Tuning

Key hyperparameters to experiment with:

```python
# In train_gopro.py or via command line
--learning_rate 1e-3     # Try: 5e-4, 1e-3, 2e-3
--batch_size 16          # Try: 8, 16, 32
--patch_size 17          # Try: 17, 33, 48
--stride 13              # Try: 10, 13, 17
```

### Multi-GPU Training

Add to `train_gopro.py`:

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

---

## Performance Tips

1. **Use GPU:** Essential for reasonable training times
2. **Larger batch size:** Better gradient estimates (if GPU memory allows)
3. **Data augmentation:** Reduces overfitting
4. **Mixed precision:** Faster training with `torch.cuda.amp`
5. **Checkpointing:** Save every 25-50 epochs to resume if interrupted

---

## Support

For issues, questions, or contributions:

1. Check [Issues](https://github.com/YOUR_USERNAME/ESPCN-GOPRO/issues)
2. Open a new issue with details
3. Join discussions

---

**Happy Super-Resolving! 🚀**
