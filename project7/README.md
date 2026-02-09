# HRNet Blur Classification Project

This project uses an HRNet-W18 model to classify the blur radius (0-9) of images.

## Recent Updates

### 1. Label Range Fix (0-9)
- **Problem**: The training script `train.py` was previously using a 1-10 label range and subtracting 1, which caused errors when the dataset was updated to use 0-9 directly.
- **Fix**: Updated `train.py` to use labels 0-9 directly without modification.

### 2. Evaluation Metadata Format
- **Update**: The `evaluate.py` script has been updated to handle the new metadata format which includes 3 elements: `(key, label, sharpness_map)`.

### 3. Tensor Database Generation (New)
We have added a new pipeline to export the HRNet features (10x15x20 tensors) to a separate LMDB database for future use.

- **Script**: `generate_tensor_db.py`
  - Loads images from `coc_train.lmdb`.
  - Runs HRNet inference.
  - Saves the output tensor `[10, 15, 20]` (Channels, Height, Width) to `coc_tensor_10x15x20.lmdb`.
  - **Note**: The resolution is naturally downsampled by 32x (480/32=15, 640/32=20).

- **Verification**: `verify_tensor_performance.py`
  - Reads the generated tensors.
  - Applies **Top-k Soft Expectation**:
    1.  Computes Softmax probabilities over the 10 classes.
    2.  Selects Top-3 probabilities (masks others to 0).
    3.  Re-normalizes probabilities to sum to 1.
    4.  Calculates weighted expected radius.
  - Validates that the generated features match the ground truth labels.
  - Outputs detailed per-label error histograms.

## Key Scripts

| Script | Description |
| :--- | :--- |
| `preprocessing.py` | Generates the training dataset (LMDB) from source images. |
| `train.py` | Trains the HRNet-W18 model on the dataset. |
| `evaluate.py` | Runs evaluation loop on the test set. |
| `evaluate_single_image.py` | Visualizes model predictions on a single image (with crops). |
| `generate_tensor_db.py` | **[NEW]** Generates feature tensor database. |
| `verify_tensor_performance.py` | **[NEW]** Verifies tensor database quality. |

## Usage

### Generate Tensor Database
```bash
python generate_tensor_db.py
```

### Verify Tensor Database
```bash
python verify_tensor_performance.py
```
This will print a detailed error analysis table for each label (0-9), showing accuracy and error distribution.
