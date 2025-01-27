# Image-Summarization

This project implements an image-to-paragraph captioning system using a combination of the EfficientNetB7 model for image feature extraction and the T5 language model for generating natural language summaries. The workflow includes preprocessing, feature extraction, training, evaluation, and visualization.

---

## Project Directory Structure

- **`dataset_im2p`**: Contains images and caption data.
  - `stanford_images/`: Folder containing input images.
  - `stanford_df_rectified.csv`: CSV file containing captions and metadata.
  -  `Dataset Link` : http://cs.stanford.edu/people/ranjaykrishna/im2p/index.html
- **`scene_t5_final.py`**: Python script implementing the captioning pipeline.
- **`features_captions_cross.pkl`**: Precomputed features and captions for faster loading.

---

## Key Files and Usage

### `scene_t5_final.py`

This script performs the following tasks:

1. **Data Loading and Preprocessing**
   - Loads image-paragraph pairs from the dataset CSV.
   - Splits data into training, validation, and test sets.

2. **Model Initialization**
   - **EfficientNetB7**: Pre-trained CNN for feature extraction (non-trainable).
   - **T5 (Text-to-Text Transfer Transformer)**: Pre-trained sequence-to-sequence model for text generation.

3. **Feature Extraction**
   - Extracts image features using EfficientNetB7.
   - Projects image features to the T5 embedding dimension using a custom cross-attention mechanism.

4. **Training**
   - Optimizes the T5 model using a sparse categorical cross-entropy loss function.
   - Includes an early stopping mechanism to prevent overfitting.

5. **Evaluation**
   - Calculates BLEU scores and other metrics (precision, recall, F1-score, and accuracy).
   - Generates captions for test images and visualizes attention maps.

6. **Visualization**
   - Generates and saves images with reference and generated captions.
   - Visualizes attention scores overlayed on the input image.

---

## Environment Setup

### Prerequisites

1. **Python version**: 3.8+
2. **Required Libraries**:

```bash
pip install tensorflow transformers nltk scikit-learn matplotlib pandas numpy pickle-mixin
```

3. **Hardware Requirements**:
   - GPU support is recommended (e.g., NVIDIA CUDA-enabled GPUs).
   - Ensure CUDA drivers and cuDNN are correctly installed.

---

## How to Run

### 1. Clone Repository
```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Prepare Dataset

- Place images in the `stanford_images/` folder.
- Ensure `stanford_df_rectified.csv` contains:
  - `Image_name`: Image filenames (without extensions).
  - `Paragraph`: Corresponding captions.
  - `train`, `val`, `test`: Flags for dataset split.

### 3. Run Script

```bash
python scene_t5_final.py
```

- The script will:
  - Compute or load precomputed image features and captions.
  - Train the model (unless pre-trained weights are provided).
  - Evaluate the model and save results.

---

## Configuration

### Adjustable Parameters

- **Global Constants**:
  - `IMAGE_SIZE`: Target size for resizing images (default: `(224, 224)`).
  - `SEQ_LENGTH`: Maximum sequence length for captions (default: `128`).
  - `BATCH_SIZE`: Batch size for training (default: `8`).
  - `EPOCHS`: Number of epochs for training (default: `50`).
- **Paths**:
  - `IMAGES_PATH`: Path to the folder containing images.
  - `CAPTIONS_FILE`: Path to the CSV file with captions.
  - `FEATURES_CAPTIONS_FILE`: Path to save/load precomputed features and captions.

---

## Evaluation Metrics

The following metrics are computed during evaluation:

1. **BLEU Scores**:
   - BLEU-1 to BLEU-4 using NLTK.
2. **Text Matching Metrics**:
   - Precision, Recall, F1-score, and Accuracy.

---

## Results

### Sample Evaluation

The script saves:
- Images with reference and generated captions.
- Attention maps visualizing areas of focus during caption generation.

### BLEU Scores

- Displays BLEU scores for each image and averages across the test set.

---

## Troubleshooting

### Common Errors

1. **Missing CUDA Drivers**
   - Ensure NVIDIA drivers and CUDA toolkit are correctly installed.
2. **FileNotFoundError**
   - Verify that image and caption paths in the script are correct.

### Debugging

- Modify `warnings.filterwarnings("ignore")` to show detailed warnings.
- Insert `print()` statements to inspect intermediate outputs.

---

## Future Work

1. Integrate advanced decoding methods (e.g., diverse beam search).
2. Explore other pre-trained models for text generation (e.g., GPT).
3. Implement additional evaluation metrics (e.g., ROUGE, CIDEr).

---

## Contact

For questions or issues, please contact [Mohan Kumar B Chandrashekar] at [mohanchandru920@gmail.com].

