# TF-IDF Text Classification Model

This project trains a neural-network text classifier on the 20 Newsgroups dataset using TF-IDF features and a multilayer perceptron (MLP). The implementation is contained in the notebook:

```text
C:\Users\sudee\Downloads\Paper_Latex_Code\TF-IDF_Model.ipynb
```

The model converts text documents into high-dimensional TF-IDF vectors, trains a dense Keras classifier, and evaluates performance using accuracy and a detailed classification report.

## Project Overview

The notebook demonstrates a supervised text classification pipeline for assigning news documents to one of 20 topic categories from the 20 Newsgroups dataset.

The workflow includes:

- Loading the full 20 Newsgroups dataset from scikit-learn
- Creating a stratified train-test split
- Converting raw text into TF-IDF features
- Training a dense neural network classifier
- Evaluating the model with test accuracy and per-class metrics

## Dataset

The project uses the built-in `fetch_20newsgroups` dataset from scikit-learn.

Dataset details:

- Source: scikit-learn 20 Newsgroups dataset
- Subset used: `all`
- Number of classes: 20
- Input type: raw text documents
- Target type: integer class labels mapped to topic names
- Split: 80% training and 20% testing
- Split strategy: stratified split to preserve class distribution

The dataset is downloaded automatically the first time the notebook is run.

## Methodology

### Text Vectorization

The notebook uses `TfidfVectorizer` to convert documents into numerical features.

Configuration:

```python
TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    ngram_range=(1, 2),
    max_features=50000
)
```

This means the model uses:

- English stop-word removal
- Unigrams and bigrams
- Terms appearing in no more than 70% of documents
- A maximum vocabulary size of 50,000 features

The sparse TF-IDF matrices are converted to dense `float32` arrays before being passed to Keras.

## Model Architecture

The classifier is a Keras Sequential MLP.

Architecture:

| Layer | Output Size | Activation | Notes |
| --- | ---: | --- | --- |
| Input | 50,000 | None | TF-IDF feature vector |
| Dense | 512 | ReLU | First hidden layer |
| Dropout | 512 | None | Dropout rate: 0.3 |
| Dense | 256 | ReLU | Second hidden layer |
| Dropout | 256 | None | Dropout rate: 0.3 |
| Dense | 20 | Softmax | Class probabilities |

Model summary from the saved notebook output:

```text
Total params: 25,736,980
Trainable params: 25,736,980
Non-trainable params: 0
Approximate model size: 98.18 MB
```

## Training Configuration

The model is compiled with:

- Framework: TensorFlow / Keras
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: categorical cross-entropy
- Metric: accuracy
- Epochs: 10
- Batch size: 128
- Validation split: 10% of the training data

Labels are one-hot encoded using `LabelBinarizer`.

## Evaluation

After training, the notebook evaluates the model on the held-out test set.

Evaluation outputs:

- Test loss
- Test accuracy
- Classification report with precision, recall, and F1-score for each newsgroup class

The final test accuracy is printed by the notebook with:

```python
print(f"Test Accuracy: {test_acc:.4f}")
```

The saved notebook output includes the model summary and early training logs, but it does not include the final printed test accuracy or full classification report. Re-run the notebook to regenerate the final metrics.

## Requirements

Install the required Python packages before running the notebook:

```bash
pip install numpy scikit-learn tensorflow jupyter
```

Recommended environment:

- Python 3.9 or later
- TensorFlow 2.x
- scikit-learn
- Jupyter Notebook or JupyterLab

## How to Run

1. Open the notebook:

   ```text
   C:\Users\sudee\Downloads\Paper_Latex_Code\TF-IDF_Model.ipynb
   ```

2. Install dependencies if needed:

   ```bash
   pip install numpy scikit-learn tensorflow jupyter
   ```

3. Start Jupyter:

   ```bash
   jupyter notebook
   ```

4. Run all notebook cells from top to bottom.

5. Review the printed test accuracy and classification report at the end of the notebook.

## Generated Artifacts

The notebook folder also contains the following related image files:

- `accuracy_loss.png`
- `categorization_system.png`
- `confusion_matrix.png`
- `per_class_accuracy.png`
- `text_classification_architecture.png`

These files can be used in reports, presentations, or documentation to explain the model architecture and evaluation results.

## Notes

- Converting a 50,000-feature sparse TF-IDF matrix into a dense array can require significant memory.
- If memory usage is too high, reduce `max_features`, use a smaller batch size, or train a model that supports sparse inputs.
- The dataset download requires internet access the first time `fetch_20newsgroups` is run.
- Results may vary slightly depending on TensorFlow version, hardware, and random initialization.

## Project Structure

```text
Paper_Latex_Code/
+-- TF-IDF_Model.ipynb
+-- accuracy_loss.png
+-- categorization_system.png
+-- confusion_matrix.png
+-- per_class_accuracy.png
+-- text_classification_architecture.png
```

## Summary

This notebook provides a complete baseline for multi-class text classification using TF-IDF feature extraction and a dense neural network. It is suitable for comparing traditional text representations with neural classifiers and for generating classification metrics on the 20 Newsgroups benchmark dataset.
