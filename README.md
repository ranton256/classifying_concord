# Classifying Concord: Machine Learning Meets Transcendentalism

This project demonstrates how the use of machine learning to distinguish between the writing styles of Ralph Waldo Emerson and Henry David Thoreau—two central figures of American Transcendentalism. It show how to use NLP techniques to construct a novel dataset from their public domain works and provides a practical example of ML classification techniques.

## Project Overview

The dataset consists of passages (3-5 sentences each) which were extracted from Emerson's *Essays: First Series* and Thoreau's *Walden, and On The Duty Of Civil Disobedience*, resulting in 1,911 labeled text segments.

Each passage was classified to assign an author using a range of machine learning models, from traditional algorithms to modern transformer-based approaches.

## Methods

- **Preprocessing**: Texts were cleaned, segmented, and lemmatized using spaCy. Stopwords and boilerplate were removed.
- **Feature Engineering**: Both TF-IDF vectorization and transformer-based embeddings (DistilBERT) were used to represent text.
- **Models Compared**:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - DistilBERT
    - For feature extraction for other models.
    - Directly as a classifier after fine-tuning.

## Results

- **Traditional ML (TF-IDF + classifiers)**: 83–86% accuracy
- **DistilBERT features + classifiers**: 89–90% accuracy
- **Fine-tuned DistilBERT**: 92% accuracy (best)

Analysis showed that misclassifications often involved boilerplate, short segments, or philosophical passages where both authors' styles converged. Thoreau's concrete nature descriptions were rarely confused for Emerson's more abstract prose.

## Why This Matters

This project bridges machine learning and literary analysis, providing a reproducible case study in authorship attribution and stylistic quantification. It demonstrates the progression from classic ML to state-of-the-art NLP, and offers insights for both technical and humanities audiences.

## Getting Started

1. **Clone the repository:**

   ```shell
   git clone https://github.com/yourusername/classifying_concord.git
   cd classifying_concord
   ```

2. **Install dependencies:**

   ```shell
   pip install -r requirements.txt
   ```

3. **Download spaCy English model:**

   ```shell
   python -m spacy download en_core_web_sm
   ```

4. **Run the Jupyter notebook:**

   ```shell
   jupyter notebook supervised_ML_identify_author.ipynb
   ```

For more details, see the full article in `classifying_condord.md` and the code in this repository.
