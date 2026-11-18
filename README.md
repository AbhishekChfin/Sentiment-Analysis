# Amazon Reviews Sentiment Analysis

This is a machine learning and sentiment analysis project for classifying Amazon product reviews as positive or negative sentiment using natural language processing and traditional ML linear models from scratch.

## Linear Models and accuracy

- **Logistic Regression** 
- **Linear Support Vector Classifier (SVC)**

Both models achieve ~88% accuracy on the test set using HashingVectorizer with n-gram features.

## Dataset

### Source: Download the dataset using the link below.
- **Kaggle Dataset**: [Amazon Reviews](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)
- **Format**: Data file is in bz2 compressed format
- **Size**: around 3.6M reviews (train) + 0.4M reviews (test)

## Project Structure

```
Sentiment Analysis/
├── README.md                              # Detailed project walkthrough
├── amazon-reviews-sentiment-analysis.ipynb # Main analysis notebook
```

## Installation

### Dependencies

Install required packages using pip:

```bash
pip install pandas nltk scikit-learn tqdm numpy scipy
```

Or install from a requirements file:

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

1. **Clone/Download the repository**:
   ```bash
   cd "Sentiment Analysis"
   ```

## Results

### Model Performance

Both models were evaluated on 40K test samples and the results are following:

```
Logistic Regression:
  - Accuracy: ~88%

Linear SVC:
  - Accuracy: ~88%
```

View exact metrics in the python notebook.

### Feature Statistics

- **Vocabulary Size**: ~40K unique words (from training set)
- **Feature Dimension**: 30,000 (HashingVectorizer with n-grams 1-2)
- **Sparsity**: ~95% (typical for text features)
- **Training Set**: 90% (~3.2M samples)
- **Test Set**: 10% (~0.4M samples)


## Model Configuration

- You can compare different configuration for both models, but the configurations mentioned below resulted with the least loss and better accuracy.

**Logistic Regression**:
- Solver: `lbfgs`
- C: 5.0 (inverse regularization strength)
- Max iterations: 1000

**Linear SVC**:
- Dual: False (for large sample count)
- Max iterations: 1000

## References

- [scikit-learn Documentation](https://scikit-learn.org)
- [NLTK Book](https://www.nltk.org/book/)
- [FastText Format](https://fasttext.cc/docs/en/supervised-tutorial.html)

## License

This project uses publicly available data from Kaggle.

## Author

Created as a comprehensive sentiment analysis pipeline demonstrating end-to-end machine learning workflows for NLP tasks.

---
