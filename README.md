# Predicting Fake Job Postings Using Text + Metadata Patterns in pattern recognition

A Machine Learning Approach to Online Recruitment Fraud Detection

## Overview

Online job platforms have made recruitment more accessible, but fraudulent job advertisements remain a serious concern. Fake job postings can mislead job seekers, collect personal information, and expose applicants to financial scams.

This project investigates the detection of fraudulent job postings using the Employment Scam Aegean Dataset (EMSCAD). It combines natural language processing (NLP) techniques with structured metadata to classify job advertisements as real or fake.

The project evaluates multiple machine learning models and an LSTM-based recurrent neural network to explore how textual content and job-posting metadata can support recruitment fraud detection.

## Objectives

- Analyze the distribution and characteristics of real and fraudulent job postings.
- Apply text preprocessing and TF-IDF feature extraction to job-related text.
- Combine textual features with structured metadata.
- Compare multiple machine learning and deep learning approaches.
- Evaluate model performance using classification metrics.
- Investigate influential textual patterns and metadata associated with fraudulent postings.

## Dataset

The project uses the Employment Scam Aegean Dataset (EMSCAD), a publicly available dataset for online recruitment fraud detection.

**Dataset source:** [Real / Fake Job Posting Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

The report describes a working dataset containing approximately 18,000 job postings, including both legitimate and fraudulent advertisements. The original EMSCAD dataset contains 17,880 records, so the exact number of records used in the experiments should be verified against the processed CSV file.

### Key Features

| Feature | Type | Description |
|---|---|---|
| Job-related text | Text | Job title, description, company profile, and other available textual fields |
| `has_salary` | Binary | Indicates whether salary information is available |
| `has_company` | Binary | Indicates whether company profile information is available |
| Target variable | Binary | Real job (0) or fake job (1) |

The target labels are defined as:

- `0`: Real job posting
- `1`: Fake job posting

## Methodology

The project follows a machine learning pipeline:

1. **Exploratory Data Analysis (EDA):** Examine class distribution, missing values, text length, and metadata patterns.
2. **Data Preprocessing:** Handle missing values, clean textual data, and prepare relevant features.
3. **Feature Engineering:** Transform textual information using TF-IDF and encode selected metadata features.
4. **Multimodal Feature Integration:** Combine sparse text vectors and structured features using `scipy.sparse.hstack` where applicable.
5. **Model Training:** Train and evaluate machine learning classifiers and an LSTM-based model.
6. **Performance Evaluation:** Compare classification results using multiple evaluation metrics.
7. **Interpretation:** Examine feature importance, clustering patterns, and model errors.

## Models Evaluated

### Traditional Machine Learning

- **Logistic Regression:** Baseline classification model for high-dimensional text features.
- **Random Forest:** Ensemble learning model for identifying potentially nonlinear relationships.
- **Support Vector Machine (SVM):** Classifier suited to high-dimensional feature representations.
- **K-Nearest Neighbors (KNN):** Distance-based classification approach.

### Deep Learning

- **Long Short-Term Memory (LSTM):** Recurrent neural network architecture used to learn sequential patterns in job-posting text.

The reported LSTM configuration uses three training epochs, a batch size of 32, and a validation split of 0.1. The exact preprocessing and metadata-integration strategy should be confirmed from the notebook implementation.

## Exploratory Data Analysis

The analysis investigates:

- Real versus fake job-posting distribution.
- Distribution of job-description lengths.
- Missing values across relevant fields.
- Relationship between company-profile availability and fraudulent postings.
- K-means clustering for exploratory pattern analysis.
- Textual features that may help distinguish fraudulent advertisements.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC curve and ROC-AUC
- Confusion matrix

Because fraudulent job postings are the minority class, accuracy alone is insufficient. Fake-class precision, recall, F1-score, and ROC-AUC are important for understanding detection quality.

## Reported Experimental Results

The project report records an LSTM test accuracy of approximately **96.78%**.

However, high accuracy does not necessarily imply effective fraud detection. The reported LSTM results indicate that recall for fake job postings is lower than recall for real postings, highlighting the difficulty of identifying the minority class.

The reported metrics and confusion matrices should be revalidated against the actual test predictions before drawing final conclusions or making model performance claims.

## Technologies Used

- Python
- Google Colab
- Pandas and NumPy
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- SciPy sparse matrices
- TensorFlow/Keras or the framework used in the notebook for LSTM
- Matplotlib and Seaborn, if used in the notebook

## Repository Structure

```text
fake-job-postings-detection/
├── notebooks/
│   └── Fake Job Posting.ipynb
├── report/
│   └── Fake Job Posting.pdf
└── README.md
```

*The filenames above are suggested. Update them to match the files uploaded to this repository.*

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
   ```

2. Open the repository folder.

3. Open the notebook in Google Colab or Jupyter Notebook.

4. Download the dataset from the linked Kaggle source.

5. Upload the CSV file to your runtime or configure the notebook to use its actual dataset path.

6. Run the notebook cells in order to reproduce the preprocessing, training, evaluation, and visualization steps.

### Dependencies

Install the commonly used Python packages with:

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn tensorflow
```

Adjust the dependencies to match the libraries actually imported by the notebook.

## Limitations

- The dataset is imbalanced, with fewer fraudulent postings than legitimate ones.
- Text and metadata alone may not capture all fraud signals.
- Results can vary depending on preprocessing, data splitting, and model configuration.
- Performance should be validated with consistent evaluation metrics and, ideally, stratified cross-validation.
- The experiments do not establish real-world fraud detection performance without testing on independent, unseen data.

## Future Work

- Apply class-imbalance handling techniques and threshold tuning.
- Use stratified k-fold cross-validation for more reliable evaluation.
- Explore pretrained language models such as BERT.
- Investigate contextual text embeddings and additional metadata.
- Improve fake-class recall while controlling false positives.
- Validate models on newer, independently collected job-posting data.

## References

1. S. Vidros, C. Kolias, G. Kambourakis, and L. Akoglu, “Automatic Detection of Online Recruitment Frauds: Characteristics, Methods, and a Public Dataset,” *Future Internet*, vol. 9, no. 1, 2017. [Read paper](https://www.mdpi.com/1999-5903/9/1/6).
2. S. Dutta and S. Kumar Bandyopadhyay, “Fake Job Recruitment Detection Using Machine Learning Approach,” *International Journal of Engineering Trends and Technology*, 2019.
3. K. Taneja, J. Vashishtha, and S. Ratnoo, “Fraud-BERT: Transformer Based Context Aware Online Recruitment Fraud Detection,” 2025.
4. A. S. Pillai, “Detecting Fake Job Postings Using Bidirectional LSTM,” 2023.
5. A. K. Praveen et al., “Detecting Fake Job Posting Using ML Classifications and Ensemble Model,” 2023.

## Author

**Project:** Predicting Fake Job Postings Using Text + Metadata Patterns

**Purpose:** Academic machine learning project on online recruitment fraud detection.

For dataset details and reproducibility, refer to the notebook and project report included in this repository.
