# AI-Based Personalized Learning and Student Performance Predictor (P38)

A small, reproducible machine-learning project that predicts an **engineered student mastery category** from five categorical student features and provides simple practice recommendations.

> **Important scope note:** The dataset contains overall Mathematics, Reading, and Writing scores. It does not contain individual questions, question-difficulty labels, topic-level scores, or answer histories. Therefore, this project does **not** directly predict the difficulty of individual questions. It predicts a broad performance category and maps that category to a general practice suggestion.

## Project objectives

- Explore and validate the Students Performance in Exams dataset.
- Create Low, Medium, and High mastery labels from the three exam scores.
- Compare a simple baseline, Logistic Regression, and a small neural network.
- Evaluate performance using accuracy, macro-F1, per-class metrics, and confusion matrices.
- Test whether class weighting helps with the minority Low category.
- Load a saved model for a pre-exam estimate and provide a separate score-based post-exam recommendation.

## Dataset

**Source:** [Students Performance in Exams — Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

The dataset contains 1,000 student records and eight original columns:

- `gender`
- `race/ethnicity`
- `parental level of education`
- `lunch`
- `test preparation course`
- `math score`
- `reading score`
- `writing score`

The CSV is deliberately kept out of Git by `.gitignore`. Download the dataset from Kaggle, locate `StudentsPerformance.csv`, and place it in the project's `data/` directory. Review the dataset's current licence and terms before redistributing the file.

Expected local path:

```text
data/StudentsPerformance.csv
```

## How the target is created

The original dataset has no mastery label, so the project calculates the arithmetic mean of each student's three exam scores:

```text
average_score = (math score + reading score + writing score) / 3
```

It then assigns categories using fixed thresholds:

| Average score | Engineered mastery category |
|---|---|
| Below 50 | Low |
| 50 to below 75 | Medium |
| 75 or above | High |

These thresholds are project-defined rules, not labels supplied by the dataset. The observed full-dataset distribution was:

| Category | Students | Percentage |
|---|---:|---:|
| Low | 103 | 10.3% |
| Medium | 573 | 57.3% |
| High | 324 | 32.4% |

Because the classes are imbalanced, accuracy is not sufficient on its own. Macro-F1 and the precision, recall, and F1-score for each class are also reported.

## Input features and leakage prevention

The pre-exam model uses these five categorical features:

- `gender`
- `race/ethnicity`
- `parental level of education`
- `lunch`
- `test preparation course`

Math, Reading, Writing, and `average_score` are **not** used as model inputs because they are used to create the target. Including them would leak the answer into the inputs.

The features are observational and have important limitations, including possible demographic bias. Their associations must not be interpreted as causal effects or used as the basis for high-stakes educational decisions.

## Project structure

```text
AI-Student-Performance-Predictor/
├── data/
│   └── StudentsPerformance.csv           # Download locally; ignored by Git
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_Model_Training_and_Evaluation.ipynb
│   ├── 03_Personalized_Recommendations.ipynb
│   └── P38_Student_Performance.ipynb      # Unified end-to-end submission notebook
├── models/
│   ├── logistic_regression.joblib
│   ├── preprocessor.joblib
│   ├── student_mastery_model.keras
│   ├── logistic_regression_class_weighted.joblib
│   └── student_mastery_model_class_weighted.keras
├── reports/
│   ├── model_comparison.csv
│   ├── test_classification_report.csv
│   ├── confusion_matrices.png
│   ├── training_curves.png
│   ├── logistic_regression_coefficients.png
│   ├── validation_class_weighting_comparison.csv
│   └── weighted_training_curves.png
├── .gitignore
├── README.md
└── requirements.txt
```

Files under `models/` and `reports/` are generated when the corresponding notebook cells run. The exact contents of these directories may vary as experiments are added.

## Technology stack

- Python 3.11
- Pandas and NumPy for data processing
- Matplotlib and Seaborn for visualisation
- Scikit-learn for preprocessing, baseline models, Logistic Regression, metrics, and class weighting
- TensorFlow/Keras for the multilayer perceptron
- Joblib for saving and loading Scikit-learn objects
- Jupyter Notebook for the modular workflow

## Setup

### 1. Clone or open the repository

Open the repository folder in VS Code or Antigravity. The Conda environment below is the one used during development.

### 2. Create and activate the Conda environment (new setup only)

If you do not already have the environment, create it with Python 3.11:

```bash
conda create -n deep-learning python=3.11
conda activate deep-learning
```

If the `deep-learning` environment already exists, just activate it:

```bash
conda activate deep-learning
```

### 3. Install dependencies

From the project root, with the environment active:

```bash
python -m pip install -r requirements.txt
```

### 4. Add the dataset

Create the `data/` directory if it does not exist, then place the downloaded `StudentsPerformance.csv` file in it. The notebooks expect that filename and use relative paths, so they do not depend on a particular computer username.

### 5. Select the notebook kernel

In VS Code or Antigravity, select the Jupyter kernel associated with the `deep-learning` Conda environment (Python 3.11). You can check the active interpreter in a notebook with:

```python
import sys
print(sys.executable)
```

The path should point to the `deep-learning` environment.

## Run the notebooks in order

Run the notebooks from top to bottom and execute the cells in order.

### 1. `01_EDA_and_Preprocessing.ipynb`

- Load and validate the dataset.
- Inspect missing values, duplicates, data types, and score ranges.
- Plot score distributions and the relationship between subject scores.
- Compare average scores across selected student groups.
- Engineer the `average_score` and `mastery_level` columns and inspect the class distribution.

### 2. `02_Model_Training_and_Evaluation.ipynb`

- Recreate the target from the source score columns.
- Split the data into training, validation, and test sets (60% / 20% / 20%), using stratification and a fixed random seed.
- Fit one-hot encoding on training data only, then apply the fitted preprocessor to validation and test data.
- Train a majority-class `DummyClassifier`, Logistic Regression, and a small Keras neural network.
- Evaluate the baseline models and save metrics and plots.
- Run a class-weighting experiment and compare the weighted models on validation data.
- Save the fitted preprocessor, models, and reports under `models/` and `reports/`.

### 3. `03_Personalized_Recommendations.ipynb`

- Load `models/logistic_regression.joblib` and `models/preprocessor.joblib` created by Notebook 02.
- Demonstrate pre-exam prediction from the five categorical features, including class probabilities and a broad practice suggestion.
- Demonstrate post-exam recommendations from actual Mathematics, Reading, and Writing scores.
- Run the notebook's checks to verify the saved artifacts and recommendation functions.

**Notebook 03 depends on the saved files generated by Notebook 02.** Run Notebook 02 successfully before running Notebook 03.

### 4. `P38_Student_Performance.ipynb` — Unified Submission Notebook

This notebook combines the main workflow from the three modular notebooks into a single file for project review or academic submission.

It covers dataset exploration, target engineering, preprocessing, model training, evaluation, the class-weighting experiment, and personalized recommendations.

The three numbered notebooks remain the preferred workflow for learning, debugging, and modifying individual stages.

Run this notebook from top to bottom using the same Conda environment and local dataset. Note that it can regenerate and overwrite files under `models/` and `reports/`.

## Modeling approach

### Models

1. **Dummy Classifier (`most_frequent`)** — a simple reference point that predicts the most common training category.
2. **Logistic Regression** — the conventional feature-based classification baseline.
3. **Small neural network (MLP)** — a Keras model with Dense layers, ReLU activations, dropout, and a three-class softmax output. It uses Adam, sparse categorical cross-entropy, and early stopping.
4. **Class-weighted experiments** — weighted versions of Logistic Regression and the neural network to investigate performance on the less frequent categories.

The same training/validation/test split and training-only preprocessing are used for the baseline comparison. Validation results are used for model selection; the held-out test results estimate final baseline performance.

### Baseline test results

The first baseline experiment produced the following results on the held-out test set (200 students):

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Dummy Classifier | 57.5% | 0.243 |
| Logistic Regression | 62.0% | **0.390** |
| Neural Network | **62.5%** | 0.382 |

The neural network had the highest accuracy, while Logistic Regression had the highest macro-F1. The accuracy difference was small, so the neural network did not clearly outperform the simpler baseline across classes.

The original Logistic Regression achieved a validation macro-F1 of **0.439** in the model-selection comparison and is the model currently used by Notebook 03. Its baseline test report showed **0% recall for Low mastery**, so it does not reliably identify students in that category.

### Class-weighting experiment (validation results)

Class weights were calculated using training labels only. Results below are from the validation set and must not be directly compared as if they were the same evaluation split as the baseline test results above.

| Model | Validation accuracy | Validation macro-F1 | Low-class recall |
|---|---:|---:|---:|
| Logistic Regression | 58.5% | **0.439** | 0.143 |
| Neural Network (class weighted) | 40.5% | 0.389 | 0.571 |
| Logistic Regression (class weighted) | 38.0% | 0.362 | 0.524 |
| Neural Network | 56.5% | 0.331 | 0.000 |
| Dummy Classifier | 57.0% | 0.242 | 0.000 |

Class weighting improved Low-class recall in the validation split but reduced overall macro-F1 for both weighted models. For the current recommendation demonstration, the original Logistic Regression was retained because it achieved the best validation macro-F1. The class-weighted models and validation results are preserved as experiments rather than replacing the selected model.

These measurements are from one seeded split of a small dataset; they should not be interpreted as a guarantee of performance on other student populations.

## Recommendation logic

### Pre-exam mode

The saved Logistic Regression model accepts the five categorical features, predicts a Low/Medium/High category, returns class probabilities, and maps the predicted category to a broad practice suggestion. The probabilities are model estimates, not guarantees about an individual student's result.

### Post-exam mode

A separate rule-based function accepts the actual three subject scores, identifies the lowest-scoring subject (handling ties), and recommends reviewing its foundational concepts and practising related questions. This mode is not a separately trained ML model and does not diagnose specific topics, because the dataset has no topic-level scores.

## Limitations and responsible use

- The target categories are engineered from score thresholds; they are not labels supplied by the dataset.
- The input features provide limited information about an individual student's actual knowledge or learning history.
- Low-class recall was 0% for the selected Logistic Regression model on the original test set. The system must not be presented as a reliable tool for identifying every student who needs support.
- Class-weighting improved Low-class recall in validation but reduced overall validation macro-F1; more work and independent evaluation would be needed before deciding whether it is preferable.
- Correlations and model coefficients describe associations, not causation.
- Demographic features can introduce fairness and privacy concerns. This prototype is for educational demonstration only, not high-stakes decisions about students.
- The recommendations are broad practice suggestions, not question-level difficulty predictions or topic-level diagnosis.

## Potential future improvements

- Investigate additional features that directly measure learning progress, topic mastery, or responses to questions, with appropriate consent and data governance.
- Evaluate on a separate dataset or with repeated cross-validation to examine how stable the results are.
- Consider a simple local Streamlit interface for interactive demonstration, if added as a separate, optional step.

## Current Project Status

The core notebook-based machine learning workflow is complete. The project includes three modular development notebooks, one unified submission notebook, baseline model comparisons, a class-weighting experiment, saved model artifacts, evaluation reports, and both pre-exam and post-exam recommendation modes.

An interactive Streamlit interface remains an optional future enhancement.

## Author's note

This project is designed to demonstrate a complete, understandable ML workflow: data validation and exploration, target engineering, leakage-aware preprocessing, baseline comparison, a small deep-learning model, evaluation, saved artifacts, and a simple recommendation layer. The results and limitations are reported transparently rather than optimised for a headline accuracy score.
