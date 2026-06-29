# MistakeCoach: Eedi Public Test Correctness Prediction

This repository contains a course project for analyzing Eedi math diagnostic question data and predicting whether a student will answer a question correctly.

The final modeling setup uses the dataset-provided train/test split:

- Train: `train_data/train_task_1_2.csv`
- Test: `test_data/test_public_answers_task_1.csv`

## Project Structure

```text
app/                         Streamlit prototype
src/                         Tutor, data, analytics, and modeling modules
scripts/                     EDA and model training scripts
reports/                     Final report and supporting analysis notes
data/raw/                    Small demo data for the Streamlit app
data/processed/              Lightweight public-test result summaries
models/                      Best public-test model artifact
tests/                       Unit tests
```

## Main Reports

- `reports/final_project_report_en.md`
- `reports/eedi_official_split_model_report_en.md`
- `reports/feature_engineering_rationale_en.md`
- `reports/eedi_eda_report_en.md`

## Final Public Test Result

Best model: `neural_network_mlp`

| Model | Accuracy | ROC-AUC | F1 |
|---|---:|---:|---:|
| Neural Network MLP | 73.0% | 78.5% | 79.6% |
| Logistic Regression | 72.9% | 78.4% | 80.2% |
| Linear SVM | 72.6% | 78.0% | 79.8% |
| Gradient Boosting | 72.6% | 77.8% | 79.8% |
| Random Forest | 72.1% | 77.6% | 80.0% |
| Decision Tree | 72.4% | 77.6% | 79.4% |
| Gaussian Naive Bayes | 63.3% | 53.9% | 77.1% |
| Baseline Most Frequent | 64.3% | 50.0% | 78.3% |

Full public-test metrics are in:

`data/processed/eedi_official_split_models/public_test_model_metrics.csv`

## How to Run

Start from the repository root:

```bash
cd MistakeCoach_GitHub_Submission_Public
```

Create and activate a Python virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are using Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### 1. Run the Streamlit Demo

The Streamlit app uses the small demo files already included in `data/raw/`, so it can run without the full Eedi dataset.

```bash
streamlit run app/streamlit_app.py
```

After the command starts, open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

### 2. Run the Public-Test Model Training

The large Eedi raw data is not included in this GitHub package. Place the Eedi data beside this project using this structure:

```text
../data/
  train_data/train_task_1_2.csv
  test_data/test_public_answers_task_1.csv
  metadata/question_metadata_task_1_2.csv
  metadata/student_metadata_task_1_2.csv
  metadata/subject_metadata.csv
```

Then run:

```bash
PYTHONPATH=. python scripts/train_eedi_official_split_models.py
```

For a quick smoke test:

```bash
PYTHONPATH=. python scripts/train_eedi_official_split_models.py --train-sample-size 5000 --test-limit 5000
```

The training script writes public-test metrics, coverage summaries, feature importance, and the best model artifact into `data/processed/eedi_official_split_models/` and `models/`.

### 3. Run EDA

```bash
PYTHONPATH=. python scripts/analyze_eedi_data.py --task 1_2
```

The EDA script writes summary tables and an HTML dashboard into `data/processed/eedi_eda/`.

### 4. Run Tests

```bash
PYTHONPATH=. pytest
```

## Notes

- Raw Eedi data is excluded because it is large.
- Local dependency folder `.deps/`, cache files, and large prediction CSVs are excluded.
- The final report uses public test only.
