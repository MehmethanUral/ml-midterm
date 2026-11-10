# Machine Learning Midterm Project

Classification (Decision Tree, KNN, Gaussian Naive Bayes) and K-Means clustering on two scikit-learn datasets: **Breast Cancer Wisconsin** and **Wine Classification**.

- **Repository:** https://github.com/mehmethanural/ml-midterm
- **Author:** Mehmethan Ural

## Project structure

```
ml-midterm/
├── breast_cancer.ipynb        # Sections 1–10 (+12 inventory) for Breast Cancer Wisconsin
├── wine.ipynb                 # Sections 1–10 (+12 inventory) for Wine Classification
├── report.docx                # Written report (Sections 1–12, figures, summaries, conclusions)
├── classifier_runs.csv        # 132 rows – one row per classifier fit
├── classifier_summary.csv     # 44 rows – 22 configurations per dataset, mean and std (ddof=1)
├── classifier_test.csv        # 2 rows – selected seed-42 classifier per dataset, test metrics
├── kmeans_runs.csv            # 12 rows – one row per K-Means fit
├── kmeans_summary.csv         # 4 rows – 2 representations per dataset, across-seed summary
├── kmeans_test.csv            # 4 rows – seed-42 K-Means test evaluation
├── figures/
│   ├── breast_cancer/         # 13 figures (EDA, parameter comparison, confusion matrix, ROC, K-Means)
│   └── wine/                  # 11 figures
├── result/                    # Per-dataset intermediate CSV files and split row indices
├── requirements.txt
├── AI_DISCLOSURE.md
└── README.md
```

## Environment

| Package | Version |
|---|---|
| Python | 3.11.16 |
| scikit-learn | 1.9.1 |
| pandas | 3.0.6 |
| numpy | 2.4.6 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |

## How to install and run

```bash
# 1. Create and activate a virtual environment (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install the dependencies
pip install -r requirements.txt

# 3. Start Jupyter and run both notebooks from top to bottom
jupyter lab
```

Open `breast_cancer.ipynb` and choose **Kernel → Restart Kernel and Run All Cells**, then do the same for `wine.ipynb`. Each notebook runs on its own. The six combined CSV files in the project root are written by the last cell of whichever notebook runs second (both per-dataset result files must exist in `result/`). Figures are saved to `figures/<dataset>/`.

## Dataset sources

Both datasets are loaded directly from scikit-learn; no files need to be downloaded.

- Breast Cancer Wisconsin (Diagnostic): `sklearn.datasets.load_breast_cancer` – 569 samples, 30 features, classes 0 = malignant, 1 = benign. Original source: UCI Machine Learning Repository.
- Wine Classification: `sklearn.datasets.load_wine` – 178 samples, 13 features, classes 0, 1, 2. Original source: UCI Machine Learning Repository.

## Experimental design (summary)

- Fixed 20% stratified test set (`random_state=42`), then three stratified 70/10 training/validation splits with seeds 42, 52 and 62.
- StandardScaler fitted on each seed's training data only; unscaled and scaled representations.
- 66 classifier fits per dataset (132 total) and 6 K-Means fits per dataset (12 total): 144 model fits.
- Model selection by highest mean validation macro-F1 (ties: mean validation accuracy, lower std, alphabetical ID). Only the seed-42 fitted instance of the selected configuration is evaluated on the test set.
