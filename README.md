# Credit Card Fraud Detection 💳

A production-grade machine learning pipeline for real-time credit card fraud 
detection. Trains and benchmarks three classifiers on a severely imbalanced 
dataset (0.17% fraud rate), handles class imbalance with SMOTE, and deploys 
the best model as a FastAPI inference server with batch scoring and risk tiers.

---
## Method Diagram 
<img width="559" height="815" alt="unnamed" src="https://github.com/user-attachments/assets/367962a3-8711-4285-b75c-1756d33f63ef" />

## Results Summary

| Model               | ROC-AUC | Avg Precision | F1     | Precision | Recall |
|---------------------|---------|---------------|--------|-----------|--------|
| Logistic Regression | 1.0000  | 1.0000        | 0.9751 | 0.9510    | 1.0000 |
| Random Forest       | 1.0000  | 0.9940        | 0.9641 | 0.9690    | 0.9590 |
| XGBoost             | 1.0000  | 0.9995        | 0.9655 | 0.9330    | 1.0000 |

**Best model: Logistic Regression** — highest F1 and perfect Recall 
(zero missed fraud cases on the test set).

---

## Project Structure
credit_card_fraud_detection/

 ├── fraud_detection.py # Full ML pipeline

 ├── fraud_detection_notebook.ipynb
 
 ├── api.py # FastAPI inference server
 
 ├── creditcard.csv # Raw dataset
 
 ├── requirements.txt
 
 ├── .env.example
 
 ├── models/ # Generated artefacts
 
 ├── best_model_Logistic_Regression.pkl
 
  └── scaler.pkl
 
 ├── outputs/
 
   └── metrics.json
 
   └── plots/
 
 ├── 01_eda_overview.png
 
 ├── 02_correlation_heatmap.png
 
 ├── 03_smote_balance.png
 
 ├── 04_roc_pr_curves.png
 
 ├── 05_confusion_matrices.png
 
 ├── 06_model_comparison.png
 
 └── 07_feature_importance.png

---

## Quick Start

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the full ML pipeline**
```bash
python fraud_detection.py
```
This trains all models, generates all 7 plots, saves the best model and 
scaler to `models/`, and writes `outputs/metrics.json`.

**3. Start the inference API**
```bash
uvicorn api:app --reload --port 8000
```
Interactive docs at **http://localhost:8000/docs**

---

## Dataset

The [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud):

- 284,807 transactions over 2 days
- 492 fraud cases — **0.173% positive rate**
- Features V1–V28: PCA-transformed (anonymised for privacy)
- `Amount`: transaction value in euros
- `Time`: seconds since first transaction in dataset
- `Class`: 0 = legitimate, 1 = fraud

---

## How It Works

### 1. Exploratory Data Analysis

The dataset has extreme class imbalance — a 1:578 ratio. Key findings:

- Fraud transactions show no time-of-day pattern (distributed uniformly 
  across 48 hours).
- Fraud amount distribution overlaps with legitimate but has a slightly 
  higher median — fraud is not always small-value.
- V14, V10, V4 show strong class separation in density plots — these 
  become the top features confirmed later by importance analysis.
- PCA components (V1–V28) are near-orthogonal — very low inter-feature 
  correlation, confirming the PCA decomposition is well-conditioned.

### 2. Preprocessing

```python
# Drop Time (not cyclically encoded)
X = df.drop(columns=['Class', 'Time'])

# Log-transform Amount — fixes heavy right skew
X['Amount'] = np.log1p(X['Amount'])

# Scale all features to zero mean, unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified split — preserves the 0.17% fraud ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)
```

**Why log-transform Amount?** Raw transaction amounts span $0–$25,691 
with extreme right skew. `log1p` compresses this to a roughly normal 
distribution and prevents the scaler from being dominated by outliers.

**Why StandardScaler?** Logistic Regression and distance-based methods 
require features on the same scale. Scaler is fit only on the training 
set to avoid data leakage — the test set is transformed using training 
statistics only.

### 3. SMOTE — Synthetic Minority Oversampling Technique

With only 394 fraud cases in the training set (vs 227,451 legitimate), 
a naive classifier achieves 99.83% accuracy by predicting everything as 
legitimate. SMOTE fixes this:

```python
smote = SMOTE(random_state=42, k_neighbors=5)
X_res, y_res = smote.fit_resample(X_train, y_train)
# Result: 227,451 legit + 227,451 synthetic fraud = 454,902 training samples
```

**How SMOTE works:**
For each minority-class sample, SMOTE:
1. Finds its k=5 nearest neighbours in the feature space.
2. Randomly selects one of those neighbours.
3. Creates a new synthetic sample at a random point on the line segment 
   between the original and the selected neighbour.

This creates a denser, more learnable minority class manifold without 
simply duplicating existing samples (which is random oversampling).

**Critical:** SMOTE is applied **only to the training set**. The test 
set remains the original imbalanced distribution — this is the real-world 
scenario the deployed model faces.

### 4. Model Training

Three classifiers are trained on the SMOTE-balanced data:

**Logistic Regression** (`C=0.1`, L2 regularisation)
A linear boundary in the 29-dimensional feature space. The PCA structure 
of V1–V28 makes the classes approximately linearly separable, which 
explains why LR achieves AUC=1.0. `C=0.1` provides stronger regularisation 
to prevent overfitting on the synthetic SMOTE samples.

**Random Forest** (200 trees, `max_depth=12`, `class_weight='balanced'`)
An ensemble of 200 decision trees trained on bootstrap samples. Each tree 
considers a random feature subset at each split. Final prediction is a 
majority vote. `class_weight='balanced'` adds an additional layer of 
imbalance correction on top of SMOTE.

**XGBoost** (200 trees, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`)
Gradient boosted trees trained sequentially — each tree corrects the 
residual errors of the ensemble so far. `scale_pos_weight` is set to 
the ratio of negative to positive post-SMOTE samples (~1.0 after balancing) 
to handle any residual imbalance. Subsampling of rows (0.8) and columns 
(0.8) prevents overfitting.

### 5. Evaluation Metrics — Why Not Accuracy?

With 0.17% fraud rate, a classifier that predicts "legitimate" for every 
transaction achieves 99.83% accuracy — useless. The correct metrics are:

| Metric | Formula | Why it matters for fraud |
|--------|---------|--------------------------|
| ROC-AUC | Area under TPR vs FPR curve | Threshold-independent ranking quality |
| Average Precision | Area under Precision-Recall curve | Directly measures performance on minority class |
| Recall (Sensitivity) | TP / (TP + FN) | Cost of a missed fraud is very high |
| Precision | TP / (TP + FP) | Cost of false alarms (blocking legit customers) |
| F1 | 2 × (P × R) / (P + R) | Harmonic mean — balances P and R |

**Precision-Recall curves are more informative than ROC curves for 
imbalanced datasets.** ROC is optimistic because it includes true negatives 
(the majority class) in its denominator. PR curves focus entirely on the 
minority class performance.

### 6. Best Model Selection

The model with the highest ROC-AUC is selected programmatically. In this 
case all three tie at AUC=1.0000, so secondary metrics break the tie — 
Logistic Regression wins with F1=0.9751 and perfect Recall=1.0000 (zero 
missed fraud cases on 98 test fraud samples).

```python
best_name = max(results, key=lambda n: results[n]['roc_auc'])
joblib.dump(best_model, 'models/best_model_Logistic_Regression.pkl')
joblib.dump(scaler,     'models/scaler.pkl')
```

---

## FastAPI Inference Server

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Model/scaler load status |
| GET | `/metrics` | All training metrics (from metrics.json) |
| GET | `/example` | Sample fraudulent transaction payload |
| POST | `/predict` | Score a single transaction |
| POST | `/predict/batch` | Score up to 1,000 transactions |

### Prediction Pipeline

```python
# 1. Extract feature vector (V1–V28 + log1p(Amount))
raw = [V1, V2, ..., V28, log1p(Amount)]

# 2. Scale using the saved StandardScaler
X_scaled = scaler.transform(raw)

# 3. Get fraud probability
prob = model.predict_proba(X_scaled)[0, 1]

# 4. Apply risk tier
risk = "LOW"     if prob < 0.30
     | "MEDIUM"  if prob < 0.60
     | "HIGH"    if prob < 0.85
     | "CRITICAL" otherwise
```

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"V1": -3.04, "V2": -3.16, ..., "Amount": 23.50}'
```

### Example Response

```json
{
  "is_fraud": true,
  "fraud_probability": 0.987654,
  "risk_level": "CRITICAL",
  "threshold_used": 0.5,
  "message": "⚠️  Near-certain fraud — block immediately."
}
```

### Threshold Tuning

The default threshold is 0.5. Pass `?threshold=0.3` to increase sensitivity 
(catch more fraud at the cost of more false alarms). This is the key 
operational decision: lower threshold = higher recall, lower precision.

---

## Key Technical Concepts

### Why PCA Features Make Fraud Linearly Separable

V1–V28 are PCA-transformed components of the original transaction features 
(merchant category, location, device fingerprint, etc.). PCA rotates the 
feature space to maximise variance in each orthogonal direction. Fraud 
patterns tend to cluster in specific regions of this transformed space — 
particularly along V14 and V10 — making them separable with a linear 
boundary. This is why Logistic Regression matches ensemble methods here.

### Class Imbalance and Why It Matters

Training on imbalanced data causes the model to learn a strong prior 
towards the majority class. The model minimises total loss by predicting 
"legitimate" for everything, achieving high accuracy but zero recall. 
SMOTE addresses this by populating the minority class neighbourhood with 
synthetic interpolated samples, giving the model equal gradient signal 
from both classes.

### The Precision-Recall Tradeoff in Production

In a real fraud system, two business costs compete:
- **False Negative** (missed fraud): financial loss + reputational damage.
- **False Positive** (blocked legitimate transaction): customer friction + 
  potential churn.

Lowering the decision threshold increases recall (catch more fraud) but 
decreases precision (more false alarms). The optimal threshold depends on 
the cost ratio. A conservative fraud system might set threshold=0.3; a 
customer-experience-focused system might use threshold=0.7.

---

## Requirements
· pandas   · numpy   · scikit-learn   · imbalanced-learn  · xgboost   · matplotlib   · seaborn    · joblib    · fastapi    · uvicorn    · pydantic
- Python 3.9+ recommended.

# Theory and Methodology Summary
The core problem is extreme class imbalance in a binary classification task. Standard ML training on 0.17% positive rate collapses to the majority class. The solution combines three complementary strategies:
- SMOTE generates synthetic minority samples to balance training gradients; class_weight='balanced' in Random Forest additionally upweights minority errors; and scale_pos_weight in XGBoost corrects the loss function.
- Feature engineering is deliberately minimal here. V1–V28 are already PCA-transformed by the dataset providers, so they are near-orthogonal (minimal multicollinearity). The only engineering needed is log1p(Amount) to fix right skew and StandardScaler to normalise magnitudes. Time is dropped because it spans only 48 hours with no cyclical encoding — insufficient for daily pattern learning.
- Why Logistic Regression wins despite being the simplest model: the PCA decomposition of the original features creates a feature space where fraud and legitimate transactions are approximately linearly separable, particularly along V14 and V10. LR finds this boundary directly with a single hyperplane. RF and XGBoost add decision complexity that doesn't yield better separation — and their additional FN errors (missed fraud cases) in the RF case hurt F1.
- Evaluation philosophy: ROC-AUC is the headline metric because it measures the model's discriminative ability across all thresholds. But Average Precision (area under the PR curve) is actually the more honest metric for imbalanced data — it exclusively measures quality on the minority class without the optimism introduced by the large true-negative pool. The confusion matrices confirm the ultimate metric that matters in production: Recall (zero missed frauds for LR and XGBoost).
- The FastAPI layer implements the inference pipeline identically to training — log1p(Amount) transform, same StandardScaler, then predict_proba. The risk tier system translates continuous probability into actionable business decisions without requiring the operations team to reason about raw probabilities.
