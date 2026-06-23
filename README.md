# 📊 Customer Intelligence Platform

An end-to-end machine learning project that transforms raw telecom customer data into actionable business intelligence — covering customer segmentation, churn prediction, and revenue forecasting.

Built as a comprehensive Data Science portfolio project demonstrating the full ML lifecycle: from data cleaning and exploratory analysis through clustering, classification, regression, and hyperparameter tuning.

---

## 🧩 Business Problem

Customer churn is one of the most critical challenges in the telecommunications industry. Acquiring a new customer costs **5–7x more** than retaining an existing one, making churn prevention a top business priority.

This project addresses three core business questions:

1. **Who are our customers?** — Segment the customer base into distinct groups to enable targeted strategies.
2. **Who is likely to leave?** — Predict which customers are at risk of churning so the business can intervene proactively.
3. **How much revenue will each customer generate?** — Forecast customer lifetime revenue to prioritize high-value retention efforts.

---

## 🔄 Project Workflow

```
Raw Data → Data Cleaning → EDA → Feature Engineering → Modeling → Evaluation → Tuning
```

| Phase | Description |
|---|---|
| **Data Cleaning** | Handle data types, encode categorical variables, drop irrelevant columns, resolve missing values |
| **Exploratory Data Analysis** | Analyze churn distribution, feature correlations, and customer behavior patterns |
| **Feature Engineering** | Binary encoding, ordinal encoding, one-hot encoding, standard scaling, SMOTE oversampling |
| **Customer Segmentation** | KMeans clustering with Silhouette Score evaluation to identify 4 distinct customer segments |
| **Churn Prediction** | Train and compare 5 classification models to predict customer churn |
| **Revenue Prediction** | Train and compare 5 regression models to forecast customer total charges |
| **Hyperparameter Tuning** | RandomizedSearchCV to optimize the best classification and regression models |

---

## 📁 Dataset

**Source:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM Sample Dataset)

| Property | Value |
|---|---|
| Total Records | 7,032 (7,032 after cleaning) |
| Features | 21 columns (including customerID and target) |
| Target (Classification) | `Churn` — Whether the customer left (Yes/No) |
| Target (Regression) | `TotalCharges` — Cumulative revenue from the customer |
| Churn Rate | **26.6%** churned vs **73.4%** retained |
| Missing Values | 0 (original), 11 rows dropped after TotalCharges type conversion |
| Duplicates | 0 |
| Outliers | None detected |

**Key Features:**
- **Demographics:** Gender, SeniorCitizen, Partner, Dependents
- **Services:** PhoneService, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Account:** Contract type, PaymentMethod, PaperlessBilling, MonthlyCharges, TotalCharges, Tenure

---

## 📂 Project Structure

```
customer-intelligence-platform/
│
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv          # Original dataset
│   └── processed/
│       └── telco_clean.csv                    # Cleaned & encoded dataset
│
├── models/
│   ├── random_forest_churn.pkl                # Baseline Random Forest (Churn)
│   ├── best_rf_churn.pkl                      # Tuned Random Forest (Churn)
│   ├── scaler_churn.pkl                       # StandardScaler for churn features
│   ├── revenue_model.pkl                      # Polynomial Regression (Revenue)
│   ├── polynomial_transformer.pkl             # PolynomialFeatures transformer
│   ├── best_dtr_revenue.pkl                   # Tuned Decision Tree (Revenue)
│   └── scaler_revenue.pkl                     # StandardScaler for revenue features
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb                 # Data preprocessing pipeline
│   ├── 02_eda.ipynb                           # Exploratory Data Analysis
│   ├── 03_customer_segmentation.ipynb         # KMeans clustering
│   ├── 04_churn_prediction.ipynb              # Classification models
│   ├── 05_revenue_prediction.ipynb            # Regression models
│   └── 06_hyperparaneter_tuning.ipynb         # Model optimization
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 👥 Customer Segmentation

### Methodology

| Component | Detail |
|---|---|
| Algorithm | **KMeans Clustering** |
| Clusters Selected | **K = 4** |
| Evaluation Method | Elbow Method (WCSS) + Silhouette Score |
| Scaling | StandardScaler on all features |
| Key Features Analyzed | Tenure, MonthlyCharges, TotalCharges |

### Cluster Profiles

| Cluster | Name | Customers | Avg Tenure | Avg Monthly Charges | Avg Total Charges | Churn Rate |
|---|---|---|---|---|---|---|
| 0 | **Loyal High-Spenders** | 1,984 | 56.9 months | $91.86 | $5,255.30 | 13.7% |
| 1 | **Mid-Tier Core Users** | 680 | 31.8 months | $41.99 | $1,500.43 | 25.0% |
| 2 | **Low Revenue Customers** | 1,747 | 29.0 months | $24.88 | $697.56 | 9.4% |
| 3 | **New High-Spenders** ⚠️ | 2,621 | 16.4 months | $76.84 | $1,293.68 | **48.2%** |

### Cluster Insights & Recommendations

**🟢 Cluster 0 — Loyal High-Spenders (VIP Customers)**
- The most valuable segment: long-tenured, premium-plan subscribers generating the highest lifetime revenue.
- **Churn Rate: 13.7%** — Low risk, but still needs protection.
- **Action:** Implement loyalty programs, exclusive perks, and premium customer service to prevent competitor poaching.

**🟡 Cluster 1 — Mid-Tier Core Users**
- Moderate tenure and spending. Consistent users on standard/essential plans.
- **Churn Rate: 25.0%** — Moderate risk.
- **Action:** Target for upselling campaigns — offer discounted upgrades to premium packages to increase their lifetime value.

**🟢 Cluster 2 — Low Revenue Customers (Price-Sensitive Loyalists)**
- Long tenure but lowest monthly charges. These customers use basic plans and are highly price-sensitive.
- **Churn Rate: 9.4%** — Lowest risk across all segments.
- **Action:** Avoid aggressive upselling. Instead, offer affordable add-ons and budget-friendly bundles to incrementally grow revenue without triggering churn.

**🔴 Cluster 3 — New High-Spenders (Highest Churn Risk)**
- The largest segment (2,621 customers) with the shortest tenure but high monthly charges. These are new customers who subscribed to premium services immediately.
- **Churn Rate: 48.2%** — Nearly half of these customers churn.
- **Action:** Deploy targeted retention strategies — active engagement during onboarding, usage optimization guidance, and long-term contract incentives (1–2 year deals with discounts).

---

## 🔮 Churn Prediction

### Problem Setup

| Component | Detail |
|---|---|
| Target Variable | `Churn` (Binary: 0 = Retained, 1 = Churned) |
| Class Imbalance Handling | **SMOTE** (Synthetic Minority Oversampling) |
| Train/Test Split | 80/20 with stratification |
| Feature Scaling | StandardScaler on numerical columns |
| Evaluation Focus | **F1 Score** (balances precision and recall for imbalanced data) |

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Random Forest** 🏆 | **0.7775** | **0.5698** | **0.6658** | **0.6141** |
| SVM | 0.7491 | 0.5202 | 0.7246 | 0.6056 |
| Naive Bayes | 0.7363 | 0.5027 | 0.7460 | 0.6006 |
| KNN | 0.7171 | 0.4792 | 0.7406 | 0.5819 |
| Decision Tree | 0.7122 | 0.4678 | 0.6016 | 0.5263 |

### Best Model: Random Forest

**Why Random Forest was selected:**
- **Highest Accuracy** (77.8%) — correctly classified 1,104 out of 1,407 test samples.
- **Highest Precision** (57.0%) — best at minimizing false churn alerts, reducing unnecessary retention spend.
- **Highest F1 Score** (0.614) — most balanced trade-off between catching churners and avoiding false positives.

**Detailed Performance (Classification Report):**
- **Class 0 (Retained):** Precision 87%, Recall 82% — strong at identifying loyal customers.
- **Class 1 (Churned):** Precision 57%, Recall 67% — successfully identifies 2 out of 3 at-risk customers.

> **Business Impact:** The model can flag approximately 67% of customers who would churn, enabling the retention team to intervene proactively. While 43% of flagged customers may be false positives, the cost of a retention offer is typically far lower than the cost of losing a customer.

---

## 💰 Revenue Prediction

### Problem Setup

| Component | Detail |
|---|---|
| Target Variable | `TotalCharges` (Continuous — cumulative customer revenue) |
| Train/Test Split | 80/20 |
| Feature Scaling | StandardScaler on tenure and MonthlyCharges |
| Note | Churn was kept as a feature since it provides useful predictive signal |

### Model Comparison

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| **Polynomial Regression** 🏆 | **49.15** | **71.23** | **0.9990** |
| Neural Network (MLP) | 53.25 | 76.84 | 0.9989 |
| Decision Tree | 82.60 | 122.96 | 0.9971 |
| Linear Regression | 539.71 | 673.51 | 0.9122 |
| SVR | 1,453.59 | 2,091.56 | 0.1534 |

### Best Model: Polynomial Regression (Degree 2)

**Why Polynomial Regression dominated:**
- **R² = 0.9990** — The model explains 99.9% of the variance in customer revenue, indicating near-perfect predictive capability.
- **MAE = $49.15** — On average, predictions deviate by only ~$49 from the actual total charges.
- The strong performance of Polynomial Regression over Linear Regression confirms that the relationship between customer features and revenue is **non-linear** — interaction effects between features (like tenure × monthly charges) are significant drivers.

> **Key Insight:** Customer revenue is highly predictable using existing customer attributes. The available features (tenure, monthly charges, contract type, services subscribed) capture almost all the variation in total revenue, making this model reliable for revenue forecasting and customer lifetime value estimation.

**Why SVR performed poorly:**
- SVR achieved an R² of only 0.15, indicating it failed to learn the underlying data patterns. This is likely due to the default hyperparameters being unsuitable for the data scale and feature space without extensive tuning.

---

## ⚙️ Hyperparameter Tuning

### Strategy

| Component | Detail |
|---|---|
| Search Method | **RandomizedSearchCV** |
| Cross-Validation | 3-fold CV |
| Iterations | 20 random combinations per model |
| Parallel Processing | `n_jobs=-1` (all CPU cores) |

### Classification Tuning — Random Forest

**Parameter Search Space:**
```python
{
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]
}
```

**Best Parameters Found:**
| Parameter | Value |
|---|---|
| n_estimators | 100 |
| max_depth | 10 |
| min_samples_split | 5 |
| min_samples_leaf | 1 |
| criterion | entropy |

**Performance Comparison:**

| Metric | Before Tuning | After Tuning |
|---|---|---|
| Accuracy | 0.7775 | 0.7632 |
| Precision (Class 1) | 0.5698 | 0.5400 |
| Recall (Class 1) | 0.6658 | 0.6900 |
| F1 Score (Class 1) | 0.6141 | 0.6000 |

> **Interpretation:** The tuned model trades a small amount of precision for improved recall (69% vs 66.6%), meaning it catches more actual churners at the cost of slightly more false alarms. Depending on the business context — where missing a churner is more costly than a false retention offer — this trade-off may be favorable.

---

### Regression Tuning — Decision Tree Regressor

**Parameter Search Space:**
```python
{
    "max_depth": [3, 5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["squared_error", "friedman_mse"]
}
```

**Best Parameters Found:**
| Parameter | Value |
|---|---|
| max_depth | None |
| min_samples_split | 10 |
| min_samples_leaf | 2 |
| criterion | friedman_mse |

**Performance:**

| Metric | Before Tuning | After Tuning |
|---|---|---|
| R² Score | 0.9971 | 0.9976 |

> **Interpretation:** The tuned Decision Tree achieved a marginal improvement in R² (0.9976), confirming that the default Decision Tree was already well-suited for this dataset. The regularization parameters (`min_samples_split=10`, `min_samples_leaf=2`) help reduce overfitting while maintaining near-perfect accuracy.

---

## 💡 Key Business Insights

### Customer Segments
- The customer base splits into **4 distinct segments**, ranging from price-sensitive loyalists to high-spending newcomers.
- **Cluster 3 (New High-Spenders)** is the largest segment (2,621 customers) and also the most at-risk, with a **48.2% churn rate** — nearly half of these customers leave.
- **Cluster 2 (Low Revenue Customers)** has the lowest churn rate at **9.4%**, demonstrating that affordable plans drive long-term retention.

### Churn Behavior
- **26.6% of all customers churned**, representing a significant revenue loss opportunity.
- **Tenure is the strongest retention factor** (correlation: -0.35 with churn) — the longer a customer stays, the less likely they are to leave.
- **Monthly charges are the primary churn trigger** (correlation: +0.19) — customers with higher bills are more likely to leave.
- The critical churn window is the **first 12 months** — churn risk drops dramatically after the first year.
- **Month-to-month contracts** have the highest churn concentration; long-term contracts significantly improve retention.

### Revenue Patterns
- Customer revenue is **99.9% predictable** using existing attributes, driven by non-linear interactions between tenure, monthly charges, and service subscriptions.
- **High monthly charges + low tenure = high churn risk** — new premium subscribers leave before generating significant lifetime value.
- Tenure and TotalCharges have a **strong multicollinearity** (r = 0.83), confirming that customer longevity is the primary driver of cumulative revenue.

### Actionable Recommendations

| Priority | Action | Target Segment | Expected Impact |
|---|---|---|---|
| 🔴 **Critical** | Deploy onboarding retention programs in the first 12 months | New High-Spenders (Cluster 3) | Reduce 48.2% churn rate |
| 🔴 **Critical** | Offer long-term contract incentives with discounts | Month-to-month subscribers | Lock in at-risk customers |
| 🟡 **High** | Launch upselling campaigns for mid-tier users | Mid-Tier Core Users (Cluster 1) | Increase ARPU |
| 🟡 **High** | Implement VIP loyalty programs | Loyal High-Spenders (Cluster 0) | Protect top revenue generators |
| 🟢 **Medium** | Introduce affordable add-on bundles | Low Revenue Customers (Cluster 2) | Grow revenue without churn risk |

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3 |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Imbalanced Data** | imbalanced-learn (SMOTE) |
| **Model Persistence** | Joblib |
| **Environment** | Jupyter Notebook |

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/customer-intelligence-platform.git
cd customer-intelligence-platform
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install scikit-learn seaborn matplotlib jupyter
```

### 4. Run the Notebooks
```bash
jupyter notebook
```

Navigate to the `notebooks/` folder and run the notebooks in order:
1. `01_data_cleaning.ipynb`
2. `02_eda.ipynb`
3. `03_customer_segmentation.ipynb`
4. `04_churn_prediction.ipynb`
5. `05_revenue_prediction.ipynb`
6. `06_hyperparaneter_tuning.ipynb`

---

## 🔮 Future Improvements

- **Feature Engineering:** Create derived features such as `AverageMonthlySpend` (TotalCharges / tenure) and interaction features to potentially improve churn prediction.
- **Advanced Models:** Experiment with XGBoost, LightGBM, or CatBoost for classification — these gradient boosting methods often outperform Random Forest on tabular data.
- **Deep Learning:** Implement neural network architectures (e.g., TabNet) for churn prediction to capture complex non-linear patterns.
- **Model Deployment:** Build a REST API using Flask or FastAPI to serve predictions in real-time for integration with CRM systems.
- **Dashboard:** Create an interactive Streamlit or Plotly Dash dashboard for business stakeholders to explore customer segments and churn risk scores.
- **Time-Series Analysis:** Incorporate temporal patterns to predict *when* a customer is likely to churn, not just *if*.
- **A/B Testing Framework:** Design an experimentation pipeline to measure the actual impact of retention interventions on each customer segment.
- **Pipeline Automation:** Refactor the notebook workflow into modular Python scripts with an orchestrated ML pipeline (e.g., using MLflow or DVC).

---

<p align="center">
  <i>Built with 💡 by a Data Science enthusiast passionate about turning data into business value.</i>
</p>
