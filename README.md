# Customer-Lifetime-Value-Churn-Intelligence
Customer Churn Prediction, Lifetime Value Estimation and Retention Intelligence using Machine Learning
# Customer Lifetime Value and Churn Intelligence System

### Predicting Customer Churn, Understanding Customer Value and Supporting Data-Driven Retention Decisions

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-LightGBM-green)
![Explainable AI](https://img.shields.io/badge/Explainability-SHAP-orange)
![Database](https://img.shields.io/badge/Database-PostgreSQL-blue)
![Deployment](https://img.shields.io/badge/Deployment-Streamlit-red)
![Status](https://img.shields.io/badge/Project-Deployed-success)

---

## 1. Project Statement

Customer retention is a significant challenge for businesses operating in competitive markets. Acquiring new customers often requires substantial investment, whereas retaining existing customers can contribute to sustainable revenue and long-term business growth.

However, organisations frequently struggle to identify customers who are likely to discontinue their relationship, understand the factors influencing their decisions, estimate their potential long-term value and determine which retention interventions are economically worthwhile.

The **Customer Lifetime Value and Churn Intelligence System** addresses these challenges through an integrated Data Science and Machine Learning framework.

The system analyses customer characteristics, predicts churn probability, explains model decisions, identifies customer segments and demonstrates experimental uplift modelling. It also provides an interactive Return on Investment (ROI) simulator to assess the potential financial implications of retention campaigns.

The ultimate objective is to transform historical customer information into actionable intelligence that supports informed customer retention and business decision-making.

### Core Business Question

**Which customers are likely to leave, what factors influence their behaviour, which customers represent greater potential value, and where should a business allocate its retention budget?**

---

## 2. Project Objectives

The primary objectives of this project are:

1. Analyse customer demographics, account characteristics and behavioural patterns to identify factors associated with customer churn.
2. Develop a machine learning classification model using LightGBM to estimate customer churn probability.
3. Apply SHAP to interpret model predictions and identify influential customer attributes.
4. Investigate survival analysis techniques to estimate customer retention duration where valid time-to-event information is available.
5. Demonstrate uplift modelling using CausalML to distinguish customers who may respond incrementally to a marketing intervention.
6. Develop customer value scoring and clustering techniques to identify meaningful customer segments.
7. Integrate customer records, predictions and analytical outputs into PostgreSQL.
8. Develop an interactive Streamlit application to communicate analytical findings.
9. Implement an ROI simulator to evaluate hypothetical retention campaign costs and benefits.
10. Present the results through an accessible, reproducible and business-oriented analytical platform.

---

## 3. Business Problem and Proposed Solution

| Business Challenge | Proposed Solution |
|---|---|
| Customers leave without sufficient warning | LightGBM churn prediction |
| Business teams cannot easily understand churn drivers | SHAP explainability |
| Customer retention duration is difficult to estimate | Survival analysis feasibility assessment |
| Marketing campaigns target customers indiscriminately | Uplift modelling demonstration |
| Customer value is difficult to compare | Customer value scoring |
| Similar customers are not grouped effectively | K-Means clustering |
| Analytical results are distributed across files | PostgreSQL integration |
| Technical findings are difficult for business users to interpret | Streamlit dashboard |
| Retention campaign profitability is uncertain | ROI simulation |

---

## 4. Technology Stack

| Category | Technology | Purpose |
|---|---|---|
| Programming Language | Python | Core implementation |
| Data Manipulation | Pandas, NumPy | Data processing and analysis |
| Exploratory Analysis | Matplotlib, Seaborn | Statistical visualisation |
| Interactive Visualisation | Plotly | Dashboard charts |
| Machine Learning | LightGBM | Customer churn classification |
| Machine Learning Utilities | Scikit-learn | Preprocessing, evaluation and clustering |
| Explainable AI | SHAP | Model interpretation |
| Uplift Modelling | CausalML | Treatment-effect estimation |
| Survival Analysis | Lifelines | Planned time-to-event modelling extension |
| Database | PostgreSQL | Structured data storage |
| Cloud Database | Neon PostgreSQL | Managed database hosting |
| Application Framework | Streamlit | Interactive application |
| Model Persistence | Joblib | Saving and loading trained models |
| Version Control | Git and GitHub | Source code management |
| Development Environment | GitHub Codespaces | Cloud-based development |
| Deployment | Streamlit Community Cloud | Application hosting |

---

## 5. Dataset Description

This project uses two separate datasets for different analytical objectives.

### 5.1 Primary Dataset: Bank Customer Churn

The primary dataset is a bank customer churn dataset containing 10,000 records and 18 original attributes.

It includes information relating to:

- Customer demographics.
- Credit scores.
- Geographic information.
- Account tenure.
- Account balances.
- Product ownership.
- Credit card status.
- Customer activity.
- Estimated salary.
- Customer complaints.
- Satisfaction scores.
- Customer churn status.

**Target variable:** `Exited`

| Value | Interpretation |
|---|---|
| 0 | Customer retained |
| 1 | Customer churned |

#### Initial Dataset Statistics

| Metric | Result |
|---|---:|
| Total records | 10,000 |
| Total original attributes | 18 |
| Retained customers | 7,962 |
| Churned customers | 2,038 |
| Actual churn rate | 20.38% |
| Missing values | 0 |
| Duplicate records | 0 |

The initial analysis identified a significant relationship between the complaint attribute and the churn target. To avoid potential target leakage and misleading predictive performance, the complaint feature was excluded from the primary model.

### 5.2 Secondary Dataset: Hillstrom Email Marketing Experiment

A separate Hillstrom MineThatData email marketing dataset was used to demonstrate uplift modelling.

The dataset contains approximately 64,000 customer records and three experimental campaign groups:

- Mens E-Mail
- Womens E-Mail
- No E-Mail

For the uplift modelling experiment:

| Component | Selection |
|---|---|
| Treatment group | Womens E-Mail |
| Control group | No E-Mail |
| Outcome variable | Conversion |
| Modelling approach | T-Learner |
| Library | CausalML |

The secondary dataset is used exclusively for experimental marketing uplift analysis.

Its results are not treated as evidence of how the bank's customers would respond to retention campaigns.

---

## 6. End-to-End Project Workflow

The project follows a structured Data Science lifecycle, beginning with raw data acquisition and ending with an interactive cloud application.

### Stage 1: Data Acquisition and Understanding

- Loaded the bank customer churn dataset.
- Examined the dataset dimensions and attribute types.
- Identified numerical and categorical variables.
- Analysed missing values and duplicate records.
- Investigated the target variable distribution.
- Examined relationships between customer characteristics and churn.

**Outcome:** Established an understanding of the dataset and its analytical suitability.

### Stage 2: Exploratory Data Analysis

Conducted exploratory analysis to investigate customer behaviour and identify potential churn patterns.

Key analyses included:

- Churn distribution.
- Age versus churn.
- Geography versus churn.
- Tenure versus churn.
- Customer activity versus churn.
- Account balance distribution.
- Number of products versus churn.
- Feature correlation analysis.
- Complaint versus churn analysis.

**Outcome:** Identified patterns and potential predictive relationships that informed subsequent feature engineering and modelling decisions.

### Stage 3: Data Preprocessing and Feature Engineering

Implemented a structured preprocessing pipeline.

Activities included:

- Removing irrelevant identifiers and non-predictive attributes.
- Excluding the complaint feature from the primary churn model due to potential target leakage.
- Creating additional customer attributes.
- Engineering age groups and tenure groups.
- Creating a zero-balance indicator.
- Handling numerical and categorical variables.
- Applying median imputation for numerical features.
- Applying most-frequent imputation and one-hot encoding for categorical features.
- Performing stratified training and testing splits.
- Saving the fitted preprocessing pipeline.

**Dataset partitioning:**

| Dataset | Records |
|---|---:|
| Training | 8,000 |
| Testing | 2,000 |
| Total | 10,000 |

**Outcome:** Produced consistent, model-ready training and testing datasets.

### Stage 4: Customer Churn Prediction Using LightGBM

Developed a binary classification model using LightGBM.

The objective was to estimate the probability that a customer would churn based on available customer characteristics.

The modelling workflow included:

1. Establishing a baseline using DummyClassifier.
2. Training a LightGBM classification model.
3. Addressing class imbalance using class weighting.
4. Performing five-fold stratified cross-validation.
5. Evaluating performance on the held-out test dataset.
6. Generating churn probability predictions.
7. Saving the trained model for subsequent application use.

#### Evaluation Metrics

The model evaluation framework includes:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-Recall AUC
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

The actual numerical evaluation results are maintained in the project's evaluation reports.

**Outcome:** Developed a reusable churn prediction pipeline and generated predictions for 2,000 held-out customers.

### Stage 5: Explainable AI Using SHAP

Integrated SHAP to interpret the LightGBM model.

The purpose was to move beyond prediction and understand which customer characteristics influence the model's decisions.

Implemented visualisations include:

- SHAP summary plot.
- SHAP feature importance.
- Individual customer waterfall explanation.
- Age dependence analysis.
- Balance dependence analysis.
- Credit score dependence analysis.
- Tenure dependence analysis.

**Outcome:** Provided global and individual-level model explanations to improve transparency and interpretability.

SHAP explanations describe the behaviour of the trained model; they do not establish causal relationships.

### Stage 6: Survival Analysis Feasibility Assessment

The original problem statement requires survival analysis using Kaplan-Meier estimation and Cox proportional hazards modelling.

The primary bank dataset provides a customer tenure attribute and a churn indicator. However, it does not provide sufficiently documented event dates, observation dates or censoring information to establish a reliable time-to-event model.

Consequently, the current implementation assesses the feasibility of survival modelling rather than presenting an unsupported Kaplan-Meier or Cox model.

**Outcome:** Identified a data limitation and established the requirements for implementing valid survival analysis.

A suitable time-to-event dataset, such as one containing documented event durations and censoring indicators, is required to complete this component.

### Stage 7: Customer Value Scoring and Segmentation

Developed a heuristic customer value scoring framework using available customer attributes.

The scoring methodology considers:

- Account balance.
- Estimated salary.
- Number of products.
- Customer activity.

The resulting customer value score is used to group customers into four segments:

1. Low Value
2. Medium Value
3. High Value
4. Premium Value

K-Means clustering was also implemented using selected customer attributes, including:

- Age.
- Tenure.
- Balance.
- Estimated salary.
- Number of products.
- Customer activity.
- Credit score.

The clustering workflow includes feature transformation, standardisation, silhouette-based cluster selection and final model training.

**Outcome:** Generated customer value segments and cluster assignments for 10,000 customers.

**Important methodological limitation:** The calculated customer value score is a heuristic proxy, not actual financial Customer Lifetime Value. Reliable financial CLV estimation requires historical revenue, transaction frequency, contribution margin, customer acquisition cost and appropriate observation periods.

### Stage 8: Uplift Modelling Using CausalML

Implemented an experimental uplift modelling workflow using the Hillstrom email marketing dataset.

A T-Learner approach was used to estimate differences in conversion probability between the treatment and control groups.

The workflow included:

1. Selecting the Womens E-Mail treatment group.
2. Selecting the No E-Mail control group.
3. Defining conversion as the outcome.
4. Preparing pre-campaign customer attributes.
5. Applying feature preprocessing.
6. Training treatment and control models through CausalML.
7. Generating individual uplift predictions.
8. Segmenting customers according to predicted uplift.
9. Evaluating uplift ranking using Qini analysis.

**Outcome:** Generated 8,539 uplift predictions for the selected experimental population.

The uplift modelling component demonstrates how experimental campaign data can support targeted marketing decisions.

However, the results must not be directly generalised to the bank customer population without a suitable bank retention experiment.

### Stage 9: PostgreSQL Database Integration

Integrated the analytical pipeline with Neon PostgreSQL.

Five structured tables were created.

| Database Table | Purpose |
|---|---|
| `customers` | Original customer profiles |
| `churn_predictions` | LightGBM churn predictions |
| `customer_segments` | Customer value scores and cluster assignments |
| `uplift_predictions` | Experimental uplift predictions |
| `model_metrics` | Model evaluation results |

#### Database Loading Results

| Table | Records Loaded |
|---|---:|
| customers | 10,000 |
| churn_predictions | 2,000 |
| customer_segments | 10,000 |
| uplift_predictions | 8,539 |
| model_metrics | 10 |
| **Total** | **30,549** |

**Outcome:** Successfully established a cloud PostgreSQL connection and loaded 30,549 records across five database tables.

### Stage 10: Streamlit Application Development

Developed an interactive Streamlit application to present customer intelligence and analytical results.

The application provides ten navigation sections.

| Section | Functionality |
|---|---|
| Home | Platform overview and key information |
| Executive Dashboard | Business KPIs and customer distribution |
| Customer Intelligence | Customer search and profile exploration |
| Churn Prediction | Churn probability and risk analysis |
| Explainable AI | SHAP visualisations and model interpretation |
| Survival Analysis | Survival modelling feasibility and limitations |
| Customer Segmentation | Customer value scoring and clustering |
| Retention Intelligence | Experimental uplift analysis |
| ROI Simulator | Hypothetical retention campaign profitability |
| Model Performance | Evaluation metrics and model visualisations |

### Stage 11: ROI Simulation

Developed an interactive simulator to evaluate hypothetical retention campaign scenarios.

The user can specify:

- Number of customers targeted.
- Campaign cost per customer.
- Assumed incremental retention rate.
- Estimated value per retained customer.

The simulator calculates:

**Total Campaign Cost**

`Number of Targeted Customers × Campaign Cost per Customer`

**Expected Retained Customers**

`Targeted Customers × Assumed Incremental Retention Rate`

**Estimated Benefit**

`Expected Retained Customers × Assumed Value per Retained Customer`

**Estimated Net Benefit**

`Estimated Benefit − Total Campaign Cost`

**Return on Investment**

`((Estimated Benefit − Total Campaign Cost) / Total Campaign Cost) × 100`

The simulator allows business users to explore the financial implications of different retention assumptions.

The resulting ROI is illustrative and must not be interpreted as measured campaign performance.

### Stage 12: Cloud Deployment

The application was developed using GitHub Codespaces and deployed through Streamlit Community Cloud.

The deployment architecture includes:

- GitHub for source code management.
- Streamlit Community Cloud for application hosting.
- Neon PostgreSQL for cloud database services.
- Streamlit Secrets for secure database credential management.

**Outcome:** Successfully deployed the interactive application to a cloud-hosted environment.

---

## 7. Evaluation and Project Results

The project was evaluated across data quality, machine learning, database integration, uplift modelling and application functionality.

### 7.1 Data Quality Evaluation

| Evaluation | Result |
|---|---|
| Total bank customer records | 10,000 |
| Original features | 18 |
| Missing values | 0 |
| Duplicate records | 0 |
| Training records | 8,000 |
| Testing records | 2,000 |
| Actual churn rate | 20.38% |

### 7.2 Machine Learning Evaluation

The churn classification model was evaluated using a five-fold stratified cross-validation framework and a separate held-out test dataset.

The evaluation framework includes:

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Reliability of positive churn predictions |
| Recall | Proportion of actual churners identified |
| F1-score | Balance between precision and recall |
| ROC-AUC | Discrimination between churn and retained customers |
| PR-AUC | Performance under class imbalance |

Numerical performance values should be taken directly from:

`reports/evaluation/model_evaluation.csv`

Cross-validation results are maintained in:

`reports/evaluation/cross_validation_results.csv`

### 7.3 Explainability Evaluation

SHAP analysis was performed to examine global feature importance and individual customer explanations.

Generated visualisations are stored in:

`reports/figures/`

### 7.4 Uplift Modelling Evaluation

The experimental uplift model generated:

- 8,539 customer uplift predictions.
- Treatment-versus-control conversion estimates.
- Customer uplift categories.
- Qini evaluation outputs.

The evaluation is specific to the Hillstrom experimental dataset.

### 7.5 Database Evaluation

Successfully verified five PostgreSQL tables containing a combined total of 30,549 records.

### 7.6 Application Evaluation

The Streamlit application was deployed with ten navigation sections and integrated with the PostgreSQL backend.

Final acceptance testing should verify database queries, model outputs, visualisations, navigation and application behaviour.

---

## 8. System Architecture

The overall architecture follows a modular data processing and application integration approach.

```text
                  DATA SOURCES
                       |
          +------------+------------+
          |                         |
          v                         v
   BANK CHURN DATASET        HILLSTROM DATASET
          |                         |
          v                         v
     DATA CLEANING            EXPERIMENTAL
     AND VALIDATION           DATA PREPARATION
          |                         |
          v                         v
       EDA AND                CAUSALML
  FEATURE ENGINEERING         UPLIFT MODEL
          |
          v
   LIGHTGBM CLASSIFIER
          |
          +-------------------+
          |                   |
          v                   v
     CHURN SCORES        SHAP EXPLANATIONS
          |
          v
   CUSTOMER SEGMENTATION
          |
          v
   CUSTOMER INTELLIGENCE
          |
          v
    NEON POSTGRESQL
          |
          v
    STREAMLIT APPLICATION
          |
          +---------------------------+
          |             |             |
          v             v             v
      DASHBOARDS    ROI SIMULATOR   MODEL REPORTS
          |
          v
      BUSINESS INSIGHTS
```

---

## 9. Project Directory Structure

```text
Customer-Lifetime-Value-Churn-Intelligence/
│
├── app.py
│
├── app/
│   ├── Home.py
│   │
│   ├── pages/
│   │   ├── 01_Executive_Dashboard.py
│   │   ├── 02_Customer_Intelligence.py
│   │   ├── 03_Churn_Prediction.py
│   │   ├── 04_Explainable_AI.py
│   │   ├── 05_Survival_Analysis.py
│   │   ├── 06_Customer_Segmentation.py
│   │   ├── 07_Retention_Intelligence.py
│   │   ├── 08_ROI_Simulator.py
│   │   └── 09_Model_Performance.py
│   │
│   └── utils/
│       └── database.py
│
├── data/
│   ├── raw/
│   │   └── Customer-Churn-Records.csv
│   │
│   ├── processed/
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   ├── y_test.csv
│   │   ├── customer_id_train.csv
│   │   ├── customer_id_test.csv
│   │   ├── customer_intelligence.csv
│   │   └── uplift_predictions.csv
│   │
│   └── external/
│       ├── hillstrom.csv
│       └── hillstrom.csv.gz
│
├── database/
│   ├── __init__.py
│   ├── db_connection.py
│   ├── schema.sql
│   ├── create_tables.py
│   └── load_data.py
│
├── models/
│   ├── preprocessor.pkl
│   ├── lightgbm_churn_model.pkl
│   ├── uplift_model.pkl
│   ├── uplift_preprocessor.pkl
│   └── kmeans_model.pkl
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   │   ├── 01_churn_distribution.png
│   │   ├── 02_age_vs_churn.png
│   │   ├── 03_geography_vs_churn.png
│   │   ├── 04_tenure_vs_churn.png
│   │   ├── 05_membership_vs_churn.png
│   │   ├── 06_balance_vs_churn.png
│   │   ├── 07_products_vs_churn.png
│   │   ├── 08_correlation_matrix.png
│   │   ├── 09_complaints_vs_churn.png
│   │   ├── 10_lightgbm_confusion_matrix.png
│   │   ├── 11_lightgbm_roc_curve.png
│   │   ├── 12_lightgbm_pr_curve.png
│   │   ├── 13_feature_importance.png
│   │   ├── 14_shap_summary.png
│   │   ├── 15_shap_feature_importance.png
│   │   ├── 16_customer_shap_waterfall.png
│   │   ├── 17_customer_value_distribution.png
│   │   ├── 18_customer_segmentation.png
│   │   ├── 19_cluster_churn_analysis.png
│   │   └── 20_uplift_qini_curve.png
│   │
│   └── evaluation/
│       ├── data_quality_report.txt
│       ├── model_evaluation.csv
│       ├── cross_validation_results.csv
│       ├── feature_importance.csv
│       ├── shap_values.csv
│       ├── customer_shap_explanation.csv
│       ├── clustering_evaluation.csv
│       ├── customer_segment_summary.csv
│       ├── qini_curve.csv
│       └── uplift_segment_summary.csv
│
├── src/
│   ├── data/
│   │   └── inspect_dataset.py
│   │
│   ├── preprocessing/
│   │   └── preprocess.py
│   │
│   ├── features/
│   │
│   ├── models/
│   │   ├── train_churn.py
│   │   └── uplift_model.py
│   │
│   ├── evaluation/
│   │   ├── eda.py
│   │   ├── explainability.py
│   │   ├── test_shap.py
│   │   └── survival_analysis.py
│   │
│   └── business_logic/
│       └── customer_clv.py
│
├── tests/
│
├── requirements.txt
├── requirements-app.txt
├── .gitignore
├── README.md
└── .env (local only - excluded from GitHub)
```

*Note: This structure represents the project's intended organisation. Files excluded from GitHub, optional files and empty directories may not appear in the public repository. The deployed application uses the root-level `app.py`.*

---

## 10. Repository and Deployment

### GitHub Repository

[Customer Lifetime Value and Churn Intelligence System](https://github.com/Raajarapu/Customer-Lifetime-Value-Churn-Intelligence)

### Live Application

**Streamlit Application:** [Insert your deployed Streamlit URL here]

### Database

Neon PostgreSQL is used as the managed database service.

Database credentials are stored in Streamlit Community Cloud Secrets and local environment variables.

Sensitive credentials are excluded from version control.

---

## 11. Challenges Encountered and Technical Decisions

Several practical challenges were addressed during development.

### 11.1 Potential Target Leakage

The complaint attribute showed an unusually strong relationship with the churn target.

Rather than relying on potentially misleading correlations, the feature was excluded from the primary churn model.

### 11.2 Feature Transformation Consistency

The preprocessing pipeline generated transformed feature names with numerical and categorical prefixes.

These transformed names needed to be used correctly during SHAP dependence analysis.

### 11.3 Survival Data Limitations

The available bank dataset lacked sufficiently documented event and censoring information.

Rather than inventing survival durations, the project explicitly identified the limitation and documented the additional data requirements.

### 11.4 Separate Experimental Data for Uplift Modelling

The bank dataset did not contain randomised treatment assignments.

The Hillstrom dataset was therefore introduced as a separate experimental dataset to demonstrate treatment-effect estimation.

### 11.5 Database Connectivity

Integrated GitHub Codespaces with Neon PostgreSQL using SQLAlchemy and a PostgreSQL driver.

Database credentials were configured through environment variables and Streamlit Secrets.

### 11.6 Cloud Deployment

Resolved application dependency and configuration issues during Streamlit Community Cloud deployment.

The final deployment separates application source code from sensitive database credentials.

---

## 12. Limitations and Future Improvements

Although the project demonstrates a broad range of Data Science techniques, several improvements are necessary to establish a fully production-grade customer intelligence platform.

### Survival Analysis

Implement Kaplan-Meier estimation and Cox proportional hazards regression using a dataset with valid event durations, event indicators and censoring information.

### Financial Customer Lifetime Value

Replace heuristic customer value scoring with a financial CLV model using transaction-level revenue, purchase frequency, contribution margin and customer retention information.

### Bank Retention Uplift Modelling

Develop a bank-specific randomised retention campaign dataset and evaluate treatment effects using appropriate uplift metrics.

### Model Validation

Introduce comprehensive automated testing, probability calibration, threshold optimisation and monitoring for data and model drift.

### Production Readiness

Implement role-based access control, logging, automated retraining, database backups and scheduled data pipelines.

### Business Validation

Compare predicted customer risk and estimated campaign benefits against actual business outcomes before recommending operational interventions.

---

## 13. Conclusion

The Customer Lifetime Value and Churn Intelligence System demonstrates how Data Science and Machine Learning can be integrated into a unified customer analytics and decision-support platform.

The project began with the examination of a bank customer churn dataset and progressed through exploratory data analysis, feature engineering, predictive modelling and model interpretation.

LightGBM was implemented to estimate customer churn probability, while SHAP was used to explain the factors influencing model predictions. Customer value scoring and K-Means clustering were developed to identify distinct customer groups and support more structured customer analysis.

Furthermore, the project explored experimental uplift modelling through CausalML using the Hillstrom email marketing dataset. This demonstrated the distinction between identifying customers who are likely to churn and estimating which customers may respond incrementally to a marketing intervention.

The analytical outputs were integrated into Neon PostgreSQL, providing a centralised data storage layer. An interactive Streamlit application was subsequently developed and deployed, bringing together customer intelligence, churn analysis, model explainability, segmentation, experimental uplift insights and hypothetical retention ROI calculations.

The implementation successfully integrated five PostgreSQL tables containing **30,549 records**, including 10,000 customer profiles, 2,000 churn predictions, 10,000 customer segmentation records, 8,539 experimental uplift predictions and 10 model evaluation records.

Importantly, the project also identified methodological limitations. The available bank dataset does not support reliable time-to-event modelling, the customer value score is a heuristic proxy rather than financial CLV, and the uplift results originate from a separate marketing experiment. These limitations establish clear directions for further development rather than being presented as completed analytical capabilities.

Overall, the project demonstrates the practical integration of data engineering, predictive analytics, explainable machine learning, experimental modelling, cloud database management and application deployment.

**The central achievement is the transformation of raw customer information into an integrated analytical platform that supports more informed, transparent and economically conscious customer retention decisions.**

---

## 14. Individual Learning and Technical Ownership

This project provided practical experience across the complete Data Science development lifecycle.

Key areas of learning and technical ownership included:

- Understanding and validating real-world customer datasets.
- Performing exploratory data analysis and feature engineering.
- Developing and evaluating machine learning classification models.
- Applying explainability techniques to interpret model behaviour.
- Understanding the distinction between classification, survival analysis and uplift modelling.
- Developing customer segmentation and value-scoring approaches.
- Integrating Python applications with PostgreSQL.
- Building interactive analytical dashboards using Streamlit.
- Managing source code through Git and GitHub.
- Deploying a database-connected application using cloud services.
- Identifying methodological limitations and communicating them transparently.

The project strengthened the understanding that successful Data Science involves more than developing predictive models. It requires reliable data, appropriate evaluation, sound statistical reasoning, effective communication and the ability to translate analytical results into practical business insights.

---

## 15. References and Resources

### Datasets

- IBM Telco Customer Churn Dataset.
- Bank Customer Churn Dataset.
- KKBox Churn Prediction Challenge.
- Online Retail II Dataset – UCI.
- Criteo Uplift Modelling Dataset.
- Hillstrom MineThatData Email Marketing Dataset.

### Documentation

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [CausalML Documentation](https://causalml.readthedocs.io/)
- [Lifelines Documentation](https://lifelines.readthedocs.io/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)

---

## 16. Project Information

**Project Title:** Customer Lifetime Value and Churn Intelligence System

**Domain:** Data Science and Machine Learning

**Primary Application:** Customer Analytics and Retention Intelligence

**Programming Language:** Python

**Database:** PostgreSQL

**Application Framework:** Streamlit

**Deployment Platform:** Streamlit Community Cloud

**Source Code:** GitHub

**Project Status:** Application deployed; final validation and methodological extensions identified.

---

### Final Statement

**Predicting customer churn is only the beginning. Understanding customer behaviour, identifying potential value and making better retention decisions is the real objective.**
