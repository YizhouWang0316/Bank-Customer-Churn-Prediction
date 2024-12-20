# **Bank Customer Churn Prediction**

This project aims to predict customer churn in a bank using machine learning techniques. Churn prediction helps banks identify customers who are likely to leave and implement strategies to retain them, ultimately improving customer satisfaction and reducing revenue loss.

---

## **Table of Contents**
- Project Overview
- Dataset
  Technologies Used
- Project Workflow
- Key Results
- How to Run
- Future Enhancements

---
## **Project Overview**
Customer churn prediction is a critical business challenge, particularly in the banking sector where retaining customers is more cost-effective than acquiring new ones. This project utilizes data preprocessing, feature engineering, and machine learning models to predict churn behavior based on customer demographic and transactional data.

---

## **Dataset**
The dataset used in this project includes:
- Customer demographics (e.g., age, gender, location)
- Account details (e.g., balance, tenure)
- Transactional behavior (e.g., transaction frequency, credit score)

The dataset includes a target variable, `Churn`, which indicates whether a customer has churned (`1`) or not (`0`).

---

## **Technologies Used**
- **Programming Language:** Python
- **Libraries:**
  - **Data Analysis & Preprocessing:** Pandas, NumPy
  - **Data Visualization:** Matplotlib, Seaborn
  - **Machine Learning:** scikit-learn
- **Jupyter Notebook:** For interactive development and analysis

---

## **Project Workflow**
1. **Data Exploration and Cleaning:**
   - Handled missing values and outliers.
   - Preprocessed categorical and numerical features (e.g., one-hot encoding, standardization).

2. **Feature Engineering:**
   - Analyzed feature importance.
   - Engineered new variables to enhance prediction accuracy.

3. **Model Selection and Training:**
   - Built multiple machine learning models, including:
     - Logistic Regression
     - Random Forest
     - Gradient Boosting
   - Used cross-validation and hyperparameter tuning to optimize model performance.

4. **Model Evaluation:**
   - Evaluated models using metrics like:
     - Accuracy
     - Precision, Recall, F1-score
     - ROC-AUC Score
   - Visualized confusion matrices and ROC curves for performance analysis.

5. **Insights and Recommendations:**
   - Highlighted key drivers of customer churn.
   - Proposed actionable strategies to reduce churn rates.

---


## Key Results

### Model Performance

- **Random Forest (RF):** Achieved the highest accuracy (86.16%) and AUC (0.8476) but had a moderate recall (41.06%), indicating it missed a significant number of actual churners.
- **K-Nearest Neighbors (KNN):** Delivered an accuracy of 84.28% and AUC of 0.7986 with a better precision (72.83%) than Logistic Regression but still low recall (36.35%).
- **Logistic Regression (LR):** Had the lowest performance with 80.92% accuracy and AUC of 0.7722, exhibiting very low recall (19.45%).

### Feature Importance

- **Top Influencers:** Age, Estimated Salary, Credit Score, Balance, and Number of Products were the most significant predictors of churn.
- **Customer Engagement:** Active membership and a higher number of banking products significantly reduced the likelihood of churn.
- **Geographical Impact:** Customers from Germany showed a higher propensity to churn compared to other regions.

## Insights

- **Demographics:** Older customers are more likely to churn, and female customers exhibit higher churn rates than males.
- **Financial Metrics:** Higher salaries and account balances correlate with increased churn, suggesting affluent customers may seek better offers elsewhere. Conversely, higher credit scores are associated with lower churn rates.
- **Customer Engagement:** Active members and those with multiple banking products are less likely to leave, highlighting the importance of engagement and cross-selling.
- **Geographical Factors:** Regional differences, particularly higher churn rates in Germany, indicate the need for localized strategies.

## Recommendations

### Targeted Retention Strategies

- **Older Customers:** Develop personalized services and loyalty programs tailored to older demographics.
- **Inactive Members:** Implement engagement initiatives and activity-based rewards to convert inactive members into active ones.

### Financial Incentives

- **Affluent Customers:** Offer premium services, exclusive benefits, and tailored financial products to retain high-balance and high-salary customers.

### Cross-Selling and Product Diversification

- Encourage customers to adopt additional banking products through bundled offers, discounts, and loyalty rewards to increase their investment in the bank’s ecosystem.

### Geographical Focus

- **Germany:** Investigate and address region-specific factors contributing to higher churn rates through improved customer service and localized product offerings.

### Gender-Specific Campaigns

- Enhance services and develop targeted marketing campaigns to address the higher churn rates among female customers.

### Model Optimization

- **Improve Recall:** Adjust model parameters, use techniques like SMOTE for balancing the dataset, or explore advanced algorithms to better identify actual churners.
- **Alternative Metrics:** Focus on metrics such as F1-Score or Precision-Recall AUC to better assess model performance in predicting churn.

### Continuous Monitoring and Iteration

- Regularly update models with new data, perform periodic re-training, and implement a feedback loop to refine retention strategies based on real-world outcomes.


---

## **How to Run**
To replicate this project, follow these steps:
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repo/Bank_Customer_Churn_Prediction.git
   ```
2. **Install Required Libraries:**
   Install the dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Jupyter Notebook:**
   Open the `.ipynb` file in Jupyter Notebook and run the cells sequentially.

---

## **Future Enhancements**
- Incorporate more advanced machine learning models, such as XGBoost or CatBoost.
- Experiment with deep learning techniques for improved prediction accuracy.
- Build an interactive dashboard for real-time churn analysis and visualization.
