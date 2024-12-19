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


## **Key Results**
- The **Random Forest model** achieved the highest accuracy of 86% and an AUC score of 85%.
- **Feature importance analysis** revealed key factors influencing churn, such as:
  - Low account balance
  - High transaction frequency
  - Short customer tenure
- Recommendations included personalized retention strategies for high-risk customers.

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

---

## **Contributing**
Contributions are welcome! If you have suggestions or improvements, feel free to fork the repository and create a pull request.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Acknowledgments**
Special thanks to the data science community for providing resources and tools to build predictive models effectively.
