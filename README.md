# Customer_Churn_Prediction

![churn](https://github.com/user-attachments/assets/2a6dc421-3f7a-4bcd-b8a4-f90649f8265a)

📜 Overview
Customer churn, where customers stop using a service or product, poses significant challenges to businesses. Predicting churn accurately allows companies to implement retention strategies and reduce revenue loss. This project focuses on building and evaluating multiple machine learning models to predict customer churn using a balanced dataset created with SMOTE (Synthetic Minority Oversampling Technique).

The goal is to identify at-risk customers and help businesses take proactive measures to improve customer satisfaction and retention.

🛠 Features
Data Preprocessing:

Handled missing values, categorical features, and outliers.
Balanced the dataset using SMOTE, ensuring equal representation of churned and non-churned classes for better model performance.
Model Building and Evaluation:

Implemented six machine learning models:
Logistic Regression
K-Nearest Neighbors (KNN)
Support Vector Machines (SVM)
Random Forest
Gradient Boosting
XGBoost
Evaluated each model on multiple metrics, including:
Accuracy
Precision
Recall
F1 Score
ROC-AUC Score (to measure classification performance considering class imbalance).
Hyperparameter Tuning:

Used RandomizedSearchCV for fine-tuning hyperparameters of each model to achieve optimal performance.
Best Model Selection:

Selected XGBoost as the best-performing model based on its high accuracy, ROC-AUC score, and overall balanced performance.
Model Saving and Deployment:

Saved the trained XGBoost model using Joblib for future predictions.
Provided a framework for loading the saved model to make predictions on new, unseen data.


🔧 Tools and Technologies
Programming Language: Python
Libraries Used:
Data Preprocessing: Pandas, NumPy
Machine Learning Algorithms: Scikit-learn, XGBoost
Imbalanced Data Handling: imbalanced-learn (SMOTE)
Model Evaluation Metrics: Precision, Recall, F1 Score, ROC-AUC, Accuracy
Visualization: Matplotlib, Seaborn
Model Saving: Joblib

📝 Conclusion
Key Factors Influencing Churn:
Customer Tenure: Customers are more likely to churn during the first 5-10 months, highlighting the importance of a positive early customer experience.
Contract Type: Month-to-month contracts have higher churn rates compared to one-year or two-year contracts.
Recommendations for Vodafone:

Focus on Early Customer Experience:
Improve onboarding processes, service quality, and provide dedicated tech support during the initial months to boost customer satisfaction and reduce churn.
Promote Long-Term Contracts:
Encourage customers to choose longer-term contracts by offering incentives and additional benefits to foster loyalty.
Model Performance:

XGBoost was selected as the best model with:
Accuracy: 84%
ROC-AUC Score: 92%
Random Forest also performed well with 84% accuracy and  91% ROC-AUC Score, but XGBoost's slight edge makes it the preferred choice.

Ensemble methods perform well on classification tasks, compared to using single classifiers

🚀 How to Use
Clone the repository.
Install the required Python libraries listed in requirements.txt.
Train models or directly load the pre-trained XGBoost model saved as xgboost_model.joblib.
Run predictions using the predict_churn.py script with new customer data.
📊 Visualization
Included detailed visualizations of feature importance, data distribution, and model performance using Matplotlib and Seaborn.
Performance metrics such as confusion matrices, ROC curves, and classification reports are provided for each model.

🔮 Future Scope
Enhanced Features: Add sentiment analysis from customer reviews and time-series data for behavior trends.
Scalability & Model Improvements: Scale models to larger datasets, explore ensemble/deep learning methods, and leverage cloud platforms for distributed usage.
