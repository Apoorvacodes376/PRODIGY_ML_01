# PRODIGY_ML_01

---

#  House Prices Prediction using Machine Learning

This project focuses on predicting house sale prices using machine learning techniques on the **Kaggle House Prices – Advanced Regression Techniques** dataset. The goal is to build, evaluate, and compare regression models while following proper ML practices such as validation, preprocessing, and model evaluation.

---

##  Project Overview

Accurately estimating house prices is a classic regression problem. In this project, we:

* Built a baseline **Decision Tree Regressor**
* Improved performance using a **Random Forest Regressor**
* Evaluated models using **Mean Absolute Error (MAE)**
* Handled missing values using **median imputation**
* Generated a Kaggle-ready submission file

This project was done as a **revision exercise** to reinforce core machine learning concepts using `scikit-learn`.

---

##  Dataset

* **Source:** Kaggle – *House Prices: Advanced Regression Techniques*
* **Training data:** `train.csv`
* **Test data:** `test.csv`
* **Target variable:** `SalePrice`

Only selected numerical features were used to keep the model simple and interpretable.

---

##  Technologies & Libraries Used

* Python
* Pandas
* NumPy
* scikit-learn

---

##  Features Used

The following numerical features were selected for model training:

* `LotArea`
* `YearBuilt`
* `1stFlrSF`
* `2ndFlrSF`
* `FullBath`
* `BedroomAbvGr`
* `TotRmsAbvGrd`

---

##  Model Workflow

1. **Data Loading**

   * Load training data using Pandas
2. **Train–Validation Split**

   * Split data into training and validation sets
3. **Baseline Model**

   * Decision Tree Regressor
4. **Model Tuning**

   * Decision Tree with limited `max_leaf_nodes`
5. **Advanced Model**

   * Random Forest Regressor
6. **Preprocessing**

   * Handle missing values using `SimpleImputer (median)`
7. **Evaluation**

   * Mean Absolute Error (MAE)
8. **Prediction**

   * Generate predictions on test data
9. **Submission**

   * Create `submission.csv` for Kaggle

---

##  Evaluation Metric

**Mean Absolute Error (MAE)** was used to evaluate model performance:

[
MAE = \frac{1}{n} \sum |y_{true} - y_{predicted}|
]

Lower MAE indicates better predictive accuracy.

---

##  Key Learnings

* Random Forest models generally outperform single Decision Trees
* Proper handling of missing values is crucial
* Preprocessing steps applied to training data must also be applied to validation and test data
* Consistency in data formats (NumPy vs DataFrame) avoids warnings and bugs
* Validation-based evaluation helps prevent overfitting

---

##  Output

* **`submission.csv`**
  Contains predicted house prices for Kaggle submission in the required format:

  ```
  Id, SalePrice
  ```

---

##  Future Improvements

* Use all numerical features automatically
* Add categorical feature encoding
* Perform hyperparameter tuning
* Try advanced models like Gradient Boosting or XGBoost
* Visualize feature importance and residuals

---

##  Conclusion

This project demonstrates a complete end-to-end machine learning pipeline for a regression problem, following best practices suitable for learning, revision, and interview preparation.
Thanks to ProdigyInfoTech for providing this opportunity to take part in the process of working on this mini-project

---

