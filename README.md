# customer-spend-prediction-catboost
Advanced machine learning regression pipeline utilizing CatBoost, 5-Fold Cross-Validation, and custom feature engineering to forecast customer monthly spend.

### Key Highlights & Technical Stack:
* **Advanced Feature Engineering:** Constructed financial interaction terms (e.g., credit-to-income ratios, transaction frequencies, and spending proxies).
* **Target Transformation:** Applied log-transformations (`np.log1p`) to manage heavy right-skewness in financial spend data, reversing them post-prediction (`np.expm1`).
* **Robust Validation:** Implemented 5-Fold Cross-Validation combined with `CatBoostRegressor` to prevent overfitting and ensure reliable out-of-fold generalization.
* **Production Hygiene:** Guarded against data leakage during one-hot encoding and clipped predictions to eliminate illogical business values (negative spend).
* **Python Libraries:** Pandas, NumPy, Scikit-Learn, CatBoost.
