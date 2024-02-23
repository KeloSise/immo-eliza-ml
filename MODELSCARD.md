# Model Card

## Project Context
This project aims to predict real estate prices based on various property characteristics. The endeavor began with a simpler model and feature set, evolving over time to include more complex models and additional features to improve accuracy.

## Data
- **Initial dataset**: 'properties.csv', filtered to include properties within the top 18 most common zip codes based on value counts.
- **Target variable**: 'price'
- **Initial features**: Number of bedrooms ('nbr_bedrooms'), total area in square meters ('total_area_sqm'), and one-hot encoded 'zip_code'.
- **Expanded features**: Added 'latitude', 'longitude', and 'construction_year' for greater model complexity and potential accuracy.

## Model details
Initially started with a linear regression model. The performance was compared to a Random Forest model. The latter had better performance.
1. **Linear Regression Model**
   - **Feature preprocessing**: Simple imputation for missing numerical values, one-hot encoding for 'zip_code'.
   - **Performance**: Provided a baseline R² score for both training and testing datasets.
  
2. **Random Forest Regressor**
   - **Feature preprocessing**: Mean imputation for missing numerical values, standard scaling for numerical features.
   - **Additional features**: 'latitude', 'longitude', and 'construction_year'.
   - **Performance**: Improved R² scores, indicating better model fit and predictive accuracy.

## Performance Metrics
- **Linear Regression Model**: Initial R² scores for training and testing sets.
                                Train R² score: 0.4788421958600292
                                Test R² score: 0.4758553292258668

- **Random Forest Regressor**: Enhanced R² scores for training and testing sets after model and feature set expansion.
                                Train R² score: 0.8728922353840767
                                Test R² score: 0.6403799127825482
## Limitations
- **Linear Regression Model**: May not capture complex nonlinear relationships or interactions between features.
- **Random Forest Regressor**: More computationally intensive, risk of overfitting, and less interpretability.

## Usage
- **Dependencies**: scikit-learn, pandas, joblib, click (for `predict.py`).
- **Training**: Run `train()` in `train.py` to retrain models.
- **Predicting**: Use `predict.py` with `-i` for input dataset and `-o` for output predictions file.


