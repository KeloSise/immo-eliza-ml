import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train():

    data = pd.read_csv("data/properties.csv")
    num_features = ["nbr_bedrooms", "total_area_sqm", "surface_land_sqm", "latitude", "longitude", "construction_year"]

    X = data[num_features].copy()
    y = data["price"].copy()

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Pipeline with imputer for missing values and scaler for numerical features
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('scaler', StandardScaler()),
    ])

    # Processing the numerical features
    X_train_prepared = num_pipeline.fit_transform(X_train)
    X_test_prepared = num_pipeline.transform(X_test)

    # Training the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_prepared, y_train)

    # Evaluating the model
    train_score = r2_score(y_train, model.predict(X_train_prepared))
    test_score = r2_score(y_test, model.predict(X_test_prepared))
    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")

    # Saving the model and preprocessing pipeline
    joblib.dump((num_pipeline, model), "models/artifacts.joblib", compress=9)

if __name__ == "__main__":
    train()
