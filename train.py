import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def train():
    # Load the data
    data = pd.read_csv("data/properties.csv")

    # Correct numeric features based on the DataFrame's column names
    num_features = ["nbr_bedrooms", "total_area_sqm", "surface_land_sqm"] # assuming 'nbr_bathrooms' is the correct name

    # Assuming 'region' is the correct categorical feature name
    cat_features = ["region"]


    # Split the data into features and target
    X = data[num_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Impute missing values for numerical features
    imputer = SimpleImputer(strategy="mean")
    X_train[num_features] = imputer.fit_transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])

    # Apply one-hot encoding to categorical features
    enc = OneHotEncoder(handle_unknown='ignore')
    X_train_cat = enc.fit_transform(X_train[cat_features]).toarray()
    X_test_cat = enc.transform(X_test[cat_features]).toarray()

    # Combine numerical and one-hot encoded categorical features
    X_train_prepared = pd.concat([pd.DataFrame(X_train[num_features], index=X_train.index),
                                  pd.DataFrame(X_train_cat, index=X_train.index, columns=enc.get_feature_names_out())], axis=1)
    X_test_prepared = pd.concat([pd.DataFrame(X_test[num_features], index=X_test.index),
                                 pd.DataFrame(X_test_cat, index=X_test.index, columns=enc.get_feature_names_out())], axis=1)

    # Train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_prepared, y_train)

    # Evaluate the model
    train_score = r2_score(y_train, model.predict(X_train_prepared))
    test_score = r2_score(y_test, model.predict(X_test_prepared))
    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")

    # Save the model and preprocessing artifacts
    joblib.dump({"features": {"num_features": num_features, "cat_features": cat_features},
                 "imputer": imputer, "enc": enc, "model": model}, "models/artifacts_forest.joblib", compress=9)

if __name__ == "__main__":
    train()
