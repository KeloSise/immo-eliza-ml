import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def train():
    # Load the data
    data = pd.read_csv("data/properties.csv")

    # Define numerical features to use
    num_features = ["nbr_bedrooms", "total_area_sqm"]

    # Define categorical features to use
    cat_features = ["zip_code"]

    # Get the counts of each zip code
    zip_code_counts = data['zip_code'].value_counts()

    # Select the top 10 most common zip codes
    top_zip_codes = zip_code_counts.head(10).index

    # Filter the dataset to include only properties within the top zip codes
    data = data[data['zip_code'].isin(top_zip_codes)]

    # Split the data into features and target
    X = data[num_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=505
    )

    # Impute missing values using SimpleImputer for numerical features
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train[num_features])
    X_train[num_features] = imputer.transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])

    # Convert categorical column 'zip_code' with one-hot encoding using OneHotEncoder
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(X_train[cat_features])
    X_train_cat = enc.transform(X_train[cat_features]).toarray()
    X_test_cat = enc.transform(X_test[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    X_train_prepared = pd.concat(
        [
            pd.DataFrame(X_train[num_features], index=X_train.index),
         pd.DataFrame(X_train_cat, index=X_train.index, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )
    X_test_prepared = pd.concat(
        [
            pd.DataFrame(X_test[num_features], index=X_test.index),
            pd.DataFrame(X_test_cat, index=X_test.index, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    # Train the model
    model = LinearRegression()
    model.fit(X_train_prepared, y_train)

    # Evaluate the model
    train_score = r2_score(y_train, model.predict(X_train_prepared))
    test_score = r2_score(y_test, model.predict(X_test_prepared))
    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")

    # Save the model and other artifacts
    artifacts = {
        "features": {
            "num_features": num_features,
            "cat_features": cat_features,
        },
        "imputer": imputer,
        "enc": enc,
        "model": model,
    }
    joblib.dump(artifacts, "models/artifacts.joblib")

if __name__ == "__main__":
    train()