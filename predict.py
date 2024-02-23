import click
import joblib
import pandas as pd

@click.command()
@click.option("-i", "--input-dataset", help="Path to input .csv dataset", required=True)
@click.option("-o", "--output-dataset", default="output/predictions.csv", help="Path to save predictions", required=True)
def predict(input_dataset, output_dataset):
    # Load new data
    data = pd.read_csv(input_dataset)

    # Load saved artifacts
    artifacts = joblib.load("models/artifacts_forest.joblib")

    # Unpack artifacts
    num_features = artifacts["features"]["num_features"]
    cat_features = artifacts["features"]["cat_features"]
    imputer = artifacts["imputer"]
    enc = artifacts["enc"]
    model = artifacts["model"]

    # Prepare new data
    data_num = data[num_features]
    data_num_imputed = imputer.transform(data_num)
    data_cat = data[cat_features]
    data_cat_encoded = enc.transform(data_cat).toarray()

    # Combine numerical and one-hot encoded categorical data
    data_prepared = pd.concat([pd.DataFrame(data_num_imputed, columns=num_features),
                               pd.DataFrame(data_cat_encoded, columns=enc.get_feature_names_out())], axis=1)

    # Make predictions using the trained model
    predictions = model.predict(data_prepared)

    # Save the predictions to a CSV file
    pd.DataFrame({"predictions": predictions}).to_csv(output_dataset, index=False)

    # Print success messages
    click.echo(click.style("Predictions generated successfully!", fg="green"))
    click.echo(f"Saved to {output_dataset}")

if __name__ == "__main__":
    predict()
