import joblib
import pandas as pd


class FraudPredictor:
    def __init__(
        self,
        model_path="models/fraud_model.pkl",
        features_path="models/feature_names.pkl"
    ):
        # Load saved model and feature names
        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(features_path)

    def predict(self, input_data, threshold=0.25):
        """
        Predict fraud probability for a single transaction.
        Accepts either a dictionary or a pandas DataFrame.
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()

        # Keep the same feature order used during training
        df = df[self.feature_names]

        probability = self.model.predict_proba(df)[:, 1][0]
        is_fraud = probability >= threshold

        return {
            "fraud_probability": f"{probability:.2%}",
            "is_fraud": bool(is_fraud),
            "recommendation": "BLOCK TRANSACTION" if is_fraud else "ALLOW TRANSACTION",
            "confidence_score": round(float(probability), 4)
        }


if __name__ == "__main__":
    predictor = FraudPredictor()

    sample_data = pd.read_csv("data/raw/creditcard.csv").iloc[0:1].drop("Class", axis=1)
    result = predictor.predict(sample_data)

    print(result)