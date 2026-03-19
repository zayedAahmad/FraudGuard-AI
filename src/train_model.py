import os
import joblib
import pandas as pd

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score, confusion_matrix


class FraudDetectionTrainer:
    """
    Training pipeline for fraud detection on imbalanced transaction data.
    """

    def __init__(self, data_path, model_dir="models"):
        self.data_path = data_path
        self.model_dir = model_dir
        self.model = None
        self.feature_names = None

        os.makedirs(self.model_dir, exist_ok=True)

    def load_and_split_data(self):
        print(f"Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)

        X = df.drop("Class", axis=1)
        y = df["Class"]

        self.feature_names = X.columns.tolist()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        print("Data loaded and split successfully.")

    def apply_smote(self):
        print("Applying SMOTE on training data...")
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        print(f"Training samples after SMOTE: {len(self.y_train)}")

    def train_model(self):
        print("Training XGBoost model...")

        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=42
        )

        self.model.fit(self.X_train, self.y_train)
        print("Training completed.")

    def evaluate_model(self):
        print("\n" + "=" * 30)
        print("MODEL PERFORMANCE REPORT")
        print("=" * 30)

        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        auprc = average_precision_score(self.y_test, y_prob)
        print(f"AUPRC: {auprc:.4f}\n")

        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))

        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))

    def save_artifacts(self):
        model_path = os.path.join(self.model_dir, "fraud_model.pkl")
        features_path = os.path.join(self.model_dir, "feature_names.pkl")

        joblib.dump(self.model, model_path)
        joblib.dump(self.feature_names, features_path)

        print(f"\nModel saved to: {model_path}")
        print(f"Feature names saved to: {features_path}")


if __name__ == "__main__":
    trainer = FraudDetectionTrainer(data_path="data/raw/creditcard.csv")
    trainer.load_and_split_data()
    trainer.apply_smote()
    trainer.train_model()
    trainer.evaluate_model()
    trainer.save_artifacts()