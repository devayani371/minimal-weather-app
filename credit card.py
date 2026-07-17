import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

class FraudDetectionPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.X_train_res, self.y_train_res = None, None
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=12, 
            random_state=42, 
            n_jobs=-1
        )
        
    def load_data(self):
        """Loads dataset and checks integrity."""
        print(f"[*] Loading data from {self.data_path}...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}. Please check path.")
        
        self.df = pd.read_csv(self.data_path)
        print(f"[+] Successfully loaded data. Shape: {self.df.shape}")
        
        # Check target distribution
        fraud_percentage = (self.df['Class'].value_counts()[1] / len(self.df)) * 100
        print(f"[!] Class Imbalance: Normal={self.df['Class'].value_counts()[0]}, Fraud={self.df['Class'].value_counts()[1]} ({fraud_percentage:.3f}%)")

    def preprocess_and_split(self):
        """Splits features, applies scaling to 'Amount', and performs stratified split."""
        print("[*] Preprocessing features...")
        
        # Drop time feature as it's rarely useful without advanced feature engineering
        X = self.df.drop(columns=['Class', 'Time'])
        y = self.df['Class']
        
        # Stratified split to ensure train/test sets have the same ratio of fraud
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Scale 'Amount' (Fit only on train to avoid data leakage)
        self.X_train['Amount'] = self.scaler.fit_transform(self.X_train[['Amount']])
        self.X_test['Amount'] = self.scaler.transform(self.X_test[['Amount']])
        print("[+] Preprocessing and splitting complete.")

    def handle_imbalance(self):
        """Applies SMOTE exclusively on the training set to prevent data leakage."""
        print("[*] Applying SMOTE to balance the training classes...")
        smote = SMOTE(random_state=42)
        self.X_train_res, self.y_train_res = smote.fit_resample(self.X_train, self.y_train)
        print(f"[+] Resampled Training Shape: {self.X_train_res.shape}")

    def train_model(self):
        """Trains the Random Forest model."""
        print("[*] Training Random Forest Classifier (this may take a minute)...")
        self.model.fit(self.X_train_res, self.y_train_res)
        print("[+] Model training complete.")

    def evaluate_model(self):
        """Generates clear, deployment-ready metric reports."""
        print("[*] Evaluating model performance on untouched test set...")
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        print("\n=== CLASSIFICATION REPORT ===")
        print(classification_report(self.y_test, y_pred))
        
        print("=== CONFUSION MATRIX ===")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        roc_auc = roc_auc_score(self.y_test, y_proba)
        print(f"\n[+] ROC-AUC Score: {roc_auc:.4f}")
        
        # Plot Confusion Matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/confusion_matrix.png')
        print("[+] Confusion Matrix plot saved to plots/confusion_matrix.png")

    def save_artifacts(self):
        """Saves scaler and model files for deployment/inference."""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/fraud_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        print("[+] Saved model artifacts to 'models/' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Pipeline")
    parser.add_argument('--data_path', type=str, default='data/creditcard.csv', help='Path to transaction CSV file')
    args = parser.parse_args()
    
    pipeline = FraudDetectionPipeline(data_path=args.data_path)
    pipeline.load_data()
    pipeline.preprocess_and_split()
    pipeline.handle_imbalance()
    pipeline.train_model()
    pipeline.evaluate_model()
    pipeline.save_artifacts()