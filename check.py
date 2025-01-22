import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

class FastCancerPredictionModel:
    def __init__(self, num_samples=10000):  # Reduced from 50000
        self.num_samples = num_samples
        self.data = None
        self.X = None
        self.y = None
        self.model = None
        self.scaler = RobustScaler()
    
    def generate_synthetic_data(self):
        np.random.seed(42)
        stages = [0, 1, 2, 3, 4]
        stage_proportions = [0.4, 0.3, 0.15, 0.10, 0.05]
        
        data = {
            'CA-125': [], 'PSA': [], 'CEA': [], 
            'HER2': [], 'CA 15-3': [], 'CancerStage': []
        }
        
        for _ in range(self.num_samples):
            stage = np.random.choice(stages, p=stage_proportions)
            
            data['CA-125'].append(max(0, np.random.normal(50 + stage * 10, 10)))
            data['PSA'].append(max(0, np.random.normal(3 + stage * 2, 2)))
            data['CEA'].append(max(0, np.random.normal(3 + stage * 2, 1.5)))
            data['HER2'].append(max(0, np.random.normal(1.5 + stage * 1, 0.5)))
            data['CA 15-3'].append(max(0, np.random.normal(25 + stage * 5, 5)))
            data['CancerStage'].append(stage)
        
        self.data = pd.DataFrame(data)
        return self.data
    
    def prepare_data(self, test_size=0.2):
        self.X = self.data.drop('CancerStage', axis=1)
        self.y = self.data['CancerStage']
        
        return train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=42, 
            stratify=self.y
        )
    
    def create_model(self):
        return RandomForestClassifier(
            n_estimators=100,  # Reduced from 300
            max_depth=10,      # Reduced complexity
            random_state=42,
            class_weight='balanced'
        )
    
    def train_model(self, X_train, y_train):
        self.model = self.create_model()
        self.model.fit(X_train, y_train)
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
        return y_pred

def main():
    model = FastCancerPredictionModel()
    model.generate_synthetic_data()
    X_train, X_test, y_train, y_test = model.prepare_data()
    model.train_model(X_train, y_train)
    model.evaluate_model(X_test, y_test)

if __name__ == "__main__":
    main()