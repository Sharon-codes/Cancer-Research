import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

class CancerStagePredictionModel:
    def __init__(self, num_samples=2000):
        self.num_samples = num_samples
        self.data = None
        self.X = None
        self.y = None
        self.model = None
        self.scaler = StandardScaler()
    
    def generate_synthetic_protein_data(self):
        np.random.seed(42)
        
        stages = [0, 1, 2, 3, 4]
        stage_proportions = [0.4, 0.2, 0.2, 0.1, 0.1]
        
        data = {
            'CA-125': [],
            'PSA': [],
            'CEA': [],
            'HER2': [],
            'CA 15-3': [],
            'CancerStage': []
        }
        
        for _ in range(self.num_samples):
            stage = np.random.choice(stages, p=stage_proportions)
            
            # More distinct protein level distributions for each stage
            if stage == 0:  # No Cancer
                ca125 = max(0, np.random.normal(20, 5))
                psa = max(0, np.random.normal(2, 1))
                cea = max(0, np.random.normal(2, 0.5))
                her2 = max(0, np.random.normal(1, 0.2))
                ca153 = max(0, np.random.normal(15, 3))
            elif stage == 1:  # Stage I
                ca125 = max(0, np.random.normal(40, 10))
                psa = max(0, np.random.normal(4, 2))
                cea = max(0, np.random.normal(4, 1))
                her2 = max(0, np.random.normal(2, 0.5))
                ca153 = max(0, np.random.normal(25, 5))
            elif stage == 2:  # Stage II
                ca125 = max(0, np.random.normal(60, 15))
                psa = max(0, np.random.normal(6, 3))
                cea = max(0, np.random.normal(6, 2))
                her2 = max(0, np.random.normal(3, 1))
                ca153 = max(0, np.random.normal(35, 7))
            elif stage == 3:  # Stage III
                ca125 = max(0, np.random.normal(80, 20))
                psa = max(0, np.random.normal(8, 4))
                cea = max(0, np.random.normal(8, 3))
                her2 = max(0, np.random.normal(4, 1.5))
                ca153 = max(0, np.random.normal(45, 10))
            else:  # Stage IV
                ca125 = max(0, np.random.normal(100, 25))
                psa = max(0, np.random.normal(10, 5))
                cea = max(0, np.random.normal(10, 4))
                her2 = max(0, np.random.normal(5, 2))
                ca153 = max(0, np.random.normal(55, 15))
            
            data['CA-125'].append(ca125)
            data['PSA'].append(psa)
            data['CEA'].append(cea)
            data['HER2'].append(her2)
            data['CA 15-3'].append(ca153)
            data['CancerStage'].append(stage)
        
        self.data = pd.DataFrame(data)
        return self.data
    
    def prepare_data(self, test_size=0.2):
        self.X = self.data.drop('CancerStage', axis=1)
        self.y = self.data['CancerStage']
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            max_depth=7,
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=5)
        
        print("Model Performance Metrics:")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nCross-Validation Scores:")
        print(f"Mean CV Score: {cv_scores.mean():.4f}")
        print(f"CV Score Standard Deviation: {cv_scores.std():.4f}")
        
        return y_pred
    
    def predict_cancer_stage(self, new_patient_data):
        # Ensure input is in the same format as training data
        new_patient_scaled = self.scaler.transform(new_patient_data)
        
        # Predict stage and probabilities
        stage_prediction = self.model.predict(new_patient_scaled)
        stage_probabilities = self.model.predict_proba(new_patient_scaled)
        
        return stage_prediction[0], stage_probabilities[0]

def main():
    # Create and prepare model
    cancer_stage_model = CancerStagePredictionModel(num_samples=3000)
    synthetic_data = cancer_stage_model.generate_synthetic_protein_data()
    
    # Prepare and train model
    X_train, X_test, y_train, y_test = cancer_stage_model.prepare_data()
    trained_model = cancer_stage_model.train_model(X_train, y_train)
    
    # Evaluate model
    cancer_stage_model.evaluate_model(X_test, y_test)
    
    # Test cases
    stage_labels = ['No Cancer', 'Stage I', 'Stage II', 'Stage III', 'Stage IV']
    
    test_patients = [
        # Patient 1: Likely No Cancer
        pd.DataFrame({
            'CA-125': [22],
            'PSA': [2.5],
            'CEA': [2.2],
            'HER2': [1.1],
            'CA 15-3': [16]
        }),
        
        # Patient 2: Moderate Stage (Stage II)
        pd.DataFrame({
            'CA-125': [65],
            'PSA': [6],
            'CEA': [5.5],
            'HER2': [3],
            'CA 15-3': [35]
        }),
        
        # Patient 3: Advanced Stage (Stage IV)
        pd.DataFrame({
            'CA-125': [110],
            'PSA': [11],
            'CEA': [12],
            'HER2': [5.5],
            'CA 15-3': [60]
        })
    ]
    
    print("\nIndividual Patient Predictions:")
    for i, patient in enumerate(test_patients, 1):
        predicted_stage, stage_probabilities = cancer_stage_model.predict_cancer_stage(patient)
        
        print(f"\nPatient {i}:")
        print(f"Predicted Stage: {stage_labels[predicted_stage]}")
        print("Stage Probabilities:")
        for stage, prob in zip(stage_labels, stage_probabilities):
            print(f"{stage}: {prob * 100:.2f}%")

if __name__ == "__main__":
    main()