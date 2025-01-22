import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import skewnorm

class CancerStagePredictionModel:
    def __init__(self, num_samples=10000):
        self.num_samples = num_samples
        self.data = None
        self.X = None
        self.y = None
        self.model = None
        self.scaler = StandardScaler()
    
    def generate_advanced_synthetic_protein_data(self):
        np.random.seed(42)
        
        # More nuanced stage distribution
        stages = [0, 1, 2, 3, 4]
        stage_proportions = [0.35, 0.25, 0.2, 0.12, 0.08]
        
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
            
            # Advanced protein level distributions with skewness and more variability
            def generate_skewed_protein(mean, std, skew):
                return max(0, skewnorm.rvs(skew, loc=mean, scale=std))
            
            if stage == 0:  # No Cancer
                ca125 = generate_skewed_protein(20, 5, 2)
                psa = generate_skewed_protein(2, 1, 3)
                cea = generate_skewed_protein(2, 0.5, 2)
                her2 = generate_skewed_protein(1, 0.2, 1)
                ca153 = generate_skewed_protein(15, 3, 2)
            elif stage == 1:  # Stage I
                ca125 = generate_skewed_protein(40, 10, 1)
                psa = generate_skewed_protein(4, 2, 2)
                cea = generate_skewed_protein(4, 1, 1)
                her2 = generate_skewed_protein(2, 0.5, 1)
                ca153 = generate_skewed_protein(25, 5, 1)
            elif stage == 2:  # Stage II
                ca125 = generate_skewed_protein(60, 15, 1)
                psa = generate_skewed_protein(6, 3, 2)
                cea = generate_skewed_protein(6, 2, 1)
                her2 = generate_skewed_protein(3, 1, 1)
                ca153 = generate_skewed_protein(35, 7, 1)
            elif stage == 3:  # Stage III
                ca125 = generate_skewed_protein(80, 20, 0.5)
                psa = generate_skewed_protein(8, 4, 1)
                cea = generate_skewed_protein(8, 3, 0.5)
                her2 = generate_skewed_protein(4, 1.5, 0.5)
                ca153 = generate_skewed_protein(45, 10, 0.5)
            else:  # Stage IV
                ca125 = generate_skewed_protein(100, 25, 0.5)
                psa = generate_skewed_protein(10, 5, 1)
                cea = generate_skewed_protein(10, 4, 0.5)
                her2 = generate_skewed_protein(5, 2, 0.5)
                ca153 = generate_skewed_protein(55, 15, 0.5)
            
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
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        self.model = RandomForestClassifier(
            n_estimators=200,  # Increased from 100
            random_state=42, 
            max_depth=10,  # Slightly increased depth
            class_weight='balanced',
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.model.fit(X_train, y_train)
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=10)  # Increased from 5 to 10
        
        print("Model Performance Metrics:")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nCross-Validation Scores:")
        print(f"Mean CV Score: {cv_scores.mean():.4f}")
        print(f"CV Score Standard Deviation: {cv_scores.std():.4f}")
        
        # Feature Importance
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nFeature Importance:")
        print(feature_importance)
        
        return y_pred
    
    def predict_cancer_stage(self, new_patient_data):
        # Ensure input is in the same format as training data
        new_patient_scaled = self.scaler.transform(new_patient_data)
        
        # Predict stage and probabilities
        stage_prediction = self.model.predict(new_patient_scaled)
        stage_probabilities = self.model.predict_proba(new_patient_scaled)
        
        return stage_prediction[0], stage_probabilities[0]

def generate_more_test_patients(num_patients=10):
    np.random.seed(42)
    test_patients = []
    
    # Define ranges for each protein marker based on cancer stages
    protein_ranges = {
        'CA-125': [(10, 30), (30, 50), (50, 70), (70, 90), (90, 120)],
        'PSA': [(1, 3), (3, 5), (5, 7), (7, 9), (9, 12)],
        'CEA': [(1, 3), (3, 5), (5, 7), (7, 9), (9, 12)],
        'HER2': [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 6)],
        'CA 15-3': [(10, 20), (20, 30), (30, 40), (40, 50), (50, 65)]
    }
    
    for _ in range(num_patients):
        # Randomly select a stage for this patient
        stage = np.random.choice([0, 1, 2, 3, 4], p=[0.35, 0.25, 0.2, 0.12, 0.08])
        
        # Generate protein marker values for the selected stage
        patient_data = {}
        for protein, ranges in protein_ranges.items():
            # Remove list wrapping, use direct uniform sampling
            min_val, max_val = ranges[stage]
            patient_data[protein] = np.random.uniform(min_val, max_val)
        
        test_patients.append(pd.DataFrame([patient_data]))
    
    return test_patients

def main():
    # Create and prepare model
    cancer_stage_model = CancerStagePredictionModel(num_samples=10000)
    synthetic_data = cancer_stage_model.generate_advanced_synthetic_protein_data()
    
    # Prepare and train model
    X_train, X_test, y_train, y_test = cancer_stage_model.prepare_data()
    trained_model = cancer_stage_model.train_model(X_train, y_train)
    
    # Evaluate model
    cancer_stage_model.evaluate_model(X_test, y_test)
    
    # Generate more test patients
    stage_labels = ['No Cancer', 'Stage I', 'Stage II', 'Stage III', 'Stage IV']
    test_patients = generate_more_test_patients(num_patients=10)
    
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