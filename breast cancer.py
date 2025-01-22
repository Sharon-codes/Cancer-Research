import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
from scipy.stats import skewnorm
import joblib

class AdvancedCancerPredictionModel:
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.data = None
        self.X = None
        self.y = None
        self.model = None
        self.scaler = RobustScaler()
        
        # Medical literature-based realistic biomarker correlations
        self.biomarker_correlations = {
            'CA-125': {
                'stage_mean': [20, 40, 60, 80, 100],
                'stage_std': [5, 8, 12, 15, 20],
                'skewness': [2, 1.5, 1, 0.5, 0.3]
            },
            'PSA': {
                'stage_mean': [2, 4, 6, 8, 10],
                'stage_std': [1, 2, 3, 4, 5],
                'skewness': [3, 2, 1, 0.5, 0.2]
            },
            'CEA': {
                'stage_mean': [2, 4, 6, 8, 10],
                'stage_std': [0.5, 1, 2, 3, 4],
                'skewness': [2, 1.5, 1, 0.5, 0.3]
            },
            'HER2': {
                'stage_mean': [1, 2, 3, 4, 5],
                'stage_std': [0.2, 0.5, 1, 1.5, 2],
                'skewness': [1, 0.8, 0.5, 0.3, 0.1]
            },
            'CA 15-3': {
                'stage_mean': [15, 25, 35, 45, 55],
                'stage_std': [3, 5, 7, 10, 15],
                'skewness': [2, 1.5, 1, 0.5, 0.3]
            }
        }
    
    def generate_ultra_realistic_synthetic_data(self):
        """
        Generate highly sophisticated synthetic data mimicking real-world medical distributions
        """
        np.random.seed(42)
        
        # More nuanced stage distribution reflecting real-world cancer prevalence
        stages = [0, 1, 2, 3, 4]
        stage_proportions = [0.4, 0.3, 0.15, 0.10, 0.05]
        
        data = {
            'CA-125': [],
            'PSA': [],
            'CEA': [],
            'HER2': [],
            'CA 15-3': [],
            'CancerStage': []
        }
        
        for _ in range(self.num_samples):
            # Randomly select stage with realistic distribution
            stage = np.random.choice(stages, p=stage_proportions)
            
            patient_data = {}
            for protein, config in self.biomarker_correlations.items():
                # Generate protein levels with stage-specific distributions
                mean = config['stage_mean'][stage]
                std = config['stage_std'][stage]
                skew = config['skewness'][stage]
                
                # Use skew-normal distribution for more realistic data generation
                protein_value = max(0, skewnorm.rvs(
                    skew, 
                    loc=mean, 
                    scale=std
                ))
                
                patient_data[protein] = protein_value
            
            # Add data to respective lists
            for protein, value in patient_data.items():
                data[protein].append(value)
            data['CancerStage'].append(stage)
        
        self.data = pd.DataFrame(data)
        return self.data
    
    def prepare_advanced_data(self, test_size=0.2):
        """
        Advanced data preparation with stratified splitting and robust scaling
        """
        self.X = self.data.drop('CancerStage', axis=1)
        self.y = self.data['CancerStage']
        
        # Stratified split to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=42, 
            stratify=self.y
        )
        
        # Robust scaling to handle outliers
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def create_advanced_ensemble_model(self):
        """
        Create a sophisticated ensemble model with stacking
        """
        # Base classifiers
        base_classifiers = [
            ('rf', RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced'
            )),
            ('svm', SVC(
                kernel='rbf', 
                probability=True, 
                class_weight='balanced',
                C=1.0, 
                gamma='scale'
            ))
        ]
        
        # Meta-classifier
        meta_classifier = LogisticRegression(
            multi_class='ovr', 
            solver='lbfgs', 
            max_iter=1000
        )
        
        # Stacking Classifier
        stacking_classifier = StackingClassifier(
            estimators=base_classifiers,
            final_estimator=meta_classifier,
            cv=5,
            stack_method='predict_proba'
        )
        
        return stacking_classifier
    
    def train_advanced_model(self, X_train, y_train):
        """
        Train the advanced ensemble model
        """
        self.model = self.create_advanced_ensemble_model()
        self.model.fit(X_train, y_train)
        return self.model
    
    def evaluate_model_performance(self, X_test, y_test):
        """
        Comprehensive model performance evaluation
        """
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Performance Metrics
        print("\n--- Detailed Model Performance Metrics ---")
        print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
        print("Precision (Weighted): {:.2f}%".format(precision_score(y_test, y_pred, average='weighted') * 100))
        print("Recall (Weighted): {:.2f}%".format(recall_score(y_test, y_pred, average='weighted') * 100))
        print("F1 Score (Weighted): {:.2f}%".format(f1_score(y_test, y_pred, average='weighted') * 100))
        
        # Detailed Classification Report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Stage')
        plt.xlabel('Predicted Stage')
        plt.tight_layout()
        plt.show()
        
        # Cross-Validation
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=cv, scoring='accuracy')
        
        print("\nCross-Validation Results:")
        print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
        print(f"CV Accuracy Standard Deviation: {cv_scores.std():.4f}")
        
        return y_pred
    
    def predict_cancer_stage(self, new_patient_data):
        """
        Predict cancer stage with probabilities
        """
        # Ensure input is scaled the same way as training data
        new_patient_scaled = self.scaler.transform(new_patient_data)
        
        # Predict stage and probabilities
        stage_prediction = self.model.predict(new_patient_scaled)
        stage_probabilities = self.model.predict_proba(new_patient_scaled)
        
        return stage_prediction[0], stage_probabilities[0]
    
    def save_model(self, filename='cancer_prediction_model.joblib'):
        """
        Save trained model for future use
        """
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, filename)
    
    def load_model(self, filename='cancer_prediction_model.joblib'):
        """
        Load pre-trained model
        """
        saved_data = joblib.load(filename)
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']

def generate_advanced_test_patients(num_patients=5):
    """
    Generate more sophisticated test patients
    """
    np.random.seed(42)
    
    # Define more nuanced ranges for each protein marker
    protein_ranges = {
        'CA-125': [(10, 40), (30, 60), (50, 80), (70, 100), (90, 120)],
        'PSA': [(1, 4), (3, 6), (5, 8), (7, 10), (9, 12)],
        'CEA': [(1, 3), (3, 6), (5, 8), (7, 10), (9, 12)],
        'HER2': [(0.5, 2), (1.5, 3), (2.5, 4), (3.5, 5), (4.5, 6)],
        'CA 15-3': [(10, 25), (20, 35), (30, 45), (40, 55), (50, 65)]
    }
    
    test_patients = []
    for _ in range(num_patients):
        # Weighted stage selection to mirror real-world distribution
        stage = np.random.choice([0, 1, 2, 3, 4], p=[0.4, 0.3, 0.15, 0.10, 0.05])
        
        patient_data = {}
        for protein, ranges in protein_ranges.items():
            min_val, max_val = ranges[stage]
            patient_data[protein] = np.random.uniform(min_val, max_val)
        
        test_patients.append(pd.DataFrame([patient_data]))
    
    return test_patients

def main():
    # Create advanced cancer prediction model
    cancer_model = AdvancedCancerPredictionModel(num_samples=1000)
    
    # Generate ultra-realistic synthetic data
    synthetic_data = cancer_model.generate_ultra_realistic_synthetic_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test = cancer_model.prepare_advanced_data()
    
    # Train advanced model
    trained_model = cancer_model.train_advanced_model(X_train, y_train)
    
    # Evaluate model performance
    y_pred = cancer_model.evaluate_model_performance(X_test, y_test)
    
    # Save the model
    cancer_model.save_model()
    
    # Generate and predict for test patients
    stage_labels = ['No Cancer', 'Stage I', 'Stage II', 'Stage III', 'Stage IV']
    test_patients = generate_advanced_test_patients(num_patients=10)
    
    print("\nIndividual Patient Predictions:")
    for i, patient in enumerate(test_patients, 1):
        predicted_stage, stage_probabilities = cancer_model.predict_cancer_stage(patient)
        
        print(f"\nPatient {i}:")
        print("Biomarkers:")
        for marker, value in patient.to_dict('records')[0].items():
            print(f"{marker}: {value:.2f}")
        
        print(f"\nPredicted Stage: {stage_labels[predicted_stage]}")
        print("Stage Probabilities:")
        for stage, prob in zip(stage_labels, stage_probabilities):
            print(f"{stage}: {prob * 100:.2f}%")

if __name__ == "__main__":
    main()