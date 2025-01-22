import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_recall_fscore_support

class AdvancedCancerRiskModel:
    def __init__(self, num_samples=10000):
        self.num_samples = num_samples
        self.model = None
        self.preprocessor = None
        self.data = None
    
    def generate_comprehensive_synthetic_data(self):
        # Ensure there's a proper implementation of the method
        np.random.seed(42)
        
        # Initialize data dictionary
        data = {
            'CA-125': [], 'PSA': [], 'CEA': [], 'HER2': [], 
            'CA 15-3': [], 'p53_protein': [],
            
            'smoking_history': [], 'alcohol_consumption': [], 
            'radiation_exposure': [], 'family_cancer_history': [],
            
            'p53_mutation': [], 'genetic_predisposition_score': [],
            
            'age': [], 'bmi': [], 'exercise_frequency': [],
            
            'CancerStage': []
        }
        
        # Define stages and their probabilities
        stages = [0, 1, 2, 3, 4]
        stage_proportions = [0.2, 0.3, 0.25, 0.15, 0.1]
        
        # Generate synthetic data for each sample
        for _ in range(self.num_samples):
            # Select stage based on probabilities
            stage = np.random.choice(stages, p=stage_proportions)
            
            # Scale factors for different stages
            stage_scale = {0: 1.0, 1: 1.5, 2: 2.0, 3: 3.0, 4: 4.5}
            scale = stage_scale[stage]
            
            # Generate features with stage-dependent variations
            data['CA-125'].append(max(0, np.random.normal(50 * scale, 10)))
            data['PSA'].append(max(0, np.random.normal(3 * scale, 2)))
            data['CEA'].append(max(0, np.random.normal(3 * scale, 1.5)))
            data['HER2'].append(max(0, np.random.normal(1.5 * scale, 0.5)))
            data['CA 15-3'].append(max(0, np.random.normal(25 * scale, 5)))
            data['p53_protein'].append(max(0, np.random.normal(20 * scale, 5)))
            
            # Categorical feature generation with stage-dependent probabilities
            def stage_adjusted_choice(categories, base_probs):
                # Adjust probabilities based on stage
                adjusted_probs = [
                    p * (1 - stage * 0.1) if i == 0 else 
                    p * (1 + stage * 0.1) 
                    for i, p in enumerate(base_probs)
                ]
                # Normalize probabilities
                total = sum(adjusted_probs)
                return np.random.choice(categories, p=[p/total for p in adjusted_probs])
            
            # Generate categorical features
            data['smoking_history'].append(stage_adjusted_choice(
                ['never', 'occasional', 'heavy'], 
                [0.5, 0.3, 0.2]
            ))
            
            data['alcohol_consumption'].append(stage_adjusted_choice(
                ['low', 'moderate', 'high'], 
                [0.4, 0.4, 0.2]
            ))
            
            data['radiation_exposure'].append(stage_adjusted_choice(
                ['none', 'low', 'moderate', 'high'], 
                [0.7, 0.2, 0.07, 0.03]
            ))
            
            # Binary and continuous feature generation
            data['family_cancer_history'].append(
                1 if np.random.random() < min(0.2, stage * 0.05) else 0
            )
            
            data['p53_mutation'].append(stage_adjusted_choice(
                ['negative', 'positive'], 
                [0.9, 0.1]
            ))
            
            # Continuous risk factors
            data['genetic_predisposition_score'].append(
                max(0, min(1, np.random.normal(0.3 + stage * 0.1, 0.2)))
            )
            
            # Age and health factors
            data['age'].append(max(18, min(85, np.random.normal(50 + stage * 5, 10))))
            data['bmi'].append(max(18, min(40, np.random.normal(26 + stage * 1, 3))))
            
            data['exercise_frequency'].append(stage_adjusted_choice(
                ['active', 'moderate', 'sedentary'], 
                [0.4, 0.4, 0.2]
            ))
            
            # Assign the stage
            data['CancerStage'].append(stage)
        
        # Create and return DataFrame
        self.data = pd.DataFrame(data)
        return self.data
    
    def create_advanced_preprocessing_pipeline(self, X):
        numeric_features = ['CA-125', 'PSA', 'CEA', 'HER2', 'CA 15-3', 
                            'p53_protein', 'age', 'bmi', 
                            'genetic_predisposition_score']
        categorical_features = ['smoking_history', 'alcohol_consumption', 
                                'radiation_exposure', 'family_cancer_history', 
                                'p53_mutation', 'exercise_frequency']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        return preprocessor
    
    def build_ensemble_model(self, X):
        self.preprocessor = self.create_advanced_preprocessing_pipeline(X)
        
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', VotingClassifier([
                ('rf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')),
                ('gb', GradientBoostingClassifier(n_estimators=150, random_state=42)),
                ('svm', SVC(probability=True, random_state=42, class_weight='balanced'))
            ], voting='soft'))
        ])
        
        return self.model
    
    def train_and_evaluate_model(self, test_size=0.2):
        X = self.data.drop('CancerStage', axis=1)
        y = self.data['CancerStage']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        model = self.build_ensemble_model(X)
        model.fit(X_train, y_train)
        
        # Performance evaluation
        y_pred = model.predict(X_test)
        
        print("Model Performance Metrics:")
        print(classification_report(y_test, y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=10))
        print("\nCross-Validation Scores:")
        print(f"Mean CV Score: {cv_scores.mean():.4f}")
        print(f"CV Score Standard Deviation: {cv_scores.std():.4f}")
        
        return model
    
    def genetic_risk_assessment(self, patient_data):
        """
        Enhanced genetic risk assessment with more nuanced calculation
        """
        p53_mutation = patient_data['p53_mutation']
        p53_protein_level = patient_data['p53_protein']
        family_history = patient_data['family_cancer_history']
        genetic_predisposition_score = patient_data['genetic_predisposition_score']
        age = patient_data['age']
        
        def _compute_genetic_risk(mutation, protein_level, family_hist, pred_score, patient_age):
            # Base risk factors
            base_risk = 0.5 if mutation == 'positive' else 0.1
            
            # Protein level modifier
            protein_modifier = min(protein_level / 100, 1)
            
            # Family history modifier
            family_hist_modifier = 1.5 if family_hist == 1 else 1.0
            
            # Age modifier (risk increases with age)
            age_modifier = 1 + max(0, (patient_age - 50) / 100)
            
            # Genetic predisposition score modifier
            pred_score_modifier = 1 + (pred_score * 2)
            
            # Combine modifiers
            genetic_risk = base_risk * protein_modifier * family_hist_modifier * age_modifier * pred_score_modifier
            
            return min(max(genetic_risk, 0), 1)  # Clamp between 0 and 1
        
        genetic_risk = _compute_genetic_risk(
            p53_mutation, 
            p53_protein_level, 
            family_history, 
            genetic_predisposition_score,
            age
        )
        
        return {
            'p53_mutation_status': p53_mutation,
            'p53_protein_level': p53_protein_level,
            'genetic_cancer_risk_score': genetic_risk
        }
    
    def interpret_genetic_risk(self, genetic_risk_score):
        """
        Interpret the genetic cancer risk score and provide a detailed explanation.
        """
        risk_interpretation = {
            'low': {
                'category': 'Low Risk',
                'description': 'Your genetic risk for cancer appears to be low. This suggests a lower likelihood of inherited cancer predisposition.',
                'recommendations': [
                    'Maintain regular health check-ups',
                    'Follow standard cancer screening guidelines',
                    'Maintain a healthy lifestyle',
                    'Consider discussing family history with a healthcare professional'
                ]
            },
            'medium': {
                'category': 'Moderate Risk',
                'description': 'Your genetic risk for cancer shows moderate potential. This indicates some genetic factors that may slightly increase cancer susceptibility.',
                'recommendations': [
                    'Schedule more frequent cancer screenings',
                    'Consult with a genetic counselor',
                    'Adopt proactive health monitoring',
                    'Consider comprehensive genetic testing',
                    'Maintain a health-conscious lifestyle'
                ]
            },
            'high': {
                'category': 'High Risk',
                'description': 'Your genetic risk for cancer is elevated. This suggests a significant genetic predisposition that warrants careful medical attention.',
                'recommendations': [
                    'Immediate consultation with a genetic specialist',
                    'Comprehensive and frequent cancer screenings',
                    'Consider preventive genetic counseling',
                    'Discuss potential preventive interventions',
                    'Develop a personalized cancer prevention strategy'
                ]
            }
        }
        
        # Categorize risk based on score
        if genetic_risk_score < 0.2:
            risk_category = 'low'
        elif genetic_risk_score < 0.5:
            risk_category = 'medium'
        else:
            risk_category = 'high'
        
        return {
            'risk_category': risk_interpretation[risk_category]['category'],
            'risk_score': genetic_risk_score,
            'description': risk_interpretation[risk_category]['description'],
            'recommendations': risk_interpretation[risk_category]['recommendations']
        }
    
    def predict_patient_risk(self, patient_data):
        # Ensure patient data is a DataFrame and has the same columns as training data
        patient_df = pd.DataFrame([patient_data])
        
        # Check if all expected columns are present
        expected_columns = self.data.columns.drop('CancerStage').tolist()
        for col in expected_columns:
            if col not in patient_df.columns:
                raise ValueError(f"Missing column in patient data: {col}")
        
        # Predict cancer stage and probabilities
        predicted_stage = self.model.predict(patient_df)[0]
        stage_probabilities = self.model.predict_proba(patient_df)[0]
        
        # Genetic risk assessment
        genetic_risk = self.genetic_risk_assessment(patient_data)
        
        # Add risk interpretation
        risk_interpretation = self.interpret_genetic_risk(genetic_risk['genetic_cancer_risk_score'])
        
        return {
            'predicted_stage': predicted_stage,
            'stage_probabilities': stage_probabilities,
            'genetic_risk': {
                **genetic_risk,
                'risk_interpretation': risk_interpretation
            }
        }

def main(num_patients=5):
    # Initialize and generate synthetic data
    cancer_risk_model = AdvancedCancerRiskModel(num_samples=10000)
    comprehensive_data = cancer_risk_model.generate_comprehensive_synthetic_data()
    
    # Train and evaluate model
    trained_model = cancer_risk_model.train_and_evaluate_model()
    
    # Generate multiple example patients with significant variation
    def generate_varied_patients(num_patients):
        patients = []
        patient_variations = [
            # Varied genetic risk scenarios
            {
                'mutation_prob': 0.1,  # Low mutation probability
                'protein_level_range': (10, 30),
                'family_history_prob': 0.2,
                'predisposition_range': (0.1, 0.3)
            },
            {
                'mutation_prob': 0.3,  # Moderate mutation probability
                'protein_level_range': (20, 50),
                'family_history_prob': 0.5,
                'predisposition_range': (0.3, 0.6)
            },
            {
                'mutation_prob': 0.6,  # High mutation probability
                'protein_level_range': (40, 70),
                'family_history_prob': 0.8,
                'predisposition_range': (0.6, 0.9)
            }
        ]
        
        for i in range(num_patients):
            # Randomly select a risk variation profile
            variation = patient_variations[i % len(patient_variations)]
            
            patient = {
                'CA-125': np.random.normal(45, 10),
                'PSA': np.random.normal(3.5, 2),
                'CEA': np.random.normal(3.2, 1.5),
                'HER2': np.random.normal(1.8, 0.5),
                'CA 15-3': np.random.normal(28, 5),
                
                # Varied genetic risk factors
                'p53_mutation': 'positive' if np.random.random() < variation['mutation_prob'] else 'negative',
                'p53_protein': max(0, np.random.uniform(variation['protein_level_range'][0], variation['protein_level_range'][1])),
                'family_cancer_history': 1 if np.random.random() < variation['family_history_prob'] else 0,
                'genetic_predisposition_score': np.random.uniform(variation['predisposition_range'][0], variation['predisposition_range'][1]),
                
                # Other varied parameters
                'smoking_history': np.random.choice(['never', 'occasional', 'heavy']),
                'alcohol_consumption': np.random.choice(['low', 'moderate', 'high']),
                'radiation_exposure': np.random.choice(['none', 'low', 'moderate', 'high']),
                'age': max(30, min(80, np.random.normal(52, 10))),
                'bmi': max(18, np.random.normal(26, 3)),
                'exercise_frequency': np.random.choice(['sedentary', 'moderate', 'active'])
            }
            
            patients.append(patient)
        
        return patients
    
    # Generate varied patients
    example_patients = generate_varied_patients(num_patients)
    
    print(f"\n{'='*50}")
    print(f"MULTI-PATIENT CANCER RISK ASSESSMENT (Patients: {num_patients})")
    print(f"{'='*50}")
    
    # Predict risk for multiple patients
    for i, patient in enumerate(example_patients, 1):
        print(f"\nPatient {i} Risk Assessment:")
        patient_risk = cancer_risk_model.predict_patient_risk(patient)
        
        print(f"Predicted Cancer Stage: {patient_risk['predicted_stage']}")
        print("Stage Probabilities:")
        for stage, prob in enumerate(patient_risk['stage_probabilities']):
            print(f"Stage {stage}: {prob*100:.2f}%")
        
        print("\nGenetic Risk Assessment:")
        genetic_risk = patient_risk['genetic_risk']
        print(f"P53 Mutation Status: {genetic_risk['p53_mutation_status']}")
        print(f"P53 Protein Level: {genetic_risk['p53_protein_level']}")
        print(f"Genetic Cancer Risk Score: {genetic_risk['genetic_cancer_risk_score']:.2f}")
        
        # Detailed risk interpretation
        risk_info = genetic_risk['risk_interpretation']
        print("\nRisk Interpretation:")
        print(f"Risk Category: {risk_info['risk_category']}")
        print(f"Risk Score: {risk_info['risk_score']:.2f}")
        print("\nDescription:")
        print(risk_info['description'])
        
        print("\nRecommendations:")
        for rec in risk_info['recommendations']:
            print(f"- {rec}")
        
        print(f"\n{'*'*50}")

if __name__ == "__main__":
    # Default to 5 patients, but allows customization
    main(num_patients=5)