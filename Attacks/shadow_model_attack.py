import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import copy
from ..models.model_architectures import SensorModelFactory
from ..models.train_target_models import ModelTrainer

class ShadowModelAttack:
    def __init__(self, target_model, num_shadow_models=5, shadow_data_ratio=0.8):
        self.target_model = target_model
        self.num_shadow_models = num_shadow_models
        self.shadow_data_ratio = shadow_data_ratio
        self.shadow_models = []
        self.attack_classifier = None
        self.is_fitted = False
    
    def train_shadow_models(self, shadow_dataset, model_config=None, epochs=50):
        print(f"Training {self.num_shadow_models} shadow models...")
        X_shadow, y_shadow = shadow_dataset
        
        for i in range(self.num_shadow_models):
            print(f"Training shadow model {i+1}/{self.num_shadow_models}")
            
            split_idx = int(len(X_shadow) * self.shadow_data_ratio)
            indices = np.random.permutation(len(X_shadow))
            
            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]
            
            X_train, y_train = X_shadow[train_indices], y_shadow[train_indices]
            X_test, y_test = X_shadow[test_indices], y_shadow[test_indices]
            
            if model_config is None:
                input_dim = X_train.shape[1]
                num_classes = len(np.unique(y_shadow))
                model_config = {
                    'model_type': 'mlp',
                    'input_dim': input_dim,
                    'num_classes': num_classes,
                    'hidden_dims': [128, 64]
                }
            
            shadow_model = SensorModelFactory.create_model(**model_config)
            trainer = ModelTrainer(shadow_model)
            
            training_history, _ = trainer.train_model(
                (X_train, y_train), (X_test, y_test), epochs=epochs, batch_size=64
            )
            
            shadow_model_info = {
                'model': shadow_model,
                'trainer': trainer,
                'train_indices': train_indices,
                'test_indices': test_indices,
                'val_accuracy': training_history['best_val_acc']
            }
            
            self.shadow_models.append(shadow_model_info)
        
        print("Shadow model training completed")
    
    def _extract_prediction_features(self, model, X):
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).numpy()
            
        max_confidences = np.max(probabilities, axis=1)
        prediction_entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
        predicted_classes = np.argmax(probabilities, axis=1)
        
        features = np.column_stack([
            max_confidences,
            prediction_entropy,
            predicted_classes
        ])
        
        return features, probabilities
    
    def build_attack_dataset(self, shadow_dataset):
        print("Building attack dataset...")
        X_shadow, y_shadow = shadow_dataset
        attack_features = []
        attack_labels = []
        
        for shadow_info in self.shadow_models:
            model = shadow_info['model']
            train_indices = shadow_info['train_indices']
            test_indices = shadow_info['test_indices']
            
            X_train_member = X_shadow[train_indices]
            X_test_nonmember = X_shadow[test_indices]
            
            member_features, member_probs = self._extract_prediction_features(model, X_train_member)
            nonmember_features, nonmember_probs = self._extract_prediction_features(model, X_test_nonmember)
            
            for i in range(len(member_features)):
                attack_features.append(member_features[i])
                attack_labels.append(1)
            
            for i in range(len(nonmember_features)):
                attack_features.append(nonmember_features[i])
                attack_labels.append(0)
        
        attack_features = np.array(attack_features)
        attack_labels = np.array(attack_labels)
        
        print(f"Attack dataset: {len(attack_features)} samples ({np.sum(attack_labels)} members, {len(attack_labels)-np.sum(attack_labels)} non-members)")
        
        return attack_features, attack_labels
    
    def train_attack_model(self, shadow_dataset):
        attack_features, attack_labels = self.build_attack_dataset(shadow_dataset)
        
        self.attack_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.attack_classifier.fit(attack_features, attack_labels)
        self.is_fitted = True
        
        train_accuracy = accuracy_score(attack_labels, self.attack_classifier.predict(attack_features))
        print(f"Attack model trained with accuracy: {train_accuracy:.4f}")
    
    def predict_membership(self, X_target):
        if not self.is_fitted:
            raise ValueError("Attack model not trained. Call train_attack_model first.")
        
        target_features, target_probs = self._extract_prediction_features(self.target_model, X_target)
        membership_predictions = self.attack_classifier.predict(target_features)
        membership_probs = self.attack_classifier.predict_proba(target_features)[:, 1]
        
        return membership_predictions, membership_probs, target_features
    
    def evaluate_attack(self, X_member, X_nonmember):
        if not self.is_fitted:
            raise ValueError("Attack model not trained. Call train_attack_model first.")
        
        X_combined = np.vstack([X_member, X_nonmember])
        y_true = np.array([1] * len(X_member) + [0] * len(X_nonmember))
        
        predictions, probabilities, features = self.predict_membership(X_combined)
        
        accuracy = accuracy_score(y_true, predictions)
        member_accuracy = accuracy_score(y_true[:len(X_member)], predictions[:len(X_member)])
        nonmember_accuracy = accuracy_score(y_true[len(X_member):], predictions[len(X_member):])
        
        results = {
            'overall_accuracy': accuracy,
            'member_accuracy': member_accuracy,
            'nonmember_accuracy': nonmember_accuracy,
            'attack_advantage': member_accuracy - (1 - nonmember_accuracy),
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        print(f"Attack Evaluation:")
        print(f"  Overall Accuracy: {accuracy:.4f}")
        print(f"  Member Accuracy: {member_accuracy:.4f}")
        print(f"  Non-member Accuracy: {nonmember_accuracy:.4f}")
        print(f"  Attack Advantage: {results['attack_advantage']:.4f}")
        
        return results

def test_shadow_attack():
    print("Testing Shadow Model Attack...")
    
    from ..models.model_architectures import MLPModel
    
    np.random.seed(42)
    X_demo = np.random.randn(2000, 20)
    y_demo = np.random.randint(0, 3, 2000)
    
    target_model = MLPModel(20, 3)
    
    attack = ShadowModelAttack(target_model, num_shadow_models=3)
    attack.train_shadow_models((X_demo, y_demo), epochs=5)
    attack.train_attack_model((X_demo, y_demo))
    
    X_member_test = np.random.randn(100, 20)
    X_nonmember_test = np.random.randn(100, 20)
    
    results = attack.evaluate_attack(X_member_test, X_nonmember_test)
    
    print("Shadow attack test completed!")

if __name__ == "__main__":
    test_shadow_attack()
