import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class LabelOnlyAttack:
    def __init__(self, target_model, num_perturbations=10, perturbation_scale=0.1):
        self.target_model = target_model
        self.num_perturbations = num_perturbations
        self.perturbation_scale = perturbation_scale
        self.attack_classifier = None
        self.is_fitted = False
    
    def _perturb_samples(self, X):
        perturbations = []
        for _ in range(self.num_perturbations):
            noise = np.random.normal(0, self.perturbation_scale, X.shape)
            perturbed_X = X + noise
            perturbations.append(perturbed_X)
        return np.array(perturbations)
    
    def _get_model_predictions(self, X):
        self.target_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.target_model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).numpy()
        return predictions
    
    def _compute_label_stability(self, X):
        perturbations = self._perturb_samples(X)
        original_predictions = self._get_model_predictions(X)
        
        stability_scores = []
        
        for i in range(len(X)):
            sample_perturbations = perturbations[:, i]
            perturbed_predictions = self._get_model_predictions(sample_perturbations)
            
            consistent_predictions = np.sum(perturbed_predictions == original_predictions[i])
            stability_score = consistent_predictions / self.num_perturbations
            stability_scores.append(stability_score)
        
        return np.array(stability_scores)
    
    def build_attack_dataset(self, member_data, nonmember_data):
        print("Building label-only attack dataset...")
        
        X_member, y_member = member_data
        X_nonmember, y_nonmember = nonmember_data
        
        print("Computing member stability scores...")
        member_stability = self._compute_label_stability(X_member)
        
        print("Computing non-member stability scores...")
        nonmember_stability = self._compute_label_stability(X_nonmember)
        
        stability_features = np.concatenate([
            member_stability.reshape(-1, 1),
            nonmember_stability.reshape(-1, 1)
        ])
        
        attack_labels = np.concatenate([
            np.ones(len(member_stability)),
            np.zeros(len(nonmember_stability))
        ])
        
        print(f"Attack dataset: {len(stability_features)} samples")
        print(f"Member avg stability: {np.mean(member_stability):.4f}")
        print(f"Non-member avg stability: {np.mean(nonmember_stability):.4f}")
        
        return stability_features, attack_labels
    
    def train_attack_model(self, member_data, nonmember_data):
        attack_features, attack_labels = self.build_attack_dataset(member_data, nonmember_data)
        
        self.attack_classifier = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        
        self.attack_classifier.fit(attack_features, attack_labels)
        self.is_fitted = True
        
        train_predictions = self.attack_classifier.predict(attack_features)
        train_accuracy = accuracy_score(attack_labels, train_predictions)
        
        print(f"Label-only attack model trained with accuracy: {train_accuracy:.4f}")
    
    def predict_membership(self, X_target):
        if not self.is_fitted:
            raise ValueError("Attack model not trained. Call train_attack_model first.")
        
        stability_scores = self._compute_label_stability(X_target)
        features = stability_scores.reshape(-1, 1)
        
        membership_predictions = self.attack_classifier.predict(features)
        membership_probs = self.attack_classifier.predict_proba(features)[:, 1]
        
        return membership_predictions, membership_probs, stability_scores
    
    def evaluate_attack(self, X_member, X_nonmember):
        if not self.is_fitted:
            raise ValueError("Attack model not trained. Call train_attack_model first.")
        
        X_combined = np.vstack([X_member, X_nonmember])
        y_true = np.array([1] * len(X_member) + [0] * len(X_nonmember))
        
        predictions, probabilities, stability_scores = self.predict_membership(X_combined)
        
        accuracy = accuracy_score(y_true, predictions)
        member_accuracy = accuracy_score(y_true[:len(X_member)], predictions[:len(X_member)])
        nonmember_accuracy = accuracy_score(y_true[len(X_member):], predictions[len(X_member):])
        
        member_stability = np.mean(stability_scores[:len(X_member)])
        nonmember_stability = np.mean(stability_scores[len(X_member):])
        
        results = {
            'overall_accuracy': accuracy,
            'member_accuracy': member_accuracy,
            'nonmember_accuracy': nonmember_accuracy,
            'member_stability': member_stability,
            'nonmember_stability': nonmember_stability,
            'stability_gap': member_stability - nonmember_stability,
            'predictions': predictions,
            'probabilities': probabilities,
            'stability_scores': stability_scores
        }
        
        print(f"Label-Only Attack Evaluation:")
        print(f"  Overall Accuracy: {accuracy:.4f}")
        print(f"  Member Accuracy: {member_accuracy:.4f}")
        print(f"  Non-member Accuracy: {nonmember_accuracy:.4f}")
        print(f"  Member Stability: {member_stability:.4f}")
        print(f"  Non-member Stability: {nonmember_stability:.4f}")
        print(f"  Stability Gap: {results['stability_gap']:.4f}")
        
        return results

class AdvancedLabelOnlyAttack(LabelOnlyAttack):
    def __init__(self, target_model, num_perturbations=15, perturbation_scales=[0.05, 0.1, 0.15]):
        super().__init__(target_model, num_perturbations, perturbation_scales[0])
        self.perturbation_scales = perturbation_scales
    
    def _compute_multi_scale_stability(self, X):
        multi_scale_features = []
        
        for scale in self.perturbation_scales:
            self.perturbation_scale = scale
            stability = self._compute_label_stability(X)
            multi_scale_features.append(stability)
        
        features = np.column_stack(multi_scale_features)
        avg_stability = np.mean(features, axis=1)
        std_stability = np.std(features, axis=1)
        
        final_features = np.column_stack([
            avg_stability,
            std_stability,
            features
        ])
        
        return final_features
    
    def build_attack_dataset(self, member_data, nonmember_data):
        print("Building advanced label-only attack dataset...")
        
        X_member, y_member = member_data
        X_nonmember, y_nonmember = nonmember_data
        
        print("Computing multi-scale member stability...")
        member_features = self._compute_multi_scale_stability(X_member)
        
        print("Computing multi-scale non-member stability...")
        nonmember_features = self._compute_multi_scale_stability(X_nonmember)
        
        attack_features = np.vstack([member_features, nonmember_features])
        attack_labels = np.concatenate([
            np.ones(len(member_features)),
            np.zeros(len(nonmember_features))
        ])
        
        print(f"Advanced attack dataset: {len(attack_features)} samples with {attack_features.shape[1]} features")
        
        return attack_features, attack_labels

def test_label_only_attack():
    print("Testing Label-Only Attack...")
    
    from ..models.model_architectures import MLPModel
    
    np.random.seed(42)
    X_member = np.random.randn(200, 10)
    X_nonmember = np.random.randn(200, 10)
    
    target_model = MLPModel(10, 3)
    
    attack = LabelOnlyAttack(target_model, num_perturbations=5)
    attack.train_attack_model((X_member, None), (X_nonmember, None))
    
    X_member_test = np.random.randn(50, 10)
    X_nonmember_test = np.random.randn(50, 10)
    
    results = attack.evaluate_attack(X_member_test, X_nonmember_test)
    
    print("Label-only attack test completed!")

if __name__ == "__main__":
    test_label_only_attack()
