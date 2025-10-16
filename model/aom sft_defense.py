import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

class AOMDefense:
    def __init__(self, model, validation_data, percentile=90, top_k=3, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.percentile = percentile
        self.top_k = top_k
        self.confidence_threshold = None
        self._compute_threshold(validation_data)
    
    def _compute_threshold(self, validation_data):
        print("Computing AOM confidence threshold...")
        X_val, y_val = validation_data
        
        self.model.eval()
        all_confidences = []
        
        with torch.no_grad():
            batch_size = 64
            for i in range(0, len(X_val), batch_size):
                batch_x = X_val[i:i+batch_size]
                x_tensor = torch.FloatTensor(batch_x).to(self.device)
                outputs = self.model(x_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidences, _ = torch.max(probabilities, 1)
                all_confidences.extend(confidences.cpu().numpy())
        
        self.confidence_threshold = np.percentile(all_confidences, self.percentile)
        print(f"AOM threshold ({self.percentile}th percentile): {self.confidence_threshold:.4f}")
    
    def _mask_predictions(self, probabilities):
        batch_size, num_classes = probabilities.shape
        
        masked_probs = torch.zeros_like(probabilities)
        
        for i in range(batch_size):
            sample_probs = probabilities[i]
            max_confidence = torch.max(sample_probs)
            
            if max_confidence > self.confidence_threshold:
                _, top_indices = torch.topk(sample_probs, self.top_k)
                masked_probs[i, top_indices] = 1.0 / self.top_k
            else:
                masked_probs[i] = sample_probs
        
        return masked_probs
    
    def defend(self, X):
        self.model.eval()
        
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(x_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            protected_probs = self._mask_predictions(probabilities)
            
            protected_confidences, protected_predictions = torch.max(protected_probs, 1)
            original_confidences, original_predictions = torch.max(probabilities, 1)
        
        defense_stats = {
            'original_confidences': original_confidences.cpu().numpy(),
            'protected_confidences': protected_confidences.cpu().numpy(),
            'masking_rate': np.mean(protected_confidences.cpu().numpy() != original_confidences.cpu().numpy()),
            'avg_confidence_reduction': np.mean(original_confidences.cpu().numpy() - protected_confidences.cpu().numpy())
        }
        
        return protected_probs.cpu().numpy(), defense_stats
    
    def get_predictions(self, X, apply_defense=True):
        if apply_defense:
            protected_probs, stats = self.defend(X)
            predictions = np.argmax(protected_probs, axis=1)
            return predictions, protected_probs, stats
        else:
            self.model.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(X).to(self.device)
                outputs = self.model(x_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, 1)
            return predictions.cpu().numpy(), probabilities.cpu().numpy(), {}
    
    def evaluate_defense_impact(self, test_data):
        X_test, y_test = test_data
        
        original_predictions, original_probs, _ = self.get_predictions(X_test, apply_defense=False)
        protected_predictions, protected_probs, defense_stats = self.get_predictions(X_test, apply_defense=True)
        
        original_accuracy = np.mean(original_predictions == y_test)
        protected_accuracy = np.mean(protected_predictions == y_test)
        
        confidence_change = np.mean(defense_stats['original_confidences']) - np.mean(defense_stats['protected_confidences'])
        
        results = {
            'original_accuracy': original_accuracy,
            'protected_accuracy': protected_accuracy,
            'accuracy_change': protected_accuracy - original_accuracy,
            'masking_rate': defense_stats['masking_rate'],
            'confidence_reduction': confidence_change,
            'avg_original_confidence': np.mean(defense_stats['original_confidences']),
            'avg_protected_confidence': np.mean(defense_stats['protected_confidences'])
        }
        
        return results
    
    def set_parameters(self, percentile=None, top_k=None):
        if percentile is not None:
            self.percentile = percentile
        if top_k is not None:
            self.top_k = top_k
    
    def get_config(self):
        return {
            'percentile': self.percentile,
            'top_k': self.top_k,
            'confidence_threshold': self.confidence_threshold
        }


class AdaptiveAOMDefense(AOMDefense):
    def __init__(self, model, validation_data, min_percentile=80, max_percentile=95, 
                 adaptation_step=2, top_k=3, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.adaptation_step = adaptation_step
        self.current_percentile = min_percentile
        self.attack_detection_count = 0
        self.adaptation_history = []
        
        super().__init__(model, validation_data, self.current_percentile, top_k, device)
    
    def detect_attack_pattern(self, query_batch, threshold=0.8):
        if len(query_batch) < 10:
            return False
        
        with torch.no_grad():
            x_tensor = torch.FloatTensor(query_batch).to(self.device)
            outputs = self.model(x_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidences, _ = torch.max(probabilities, 1)
        
        high_confidence_ratio = np.mean(confidences.cpu().numpy() > threshold)
        
        return high_confidence_ratio > 0.7
    
    def adapt_defense(self, recent_queries):
        if self.detect_attack_pattern(recent_queries):
            self.attack_detection_count += 1
            
            if self.attack_detection_count >= 3 and self.current_percentile < self.max_percentile:
                self.current_percentile = min(self.current_percentile + self.adaptation_step, self.max_percentile)
                print(f"Attack detected! Increasing AOM percentile to {self.current_percentile}")
                self.attack_detection_count = 0
                
                self.percentile = self.current_percentile
                self.confidence_threshold = None
        else:
            if self.attack_detection_count > 0:
                self.attack_detection_count -= 1
            
            if self.attack_detection_count == 0 and self.current_percentile > self.min_percentile:
                self.current_percentile = max(self.current_percentile - self.adaptation_step, self.min_percentile)
                print(f"Reducing AOM percentile to {self.current_percentile}")
                
                self.percentile = self.current_percentile
                self.confidence_threshold = None


def test_aom_defense():
    print("Testing AOM Defense...")
    
    class SimpleModel(torch.nn.Module):
        def __init__(self, input_dim=10, num_classes=3):
            super(SimpleModel, self).__init__()
            self.fc = torch.nn.Linear(input_dim, num_classes)
        
        def forward(self, x):
            return self.fc(x)
    
    model = SimpleModel()
    
    np.random.seed(42)
    X_val = np.random.randn(100, 10)
    y_val = np.random.randint(0, 3, 100)
    
    X_test = np.random.randn(50, 10)
    y_test = np.random.randint(0, 3, 50)
    
    aom_defense = AOMDefense(model, (X_val, y_val), percentile=85, top_k=2)
    
    protected_probs, stats = aom_defense.defend(X_test)
    
    print(f"Masking rate: {stats['masking_rate']:.3f}")
    print(f"Confidence reduction: {stats['avg_confidence_reduction']:.4f}")
    print(f"Original avg confidence: {np.mean(stats['original_confidences']):.4f}")
    print(f"Protected avg confidence: {np.mean(stats['protected_confidences']):.4f}")
    
    config = aom_defense.get_config()
    print(f"AOM configuration: {config}")
    
    results = aom_defense.evaluate_defense_impact((X_test, y_test))
    print(f"Accuracy change: {results['accuracy_change']:.4f}")
    
    print("AOM defense test completed!")


if __name__ == "__main__":
    test_aom_defense()
