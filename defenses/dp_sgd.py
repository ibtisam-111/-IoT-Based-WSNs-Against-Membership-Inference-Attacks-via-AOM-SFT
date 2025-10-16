import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math

class DPSGD:
    def __init__(self, model, noise_multiplier=1.1, max_grad_norm=1.0, delta=1e-5):
        self.model = model
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self.device = next(model.parameters()).device
        
    def compute_epsilon(self, epochs, batch_size, dataset_size):
        sampling_prob = batch_size / dataset_size
        steps = epochs * math.ceil(dataset_size / batch_size)
        return self._compute_dp_sgd_epsilon(steps, sampling_prob)
    
    def _compute_dp_sgd_epsilon(self, steps, sampling_prob):
        if self.noise_multiplier == 0:
            return float('inf')
        
        from opacus import privacy_analysis
        # Simplified epsilon calculation
        orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
        epsilon, _ = privacy_analysis.compute_rdp(
            q=sampling_prob,
            noise_multiplier=self.noise_multiplier,
            steps=steps,
            orders=orders
        )
        epsilon = privacy_analysis.get_privacy_spent(orders, epsilon, self.delta)[0]
        return epsilon
    
    def train_step(self, data, target, optimizer, criterion):
        self.model.train()
        
        data, target = data.to(self.device), target.to(self.device)
        
        optimizer.zero_grad()
        output = self.model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Clip gradients and add noise
        self._clip_and_add_noise(optimizer)
        
        optimizer.step()
        
        return loss.item()
    
    def _clip_and_add_noise(self, optimizer):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        
        # Add Gaussian noise
        for p in self.model.parameters():
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * self.noise_multiplier * self.max_grad_norm
                p.grad.data.add_(noise)
    
    def train_model(self, train_data, val_data, epochs=50, batch_size=64, learning_rate=0.001):
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        epsilon = self.compute_epsilon(epochs, batch_size, len(X_train))
        print(f"Training with DP-SGD - Target epsilon: {epsilon:.2f}")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                loss = self.train_step(data, target, optimizer, criterion)
                running_loss += loss
                
                with torch.no_grad():
                    output = self.model(data.to(self.device))
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target.to(self.device)).sum().item()
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            
            # Validation
            val_loss, val_acc = self._validate(val_loader, criterion)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:3d}/{epochs} | '
                      f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                      f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'epsilon': epsilon
        }
    
    def _validate(self, val_loader, criterion):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc

class PrivacyAccountant:
    def __init__(self, delta=1e-5):
        self.delta = delta
        self.steps = 0
        
    def step(self, sampling_prob, noise_multiplier):
        self.steps += 1
        epsilon = self._compute_epsilon(sampling_prob, noise_multiplier, self.steps)
        return epsilon
    
    def _compute_epsilon(self, sampling_prob, noise_multiplier, steps):
        if noise_multiplier == 0:
            return float('inf')
        
        # Simplified RDP to DP conversion
        alpha = 1.0 + 1.0 / noise_multiplier
        rdp = steps * sampling_prob * alpha / (2 * noise_multiplier ** 2)
        epsilon = rdp + math.log(1 / self.delta) / (alpha - 1)
        
        return epsilon

def test_dp_sgd():
    print("Testing DP-SGD Defense...")
    
    from ..models.model_architectures import MLPModel
    
    np.random.seed(42)
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randint(0, 3, 1000)
    X_val = np.random.randn(200, 20)
    y_val = np.random.randint(0, 3, 200)
    
    model = MLPModel(20, 3)
    dp_trainer = DPSGD(model, noise_multiplier=1.1, max_grad_norm=1.0)
    
    history = dp_trainer.train_model(
        (X_train, y_train), (X_val, y_val), epochs=5, batch_size=64
    )
    
    print(f"Final training accuracy: {history['train_accuracies'][-1]:.2f}%")
    print(f"Final validation accuracy: {history['val_accuracies'][-1]:.2f}%")
    print(f"Privacy budget (epsilon): {history['epsilon']:.2f}")
    
    print("DP-SGD test completed!")

if __name__ == "__main__":
    test_dp_sgd()
