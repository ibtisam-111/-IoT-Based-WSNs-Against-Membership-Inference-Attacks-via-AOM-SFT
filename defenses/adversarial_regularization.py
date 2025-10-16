import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class AdversarialRegularization:
    def __init__(self, model, lambda_reg=1.0, attack_model=None):
        self.model = model
        self.lambda_reg = lambda_reg
        self.attack_model = attack_model
        self.device = next(model.parameters()).device
        
    def _compute_membership_loss(self, member_outputs, nonmember_outputs):
        member_probs = torch.softmax(member_outputs, dim=1)
        nonmember_probs = torch.softmax(nonmember_outputs, dim=1)
        
        member_confidences, _ = torch.max(member_probs, dim=1)
        nonmember_confidences, _ = torch.max(nonmember_probs, dim=1)
        
        # Encourage similar confidence distributions
        confidence_loss = torch.abs(torch.mean(member_confidences) - torch.mean(nonmember_confidences))
        
        # Encourage similar output distributions
        member_entropy = -torch.sum(member_probs * torch.log(member_probs + 1e-8), dim=1)
        nonmember_entropy = -torch.sum(nonmember_probs * torch.log(nonmember_probs + 1e-8), dim=1)
        entropy_loss = torch.abs(torch.mean(member_entropy) - torch.mean(nonmember_entropy))
        
        return confidence_loss + entropy_loss
    
    def train_step(self, member_data, member_target, nonmember_data, nonmember_target, 
                   optimizer, criterion, alpha=0.5):
        self.model.train()
        
        member_data = member_data.to(self.device)
        member_target = member_target.to(self.device)
        nonmember_data = nonmember_data.to(self.device)
        nonmember_target = nonmember_target.to(self.device)
        
        optimizer.zero_grad()
        
        # Forward pass for member data
        member_outputs = self.model(member_data)
        task_loss = criterion(member_outputs, member_target)
        
        # Forward pass for non-member data
        nonmember_outputs = self.model(nonmember_data)
        
        # Membership inference regularization loss
        reg_loss = self._compute_membership_loss(member_outputs, nonmember_outputs)
        
        # Combined loss
        total_loss = alpha * task_loss + self.lambda_reg * reg_loss
        
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'reg_loss': reg_loss.item()
        }
    
    def train_model(self, member_data, nonmember_data, val_data, epochs=50, 
                   batch_size=64, learning_rate=0.001):
        X_member, y_member = member_data
        X_nonmember, y_nonmember = nonmember_data
        X_val, y_val = val_data
        
        member_dataset = TensorDataset(torch.FloatTensor(X_member), torch.LongTensor(y_member))
        nonmember_dataset = TensorDataset(torch.FloatTensor(X_nonmember), torch.LongTensor(y_nonmember))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        member_loader = DataLoader(member_dataset, batch_size=batch_size, shuffle=True)
        nonmember_loader = DataLoader(nonmember_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        train_losses = []
        val_accuracies = []
        reg_losses = []
        
        print(f"Training with adversarial regularization (lambda={self.lambda_reg})...")
        
        for epoch in range(epochs):
            # Combine loaders
            member_iter = iter(member_loader)
            nonmember_iter = iter(nonmember_loader)
            
            epoch_losses = []
            epoch_reg_losses = []
            
            for batch_idx in range(min(len(member_loader), len(nonmember_loader))):
                try:
                    member_batch = next(member_iter)
                    nonmember_batch = next(nonmember_iter)
                    
                    member_data, member_target = member_batch
                    nonmember_data, nonmember_target = nonmember_batch
                    
                    loss_dict = self.train_step(
                        member_data, member_target, nonmember_data, nonmember_target,
                        optimizer, criterion
                    )
                    
                    epoch_losses.append(loss_dict['total_loss'])
                    epoch_reg_losses.append(loss_dict['reg_loss'])
                    
                except StopIteration:
                    break
            
            avg_loss = np.mean(epoch_losses)
            avg_reg_loss = np.mean(epoch_reg_losses)
            
            # Validation
            val_acc = self._validate(val_loader)
            
            train_losses.append(avg_loss)
            val_accuracies.append(val_acc)
            reg_losses.append(avg_reg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:3d}/{epochs} | '
                      f'Loss: {avg_loss:.4f} | Reg Loss: {avg_reg_loss:.4f} | '
                      f'Val Acc: {val_acc:.2f}%')
        
        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'reg_losses': reg_losses
        }
    
    def _validate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy

class ConfidenceSmoothing:
    def __init__(self, model, temperature=2.0):
        self.model = model
        self.temperature = temperature
        self.device = next(model.parameters()).device
    
    def smooth_predictions(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            smoothed_probs = torch.softmax(outputs / self.temperature, dim=1)
        return smoothed_probs.cpu().numpy()
    
    def train_with_smoothing(self, train_data, val_data, epochs=50, batch_size=64):
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(data)
                
                # Temperature scaling during training
                smoothed_outputs = outputs / self.temperature
                loss = criterion(smoothed_outputs, target)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                val_acc = self._validate(val_loader)
                print(f'Epoch {epoch+1:3d}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%')
    
    def _validate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return 100. * correct / total

def test_adversarial_regularization():
    print("Testing Adversarial Regularization...")
    
    from ..models.model_architectures import MLPModel
    
    np.random.seed(42)
    X_member = np.random.randn(500, 20)
    y_member = np.random.randint(0, 3, 500)
    X_nonmember = np.random.randn(500, 20)
    y_nonmember = np.random.randint(0, 3, 500)
    X_val = np.random.randn(200, 20)
    y_val = np.random.randint(0, 3, 200)
    
    model = MLPModel(20, 3)
    adv_reg = AdversarialRegularization(model, lambda_reg=1.0)
    
    history = adv_reg.train_model(
        (X_member, y_member), (X_nonmember, y_nonmember), (X_val, y_val), epochs=5
    )
    
    print(f"Final validation accuracy: {history['val_accuracies'][-1]:.2f}%")
    print("Adversarial regularization test completed!")

if __name__ == "__main__":
    test_adversarial_regularization()
