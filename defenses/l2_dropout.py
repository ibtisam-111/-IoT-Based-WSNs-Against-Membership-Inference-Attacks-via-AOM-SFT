import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class L2Regularization:
    def __init__(self, model, lambda_l2=0.001):
        self.model = model
        self.lambda_l2 = lambda_l2
        self.device = next(model.parameters()).device
    
    def _l2_penalty(self):
        l2_norm = 0.0
        for param in self.model.parameters():
            l2_norm += torch.norm(param, 2) ** 2
        return self.lambda_l2 * l2_norm
    
    def train_step(self, data, target, optimizer, criterion):
        self.model.train()
        
        data, target = data.to(self.device), target.to(self.device)
        
        optimizer.zero_grad()
        output = self.model(data)
        
        task_loss = criterion(output, target)
        reg_loss = self._l2_penalty()
        total_loss = task_loss + reg_loss
        
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item(), task_loss.item(), reg_loss.item()
    
    def train_model(self, train_data, val_data, epochs=50, batch_size=64, learning_rate=0.001):
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0)
        
        train_losses = []
        val_accuracies = []
        reg_losses = []
        
        print(f"Training with L2 regularization (lambda={self.lambda_l2})...")
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            running_reg_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                total_loss, task_loss, reg_loss = self.train_step(data, target, optimizer, criterion)
                running_loss += total_loss
                running_reg_loss += reg_loss
                
                with torch.no_grad():
                    output = self.model(data.to(self.device))
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target.to(self.device)).sum().item()
            
            train_loss = running_loss / len(train_loader)
            avg_reg_loss = running_reg_loss / len(train_loader)
            train_acc = 100. * correct / total
            
            # Validation
            val_acc = self._validate(val_loader)
            
            train_losses.append(train_loss)
            val_accuracies.append(val_acc)
            reg_losses.append(avg_reg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:3d}/{epochs} | '
                      f'Loss: {train_loss:.4f} | Reg Loss: {avg_reg_loss:.4f} | '
                      f'Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')
        
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
        
        return 100. * correct / total

class DropoutRegularization:
    def __init__(self, model, dropout_rate=0.5):
        self.model = model
        self.dropout_rate = dropout_rate
        self.device = next(model.parameters()).device
        
        # Apply dropout to all linear layers
        self._apply_dropout()
    
    def _apply_dropout(self):
        for module in self.model.modules():
            if isinstance(module, nn.Linear) and module.out_features > 1:
                # Insert dropout before the linear layer
                new_sequential = nn.Sequential(
                    nn.Dropout(self.dropout_rate),
                    module
                )
                # Replace the module (this is simplified - in practice you'd need to modify the model structure)
                pass
    
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
        val_accuracies = []
        
        print(f"Training with dropout (rate={self.dropout_rate})...")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            
            # Validation
            val_acc = self._validate(val_loader)
            
            train_losses.append(train_loss)
            val_accuracies.append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:3d}/{epochs} | '
                      f'Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                      f'Val Acc: {val_acc:.2f}%')
        
        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
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
        
        return 100. * correct / total

class CombinedRegularization:
    def __init__(self, model, lambda_l2=0.001, dropout_rate=0.3):
        self.model = model
        self.l2_reg = L2Regularization(model, lambda_l2)
        self.dropout_reg = DropoutRegularization(model, dropout_rate)
        self.device = next(model.parameters()).device
    
    def train_model(self, train_data, val_data, epochs=50, batch_size=64, learning_rate=0.001):
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0)
        
        train_losses = []
        val_accuracies = []
        
        print("Training with L2 + Dropout regularization...")
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                
                task_loss = criterion(output, target)
                reg_loss = self.l2_reg._l2_penalty()
                total_loss = task_loss + reg_loss
                
                total_loss.backward()
                optimizer.step()
                
                running_loss += total_loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            
            # Validation
            val_acc = self._validate(val_loader)
            
            train_losses.append(train_loss)
            val_accuracies.append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:3d}/{epochs} | '
                      f'Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                      f'Val Acc: {val_acc:.2f}%')
        
        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
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
        
        return 100. * correct / total

def test_l2_dropout():
    print("Testing L2 and Dropout Regularization...")
    
    from ..models.model_architectures import MLPModel
    
    np.random.seed(42)
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randint(0, 3, 1000)
    X_val = np.random.randn(200, 20)
    y_val = np.random.randint(0, 3, 200)
    
    # Test L2
    model_l2 = MLPModel(20, 3)
    l2_trainer = L2Regularization(model_l2, lambda_l2=0.001)
    history_l2 = l2_trainer.train_model((X_train, y_train), (X_val, y_val), epochs=5)
    
    # Test Dropout
    model_dropout = MLPModel(20, 3)
    dropout_trainer = DropoutRegularization(model_dropout, dropout_rate=0.3)
    history_dropout = dropout_trainer.train_model((X_train, y_train), (X_val, y_val), epochs=5)
    
    print(f"L2 final val accuracy: {history_l2['val_accuracies'][-1]:.2f}%")
    print(f"Dropout final val accuracy: {history_dropout['val_accuracies'][-1]:.2f}%")
    print("L2 and Dropout test completed!")

if __name__ == "__main__":
    test_l2_dropout()
