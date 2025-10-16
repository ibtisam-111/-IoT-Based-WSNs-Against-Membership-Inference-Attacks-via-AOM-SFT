import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import json
import os
from .model_architectures import SensorModelFactory

class ModelTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader, criterion, optimizer):
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
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
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
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train_model(self, train_data, val_data, epochs=100, batch_size=64, 
                   learning_rate=0.001, patience=10, model_dir='./saved_models'):
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        os.makedirs(model_dir, exist_ok=True)
        
        best_val_acc = 0.0
        patience_counter = 0
        best_model_path = None
        
        print(f"Training model for {epochs} epochs...")
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            scheduler.step(val_loss)
            
            epoch_time = time.time() - start_time
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:3d}/{epochs} | Time: {epoch_time:.2f}s')
                print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
                print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
                print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                model_name = f"best_model_{int(best_val_acc)}.pth"
                best_model_path = os.path.join(model_dir, model_name)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_acc': train_acc
                }, best_model_path)
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if best_model_path and os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model with val accuracy: {checkpoint['val_acc']:.2f}%")
        
        training_history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': best_val_acc
        }
        
        return training_history, best_model_path
    
    def evaluate_model(self, test_data):
        X_test, y_test = test_data
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        test_loss, test_acc = self.validate(test_loader, nn.CrossEntropyLoss())
        
        self.model.eval()
        all_predictions = []
        all_confidences = []
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device)
                output = self.model(data)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_confidences.extend(confidence.cpu().numpy())
        
        results = {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'predictions': np.array(all_predictions),
            'confidences': np.array(all_confidences)
        }
        
        return results
    
    def get_predictions(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy(), confidences.cpu().numpy()

def train_dataset_model(dataset_name, train_data, val_data, test_data, model_config=None):
    if model_config is None:
        model_config = SensorModelFactory.get_default_config(dataset_name)
    
    model = SensorModelFactory.create_model(**model_config)
    
    print(f"Training {dataset_name} model...")
    print(f"Model type: {model_config['model_type']}")
    print(f"Input dim: {model_config['input_dim']}, Classes: {model_config['num_classes']}")
    
    trainer = ModelTrainer(model)
    
    training_history, model_path = trainer.train_model(
        train_data, val_data, epochs=100, batch_size=64
    )
    
    test_results = trainer.evaluate_model(test_data)
    
    print(f"\nFinal Results for {dataset_name}:")
    print(f"Best validation accuracy: {training_history['best_val_acc']:.2f}%")
    print(f"Test accuracy: {test_results['test_accuracy']:.2f}%")
    print(f"Model saved: {model_path}")
    
    return trainer, training_history, test_results

def demo_training():
    print("Demonstrating model training...")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    X_demo = np.random.randn(1000, 50)
    y_demo = np.random.randint(0, 5, 1000)
    
    split_idx = int(0.8 * len(X_demo))
    X_train, X_val = X_demo[:split_idx], X_demo[split_idx:]
    y_train, y_val = y_demo[:split_idx], y_demo[split_idx:]
    
    model_config = {
        'model_type': 'mlp',
        'input_dim': 50,
        'num_classes': 5,
        'hidden_dims': [128, 64]
    }
    
    model = SensorModelFactory.create_model(**model_config)
    trainer = ModelTrainer(model, device='cpu')
    
    training_history, _ = trainer.train_model(
        (X_train, y_train), (X_val, y_val), epochs=5, batch_size=32
    )
    
    print("Demo training completed!")

if __name__ == "__main__":
    demo_training()
