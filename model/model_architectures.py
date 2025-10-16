import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[256, 128, 64], dropout_rate=0.2):
        super(MLPModel, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x)

class CNN1DModel(nn.Module):
    def __init__(self, input_channels, num_classes, seq_length=100):
        super(CNN1DModel, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        self.fc_input_size = self._get_fc_input_size(seq_length)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def _get_fc_input_size(self, seq_length):
        x = torch.zeros(1, self.input_channels, seq_length)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        lstm_out, (hidden, cell) = self.lstm(x)
        
        last_hidden = hidden[-1]
        last_hidden = self.dropout(last_hidden)
        
        output = self.classifier(last_hidden)
        
        return output

class SensorModelFactory:
    @staticmethod
    def create_model(model_type, input_dim, num_classes, **kwargs):
        if model_type == 'mlp':
            return MLPModel(input_dim, num_classes, **kwargs)
        elif model_type == 'cnn':
            return CNN1DModel(input_dim, num_classes, **kwargs)
        elif model_type == 'lstm':
            return LSTMModel(input_dim, num_classes, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_default_config(dataset_name):
        configs = {
            'uci_har': {
                'model_type': 'mlp',
                'input_dim': 561,
                'num_classes': 6
            },
            'intel_lab': {
                'model_type': 'cnn', 
                'input_dim': 4,
                'num_classes': 1
            },
            'gas_sensor': {
                'model_type': 'lstm',
                'input_dim': 16,
                'num_classes': 6
            }
        }
        return configs.get(dataset_name, {})

def test_model_architectures():
    print("Testing model architectures...")
    
    test_cases = [
        ('mlp', 100, 5),
        ('cnn', 4, 3),
        ('lstm', 16, 6)
    ]
    
    for model_type, input_dim, num_classes in test_cases:
        print(f"\n--- Testing {model_type.upper()} ---")
        
        model = SensorModelFactory.create_model(model_type, input_dim, num_classes)
        
        batch_size = 8
        if model_type == 'cnn':
            x = torch.randn(batch_size, input_dim, 50)
        elif model_type == 'lstm':
            x = torch.randn(batch_size, 10, input_dim)
        else:
            x = torch.randn(batch_size, input_dim)
        
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    test_model_architectures()
