class ExperimentConfig:
    def __init__(self):
        # Dataset configurations
        self.dataset_configs = {
            'uci_har': {
                'input_dim': 561,
                'num_classes': 6,
                'model_type': 'mlp',
                'hidden_dims': [256, 128, 64],
                'batch_size': 64,
                'learning_rate': 0.001
            },
            'intel_lab': {
                'input_dim': 4,
                'num_classes': 1,
                'model_type': 'cnn',
                'seq_length': 100,
                'batch_size': 64,
                'learning_rate': 0.001
            },
            'gas_sensor': {
                'input_dim': 16,
                'num_classes': 6,
                'model_type': 'lstm',
                'hidden_size': 64,
                'num_layers': 2,
                'batch_size': 64,
                'learning_rate': 0.001
            }
        }
        
        # Model training parameters
        self.training_params = {
            'epochs': 100,
            'batch_size': 64,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'patience': 10,
            'validation_split': 0.2,
            'random_seed': 42
        }
        
        # AOM-SFT defense parameters
        self.defense_params = {
            'aom_percentile': 90,
            'aom_top_k': 3,
            'sft_noise_level': 0.1,
            'sft_window_size': 5,
            'sft_transformation': 'noise'  # 'noise', 'smoothing', 'quantization'
        }
        
        # Attack parameters
        self.attack_params = {
            'num_shadow_models': 5,
            'shadow_data_ratio': 0.8,
            'num_perturbations': 10,
            'perturbation_scale': 0.1,
            'attack_model_type': 'random_forest'
        }
        
        # Baseline defense parameters
        self.baseline_defenses = {
            'dp_sgd': {
                'noise_multiplier': 1.1,
                'max_grad_norm': 1.0,
                'delta': 1e-5
            },
            'adversarial_regularization': {
                'lambda_reg': 1.0,
                'alpha': 0.5
            },
            'l2_regularization': {
                'lambda_l2': 0.001
            },
            'dropout': {
                'dropout_rate': 0.5
            }
        }
        
        # Simulation parameters
        self.simulation_params = {
            'area_width': 100,
            'area_height': 100,
            'num_nodes': 20,
            'num_clusters': 5,
            'simulation_steps': 50,
            'attack_range': 40
        }
        
        # Path configurations
        self.paths = {
            'data_dir': './data',
            'model_dir': './saved_models',
            'results_dir': './results',
            'logs_dir': './logs'
        }

    def get_dataset_config(self, dataset_name):
        return self.dataset_configs.get(dataset_name, {})
    
    def update_training_params(self, **kwargs):
        self.training_params.update(kwargs)
    
    def update_defense_params(self, **kwargs):
        self.defense_params.update(kwargs)
    
    def update_attack_params(self, **kwargs):
        self.attack_params.update(kwargs)
    
    def get_full_config(self):
        return {
            'dataset_configs': self.dataset_configs,
            'training_params': self.training_params,
            'defense_params': self.defense_params,
            'attack_params': self.attack_params,
            'baseline_defenses': self.baseline_defenses,
            'simulation_params': self.simulation_params,
            'paths': self.paths
        }


class DefenseConfig:
    def __init__(self, defense_type='aom_sft'):
        self.defense_type = defense_type
        self.configs = {
            'aom_sft': {
                'aom_percentile': 90,
                'aom_top_k': 3,
                'sft_noise_level': 0.1,
                'sft_window_size': 5
            },
            'dp_sgd': {
                'noise_multiplier': 1.1,
                'max_grad_norm': 1.0,
                'delta': 1e-5
            },
            'adv_reg': {
                'lambda_reg': 1.0,
                'alpha': 0.5
            },
            'l2': {
                'lambda_l2': 0.001
            },
            'dropout': {
                'dropout_rate': 0.5
            }
        }
    
    def get_config(self, defense_type=None):
        if defense_type is None:
            defense_type = self.defense_type
        return self.configs.get(defense_type, {})
    
    def set_defense_type(self, defense_type):
        self.defense_type = defense_type


class AttackConfig:
    def __init__(self, attack_type='shadow_model'):
        self.attack_type = attack_type
        self.configs = {
            'shadow_model': {
                'num_shadow_models': 5,
                'shadow_data_ratio': 0.8,
                'attack_model': 'random_forest'
            },
            'label_only': {
                'num_perturbations': 10,
                'perturbation_scale': 0.1,
                'stability_threshold': 0.7
            }
        }
    
    def get_config(self, attack_type=None):
        if attack_type is None:
            attack_type = self.attack_type
        return self.configs.get(attack_type, {})


def create_default_config():
    return ExperimentConfig()


def save_config(config, filepath):
    import json
    import os
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(config.get_full_config(), f, indent=2)


def load_config(filepath):
    import json
    
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    
    config = ExperimentConfig()
    
    # Update all configurations
    for key, value in config_dict.items():
        if hasattr(config, key):
            current_value = getattr(config, key)
            if isinstance(current_value, dict):
                current_value.update(value)
            else:
                setattr(config, key, value)
    
    return config


def print_config_summary(config):
    print("Experiment Configuration Summary")
    print("=" * 50)
    
    print(f"\nDataset Configurations:")
    for dataset, params in config.dataset_configs.items():
        print(f"  {dataset}: {params}")
    
    print(f"\nTraining Parameters:")
    for key, value in config.training_params.items():
        print(f"  {key}: {value}")
    
    print(f"\nDefense Parameters:")
    for key, value in config.defense_params.items():
        print(f"  {key}: {value}")
    
    print(f"\nAttack Parameters:")
    for key, value in config.attack_params.items():
        print(f"  {key}: {value}")


# Example usage
if __name__ == "__main__":
    config = create_default_config()
    print_config_summary(config)
    
    # Save config
    save_config(config, './configs/default_config.json')
    
    # Load config
    loaded_config = load_config('./configs/default_config.json')
    print("\nLoaded configuration verified!")
