import numpy as np
from scipy import signal
import warnings

class SFTTransformer:
    def __init__(self, method='noise', noise_level=0.1, window_size=5):
        self.method = method
        self.noise_level = noise_level
        self.window_size = window_size
        self.feature_stats = None
        
    def fit(self, X):
        if self.method == 'noise':
            self.feature_stats = {
                'mean': np.mean(X, axis=0),
                'std': np.std(X, axis=0)
            }
        return self
    
    def transform(self, X):
        if self.method == 'noise':
            return self._add_controlled_noise(X)
        elif self.method == 'smoothing':
            return self._apply_temporal_smoothing(X)
        elif self.method == 'quantization':
            return self._apply_value_quantization(X)
        else:
            raise ValueError("Method must be 'noise', 'smoothing', or 'quantization'")
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def _add_controlled_noise(self, X):
        if self.feature_stats is None:
            self.fit(X)
            
        noise = np.random.normal(0, self.noise_level, X.shape)
        scaled_noise = noise * self.feature_stats['std']
        X_transformed = X + scaled_noise
        
        return X_transformed
    
    def _apply_temporal_smoothing(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        X_transformed = np.zeros_like(X)
        n_samples, n_features = X.shape
        
        for i in range(n_features):
            feature_series = X[:, i]
            
            if len(feature_series) < self.window_size:
                X_transformed[:, i] = feature_series
                continue
                
            try:
                window = np.ones(self.window_size) / self.window_size
                smoothed = np.convolve(feature_series, window, mode='same')
                X_transformed[:, i] = smoothed
            except:
                X_transformed[:, i] = feature_series
        
        return X_transformed
    
    def _apply_value_quantization(self, X):
        if self.feature_stats is None:
            self.fit(X)
            
        quantization_levels = 10
        X_transformed = np.zeros_like(X)
        
        for i in range(X.shape[1]):
            feature_min = np.min(X[:, i])
            feature_max = np.max(X[:, i])
            
            if feature_max == feature_min:
                X_transformed[:, i] = X[:, i]
                continue
                
            step = (feature_max - feature_min) / quantization_levels
            quantized = np.round((X[:, i] - feature_min) / step) * step + feature_min
            X_transformed[:, i] = quantized
        
        return X_transformed
    
    def set_parameters(self, noise_level=None, window_size=None):
        if noise_level is not None:
            self.noise_level = noise_level
        if window_size is not None and window_size % 2 == 1:
            self.window_size = window_size
        elif window_size is not None:
            warnings.warn("Window size must be odd, using default")
    
    def get_transformation_info(self):
        return {
            'method': self.method,
            'noise_level': self.noise_level,
            'window_size': self.window_size
        }


class BatchSFTProcessor:
    def __init__(self, transformation_configs):
        self.transformers = []
        self.configs = transformation_configs
        
        for config in transformation_configs:
            transformer = SFTTransformer(
                method=config.get('method', 'noise'),
                noise_level=config.get('noise_level', 0.1),
                window_size=config.get('window_size', 5)
            )
            self.transformers.append(transformer)
    
    def fit(self, X_list):
        for i, (transformer, X) in enumerate(zip(self.transformers, X_list)):
            if X is not None:
                transformer.fit(X)
        return self
    
    def transform(self, X_list):
        transformed_data = []
        for i, (transformer, X) in enumerate(zip(self.transformers, X_list)):
            if X is not None:
                transformed_data.append(transformer.transform(X))
            else:
                transformed_data.append(None)
        return transformed_data
    
    def fit_transform(self, X_list):
        self.fit(X_list)
        return self.transform(X_list)


def test_sft_transformations():
    print("Testing SFT transformations...")
    
    np.random.seed(42)
    sample_data = np.random.randn(100, 8)
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Original stats - Mean: {np.mean(sample_data):.3f}, Std: {np.std(sample_data):.3f}")
    
    methods = ['noise', 'smoothing', 'quantization']
    
    for method in methods:
        print(f"\n--- Testing {method} ---")
        
        transformer = SFTTransformer(method=method, noise_level=0.1, window_size=5)
        transformed = transformer.fit_transform(sample_data)
        
        diff = transformed - sample_data
        print(f"Transformed stats - Mean: {np.mean(transformed):.3f}, Std: {np.std(transformed):.3f}")
        print(f"Difference - Mean: {np.mean(diff):.3f}, Std: {np.std(diff):.3f}")
        
        if method == 'noise':
            noise_ratio = np.std(diff) / np.std(sample_data)
            print(f"Noise ratio: {noise_ratio:.3f}")


if __name__ == "__main__":
    test_sft_transformations()
