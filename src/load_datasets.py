import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import urllib.request
import zipfile

class DataLoader:
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def _download_uci_har(self):
        uci_dir = os.path.join(self.data_dir, 'UCI_HAR')
        if os.path.exists(uci_dir):
            return True
            
        print("Downloading UCI HAR dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
        zip_path = os.path.join(self.data_dir, 'uci_har.zip')
        
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            os.rename(os.path.join(self.data_dir, 'UCI HAR Dataset'), uci_dir)
            os.remove(zip_path)
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False

    def load_uci_har(self):
        if not self._download_uci_har():
            return None
            
        print("Loading UCI HAR dataset...")
        uci_dir = os.path.join(self.data_dir, 'UCI_HAR')
        
        X_train_path = os.path.join(uci_dir, 'train', 'X_train.txt')
        y_train_path = os.path.join(uci_dir, 'train', 'y_train.txt')
        X_test_path = os.path.join(uci_dir, 'test', 'X_test.txt')
        y_test_path = os.path.join(uci_dir, 'test', 'y_test.txt')
        
        X_train = pd.read_csv(X_train_path, delim_whitespace=True, header=None)
        y_train = pd.read_csv(y_train_path, header=None).squeeze()
        X_test = pd.read_csv(X_test_path, delim_whitespace=True, header=None)
        y_test = pd.read_csv(y_test_path, header=None).squeeze()
        
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"UCI HAR: {X_train_scaled.shape[0]} train, {X_test_scaled.shape[0]} test samples")
        
        return (X_train_scaled, y_train_encoded), (X_test_scaled, y_test_encoded)

    def _download_intel_lab(self):
        intel_dir = os.path.join(self.data_dir, 'Intel_Lab')
        if os.path.exists(intel_dir):
            return True
            
        print("Downloading Intel Lab dataset...")
        url = "http://db.csail.mit.edu/labdata/labdata.html"
        data_url = "http://db.csail.mit.edu/labdata/data.txt"
        
        try:
            os.makedirs(intel_dir, exist_ok=True)
            data_path = os.path.join(intel_dir, 'data.txt')
            urllib.request.urlretrieve(data_url, data_path)
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False

    def load_intel_lab(self, sample_size=50000):
        if not self._download_intel_lab():
            return None
            
        print("Loading Intel Lab dataset...")
        data_path = os.path.join(self.data_dir, 'Intel_Lab', 'data.txt')
        
        data = pd.read_csv(data_path, sep='\s+', header=None, 
                          names=['date', 'time', 'epoch', 'nodeid', 'temp', 'humidity', 'light', 'voltage'])
        
        data = data.dropna()
        data = data[data['light'] >= 0]
        
        features = ['humidity', 'temp', 'light', 'voltage']
        X = data[features].values
        y = data['nodeid'].values
        
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X = X[indices]
            y = y[indices]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Intel Lab: {X_train_scaled.shape[0]} train, {X_test_scaled.shape[0]} test samples")
        
        return (X_train_scaled, y_train), (X_test_scaled, y_test)

    def _download_gas_sensor(self):
        gas_dir = os.path.join(self.data_dir, 'Gas_Sensor')
        if os.path.exists(gas_dir):
            return True
            
        print("Downloading Gas Sensor dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00487/ethylene_CO.zip"
        zip_path = os.path.join(self.data_dir, 'gas_sensor.zip')
        
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(gas_dir)
            os.remove(zip_path)
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False

    def load_gas_sensor(self):
        if not self._download_gas_sensor():
            return None
            
        print("Loading Gas Sensor dataset...")
        gas_dir = os.path.join(self.data_dir, 'Gas_Sensor')
        
        csv_files = [f for f in os.listdir(gas_dir) if f.endswith('.csv')]
        if not csv_files:
            print("No CSV files found in Gas Sensor directory")
            return None
            
        data_path = os.path.join(gas_dir, csv_files[0])
        data = pd.read_csv(data_path)
        
        sensor_cols = [col for col in data.columns if 'sensor' in col.lower() or col.startswith('s')]
        if len(sensor_cols) < 10:
            sensor_cols = data.columns[1:17]
        
        X = data[sensor_cols].values
        y = data.iloc[:, -1].values
        
        if len(np.unique(y)) > 10:
            y_bin = (y > np.median(y)).astype(int)
            y = y_bin
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        print(f"Gas Sensor: {X_train_scaled.shape[0]} train, {X_test_scaled.shape[0]} test samples")
        
        return (X_train_scaled, y_train_encoded), (X_test_scaled, y_test_encoded)

    def get_dataset(self, dataset_name):
        if dataset_name == 'uci_har':
            return self.load_uci_har()
        elif dataset_name == 'intel_lab':
            return self.load_intel_lab()
        elif dataset_name == 'gas_sensor':
            return self.load_gas_sensor()
        else:
            raise ValueError(f"Unknown dataset: {
