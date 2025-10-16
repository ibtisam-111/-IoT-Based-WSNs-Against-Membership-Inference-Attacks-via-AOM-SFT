Securing IoT-Based WSNs Against Membership Inference Attacks via Adaptive Output Masking and Sensor Feature Transformation
Official implementation of our paper presenting AOM-SFT, a lightweight defense framework that protects machine learning models in IoT sensor networks against privacy attacks while maintaining high accuracy and low computational overhead.

Overview
This repository contains the complete implementation of our novel defense framework against Membership Inference Attacks (MIAs) in IoT-based Wireless Sensor Networks. Our approach combines two synergistic components:

Adaptive Output Masking (AOM): Dynamic confidence-based masking of model predictions

Sensor Feature Transformation (SFT): Input-level transformation to obscure data uniqueness


Installation
git clone https://github.com/yourusername/Securing-IoT-WSNs-Against-MIA-AOM-SFT.git
cd Securing-IoT-WSNs-Against-MIA-AOM-SFT
pip install -r environment/requirements.txt


Reproducing Results
from src.models.model_architectures import MLP
from src.defenses.aom_defense import AOMDefense
from src.defenses.sft_transformation import SFTTransformer

# Initialize defense components
model = MLP(input_dim=561, num_classes=6)
aom_defense = AOMDefense(model, percentile=90)
sft_transformer = SFTTransformer(transformation_type='noise', sigma=0.1)

# Apply AOM-SFT protection
protected_output = aom_defense.defend(sft_transformer.transform(sensor_data))
