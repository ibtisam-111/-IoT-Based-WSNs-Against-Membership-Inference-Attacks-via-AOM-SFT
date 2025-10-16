IoT-Based-WSNs-Against-Membership-Inference-Attacks-via-AOM-SFT
implementation of a lightweight defense framework (AOM-SFT) that protects ML models in IoT sensor networks against privacy attacks while maintaining high accuracy and low computational overhead.
Securing-IoT-WSNs-Against-MIA-AOM-SFT/
│
├── paper/
│   ├── main.tex
├── src/
│   ├── data_processing/
│   │   ├── load_datasets.py
│   │   └── sft_transformation.py
│   │
│   ├── models/
│   │   ├── model_architectures.py
│   │   ├── train_target_models.py
│   │   └── aom_defense.py
│   │
│   ├── attacks/
│   │   ├── shadow_model_attack.py
│   │   └── label_only_attack.py
│   │
│   ├── defenses/
│   │   ├── dp_sgd.py
│   │   ├── adversarial_regularization.py
│   │   └── l2_dropout.py
│   │
│   ├── simulation/
│   │   └── iot_wsn_simulator.py
│   │
│   └── utils/
│       ├── config.py
│       ├── metrics.py
│       └── visualization.py
│
├── experiments/
│   ├── run_main_results.sh
│   ├── run_comparative_analysis.sh
│   └── run_simulation.sh
│

├── environment/
│   ├── requirements.txt
│   └── environment.yml
│
├── README.md
├── CITATION.cff
└── LICENSE
