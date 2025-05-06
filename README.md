ransomware-detection/
│
├── data/
│   ├── raw_logs/            
│   │   # Stores raw log data such as PCAP captures (benign and malicious traffic).
│   │   # These files are input to the system for feature extraction and detection.
│   │   # ➤ Place your .pcap files here for processing.
│   │
│   └── processed_data/      
│       # Contains structured datasets generated from the raw logs.
│       # These are CSV files with extracted features used for training/testing.
│       # ➤ Automatically created after preprocessing.
│
├── models/                  
│   # Contains trained machine learning models (e.g., Random Forest).
│   # Includes .pkl files for the model and scaler used during inference.
│   # ➤ Used by real-time scripts to load models and predict.
│
├── src/                     
│   ├── __init__.py
│   │   # Marks the directory as a Python package.
│   │
│   ├── data_collection.py   
│   │   # Collects data from PCAP files using PyShark and converts it into DataFrame format.
│   │   # ➤ Run automatically during real-time or batch processing.
│   │
│   ├── feature_extraction.py 
│   │   # Extracts relevant numerical features (e.g., packet size, IP count).
│   │   # ➤ Used before training or inference.
│   │
│   ├── model.py             
│   │   # Contains training code for ML models (fit, save, load).
│   │   # ➤ Run manually to train and save new models.
│   │
│   ├── evaluation.py        
│   │   # Includes functions to evaluate model performance (accuracy, confusion matrix).
│   │   # ➤ Run after training to assess model performance.
│   │
│   ├── real_time.py         
│   │   # The core real-time monitoring script.
│   │   # It loads the model, collects new PCAP logs, extracts features, and predicts threats.
│   │   # ➤ Run: `python src/real_time.py`
│   │
│   └── utils.py             
│       # Helper utilities: log parsing, normalization, config loading, etc.
│       # ➤ Imported by other scripts.
│
├── notebooks/               
│   └── exploratory_analysis.ipynb
│       # Jupyter Notebook for exploratory data analysis (EDA), visualization, and prototyping.
│       # ➤ Run in a Jupyter environment for interactive exploration.
│
├── requirements.txt         
│   # Lists all required Python packages (e.g., pandas, scikit-learn, pyshark).
│   # ➤ Run: `pip install -r requirements.txt`
│
├── config.json              
│   # Optional configuration file storing model paths, thresholds, and selected features.
│   # ➤ Loaded by code for flexibility and reproducibility.
│
├── README.md                
│   # Documentation file containing project overview, installation steps, and usage examples.
│   # ➤ Update with architecture diagram and project goals.
│
└── logs/                    
    # (Optional) Contains system-generated logs or alerts from the real-time detection script.
    # ➤ Helpful for auditing or further analysis.
