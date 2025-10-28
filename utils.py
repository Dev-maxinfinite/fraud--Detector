# utils.py (optional)
import json
from datetime import datetime

def save_model_metrics(accuracy, precision, recall, f1):
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    with open('model_metrics.json', 'a') as f:
        f.write(json.dumps(metrics) + '\n')