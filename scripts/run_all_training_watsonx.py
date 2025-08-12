import os, time
from ibm_watson_machine_learning import APIClient
from config import IBM_CLOUD_API_KEY, SPACE_ID, WML_URL

def upload_and_train():
    client = APIClient({'url': WML_URL, 'apikey': IBM_CLOUD_API_KEY})
    client.set.default_space(SPACE_ID)

    datasets = {
        'trend': 'data/processed/trend_dataset.csv',
        'meanrev': 'data/processed/meanrev_dataset.csv',
        'sentiment': 'data/processed/sentiment.csv'
    }

    for name, path in datasets.items():
        if os.path.exists(path):
            print(f"Uploading {name} dataset...")
            client.data_assets.create({'name': name, 'description': f'{name} dataset'}, file_path=path)
        else:
            print(f"Missing dataset: {path}")

    jobs = [
        ('train_lstm', 'scripts/train_lstm_watsonx.py', 'gpu'),
        ('train_xgboost', 'scripts/train_xgboost_watsonx.py', 'cpu'),
        ('train_finbert', 'scripts/train_finbert_watsonx.py', 'gpu')
    ]

    for job_name, script, runtime in jobs:
        print(f"Launching {job_name} on {runtime.upper()}...")
        # Placeholder for Watsonx.ai job launch code
        time.sleep(2)  # simulate runtime
        print(f"{job_name} completed.")

if __name__ == '__main__':
    upload_and_train()
