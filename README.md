# forex_ai_builder - Watsonx.ai Full Pipeline

## Steps
1. Install dependencies: `pip install -r requirements.txt`
2. Fill `.env` with your IBM Cloud credentials.
3. Build datasets locally:
   ```bash
   python scripts/forex_dataset_builder.py
   ```
4. Run all training on Watsonx.ai:
   ```bash
   python scripts/run_all_training_watsonx.py
   ```

All models will be trained remotely; local VM only runs dataset creation and uploads.
