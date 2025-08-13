# forex_ai_builder - Watsonx.ai Full Pipeline

## Steps
1. Install dependencies: `pip install -r requirements.txt`
2. Fill `.env` with your IBM Cloud credentials.
3. Build datasets locally:
   ```bash
   # Real market data (requires network)
   python scripts/forex_dataset_builder.py --output-dir ./data

   # Offline synthetic data (no network required)
   python scripts/forex_dataset_builder.py --offline --synthetic-rows 500 --output-dir ./data
   ```
4. Verify orchestration flow without credentials (dry-run):
   ```bash
   python scripts/run_all_training_watsonx.py --dry-run --data-dir ./data
   ```
5. Run all training on Watsonx.ai (requires `.env`):
   ```bash
   python scripts/run_all_training_watsonx.py --data-dir ./data
   ```

All models will be trained remotely; local VM only runs dataset creation and uploads. The orchestrator supports `--dry-run` to test without accessing secrets.
