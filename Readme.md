## Diffusion Model Training Boilerplate
`basic_training_diffusion.py`A minimal script for training diffusion models with 🤗 Diffusers, MLflow experiment tracking, and Perun energy/performance monitoring.
In `basic_training_diffusion.py`, see steps (1) to (8) for how MLflow and Perun are integrated into the script. 
---

## Installation
```bash
pip install -r requirements.txt
```
---

## Environment Configuration (`.env`)

Create a file named `.env` in the project root directory and fill in your credentials as follows:

```bash
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password
MLFLOW_TRACKING_URI=http://your-mlflow-tracking-server.com
HF_TOKEN=your_huggingface_token
```

**Load the variables into your shell before running any Python process:**
source .env

---

## Running the Training Script

To start training, execute:

python basic_training_diffusion.py

---

## Job Submission on HAICORE

For running the trainer on the HAICORE cluster, use the `job_submission.sh` script:
```bash 
sbatch job_submission.sh
```

**Ensure your `.env` variables are loaded in your shell before submitting the job.**

---

## Notes

- All dependencies must be installed via `requirements.txt`.
- MLflow tracking and Hugging Face authentication require valid credentials in your `.env` file.
- To use Perun energy/performance monitoring, refer to its documentation for proper setup and configuration.


