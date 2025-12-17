# Numerai Tournament Prediction System

This repository contains a complete automated system for training machine learning models and submitting predictions to the [Numerai Tournament](https://numer.ai/). The system uses XGBoost for model training, Modal for cloud-based execution, and GitHub Actions for automated daily submissions.

## 🎯 Project Overview

The goal of this project is to automatically:

1. Train an XGBoost model on Numerai's latest training and validation data
2. Generate predictions for the current tournament round
3. Submit predictions to Numerai on a daily schedule

The model architecture and hyperparameters are based on experimental results from the research notebooks in this repository.

## 📁 Repository Structure

```
numerai/
├── src/
│   └── train_and_submit.py    # Modal script for training and submission
├── notebooks/
│   ├── 1-getting-started.ipynb      # Initial exploration
│   ├── 2-feature-analysis.ipynb    # Feature analysis
│   ├── 3-experiments.ipynb          # Model experiments (source of production model)
│   └── 4-test-evaluation.ipynb      # Final model evaluation
├── .github/
│   └── workflows/
│       └── daily-predictions.yml    # GitHub Actions daily cron job
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🤖 Model Architecture

The production model is an **XGBoost Regressor** with the following hyperparameters (from `3-experiments.ipynb`):

```python
XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=5,
    colsample_bytree=0.1,
    verbosity=0,
    seed=42,
    tree_method="hist",
    device="cuda",
)
```

**Key Features:**

- Uses all features from the "all" feature set (v5.1)
- Trained on concatenated training and validation data (following notebook 4 approach)
- Trained on GPU for faster execution
- Optimized for correlation performance on validation data

## 🚀 Setup and Configuration

### Prerequisites

1. **Numerai Account**: Sign up at [numer.ai](https://numer.ai/) and create a model
2. **Modal Account**: Sign up at [modal.com](https://modal.com/) for cloud execution
3. **GitHub Account**: For automated workflows

### Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd numerai
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up Modal:

```bash
modal setup
```

### Configuration

#### 1. Modal Secrets

Create a Modal secret named `numerai` with your Numerai API credentials:

```bash
modal secret create numerai \
  NUMERAI_PUBLIC_ID=your_public_id \
  NUMERAI_SECRET_KEY=your_secret_key \
  NUMERAI_MODEL_NAME=your_model_name
```

You can find your Numerai API credentials in your [Numerai account settings](https://numer.ai/account).

#### 2. GitHub Secrets

Add the following secrets to your GitHub repository (Settings → Secrets and variables → Actions):

- `MODAL_TOKEN_ID`: Your Modal token ID (get from [Modal settings](https://modal.com/settings))
- `MODAL_TOKEN_SECRET`: Your Modal token secret

**Note**: The Numerai credentials are stored in Modal secrets (step 1), not in GitHub secrets. Modal will automatically inject them into the running function.

## 📝 Usage

### Manual Execution

Run the training and submission script locally:

```bash
modal run src/train_and_submit.py
```

This will:

1. Download the latest training and validation data from Numerai
2. Concatenate train and validation datasets for training
3. Train the XGBoost model on Modal's GPU infrastructure
4. Get the current tournament round
5. Download live tournament data for the current round
6. Generate predictions using feature pattern matching
7. Upload predictions to Numerai

### Automated Daily Execution

The GitHub Actions workflow is configured to run daily at 2:00 AM UTC. **Note**: The cron job is commented out by default - uncomment the cron line in `.github/workflows/daily-predictions.yml` to enable automatic daily runs. You can also trigger it manually from the GitHub Actions tab at any time.

The workflow:

- Checks out the repository
- Sets up Python environment
- Installs dependencies
- Runs the Modal script with your configured secrets

## 🔧 How It Works

### Training Pipeline (`src/train_and_submit.py`)

1. **Data Download**: Downloads feature metadata, training data, and validation data from Numerai v5.1
2. **Data Preparation**: Concatenates training and validation datasets (following the approach from notebook 4)
3. **Model Training**: Trains XGBoost on all available features using GPU acceleration
4. **Round Detection**: Gets the current tournament round using `get_current_round()`
5. **Live Data Download**: Downloads `live_{current_round}.parquet` for the current round
6. **Feature Extraction**: Extracts features using pattern matching (`[f for f in live_data.columns if "feature" in f]`)
7. **Prediction Generation**: Generates predictions for the live tournament data
8. **Submission**: Formats and uploads predictions to Numerai via their API

### Modal Infrastructure

- **GPU Support**: Uses NVIDIA T4 GPU for fast XGBoost training
- **Containerized**: All dependencies are containerized in a Modal image
- **Scalable**: Automatically scales based on workload
- **Secure**: API credentials stored in Modal secrets

### GitHub Actions Automation

- **Scheduled**: Configured to run daily via cron job (2:00 AM UTC) - **currently commented out by default**
- **Manual Trigger**: Can be triggered manually from GitHub UI at any time
- **Secure**: Uses GitHub Secrets for Modal authentication (Numerai credentials stored in Modal secrets)

## 📊 Research and Development

The model hyperparameters were determined through extensive experimentation documented in the Jupyter notebooks:

- **Notebook 1**: Getting started with Numerai data
- **Notebook 2**: Feature analysis and exploration
- **Notebook 3**: Model experiments and hyperparameter tuning (source of production model)
- **Notebook 4**: Final model evaluation and testing

Key findings from the experiments:

- XGBoost outperforms linear models (Ridge regression)
- Using all features performs better than feature selection methods
- The chosen hyperparameters provide good balance between performance and training time

## 📈 Monitoring

After each submission, you can monitor your model's performance on the [Numerai website](https://numer.ai/).

## 🤝 Contributing

This is a personal project, but suggestions and improvements are welcome!

## 📄 License

This project is for personal use in the Numerai Tournament.

## 🙏 Acknowledgments

- Numerai for providing the tournament platform and data
- Modal for cloud infrastructure
- The open-source community for excellent ML tools
