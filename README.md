# Air Quality Index Prediction

A beginner-friendly machine-learning project that predicts AQI from pollutant and weather readings.

## Project flow

1. `generate_data.py` creates a reproducible demonstration dataset.
2. `train.py` performs a chronological 80/20 split, compares three regression models, and saves the best model.
3. `app.py` provides an interactive Streamlit dashboard for predictions.
4. `India State Overview` displays near-real-time Open-Meteo/CAMS state-centre estimates, possible causes, and suggested actions.
5. `Live City Search` displays a current model estimate for an Indian city without requiring an account or API key.
6. `State Comparison` compares live AQI and pollutant levels for any two States or Union Territories.

## Setup on Windows

Install Python 3.11 or 3.12 and select **Add Python to PATH** during installation. Then run these commands in the project directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python generate_data.py
python train.py
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Dataset columns

`date`, `state`, `pm25`, `pm10`, `no2`, `so2`, `co`, `o3`, `temperature`, `humidity`, `wind_speed`, `rainfall`, `aqi`

To use a real dataset, place a CSV with this schema at `data/air_quality.csv`, then run `python train.py` again.

## Important concepts

- **Features:** Model inputs such as PM2.5 and humidity.
- **Target:** The output being predicted, which is `aqi`.
- **Train/test split:** The oldest 80% of records are used for training and the latest 20% for unbiased testing.
- **MAE/RMSE:** Prediction-error metrics; lower values are better.
- **R2:** The proportion of variation explained by the model; values closer to 1 are better.

The synthetic dataset is intended for learning the pipeline. Use official historical data for a final academic report.

The state and city views use the Open-Meteo Air Quality API and cache results for 30 minutes. The values are CAMS model estimates, not official CPCB monitoring-station observations.
