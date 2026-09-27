"""Train, compare and save AQI regression models."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


DATA_PATH = Path("data/air_quality.csv")
ARTIFACT_DIR = Path("artifacts")
FEATURES = [
    "pm25", "pm10", "no2", "so2", "co", "o3",
    "temperature", "humidity", "wind_speed", "rainfall",
]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    required = set(FEATURES + ["date", "state", "aqi"])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"The CSV is missing required columns: {sorted(missing)}")
    return frame.sort_values("date").reset_index(drop=True)


def evaluate(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_test)
    return {
        "MAE": round(mean_absolute_error(y_test, predictions), 3),
        "RMSE": round(mean_squared_error(y_test, predictions) ** 0.5, 3),
        "R2": round(r2_score(y_test, predictions), 4),
    }


def train_models(frame: pd.DataFrame) -> tuple[Pipeline, dict]:
    split_at = int(len(frame) * 0.8)
    train_set, test_set = frame.iloc[:split_at], frame.iloc[split_at:]
    x_train, y_train = train_set[FEATURES], train_set["aqi"]
    x_test, y_test = test_set[FEATURES], test_set["aqi"]

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=180, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=180, random_state=42, n_jobs=-1),
    }
    results = {}
    trained = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        trained[name] = pipeline
        results[name] = evaluate(pipeline, x_test, y_test)

    best_name = min(results, key=lambda name: results[name]["RMSE"])
    metadata = {
        "best_model": best_name,
        "features": FEATURES,
        "train_rows": len(train_set),
        "test_rows": len(test_set),
        "results": results,
    }
    return trained[best_name], metadata


if __name__ == "__main__":
    data = load_data()
    best_model, model_metadata = train_models(data)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, ARTIFACT_DIR / "aqi_model.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")
    print(json.dumps(model_metadata, indent=2))
