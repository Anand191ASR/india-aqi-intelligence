"""Generate a reproducible demo dataset for learning and local testing."""

from pathlib import Path

import numpy as np
import pandas as pd

from aqi_utils import calculate_reference_aqi
from india_states import STATE_DATA


OUTPUT_PATH = Path("data/air_quality.csv")


def generate_dataset(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=rows, freq="3h")
    month = dates.month.to_numpy()

    state_names = np.array(list(STATE_DATA))
    states = rng.choice(state_names, size=rows, replace=True)
    state_factor = np.array([STATE_DATA[state][2] for state in states])

    winter = np.where(np.isin(month, [11, 12, 1, 2]), 1.45, 0.85)
    monsoon = np.where(np.isin(month, [6, 7, 8, 9]), 0.65, 1.0)
    pollution_factor = winter * monsoon * state_factor

    pm25 = np.clip(rng.gamma(2.4, 28, rows) * pollution_factor, 4, 450)
    pm10 = np.clip(pm25 * rng.normal(1.65, 0.2, rows) + rng.normal(15, 8, rows), 8, 600)
    no2 = np.clip(rng.gamma(2.0, 20, rows) * pollution_factor, 3, 500)
    so2 = np.clip(rng.gamma(1.7, 10, rows) * pollution_factor, 2, 1000)
    co = np.clip(rng.gamma(1.8, 0.65, rows) * pollution_factor, 0.1, 30)
    o3 = np.clip(rng.normal(45, 20, rows) + (month - 6) ** 2, 2, 500)

    seasonal_temp = 26 + 9 * np.sin((month - 3) * np.pi / 6)
    temperature = seasonal_temp + rng.normal(0, 4, rows)
    humidity = np.clip(62 - (temperature - 25) * 1.2 + rng.normal(0, 12, rows), 15, 100)
    wind_speed = np.clip(rng.gamma(2.0, 1.2, rows), 0.1, 15)
    rainfall = np.where(np.isin(month, [6, 7, 8, 9]), rng.gamma(1.3, 4, rows), rng.gamma(0.4, 1, rows))

    frame = pd.DataFrame(
        {
            "date": dates,
            "state": states,
            "pm25": pm25,
            "pm10": pm10,
            "no2": no2,
            "so2": so2,
            "co": co,
            "o3": o3,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "rainfall": rainfall,
        }
    )
    reference = frame.apply(lambda row: calculate_reference_aqi(row.to_dict()), axis=1)
    weather_effect = np.clip((humidity - 60) * 0.08 - wind_speed * 1.2 - rainfall * 0.3, -15, 15)
    frame["aqi"] = np.clip(reference + weather_effect + rng.normal(0, 4, rows), 0, 500).round(2)
    return frame.round(2)


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Created {OUTPUT_PATH} with {len(dataset):,} rows")
