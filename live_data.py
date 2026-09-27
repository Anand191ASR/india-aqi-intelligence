"""Fetch current modelled air-quality readings for Indian states and UTs."""

from __future__ import annotations

import requests

from aqi_utils import calculate_reference_aqi
from india_states import STATE_DATA


API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
CURRENT_FIELDS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]


def fetch_live_state_data(timeout: int = 30) -> list[dict]:
    """Return current readings at each state/UT representative coordinate."""
    states = list(STATE_DATA)
    params = {
        "latitude": ",".join(str(STATE_DATA[state][0]) for state in states),
        "longitude": ",".join(str(STATE_DATA[state][1]) for state in states),
        "current": ",".join(CURRENT_FIELDS),
        "timezone": "Asia/Kolkata",
    }
    response = requests.get(API_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    locations = payload if isinstance(payload, list) else [payload]
    if len(locations) != len(states):
        raise ValueError("The API response did not include every state location.")

    records = []
    for state, location in zip(states, locations):
        current = location.get("current", {})
        pollutant_values = {
            "pm25": current.get("pm2_5"),
            "pm10": current.get("pm10"),
            "no2": current.get("nitrogen_dioxide"),
            "so2": current.get("sulphur_dioxide"),
            # Open-Meteo returns CO in ug/m3; our Indian AQI helper expects mg/m3.
            "co": (current.get("carbon_monoxide") or 0) / 1000,
            "o3": current.get("ozone"),
        }
        if any(value is None for value in pollutant_values.values()):
            continue
        latitude, longitude, _, causes = STATE_DATA[state]
        records.append(
            {
                "state": state,
                "latitude": latitude,
                "longitude": longitude,
                "time": current.get("time"),
                "aqi": calculate_reference_aqi(pollutant_values),
                "causes": causes,
                **pollutant_values,
            }
        )
    if not records:
        raise ValueError("The API returned no usable current readings.")
    return records


def search_indian_locations(query: str, timeout: int = 20) -> list[dict]:
    """Search Indian cities without requiring an API key."""
    if len(query.strip()) < 2:
        return []
    response = requests.get(
        GEOCODING_URL,
        params={"name": query.strip(), "count": 10, "language": "en", "format": "json", "countryCode": "IN"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json().get("results", [])


def fetch_live_location(latitude: float, longitude: float, timeout: int = 30) -> dict:
    """Fetch current air-quality values for one coordinate."""
    response = requests.get(
        API_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": ",".join(CURRENT_FIELDS),
            "timezone": "Asia/Kolkata",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    current = response.json().get("current", {})
    values = {
        "pm25": current.get("pm2_5"),
        "pm10": current.get("pm10"),
        "no2": current.get("nitrogen_dioxide"),
        "so2": current.get("sulphur_dioxide"),
        "co": (current.get("carbon_monoxide") or 0) / 1000,
        "o3": current.get("ozone"),
    }
    if any(value is None for value in values.values()):
        raise ValueError("Complete current readings are unavailable for this location.")
    return {"time": current.get("time"), "aqi": calculate_reference_aqi(values), **values}
