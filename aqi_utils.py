"""Shared AQI calculation and display helpers."""

from __future__ import annotations

from typing import Iterable


AQI_BANDS = [
    (0, 50, "Good", "#16875b", "Air quality is good. Normal outdoor activity is generally safe."),
    (51, 100, "Satisfactory", "#65a30d", "Sensitive people should limit unusually long outdoor activity."),
    (101, 200, "Moderate", "#d97706", "Sensitive groups may experience breathing discomfort."),
    (201, 300, "Poor", "#ea580c", "Reduce prolonged outdoor exposure and consider appropriate protection."),
    (301, 400, "Very Poor", "#dc2626", "Avoid outdoor exertion and consider keeping windows closed."),
    (401, 500, "Severe", "#7f1d1d", "Avoid outdoor activity and follow local health advisories."),
]


def aqi_category(aqi: float) -> tuple[str, str, str]:
    """Return category, color and health guidance for an AQI value."""
    value = max(0.0, min(float(aqi), 500.0))
    for low, high, label, color, guidance in AQI_BANDS:
        if low <= value <= high:
            return label, color, guidance
    return AQI_BANDS[-1][2:]


def pollutant_subindex(
    concentration: float, breakpoints: Iterable[tuple[float, float, int, int]]
) -> float:
    """Linearly interpolate an AQI sub-index from concentration breakpoints."""
    value = max(0.0, float(concentration))
    points = list(breakpoints)
    for c_low, c_high, i_low, i_high in points:
        if value <= c_high:
            return i_low + (value - c_low) * (i_high - i_low) / (c_high - c_low)
    return 500.0


BREAKPOINTS = {
    "pm25": [(0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200), (90, 120, 201, 300), (120, 250, 301, 400), (250, 500, 401, 500)],
    "pm10": [(0, 50, 0, 50), (50, 100, 51, 100), (100, 250, 101, 200), (250, 350, 201, 300), (350, 430, 301, 400), (430, 600, 401, 500)],
    "no2": [(0, 40, 0, 50), (40, 80, 51, 100), (80, 180, 101, 200), (180, 280, 201, 300), (280, 400, 301, 400), (400, 800, 401, 500)],
    "so2": [(0, 40, 0, 50), (40, 80, 51, 100), (80, 380, 101, 200), (380, 800, 201, 300), (800, 1600, 301, 400), (1600, 2400, 401, 500)],
    "co": [(0, 1, 0, 50), (1, 2, 51, 100), (2, 10, 101, 200), (10, 17, 201, 300), (17, 34, 301, 400), (34, 50, 401, 500)],
    "o3": [(0, 50, 0, 50), (50, 100, 51, 100), (100, 168, 101, 200), (168, 208, 201, 300), (208, 748, 301, 400), (748, 1000, 401, 500)],
}


def calculate_reference_aqi(row: dict[str, float]) -> float:
    """Approximate CPCB-style AQI as the largest pollutant sub-index."""
    values = [pollutant_subindex(row[name], BREAKPOINTS[name]) for name in BREAKPOINTS]
    return round(min(max(values), 500.0), 2)
