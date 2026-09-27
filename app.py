"""Streamlit dashboard for AQI prediction and India state analysis."""

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from aqi_utils import AQI_BANDS, aqi_category
from generate_data import generate_dataset
from india_states import SOLUTION_LIBRARY, STATE_DATA
from live_data import fetch_live_location, fetch_live_state_data, search_indian_locations
from train import FEATURES, train_models


MODEL_PATH = Path("artifacts/aqi_model.joblib")
METRICS_PATH = Path("artifacts/metrics.json")
DATA_PATH = Path("data/air_quality.csv")

st.set_page_config(page_title="India AQI Intelligence", page_icon="AQ", layout="wide")
st.title("India Air Quality Intelligence")
st.caption("AQI prediction, state comparison, possible causes and action ideas")


@st.cache_resource(show_spinner="Preparing the AQI model for first use...")
def ensure_project_assets() -> None:
    """Build reproducible runtime assets when a fresh deployment starts."""
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate_dataset().to_csv(DATA_PATH, index=False)

    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        training_data = pd.read_csv(DATA_PATH, parse_dates=["date"])
        trained_model, model_metrics = train_models(training_data)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(trained_model, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(model_metrics, indent=2), encoding="utf-8")


ensure_project_assets()

model = joblib.load(MODEL_PATH)
history = pd.read_csv(DATA_PATH, parse_dates=["date"])


@st.cache_data(ttl=1800, show_spinner=False)
def get_live_summary() -> pd.DataFrame:
    summary = pd.DataFrame(fetch_live_state_data())
    summary["category"] = summary["aqi"].map(lambda value: aqi_category(value)[0])
    return summary.sort_values("aqi", ascending=False)


def recommended_solutions(row: pd.Series) -> list[str]:
    particulate_score = row["pm25"] / 60 + row["pm10"] / 100
    gas_score = row["no2"] / 80 + row["so2"] / 80 + row["co"] / 2 + row["o3"] / 100
    primary = "pm" if particulate_score >= gas_score else "gas"
    return SOLUTION_LIBRARY[primary][:2] + SOLUTION_LIBRARY["general"][:2]


def pollution_reason_points(values) -> tuple[str, list[str]]:
    """Explain the current level using measured/modelled pollutant patterns."""
    aqi = float(values["aqi"])
    pm25 = float(values.get("pm25", 0) or 0)
    pm10 = float(values.get("pm10", 0) or 0)
    no2 = float(values.get("no2", 0) or 0)
    so2 = float(values.get("so2", 0) or 0)
    co = float(values.get("co", 0) or 0)
    o3 = float(values.get("o3", 0) or 0)

    if aqi > 100:
        points = []
        if pm25 > 60:
            points.append("PM2.5 is high, indicating a strong contribution from smoke, fuel burning, or fine combustion particles.")
        if pm10 > 100:
            points.append("PM10 is high, indicating possible road dust, construction dust, or other coarse suspended particles.")
        if no2 > 40 or co > 1:
            points.append("Elevated NO2 or CO can indicate emissions from traffic, generators, and fuel combustion.")
        if so2 > 40:
            points.append("Elevated SO2 can be associated with coal, thermal power generation, or industrial fuel burning.")
        if o3 > 100:
            points.append("High ozone can form when traffic and industrial gases react in sunlight.")
        if not points:
            points.append("The combined effect of multiple pollutants may be raising the AQI.")
        points.append("Low wind or regional pollution accumulation may contribute, but this view does not verify weather attribution.")
        return "Why might pollution be high?", points

    points = []
    if pm25 <= 30 and pm10 <= 50:
        points.append("PM2.5 and PM10 are low, indicating a lower current load of smoke and suspended dust.")
    if no2 <= 40 and so2 <= 40 and co <= 1:
        points.append("Combustion gases are low, suggesting a lower current impact from traffic and industrial emissions.")
    if o3 <= 50:
        points.append("Ground-level ozone is also low, indicating limited photochemical pollution at present.")
    points.append("Favourable dispersion, rainfall, or lower local activity may help, but weather data is needed for confirmation.")
    points.append("This is a current snapshot; levels can change quickly with rush hour, season, and wind.")
    return "Why might pollution be low?", points


def show_reason_points(values) -> None:
    title, points = pollution_reason_points(values)
    st.subheader(title)
    for point in points:
        st.write(f"- {point}")


@st.cache_data(ttl=1800, show_spinner=False)
def get_city_air_quality(latitude: float, longitude: float) -> dict:
    return fetch_live_location(latitude, longitude)


prediction_tab, city_tab, comparison_tab, states_tab, model_tab = st.tabs(
    ["AQI Predictor", "Live City Search", "State Comparison", "India State Overview", "Model Report"]
)

with prediction_tab:
    st.subheader("Enter air-quality readings")
    col1, col2 = st.columns(2)
    with col1:
        pm25 = st.slider("PM2.5 (ug/m3)", 0.0, 500.0, 65.0, 1.0)
        pm10 = st.slider("PM10 (ug/m3)", 0.0, 600.0, 110.0, 1.0)
        no2 = st.slider("NO2 (ug/m3)", 0.0, 500.0, 35.0, 1.0)
        so2 = st.slider("SO2 (ug/m3)", 0.0, 1000.0, 18.0, 1.0)
        co = st.slider("CO (mg/m3)", 0.0, 50.0, 1.2, 0.1)
    with col2:
        o3 = st.slider("O3 (ug/m3)", 0.0, 500.0, 45.0, 1.0)
        temperature = st.slider("Temperature (C)", -5.0, 55.0, 28.0, 0.5)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0, 1.0)
        wind_speed = st.slider("Wind speed (m/s)", 0.0, 20.0, 2.5, 0.1)
        rainfall = st.slider("Rainfall (mm)", 0.0, 100.0, 0.0, 0.5)

    input_data = pd.DataFrame(
        [[pm25, pm10, no2, so2, co, o3, temperature, humidity, wind_speed, rainfall]],
        columns=FEATURES,
    )
    prediction = float(max(0, min(500, model.predict(input_data)[0])))
    category, color, guidance = aqi_category(prediction)
    result, scale = st.columns([1, 2])
    with result:
        st.metric("Predicted AQI", f"{prediction:.0f}")
        st.markdown(
            f'<div style="background:{color};color:white;padding:12px;border-radius:6px;'
            f'font-size:20px;font-weight:700;text-align:center">{category}</div>',
            unsafe_allow_html=True,
        )
        st.info(guidance)
    with scale:
        bands = pd.DataFrame(AQI_BANDS, columns=["Start", "End", "Category", "Color", "Guidance"])
        chart = px.bar(
            bands, x="Category", y=bands["End"] - bands["Start"] + 1, base="Start",
            color="Category", color_discrete_map=dict(zip(bands["Category"], bands["Color"])),
            labels={"y": "AQI range"}, title="AQI category scale",
        )
        chart.add_hline(y=prediction, line_dash="dash", annotation_text=f"Prediction: {prediction:.0f}")
        chart.update_layout(showlegend=False, height=360)
        st.plotly_chart(chart, use_container_width=True)

with city_tab:
    st.subheader("Live city air quality - no login required")
    st.caption("Open-Meteo/CAMS model estimate. No account or API key is required.")
    city_query = st.text_input("Search for an Indian city", placeholder="Example: Ahmedabad, Surat, Delhi")
    if len(city_query.strip()) >= 2:
        try:
            locations = search_indian_locations(city_query)
            if not locations:
                st.warning("No matching location was found in India.")
            else:
                location_labels = [
                    f"{item['name']}, {item.get('admin1', '')} ({item.get('latitude'):.2f}, {item.get('longitude'):.2f})"
                    for item in locations
                ]
                selected_label = st.selectbox("Select a location", location_labels)
                location = locations[location_labels.index(selected_label)]
                city_data = get_city_air_quality(location["latitude"], location["longitude"])
                city_category, _, city_guidance = aqi_category(city_data["aqi"])
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Estimated AQI", f"{city_data['aqi']:.0f}")
                c2.metric("Category", city_category)
                c3.metric("PM2.5", f"{city_data['pm25']:.1f} ug/m3")
                c4.metric("PM10", f"{city_data['pm10']:.1f} ug/m3")
                st.caption(f"Model update: {city_data['time']} IST")
                st.info(city_guidance)
                show_reason_points(city_data)
                city_pollutants = pd.DataFrame(
                    {
                        "Pollutant": ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"],
                        "Value": [city_data[name] for name in ["pm25", "pm10", "no2", "so2", "co", "o3"]],
                    }
                )
                city_chart = px.bar(
                    city_pollutants, x="Pollutant", y="Value", color="Value",
                    color_continuous_scale="OrRd", title=f"{location['name']}: current pollutant estimate",
                )
                city_chart.update_layout(coloraxis_showscale=False, height=350)
                st.plotly_chart(city_chart, use_container_width=True)
                st.warning("This is a live model estimate, not a reading from a physical CPCB monitoring station.")
        except Exception as error:
            st.error(f"Could not load live city data: {error}")

with comparison_tab:
    st.subheader("Compare two States or Union Territories")
    st.caption("Live Open-Meteo/CAMS estimates at representative state-centre coordinates.")
    try:
        with st.spinner("Loading live state data..."):
            comparison_data = get_live_summary()
        state_names = comparison_data["state"].tolist()
        selector_one, selector_two = st.columns(2)
        with selector_one:
            first_state = st.selectbox(
                "First State / Union Territory",
                state_names,
                index=state_names.index("Delhi") if "Delhi" in state_names else 0,
            )
        with selector_two:
            second_default = "Gujarat" if "Gujarat" in state_names else state_names[1]
            second_state = st.selectbox(
                "Second State / Union Territory",
                state_names,
                index=state_names.index(second_default),
            )

        if first_state == second_state:
            st.warning("Select two different locations for comparison.")
        else:
            first = comparison_data.loc[comparison_data["state"] == first_state].iloc[0]
            second = comparison_data.loc[comparison_data["state"] == second_state].iloc[0]
            first_aqi, second_aqi = float(first["aqi"]), float(second["aqi"])
            difference = abs(first_aqi - second_aqi)
            higher_state = first_state if first_aqi > second_aqi else second_state

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"{first_state} AQI", f"{first_aqi:.0f}", aqi_category(first_aqi)[0])
            m2.metric(f"{second_state} AQI", f"{second_aqi:.0f}", aqi_category(second_aqi)[0])
            m3.metric("AQI difference", f"{difference:.0f}")
            m4.metric("Higher current AQI", higher_state)

            aqi_comparison = pd.DataFrame(
                {"Location": [first_state, second_state], "AQI": [first_aqi, second_aqi]}
            )
            aqi_chart = px.bar(
                aqi_comparison,
                x="Location",
                y="AQI",
                color="AQI",
                text="AQI",
                color_continuous_scale="RdYlGn_r",
                title="Current AQI comparison",
            )
            aqi_chart.update_layout(coloraxis_showscale=False, height=350)
            st.plotly_chart(aqi_chart, use_container_width=True)

            pollutant_names = ["pm25", "pm10", "no2", "so2", "co", "o3"]
            pollutant_comparison = pd.DataFrame(
                {
                    "Pollutant": [name.upper() for name in pollutant_names] * 2,
                    "Value": [first[name] for name in pollutant_names]
                    + [second[name] for name in pollutant_names],
                    "Location": [first_state] * len(pollutant_names)
                    + [second_state] * len(pollutant_names),
                }
            )
            pollutant_chart = px.bar(
                pollutant_comparison,
                x="Pollutant",
                y="Value",
                color="Location",
                barmode="group",
                title="Pollutant-level comparison",
            )
            pollutant_chart.update_layout(height=400)
            st.plotly_chart(pollutant_chart, use_container_width=True)

            explanation_one, explanation_two = st.columns(2)
            with explanation_one:
                st.subheader(first_state)
                show_reason_points(first)
            with explanation_two:
                st.subheader(second_state)
                show_reason_points(second)

            latest_comparison_time = max(str(first["time"]), str(second["time"]))
            st.caption(
                f"Model update: {latest_comparison_time} IST. These are state-centre estimates, not statewide averages."
            )
    except Exception as error:
        st.error(f"Could not prepare the state comparison: {error}")

with states_tab:
    header_col, refresh_col = st.columns([5, 1])
    with header_col:
        st.subheader("Current state-wise air quality")
    with refresh_col:
        if st.button("Refresh data", use_container_width=True):
            get_live_summary.clear()
            st.rerun()
    try:
        with st.spinner("Loading live air-quality data..."):
            summary = get_live_summary()
    except Exception as error:
        st.error(f"The live API is currently unavailable: {error}")
        st.stop()

    latest_time = summary["time"].dropna().max()
    st.caption(
        f"Updated: {latest_time} IST | Source: Open-Meteo Air Quality API (CAMS model, state-centre estimate). "
        "This is not an official CPCB monitoring-station reading."
    )
    map_chart = px.scatter_geo(
        summary, lat="latitude", lon="longitude", color="aqi", size="aqi",
        hover_name="state",
        hover_data={"aqi": ":.0f", "category": True, "time": True, "latitude": False, "longitude": False},
        color_continuous_scale="RdYlGn_r",
        range_color=(0, min(500, max(200, summary["aqi"].max()))),
        title="Current estimated AQI by state / UT",
    )
    map_chart.update_geos(
        scope="asia", projection_type="mercator", center={"lat": 22.5, "lon": 79},
        lataxis_range=[6, 38], lonaxis_range=[67, 98], showland=True, landcolor="#eef2f1",
        showcountries=True, countrycolor="#64748b",
    )
    map_chart.update_layout(height=590, margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(map_chart, use_container_width=True)

    selected_state = st.selectbox("Select a State or Union Territory", summary["state"].tolist())
    selected = summary.loc[summary["state"] == selected_state].iloc[0]
    category, _, guidance = aqi_category(selected["aqi"])
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Current estimated AQI", f"{selected['aqi']:.0f}")
    metric2.metric("PM2.5", f"{selected['pm25']:.1f} ug/m3")
    metric3.metric("Category", category)

    cause_col, solution_col = st.columns(2)
    with cause_col:
        show_reason_points(selected)
        st.caption(f"Regional context: {selected['causes']}")
        pollutant_values = selected[["pm25", "pm10", "no2", "so2", "co", "o3"]].astype(float)
        pollutant_chart = px.bar(
            x=pollutant_values.index.str.upper(), y=pollutant_values.values,
            labels={"x": "Pollutant", "y": "Average concentration"},
            title=f"{selected_state}: pollutant profile", color=pollutant_values.values,
            color_continuous_scale="OrRd",
        )
        pollutant_chart.update_layout(showlegend=False, coloraxis_showscale=False, height=330)
        st.plotly_chart(pollutant_chart, use_container_width=True)
    with solution_col:
        st.subheader("Suggested solutions")
        for solution in recommended_solutions(selected):
            st.write(f"- {solution}")
        st.info(guidance)

    st.subheader("All-state ranking")
    ranking = summary[["state", "aqi", "pm25", "pm10", "no2", "category", "time"]].copy()
    ranking.columns = ["State / UT", "Current AQI", "PM2.5", "PM10", "NO2", "Category", "Updated"]
    st.dataframe(ranking.round(1), use_container_width=True, hide_index=True)

with model_tab:
    if METRICS_PATH.exists():
        metadata = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.subheader("Model performance")
        metrics = pd.DataFrame(metadata["results"]).T.reset_index(names="Model")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        st.caption(f"Selected model: {metadata['best_model']} | Test rows: {metadata['test_rows']}")
    st.subheader("Recent AQI trend")
    st.line_chart(history.set_index("date")["aqi"].tail(300), height=280)

st.caption("Educational prototype: use official monitoring data for health or regulatory decisions.")
