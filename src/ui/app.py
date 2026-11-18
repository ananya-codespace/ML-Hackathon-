
import streamlit as st
import requests

from src.ui.config import (
    TRIPS_ENDPOINT,
    PREDICTION_ENDPOINT
)


# Page configuration
st.set_page_config(
    page_title="PathSnap",
    page_icon="📍",
    layout="wide"
)


# Application title
st.title("📍 PathSnap")
st.subheader("GPS Map-Matching System")

st.write(
    "Select a GPS trip to generate its predicted road route."
)


# Fetch available trips from FastAPI
@st.cache_data(ttl=10)
def get_available_trips():

    try:
        response = requests.get(
            TRIPS_ENDPOINT,
            timeout=10
        )

        response.raise_for_status()

        return response.json()["trips"]

    except requests.RequestException:
        return None


# Display available trips
trips = get_available_trips()

if trips is None:

    st.error(
        "Cannot connect to the PathSnap backend. "
        "Please make sure FastAPI is running."
    )

else:

    st.success(
        f"Connected to PathSnap API. "
        f"{len(trips)} trips available."
    )

    selected_trip = st.selectbox(
        "Select a student-test trip",
        trips
    )

    # Run prediction
    if st.button("Run Map Matching", type="primary"):

        with st.spinner("Processing GPS trip..."):

            try:
                response = requests.post(
                    f"{PREDICTION_ENDPOINT}/{selected_trip}",
                    timeout=60
                )

                response.raise_for_status()

                result = response.json()

                st.subheader("Prediction Results")

                st.write(
                    f"**Trip ID:** {result['trip_id']}"
                )

                st.write(
                    "**Predicted Route:**"
                )

                st.write(
                    result["predicted_route"]
                )

                if result.get("is_mock"):
                    st.warning(
                        "This is a temporary mock prediction. "
                        "Actual ML results will be integrated later."
                    )

            except requests.RequestException as error:

                st.error(
                    f"Prediction request failed: {error}"
                )
