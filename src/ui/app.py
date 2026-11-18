
import streamlit as st
import requests
import streamlit.components.v1 as components

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


# Header
st.title("📍 PathSnap")
st.subheader("GPS Map-Matching System")

st.write(
    "Select a GPS trip to generate its predicted road route."
)


# Fetch available trips
@st.cache_data(ttl=30)
def get_available_trips():

    response = requests.get(
        TRIPS_ENDPOINT,
        timeout=15
    )

    response.raise_for_status()

    return response.json()["trips"]


try:

    trips = get_available_trips()

    if not trips:

        st.warning("No student-test trips were found.")

    else:

        st.success(
            f"Connected to PathSnap API. "
            f"{len(trips)} student-test trips available."
        )

        selected_trip = st.selectbox(
            "Select a student-test trip",
            trips
        )

        if st.button(
            "Run Map Matching",
            type="primary"
        ):

            with st.spinner(
                "Generating candidates and predicting route..."
            ):

                response = requests.post(
                    f"{PREDICTION_ENDPOINT}/{selected_trip}",
                    timeout=300
                )

                response.raise_for_status()

                result = response.json()

            st.success("Map matching completed!")

            st.divider()

            st.subheader("Prediction Results")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Trip ID",
                result["trip_id"]
            )

            col2.metric(
                "Candidate rows",
                result["candidate_count"]
            )

            col3.metric(
                "HMM observations",
                result["hmm_info"]["n_steps"]
            )

            st.subheader("Predicted Route (HMM/Viterbi)")

            st.write(result["predicted_route"])

            with st.expander("View classifier-only route"):

                st.write(result["classifier_route"])

            with st.expander("HMM decoding information"):

                st.json(result["hmm_info"])

            st.divider()

            st.subheader("Geographical Visualisation")

            components.html(
                result["map_html"],
                height=650,
                scrolling=True
            )

except requests.exceptions.ConnectionError:

    st.error(
        "Cannot connect to the PathSnap backend. "
        "Please start the FastAPI server."
    )

except requests.exceptions.HTTPError as error:

    try:
        detail = error.response.json().get(
            "detail",
            str(error)
        )
    except Exception:
        detail = str(error)

    st.error(f"Backend error: {detail}")

except requests.exceptions.RequestException as error:

    st.error(f"Request failed: {error}")

except Exception as error:

    st.error(f"Something went wrong: {error}")
