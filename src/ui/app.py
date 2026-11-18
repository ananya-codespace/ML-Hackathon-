
import streamlit as st
import requests
import streamlit.components.v1 as components

from src.ui.config import (
    TRIPS_ENDPOINT,
    PREDICTION_ENDPOINT
)


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PathSnap",
    page_icon="📍",
    layout="wide"
)

st.title("📍 PathSnap")
st.subheader("GPS Map-Matching System")

st.write(
    "Select a GPS trip to generate its predicted road route "
    "using machine learning and HMM/Viterbi decoding."
)

st.divider()


# --------------------------------------------------
# 2. FETCH AVAILABLE TRIPS
# --------------------------------------------------

@st.cache_data(ttl=10)
def get_available_trips():
    try:
        response = requests.get(
            TRIPS_ENDPOINT,
            timeout=10
        )
        response.raise_for_status()

        data = response.json()
        return data.get("trips", [])

    except requests.RequestException as error:
        st.error(f"Unable to connect to backend: {error}")
        return None


trips = get_available_trips()


# --------------------------------------------------
# 3. TRIP SELECTION
# --------------------------------------------------

if trips is None:

    st.error(
        "Cannot connect to the PathSnap backend. "
        "Please make sure FastAPI is running."
    )

elif not trips:

    st.warning("No student-test trips are available.")

else:

    st.success(
        f"Connected to PathSnap API. "
        f"{len(trips)} trips available."
    )

    selected_trip = st.selectbox(
        "Select a student-test trip",
        trips
    )

    run_prediction = st.button(
        "🚀 Run Map Matching",
        type="primary",
        use_container_width=True
    )


    # --------------------------------------------------
    # 4. RUN PREDICTION
    # --------------------------------------------------

    if run_prediction:

        with st.spinner(
            "Generating candidates and reconstructing route..."
        ):

            try:
                response = requests.post(
                    f"{PREDICTION_ENDPOINT}/{selected_trip}",
                    timeout=180
                )

                response.raise_for_status()
                result = response.json()

                # --------------------------------------
                # 5. PREDICTION RESULTS
                # --------------------------------------

                st.divider()
                st.header("Prediction Results")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Trip ID",
                    result.get("trip_id", selected_trip)
                )

                col2.metric(
                    "Candidate rows",
                    f"{result.get('candidate_rows', 0):,}"
                )

                hmm_info = result.get("hmm_info") or {}

                observations = result.get(
                    "hmm_observations",
                    hmm_info.get("n_steps", 0)
                )

                col3.metric(
                    "HMM observations",
                    f"{observations:,}"
                )


                # --------------------------------------
                # 6. PREDICTED ROUTE
                # --------------------------------------

                predicted_route = result.get(
                    "predicted_route", []
                )

                st.subheader("Predicted Route (HMM/Viterbi)")

                if predicted_route:

                    # Display long routes in expandable groups
                    for start in range(0, len(predicted_route), 100):

                        end = min(
                            start + 100,
                            len(predicted_route)
                        )

                        with st.expander(
                            f"[ {start} – {end - 1} ]"
                        ):
                            st.write(
                                predicted_route[start:end]
                            )

                    st.caption(
                        f"Total road segments in predicted route: "
                        f"{len(predicted_route):,}"
                    )

                else:
                    st.warning(
                        "The API did not return a predicted route."
                    )


                # --------------------------------------
                # 7. PREDICTION METRICS
                # --------------------------------------

                st.divider()
                st.subheader("Prediction Metrics")

                metrics = result.get("metrics")

                if metrics:

                    if isinstance(metrics, dict):

                        metric_cols = st.columns(
                            min(len(metrics), 4)
                        )

                        for index, (name, value) in enumerate(
                            metrics.items()
                        ):
                            with metric_cols[
                                index % len(metric_cols)
                            ]:
                                st.metric(
                                    name.replace("_", " ").title(),
                                    value if isinstance(
                                        value, (int, float, str)
                                    ) else str(value)
                                )

                        with st.expander("View complete metrics"):
                            st.json(metrics)

                    else:
                        st.json(metrics)

                else:
                    st.info(
                        "No prediction metrics were included "
                        "in the API response."
                    )


                # --------------------------------------
                # 8. HMM INFORMATION
                # --------------------------------------

                if hmm_info:

                    st.divider()
                    st.subheader("HMM/Viterbi Information")
                    st.json(hmm_info)


                # --------------------------------------
                # 9. MAP VISUALIZATION
                # --------------------------------------

                map_html = result.get("map_html")

                if map_html:

                    st.divider()
                    st.subheader("Predicted Route Map")

                    components.html(
                        map_html,
                        height=650,
                        scrolling=True
                    )

                else:

                    st.info(
                        "Map HTML was not included in the API "
                        "response. Check the backend response "
                        "for the map field."
                    )


                # --------------------------------------
                # 10. PREDICTION STATUS
                # --------------------------------------

                if result.get("is_mock", False):

                    st.warning(
                        "This is a mock prediction, not actual "
                        "ML output."
                    )

                else:

                    st.success(
                        "Map-matching prediction completed successfully!"
                    )


            except requests.HTTPError as error:

                st.error(
                    f"Backend returned an HTTP error: {error}"
                )

                try:
                    st.json(response.json())
                except ValueError:
                    st.write(response.text)

            except requests.RequestException as error:

                st.error(
                    f"Prediction request failed: {error}"
                )

            except Exception as error:

                st.error(
                    f"An unexpected error occurred: {error}"
                )
