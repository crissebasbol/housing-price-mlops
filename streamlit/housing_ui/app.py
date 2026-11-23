import streamlit as st

from housing_ui import pages
from housing_ui.config import SETTINGS

st.set_page_config(page_title="Housing MLOps", page_icon=":material/home:", layout="wide")
st.sidebar.caption(f"MLflow: {SETTINGS.mlflow_public_url}")
st.navigation(
    [
        st.Page(pages.prediction_page, title="Predicción", icon=":material/sell:"),
        st.Page(pages.history_page, title="Historial de lotes", icon=":material/timeline:"),
        st.Page(pages.model_page, title="Modelo en producción", icon=":material/model_training:"),
    ]
).run()
