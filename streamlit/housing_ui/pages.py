from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

from housing_ui import api_client, db, registry, transforms

STATE_OPTIONS = [
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "District of Columbia",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Puerto Rico",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Virgin Islands",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
]
STATUS_OPTIONS = {"for_sale": "En venta", "sold": "Vendida"}
WATERFALL_COLORS = alt.Scale(
    domain=["base", "sube", "baja", "total"], range=["#64748b", "#16a34a", "#dc2626", "#2563eb"]
)


@st.cache_data(ttl=30, show_spinner=False)
def load_batch_runs() -> pd.DataFrame:
    return db.batch_runs()


@st.cache_data(ttl=30, show_spinner=False)
def load_model_actions() -> pd.DataFrame:
    return db.model_actions()


@st.cache_data(ttl=30, show_spinner=False)
def load_registry() -> tuple[dict, dict]:
    aliases = registry.aliases()
    versions = {number: registry.version(number) for number in set(aliases.values())}
    return aliases, versions


def version_label(model: dict | None) -> str:
    return f"v{model['version']}" if model else "ninguno"


def money(value: float) -> str:
    return f"US$ {value:,.0f}".replace(",", ".")


def prediction_page():
    st.title("Predicción de precio")
    st.caption("La predicción la hace la API con el modelo champion de MLflow.")
    with st.form("predict"):
        left, middle, right = st.columns(3)
        status = left.selectbox(
            "Estado de la venta", list(STATUS_OPTIONS), format_func=STATUS_OPTIONS.get
        )
        state = left.selectbox("Estado", STATE_OPTIONS, index=STATE_OPTIONS.index("Massachusetts"))
        city = left.text_input("Ciudad", "Springfield")
        zip_code = middle.text_input("Código postal", "01109", max_chars=5)
        bed = middle.number_input("Habitaciones", 1, 20, 3)
        bath = middle.number_input("Baños", 1, 20, 2)
        house_size = right.number_input("Área (pies²)", 100, 30_000, 1_800, step=50)
        acre_lot = right.number_input("Lote (acres)", 0.0, 1_000.0, 0.25, step=0.05)
        sold_before = right.checkbox("Tiene venta previa", True)
        prev_sold = right.date_input(
            "Fecha de la venta previa",
            date(2015, 6, 15),
            min_value=date(1900, 1, 1),
            max_value=date.today(),
            disabled=not sold_before,
        )
        broker = left.text_input("Agencia (opcional)", "22611")
        submitted = st.form_submit_button("Predecir", type="primary")

    if submitted:
        payload = {
            "status": status,
            "bed": bed,
            "bath": bath,
            "acre_lot": acre_lot,
            "house_size": house_size,
            "prev_sold_date": prev_sold.isoformat() if sold_before else None,
            "city": city,
            "state": state,
            "zip_code": zip_code,
            "brokered_by": broker or None,
        }
        try:
            st.session_state["last"] = {"payload": payload, "result": api_client.predict(payload)}
            st.session_state.pop("explanation", None)
        except api_client.ApiError as error:
            st.error(str(error))

    last = st.session_state.get("last")
    if not last:
        return
    result = last["result"]
    first, second, third = st.columns(3)
    first.metric("Precio estimado", money(result["predicted_price"]))
    second.metric("Versión del modelo", f"v{result['model_version']}")
    third.metric("Alias", result["model_alias"])
    st.caption(f"request_id: `{result['request_id']}`")

    if st.button("¿Por qué este precio?"):
        try:
            st.session_state["explanation"] = api_client.explain(last["payload"])
        except api_client.ApiError as error:
            st.error(str(error))
    explanation = st.session_state.get("explanation")
    if explanation:
        show_explanation(explanation)

    with st.form("feedback"):
        st.write("¿Conoce el precio real de esta propiedad?")
        actual = st.number_input("Precio real (US$)", 10_000, 20_000_000, 300_000, step=1_000)
        if st.form_submit_button("Enviar precio real"):
            try:
                api_client.feedback(result["request_id"], actual)
                st.success("Gracias: el precio real quedó asociado a esta predicción.")
            except api_client.ApiError as error:
                st.error(str(error))


def show_explanation(explanation: dict):
    steps = transforms.waterfall(explanation)
    st.subheader(f"Contribución de cada variable (modelo v{explanation['model_version']})")
    chart = (
        alt.Chart(steps)
        .mark_bar()
        .encode(
            x=alt.X("Paso:N", sort=None, title=None, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("Inicio:Q", title="Precio (US$)"),
            y2="Fin:Q",
            color=alt.Color("Tipo:N", scale=WATERFALL_COLORS, legend=None),
            tooltip=[
                "Paso",
                "Valor",
                alt.Tooltip("Efecto:Q", format=",.0f"),
                alt.Tooltip("Fin:Q", title="Precio acumulado", format=",.0f"),
            ],
        )
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        "Valores SHAP del modelo: parten del precio base y cada variable sube (verde) o baja "
        "(rojo) el precio hasta llegar a la predicción."
    )


def history_page():
    st.title("Historial de lotes")
    try:
        runs = load_batch_runs()
    except Exception as error:
        st.error(f"No se pudo leer la auditoría: {error}")
        return
    if runs.empty:
        st.info("Todavía no se ha procesado ningún lote.")
        return

    cycles = sorted(int(c) for c in runs["cycle"].unique())
    selected = st.multiselect("Ciclo", cycles, default=cycles)
    view = runs[runs["cycle"].isin(selected)]
    stats = transforms.summary(view)
    columns = st.columns(4)
    columns[0].metric("Lotes procesados", stats["batches"])
    columns[1].metric("Entrenamientos", stats["trainings"])
    columns[2].metric("Promociones", stats["promotions"])
    columns[3].metric(
        "Champion actual", f"v{stats['champion']}" if stats["champion"] else "ninguno"
    )

    st.dataframe(
        transforms.batch_table(view, registry.version_url),
        hide_index=True,
        column_config={"MLflow": st.column_config.LinkColumn("MLflow", display_text="ver")},
    )

    left, right = st.columns(2)
    mape = transforms.champion_mape_series(view)
    if not mape.empty:
        left.subheader("MAPE del champion en cada lote")
        left.altair_chart(
            alt.Chart(mape)
            .mark_line(point=True)
            .encode(
                x=alt.X("Ingreso:O"),
                y=alt.Y("MAPE %:Q"),
                color="Champion:N",
                tooltip=["Ingreso", "Champion", "Llegada", alt.Tooltip("MAPE %:Q", format=".2f")],
            ),
            width="stretch",
        )
    duel = transforms.candidate_vs_champion(view)
    if not duel.empty:
        right.subheader("Candidata frente a champion (holdout)")
        right.altair_chart(
            alt.Chart(duel)
            .mark_bar()
            .encode(
                x=alt.X("Ingreso:O"),
                xOffset="Modelo:N",
                y=alt.Y("MAPE %:Q"),
                color="Modelo:N",
                tooltip=["Ingreso", "Modelo", "Decisión", alt.Tooltip("MAPE %:Q", format=".2f")],
            ),
            width="stretch",
        )

    st.subheader("Detalle de un ingreso")
    ingestion = st.selectbox("Ingreso", view["ingestion_id"].tolist()[::-1])
    row = view[view["ingestion_id"] == ingestion].iloc[0]
    for column, title in (
        ("schema_report", "Esquema"),
        ("quality_report", "Calidad"),
        ("new_categories", "Categorías nuevas"),
        ("drift_report", "Drift"),
        ("gates", "Compuertas"),
        ("diagnostics", "Diagnóstico"),
    ):
        with st.expander(title):
            value = row[column]
            if value is None or (isinstance(value, float) and pd.isna(value)):
                st.write("Sin datos para este ingreso.")
            else:
                st.json(value)
    if row["error"]:
        st.error(row["error"])


def model_page():
    st.title("Modelo en producción")
    try:
        info = api_client.model_info()
        champion = info.get("champion")
        columns = st.columns(3)
        shadow = info.get("shadow")
        columns[0].metric("Champion en la API", version_label(champion))
        columns[1].metric("En sombra", version_label(shadow))
        columns[2].metric("Recarga cada", f"{info['reload_interval_seconds']:.0f} s")
        st.caption(f"Réplica que respondió: `{info['instance']}` · imagen `{info['git_sha'][:7]}`")
    except api_client.ApiError as error:
        st.warning(str(error))

    try:
        aliases, versions = load_registry()
        st.subheader("Alias en MLflow")
        st.dataframe(
            transforms.alias_rows(aliases, versions, registry.version_url),
            hide_index=True,
            column_config={"MLflow": st.column_config.LinkColumn("MLflow", display_text="ver")},
        )
    except Exception as error:
        st.warning(f"No se pudo consultar MLflow: {error}")

    try:
        actions = load_model_actions()
        st.subheader("Últimas acciones sobre el modelo")
        if actions.empty:
            st.write("Sin recargas ni rollbacks registrados.")
        else:
            st.dataframe(actions, hide_index=True)
    except Exception as error:
        st.warning(f"No se pudo leer audit.model_actions: {error}")
