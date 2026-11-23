FROM python:3.12-slim

ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA} \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

RUN useradd --create-home --uid 1000 app
WORKDIR /home/app
COPY --chown=app:app housing_ui ./housing_ui
ENV PYTHONPATH=/home/app
USER app

RUN python -c "import altair, psycopg2, requests, streamlit; from housing_ui import transforms"

EXPOSE 8501
CMD ["streamlit", "run", "housing_ui/app.py"]
