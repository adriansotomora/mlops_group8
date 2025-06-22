FROM python:3.10-slim

WORKDIR /code
ENV PYTHONPATH=/code/src

COPY environment.yml ./
RUN pip install conda && \
    conda env create -f environment.yml && \
    conda clean -afy

COPY app ./app
COPY src ./src
COPY config.yaml ./
COPY models ./models

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8000}/health || exit 1

CMD ["sh", "-c", "conda run -n group8_full uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
