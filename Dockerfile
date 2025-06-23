FROM continuumio/miniconda3:latest

WORKDIR /code
ENV PYTHONPATH=/code/src

# Update system packages and install build essentials
RUN apt-get update && \
    apt-get install -y build-essential libstdc++6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
 
# Copy environment file and install dependencies
COPY environment.yml .
RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict && \
    conda env create -f environment.yml && \
    conda clean -afy

# Copy application code and configuration
COPY app ./app
COPY src ./src
COPY config.yaml ./
COPY models ./models

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8000}/health || exit 1

CMD ["conda", "run", "--no-capture-output", "-n", "group8_full", "python", "-u", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]