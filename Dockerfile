FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PATH="/app/scripts:${PATH}"

WORKDIR /app

# system deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY requirements.txt ./

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# copy project
COPY . .

# make entrypoint executable
RUN chmod +x ./scripts/entrypoint.sh

# default command: run the entrypoint which launches run_batch.py
ENTRYPOINT [ "/app/scripts/entrypoint.sh" ]