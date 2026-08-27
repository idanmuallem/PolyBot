# Upgraded to Python 3.12 to satisfy SciPy 1.17+
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app

# Install system dependencies required for C-extensions (like curl_cffi)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY config/requirements.txt ./requirements.txt
# Upgrading pip first prevents weird dependency resolution bugs
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your bot's code
COPY . .

# Normalize line endings and make executable — the repo may be checked out
# with CRLF line endings on Windows, which breaks `/bin/sh` shebang parsing.
RUN sed -i 's/\r$//' config/Docker/entrypoint.sh && chmod +x config/Docker/entrypoint.sh

EXPOSE 8501

# Runs both the headless trading engine (run_engine.py) and the Streamlit
# dashboard in this one container — see config/Docker/entrypoint.sh. The
# engine no longer starts only when a browser loads the dashboard page.
CMD ["./config/Docker/entrypoint.sh"]