# Upgraded to Python 3.12 to satisfy SciPy 1.17+
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app

# Install system dependencies required for C-extensions (like curl_cffi)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt ./
# Upgrading pip first prevents weird dependency resolution bugs
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your bot's code
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "ui/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]