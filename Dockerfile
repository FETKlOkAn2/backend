FROM python:3.10-bullseye

WORKDIR /app

COPY requirements.txt .

# 1) Install dependencies and TA‑Lib C library
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        curl \
        gnupg2 \
        make \
        wget \
        apt-transport-https \
        ca-certificates \
        unixodbc \
        unixodbc-dev \
        libgssapi-krb5-2 \
        libssl-dev \
        libffi-dev \
    && \
    wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && \
    tar -xzf ta-lib-0.6.4-src.tar.gz && \
    cd ta-lib-0.6.4 && \
    ./configure && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.6.4 ta-lib-0.6.4-src.tar.gz && \
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18 && \
    rm -rf /var/lib/apt/lists/*

# 2) Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 3) Install OpenSSL, generate a self-signed cert, then remove OpenSSL to slim down
RUN apt-get update && \
    apt-get install -y openssl && \
    mkdir -p /certs && \
    openssl req -x509 -nodes -days 365 \
      -newkey rsa:2048 \
      -keyout /certs/selfsigned.key \
      -out /certs/selfsigned.crt \
      -subj "/C=US/ST=State/L=City/O=CryptoBot/CN=ec2-54-196-241-200.compute-1.amazonaws.com" && \
    apt-get purge -y --auto-remove openssl && \
    rm -rf /var/lib/apt/lists/*

# 4) Copy in your application
COPY . .

# 5) Expose HTTPS port
EXPOSE 5000

# 6) Launch Flask with SSL context
ENTRYPOINT ["python","-u","-m","core.webapp.app"]
CMD ["--certfile","/certs/selfsigned.crt","--keyfile","/certs/selfsigned.key"]

