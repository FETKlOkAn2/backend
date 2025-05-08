FROM python:3.10-bullseye

WORKDIR /app

COPY requirements.txt .

# Install dependencies and TA-Lib C library
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
    # Build and install TA-Lib C library
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz && \
    # Add Microsoft SQL Server ODBC driver
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18 && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "-u", "-m", "core.webapp.app"]
