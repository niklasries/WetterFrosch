FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel


ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=Etc/UTC
ENV PATH="/root/.local/bin:${PATH}"


RUN apt-get update && apt-get install -y --no-install-recommends \
    # Tools needed to add the Google Chrome repository
    wget \
    gnupg \
    \
    # dependencies
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    \
    # chrome dependencies
    libnss3 \
    libgconf-2-4 \
    libfontconfig1 \
    fonts-liberation \
    libappindicator3-1 \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libc6 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libexpat1 \
    libgbm1 \
    libgcc1 \
    libnspr4 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libstdc++6 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxss1 \
    libxtst6 \
    # --- End of Chrome Dependencies ---
    \
    && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    \
    # Clean up apt caches to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /wetter

# Copy and install Python requirements
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt



ENTRYPOINT ["python3", "wetter.py"]