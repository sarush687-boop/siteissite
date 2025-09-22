#!/bin/bash

# Install system dependencies needed for Playwright and Chromium
apt-get update && apt-get install -y wget gnupg curl unzip fonts-liberation libasound2 libatk-bridge2.0-0 \
  libatk1.0-0 libcups2 libdbus-1-3 libdrm2 libgbm1 libnspr4 libnss3 libx11-xcb1 libxcomposite1 libxdamage1 \
  libxrandr2 xdg-utils libu2f-udev libvulkan1 libxss1 libappindicator3-1 libasound2 libxshmfence1 libexpat1

# Install Playwright browsers (Chromium in this case)
playwright install chromium

# Optional: Create a directory if needed for caching
mkdir -p .wdm_cache

