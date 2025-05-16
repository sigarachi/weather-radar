#!/bin/bash

echo "Получение последнего релиза..."
LATEST_RELEASE_URL=$(curl -s https://api.github.com/repos/sigarachi/weather-radar/releases/latest | grep "zipball_url" | cut -d '"' -f 4)

if [ -z "$LATEST_RELEASE_URL" ]; then
    echo "Ошибка: Не удалось получить URL последнего релиза"
    exit 1
fi

if [ ! -d "unzip" ]; then
    echo "Установка unzip..."
    sudo apt-get install unzip
fi

# Download and extract the latest release
echo "Загрузка и распаковка последнего релиза..."
curl -L "$LATEST_RELEASE_URL" -o latest_release.zip
unzip -o latest_release.zip
rm latest_release.zip


# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Создание виртуального окружения..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "Установка зависимостей..."
pip install -r requirements.txt

# Get the latest release URL

# Check if screen session exists
if screen -list | grep -q "weather_api"; then
    echo "Остановка существующей сессии screen..."
    screen -X -S weather_api quit
fi

# Start new screen session with FastAPI
echo "Запуск FastAPI в сессии screen..."
screen -dmS weather_api bash -c "cd api && fastapi dev main.py"

echo "Развертывание завершено! Сессия screen 'weather_api' запущена."
echo "Для присоединения к сессии screen используйте: screen -r weather_api"
echo "Для отсоединения от сессии screen нажмите: Ctrl+A, затем D" 