# 1. Берем за основу легкую версию Python
FROM python:3.10-slim

# 2. Создаем папку внутри контейнера
WORKDIR /app

# 3. Копируем список библиотек и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Копируем весь остальной код
COPY . .

# 5. Говорим, какую команду запускать на старте
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]