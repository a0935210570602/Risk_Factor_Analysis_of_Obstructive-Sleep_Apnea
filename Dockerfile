# 使用 Python 3.8 作為基礎
FROM python:3.8-slim

# 設定工作目錄
WORKDIR /app

# 複製本機所有檔案到容器中
COPY . .

# 安裝 Python 套件
RUN pip install --upgrade pip && pip install -r requirements.txt
# RUN pip install --no-cache-dir --progress-bar=off -r requirements.txt

# 預設執行指令（可依需求調整）
# CMD ["python", "run_experiments.py"]
