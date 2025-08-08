FROM python:3.8-slim

WORKDIR /app

# 先複製 requirements.txt，利用 layer cache 加速後續 build
COPY requirements.txt .

# 安裝 Python 套件
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 複製應用程式剩餘檔案
COPY . .
# RUN pip install --no-cache-dir --progress-bar=off -r requirements.txt
CMD ["python", "main.py"]