# PyTorch公式イメージを使用（CUDA対応）
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 作業ディレクトリを設定
WORKDIR /app

# システムパッケージの更新と必要なツールのインストール
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# requirements.txtをコピーして依存関係をインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# デフォルトコマンド（必要に応じて変更可能）
CMD ["python", "71baseline.py"]
