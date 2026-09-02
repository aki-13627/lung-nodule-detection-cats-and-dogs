#!/bin/bash

REPO_URL="https://github.com/aki-13627/lung-nodule-detection-cats-and-dogs.git"
REPO_DIR="lung-nodule-detection-cats-and-dogs"

# 1. リポジトリの取得・更新
if [ -d "$REPO_DIR" ]; then
    cd $REPO_DIR
    git pull
else
    git clone $REPO_URL
    cd $REPO_DIR
fi

# 2. 仮想環境のセットアップ
sudo apt update
sudo apt install -y python3-pip python3-venv

python3 -m venv venv_dino
source venv_dino/bin/activate

# 3. ライブラリのインストール
pip install --upgrade pip
# CUDA 12.1 用のPyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# DINOやCOCOデータセット処理に必要なライブラリを追加
pip install transformers accelerate pydicom pandas opencv-python-headless tqdm timm scipy pycocotools Pillow

# 4. GPUの確認
./venv_dino/bin/python3 -c "import torch; print('PyTorch version:', torch.__version__); print('GPU available:', torch.cuda.is_available())"

# 5. バックグラウンド実行（train_dino.py を指定）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="dino_train_${TIMESTAMP}.log"

# nohup で train_dino.py を実行
nohup bash -c " ./venv_dino/bin/python3 train_dino.py" > $LOG_FILE 2>&1 &

echo "DINOv2 Training Process started in background."
echo "Log file: $LOG_FILE"
echo "To check progress: tail -f $LOG_FILE"
echo "PID: $!"