#!/bin/bash

REPO_URL="https://github.com/aki-13627/lung-nodule-detection-cats-and-dogs.git"
REPO_DIR="lung-nodule-detection-cats-and-dogs"

if [ -d "$REPO_DIR" ]; then
    cd $REPO_DIR
    git pull
else
    git clone $REPO_URL
    cd $REPO_DIR
fi

# 修正: libgl1-mesa-glx を現代の Ubuntu に合わせた libgl1 に変更
sudo apt update
sudo apt install -y python3-pip python3-venv libgl1 libglib2.0-0

python3 -m venv venv_rtdetr
source venv_rtdetr/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 先に headless 版を入れておき、後から入るパッケージの競合を防ぐ
pip install opencv-python-headless
pip install ultralytics pycocotools

./venv_rtdetr/bin/python3 -c "import torch; print('PyTorch version:', torch.__version__); print('GPU available:', torch.cuda.is_available()); import cv2; print('OpenCV imported successfully!')"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="rtdetr_train_${TIMESTAMP}.log"

nohup bash -c "./venv_rtdetr/bin/python3 train_rtdetr.py" > $LOG_FILE 2>&1 &

echo "RT-DETR Training Process started in background."
echo "Log file: $LOG_FILE"
echo "To check progress: tail -f $LOG_FILE"
echo "PID: $!"