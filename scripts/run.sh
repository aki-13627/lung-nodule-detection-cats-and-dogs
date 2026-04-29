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

sudo apt update
sudo apt install -y python3-pip python3-venv

python3 -m venv venv_yolo
source venv_yolo/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics pydicom pandas opencv-python tqdm

./venv_yolo/bin/python3 -c "import torch; print('PyTorch version:', torch.__version__); print('GPU available:', torch.cuda.is_available())"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="yolo_train_${TIMESTAMP}.log"

nohup bash -c "./venv_yolo/bin/python3 train.py" > $LOG_FILE 2>&1 &

echo "YOLO26 Process started in background."
echo "Log file: $LOG_FILE"
echo "To check progress: tail -f $LOG_FILE"
echo "PID: $!"