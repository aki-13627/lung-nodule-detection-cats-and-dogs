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

python3 -m venv venv_detr
source venv_detr/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate pydicom pandas opencv-python-headless tqdm timm scipy

./venv_detr/bin/python3 -c "import torch; print('PyTorch version:', torch.__version__); print('GPU available:', torch.cuda.is_available())"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="check_${TIMESTAMP}.log"

nohup bash -c "./venv_detr/bin/python3 utils/check.py" > $LOG_FILE 2>&1 &

echo "DETR Process started in background."
echo "Log file: $LOG_FILE"
echo "To check progress: tail -f $LOG_FILE"
echo "PID: $!"