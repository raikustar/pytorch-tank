![alt text](https://raw.githubusercontent.com/raikustar/pytorch-tank/main/prediction.png)

Trained YOLOv5 model on images of Russian tanks, BMP1/2 and BMP3. Will try to identify those 3 objects.

 py -m virtualenv env
--
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
--
git clone https://github.com/ultralytics/yolov5
--
Yolo requirements:
pip install -r requirements.txt
--
