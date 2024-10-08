![alt text](https://raw.githubusercontent.com/raikustar/pytorch-tank/main/prediction.png)

Trained YOLOv5 model on images of Russian tanks, BMP1/2 and BMP3. Will try to identify those 3 objects.

Git bash:
1. git clone https://github.com/raikustar/pytorch-tank
2. cd into pytorch-tank
3. clone yolo inside pytorch-tank | git clone https://github.com/ultralytics/yolov5
Done

--
Directory
-pytorch-tank
    - yolov5(folder)
    - other
    - stuff
    - randomimages(make folder)

--
Terminal:
../pytorch-tank/
4. Inside pytorch-tank
py -m virtualenv env
	- activate

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

