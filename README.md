![alt text](https://raw.githubusercontent.com/raikustar/pytorch-tank/main/prediction.png)

Trained YOLOv26 model on images of Russian tanks, BMP1/2 and BMP3. Will try to identify those 3 objects.


# **TBD**
In the process of getting updated to a newer version. Will include:
- docker image
- improved model
- trained from more data


# 
py -3.12 -m venv env
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt




# Dataset
path: ../pytorch-tank/datasets
train: data/train_data
val: data/test_data

names:
  0: background
  1: Tank
  2: BMP1/2
  3: BMP3