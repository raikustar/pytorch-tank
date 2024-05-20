import torch
from torchvision import transforms
from torch.nn.functional import softmax
import cv2 as cv
import numpy as np
from model import NeuralNetwork
import os
from PIL import Image 

def loadModelResultCustomCNN(model, image_path, model_file):
  model = NeuralNetwork()
  model.load_state_dict(torch.load(model_file))
  tensor_image, w, h = prepareImage(image_path=image_path)
  model.eval()
  classes, bbox = model(tensor_image)
  bbox = bbox.detach().numpy()
  print(bbox)
  class_pred = softmax(classes, dim=1)
  return tensor_image, class_pred, bbox, w, h

def prepareImage(image_path):
  original_image = cv.imread(image_path)
  h,w,_ = original_image.shape
  rgb_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
  pil_image = Image.fromarray(rgb_image)
  transform = transforms.Compose([
      transforms.Resize((224,224)),
      transforms.ToTensor()
    ])
  tensor_image = transform(pil_image)
  image = torch.unsqueeze(tensor_image, 0)
  return image, w, h

def openImageWithOpenCV(tensor_image, width, height, prediction, bounding_box):
  small_image = tensor_image[0].permute(1,2,0).numpy()
  pred_val = int(round(prediction[0][1].item(), 3) * 100)
  normal_image = cv.resize(small_image, (width, height))
  print(width, height)
  for box in bounding_box:
    x,y,w,h =  np.array([int(i) for i in box])
    cv.rectangle(normal_image, (x,y), (w+200,h+120), (0,0,250), 2)

  cv.putText(normal_image, f"{pred_val}%", (10,50), 2, 2, (255,0,0), 2)
  cv.imshow("Image", normal_image)
  if cv.waitKey(0) == ord("q"):
    cv.destroyAllWindows()

def checkForAnnotation(searchtag:str, csv_file):
  searchtag = str(searchtag[:-3])
  with open(csv_file, "r") as f:
    for i in f:
      ret = filterString(search_tag=searchtag,val=i)
      if isinstance(ret, list):
        return ret

def filterString(search_tag:str,val) -> list:

  if not isinstance(search_tag, str):
    raise TypeError(f"{search_tag} is not a string.")

  if not isinstance(val, str):
    raise TypeError(f"{val} is not a string.")

  if isinstance(search_tag,str) and search_tag in val:
    try:
      _, anno_num = val.split(",",1)
      len = int(anno_num[:1])
      arr = []
      if len != 0:
        _, anno_num, anno_coords = val.split(",", 2)
        coords = list(map(int, anno_coords.split()))
        arr = coords[:4]
        return arr

    except Exception as e:
        print(e)

def checkDirectoryForModel():
  dir = "saved_models"
  if os.path.exists(dir):
    print(f">>> Folder {dir} already exists.")
  else:
    os.makedirs(dir, exist_ok=True)
    print(f">>> Created {dir} folder.")
