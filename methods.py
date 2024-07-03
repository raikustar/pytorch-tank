import torch
from torchvision import transforms
from torchvision.ops import nms
import cv2 as cv
import numpy as np
import os
from PIL import Image 
import random
from typing import List,Tuple,Dict


class SmoothBox:
  """
  Save box coordinates of model prediction. 
  """
  def __init__(self,history_length:int = 5):
    self.history = []
    self.history_length = history_length

  def update(self, box):
    self.history.append(box)
    if len(self.history) > self.history_length:
      self.history.pop(0)
    return np.mean(self.history, axis=0)




def prepareImage(image_path):
  original_image = cv.imread(image_path)
  h,w,c = original_image.shape
  rgb_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
  pil_image = Image.fromarray(rgb_image)
  transform = transforms.Compose([
      transforms.ToTensor()
    ])
  tensor_image = transform(pil_image)
  return tensor_image, w, h

def getImageScale(re_image_path):
  splt = str(re_image_path[0])
  img = cv.imread(splt)
  w,h,_ = img.shape
  w = float("{:.2f}".format(w/224))
  h = float("{:.2f}".format(h/224))
  return w,h

def resizeBoundingBoxData(box, im_w,im_h) -> list:
  x,y,w,h = box
  x, w = int(x/im_w), int((x+w)/im_w)
  y, h = int(y/im_h), int((y+h)/im_h)
  box = [x,y,w,h]
  return box

def openImageWithOpenCV(image_path,boxes, labels, scores, nms_index, hitrate:float = 0.7):
  normal_image = cv.imread(image_path)
  print(scores)
  box_arr = []
  for idx, d in enumerate(nms_index):
    if scores[d] >= hitrate:
      box = [int(i) for i in boxes[d]]
      box_arr.append([box, scores[d]])

  if len(box_arr) != 0:
    for idx, (box, score) in enumerate(box_arr):
      if score >= hitrate:
        col = randomValue()
        x,y,w,h = adjustBoundingBox(coords=box)
        pred_val = ("{:.0f}").format(round(score,2) * 100)
        cv.rectangle(normal_image, (x,y), (w,h), col, 2)
        cv.rectangle(normal_image, (x,y), (x+65,y+30), col, -1)
        cv.putText(normal_image, f"{pred_val}%", (x+5,y+25), 2, 0.8, (0,0,0), 1)
      else:
        print("Didn't find a substantial score to show a tank.")
  elif len(scores) == 0:
    print("Didn't find a substantial score to show a tank.")
  
  cv.imshow("Image", normal_image)
  if cv.waitKey(0) == ord("q"):
    cv.destroyAllWindows()

def adjustBoundingBox(coords:list) -> list:
  height_tweak = int(25)
  x,y,w,h = coords
  width, height = w - x, h - y
  div = width/height
  y,h = y + height_tweak, h + height_tweak
  if width/height >= 3.5:
    w = x + int(width/ (div - 2.5))
  elif width/height >= 3.0:
    w = x + int(width/ (div - 1.6))
  elif width/height >= 2.5:
    w = x + int(width/ (div - 1.5))
  elif width/height >= 2.0:
    w = x + int(width/ (div - 0.9))
  elif width/height >= 1.5:
    w = x + int(width/(div - 0.5))
  elif width/height <= 1.49:
    w = x + int(width/(div - 0.25))

  return x,y,w,h

def checkWithNMS(prediction):
  b = prediction["boxes"]
  l = prediction["labels"]
  s = prediction["scores"]

  result = nms(b, s ,iou_threshold=0.3)
  return result

def checkForAnnotation(searchtag:str, csv_file):
  searchtag = str(searchtag.split("\\",2)[2:])
  search = str(searchtag[2:-2])
  with open(csv_file, "r") as f:
    for i in f:
      ret = filterString(search_tag=search, val=i)
      if isinstance(ret, list):
        return ret

def reworkFrameType(frame):
  img = cv.resize(frame, (640,640),interpolation=cv.INTER_LINEAR)
  img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
  img = img.transpose((2,0,1)).astype(np.float32)
  img /= 255.0
  img = np.expand_dims(img, axis=0)
  return torch.from_numpy(img)

def processedFrame(frame,coord,confidence,name):
    object_dict = {"Russian tank": (0,55,255), "BMP1/2": (0,255,220), "BMP3": (0,255,70)}
    text = f"{confidence}%: {name}"
    x,y,w,h = coord
    cv.rectangle(frame, (x,y), (w,h), object_dict[name], 1)
    cv.rectangle(frame, (x,y), (x+275,y+35), object_dict[name],-1)
    cv.putText(frame, text, (x+5,y+25), cv.FONT_HERSHEY_COMPLEX, 0.8, (0,0,0), 1)
    return frame

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

def randomValue(): 
  r = random.randint(0,250)
  g = random.randint(0,250)
  b = random.randint(0,250)
  col = (b,g,r)
  return col

def scale_coords(coords:list, 
  original_shape:Tuple[int,int,int], 
  size:int = 640
) -> list:
  y_scale, x_scale = original_shape[0] / size, original_shape[1] / size
  
  coords[0] *= x_scale
  coords[1] *= y_scale 
  coords[2] *= x_scale 
  coords[3] *= y_scale 
  coords[3] = min(coords[3], original_shape[0])

  coords = [int(max(i, 0)) for i in coords]
  return coords

def yoloDetection(arr:np.ndarray, 
  original_image:np.ndarray, 
  percentage_found:float = 0.25,
  box_col:Tuple[int,int,int] = (250,250,250) 
) -> None:
  
  
  scale_coord = scale_coords
  object_name = {1:"Russian tank", 2:"BMP1/2", 3:"BMP3"}
  try:
    coords, conf, name = arr[:4], arr[4], int(arr[5])
  except Exception as e:
    print(e)
    return
  
  coords = scale_coord(coords, original_image.shape, 640)
  smooth_history: int = 5
  smooth_boxes = SmoothBox(smooth_history)
  coords = smooth_boxes.update(coords)
  coords = [int(i) for i in coords]
  x,y,w,h = coords 

  box_font = cv.FONT_HERSHEY_COMPLEX
  text_color = (0,0,0)

  if conf > percentage_found:
    cv.rectangle(original_image, (x,y), (w,h), box_col, 2)
    cv.rectangle(original_image, (x,h), (x+270,h-30), box_col, -1)
    conf = round(conf * 100) 
    text = f"{object_name[name]}: {conf}%"
    cv.putText(original_image, text, (x+5,h-5),box_font,0.8, text_color)
    
