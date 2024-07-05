import sys
sys.path.append("yolov5/")

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
import cv2 as cv
import numpy as np
from yolov5.utils.general import non_max_suppression

from methods import openImageWithOpenCV, prepareImage, checkWithNMS, reworkFrameType, yoloDetection

def loadResultImageRCNN(model, image_path=None):
  model.eval()
  # image is string, gives path of file
  if image_path:
    tensor_image, w, h = prepareImage(image_path=image_path)
    
  with torch.no_grad():
    model_result = model([tensor_image.to(next(model.parameters()).device)])

  # .tolist()
  bboxes = model_result[0]["boxes"].tolist()
  labels = model_result[0]["labels"].tolist()
  scores = model_result[0]["scores"].tolist()
  nms_idx = checkWithNMS(prediction=model_result[0])
  idx = nms_idx.tolist()
  
  return bboxes, labels, scores, idx

def openImageRNN(model,image_path):
  boxes, labels, scores, idx = loadResultImageRCNN(model=model,image_path=image_path)
  openImageWithOpenCV(image_path=image_path,boxes=boxes, labels=labels, scores=scores, nms_index=idx, hitrate=0.55)

def modelReadyRCNN(model_file):
  model = fasterrcnn_resnet50_fpn(pretrained=True,weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
  model_load = torch.load(model_file)

  num_classes = 2  # Adjust according to your dataset
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  device = "cuda" if torch.cuda.is_available() else "cpu"

  model.load_state_dict(model_load["model_state_dict"])
  model.to(device=device)

  return model

def modelReadyYOLO(model_file):
  model = torch.hub.load("yolov5","custom", path=model_file, source="local")
  return model


def loadYOLOImage(real_image, model):
  image = cv.imread(real_image)
  tensor_image = reworkFrameType(image)
  yolo_detection = yoloDetection

  model.eval()
  with torch.no_grad():
    pred = model(tensor_image)[0]

  pred = pred.unsqueeze(0)
  confidence:float = 0.25
  threshold:float = 0.5

  pred = non_max_suppression(pred, conf_thres=confidence,iou_thres=threshold)
  prediction = torch.detach(pred[0]).cpu().numpy()

  colors:int = [(0,250,200), (0,250,100), (180,250,0), (250,0,100), (0,250,250), (50,250,0), (250,0,250)]

  for i, det in enumerate(prediction):
    yolo_detection(det, image, percentage_found=0.25, box_col=colors[i])
  
  cv.imshow("Detection", image)
  if cv.waitKey(0) == ord("q"):
    cv.destroyAllWindows()



def main():

  # Input data
  real_image = "randomimages/tanks.jpg"
  # YOLOv5s model
  model_yolo = modelReadyYOLO(model_file="mil.pt") 
  loadYOLOImage(real_image, model=model_yolo)



  # Faster RCNN
  #model_rnn = modelReadyRCNN(model_file="comp_models/rus_tank_fasterrcnn_resnet50.pth")
  #openImageRNN(model=model_rnn, image_path=real_image)
if __name__ == "__main__":
  main()



