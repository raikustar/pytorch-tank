import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
import cv2 as cv
import numpy as np

from methods import openImageWithOpenCV, prepareImage, checkWithNMS

def loadResultVideoRCNN(model_result):
  bboxes = model_result[0]["boxes"].cpu().numpy()
  labels = model_result[0]["labels"].cpu().numpy()
  scores = model_result[0]["scores"].cpu().numpy()
  return bboxes, labels, scores

def openVideoWithOpencv(video_path, model):
  video = cv.VideoCapture(video_path)
  fps = int(video.get(cv.CAP_PROP_FPS))

  vid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
  ])
  
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)
  model.eval()
  if not video.isOpened():
    print("Cannot open camera")
    exit()
  while True:
  # Capture frame-by-frame
    ret, frame = video.read()
    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      break
    
    input_tensor = vid_transform(frame).unsqueeze(0).to(device)
    
    
    with torch.no_grad():
      prediction = model(input_tensor)
      boxes, labels, scores = loadResultVideoRCNN(model_result=prediction)

    if len(boxes) > 0:
      col = (250,0,0)
      x,y,w,h = np.array([int(i) for i in boxes[0]])
      pred_val = ("{:.0f}").format(round(scores[0],2) * 100)
      cv.putText(frame, f"{pred_val}%", (x,y+20), 2, 1, col,2)
      cv.rectangle(frame, (x,y), (w,h), col, 2)

    cv.imshow('frame', frame)
    if cv.waitKey(fps) == ord('q'):
      break

  video.release()
  cv.destroyAllWindows()

##################################

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

def openImage(model,image_path):
  boxes, labels, scores, idx = loadResultImageRCNN(model=model,image_path=image_path)
  openImageWithOpenCV(image_path=image_path,boxes=boxes, labels=labels, scores=scores, nms_index=idx, hitrate=0.65)


def modelReady(model_file):
  model = fasterrcnn_resnet50_fpn(pretrained=True,weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
  model_load = torch.load(model_file)

  num_classes = 2  # Adjust according to your dataset
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  device = "cuda" if torch.cuda.is_available() else "cpu"

  model.load_state_dict(model_load["model_state_dict"])
  model.to(device=device)
  return model


def main():
  # Input data
  real_image = "randomimages/tanks.jpg"
  model_file = "comp_models/v8_170.pth"
  model = modelReady(model_file=model_file)

  openImage(model=model, image_path=real_image)

if __name__ == "__main__":
  main()

