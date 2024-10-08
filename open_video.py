import sys
sys.path.append('yolov5/')

import torch
import cv2 as cv
import numpy as np
from yolov5.utils.general import non_max_suppression


from methods import reworkFrameType, yoloDetection



# Collect proper data from model prediction
def openVideo(model, video_path):
    formated_frame = reworkFrameType
    yolo_detection = yoloDetection
    confidence:float = 0.25
    threshold:float = 0.5


    video = cv.VideoCapture(video_path)
    fps = int(video.get(cv.CAP_PROP_FPS))
    
    if not video.isOpened():
        print("Cannot open video")
        exit()

    model.eval()
    
    while True:
        ret, frame = video.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
    
        img = formated_frame(frame)
        with torch.no_grad():
            pred = model(img)[0]

        pred = pred.unsqueeze(0)

        pred = non_max_suppression(pred, conf_thres=confidence,iou_thres=threshold)
        prediction = torch.detach(pred[0]).cpu().numpy()

        colors:int = [(0,250,200), (0,250,100), (180,250,0), (250,0,100), (0,250,250), (50,250,0), (250,0,250)]
        
        for i, det in enumerate(prediction):
            yolo_detection(det, frame, percentage_found=0.30, box_col=colors[i])

        cv.imshow('frame', frame)
        if cv.waitKey(fps) == ord('q'):
            break

    video.release()
    cv.destroyAllWindows()


def modelReady(model_file):
    model = torch.hub.load("yolov5","custom", path=model_file, source="local")
    return model

def main():
  # Input data
  model = modelReady(model_file="mil.pt")
  openVideo(video_path="video/russiantank.mp4", model=model)


if __name__ == "__main__":
  main()
