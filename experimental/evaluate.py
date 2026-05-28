from ultralytics import YOLO

def evaluate():
    #model = YOLO("mil-yolov26n.pt")
    model = YOLO("11_20.onnx")

    result = model("test2.jpg")
    
    for res in result:
        res.show()
    

if __name__ == "__main__":
    evaluate()