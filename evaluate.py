from ultralytics import YOLO

def evaluate():
    model = YOLO("mil-yolov26.pt")

    result = model("bmp_3_5.jpg")

    for res in result:
        res.show()

if __name__ == "__main__":
    evaluate()
