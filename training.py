from ultralytics import YOLO

TRAINING_SETTINGS = {
    "image_size": 480,
    "batch_size": 16,
    "epoch_cycles": 50
}

def training():
    # build a new model from YAML
    model = YOLO("yolo26n.yaml").load("yolo26n.pt")  # build from YAML and transfer weights

    model.train(data="datasets/dataset.yaml", imgsz=TRAINING_SETTINGS["image_size"], batch=TRAINING_SETTINGS["batch_size"], epochs=TRAINING_SETTINGS["epoch_cycles"])


if __name__ == "__main__":
    training()
