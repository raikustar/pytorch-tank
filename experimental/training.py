from ultralytics import YOLO
 # 14
TRAINING_SETTINGS = {
    "image_size": 480,
    "batch_size": 20,
    "epoch_cycles": 1
}

def training():
    # build a new model from YAML
    model = YOLO("yolo26n.yaml").load("yolo26n.pt")  # build from YAML and transfer weights

    model.train(
        data="../datasets/dataset.yaml", 
        imgsz=TRAINING_SETTINGS["image_size"], 
        batch=TRAINING_SETTINGS["batch_size"], 
        epochs=TRAINING_SETTINGS["epoch_cycles"],
        workers=0,
        patience=25,
        mosaic=0.8,       # Intensity of mosaic augmentation (0.0 - 1.0)
        mixup=0.1,        # Intensity of mixup augmentation (0.0 - 1.0)
        hsv_v=0.7,      # Image Saturation Adjustment
        degrees=15,       # Image rotation (+/- degrees)
        translate=0.5,    # Image translation fraction
        shear=10
    )

def continue_training():
    # build a new model from YAML
    model = YOLO("last.pt")  # build from YAML and transfer weights

    model.train(
        data="../datasets/dataset.yaml", 
        imgsz=TRAINING_SETTINGS["image_size"], 
        batch=TRAINING_SETTINGS["batch_size"], 
        epochs=TRAINING_SETTINGS["epoch_cycles"],
        workers=0,
        resume=True,
        patience=25,
        mosaic=0.8,       # Intensity of mosaic augmentation (0.0 - 1.0)
        mixup=0.1,        # Intensity of mixup augmentation (0.0 - 1.0)
        hsv_v=0.7,      # Image Saturation Adjustment
        degrees=15,       # Image rotation (+/- degrees)
        translate=0.5,    # Image translation fraction
        shear=10
    )

def convert_to_onxx():
    model = YOLO("11_20.pt")
    model.export(format="onnx", dynamic=True, simplify=True, nms=False)


if __name__ == "__main__":
    training()
