import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import transforms, datasets
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
import datetime
import os

from methods import getImageScale, resizeBoundingBoxData, checkForAnnotation, checkDirectoryForModel


class CustomDataSetTwo(Dataset):
  train_transforms = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor()
  ])

  test_transforms = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor()
  ])

  def __init__(self, path, csv_file, train=True):
    self.path = path
    self.csv_file = csv_file
    if train:
        self.data = datasets.ImageFolder(self.path, transform=self.train_transforms)
    else:
        self.data = datasets.ImageFolder(self.path, transform=self.test_transforms)
    self.classes = {0: "notatank", 1: "tank"}
    self.samples = self.data.samples

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    w, h = getImageScale(self.samples[idx])
    image, class_id = self.data[idx]
    target = {}
    boxes = torch.zeros((0, 4), dtype=torch.float32)
    labels = torch.zeros((0,), dtype=torch.int64)


    res = str(self.samples[idx][0])
    bbox_data = checkForAnnotation(searchtag=res, csv_file=self.csv_file)
    
    if isinstance(bbox_data, list):
      new_box = resizeBoundingBoxData(box=bbox_data,im_w=w,im_h=h)
      boxes = torch.tensor(new_box, dtype=torch.float32)
      boxes = boxes.unsqueeze(0)
      labels = torch.tensor([class_id] * len(bbox_data), dtype=torch.int64)

    target["boxes"] = boxes
    target["labels"] = labels

    return image, target
    
# Custom collate function
def collate_fn(batch):
    return tuple(zip(*batch))

# Load data, prepare data
def processData():
  train_data_src = "train_data"
  test_data_src = "test_data"

  train_set = CustomDataSetTwo(path=train_data_src, csv_file="an.csv", train=True)
  test_set = CustomDataSetTwo(path=test_data_src, csv_file="an.csv", train=False)
  
  train_loader = DataLoader(dataset=train_set, batch_size=8, shuffle=True, collate_fn=collate_fn)
  test_loader = DataLoader(dataset=test_set, batch_size=8, shuffle=False, collate_fn=collate_fn)

  return train_loader, test_loader

# Training the neural network
def trainAndEvaluateRCNN(train_loader, test_loader,checkpoint_path=None, num_epochs:int=250):
  model = fasterrcnn_resnet50_fpn(pretrained=True,weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

  num_classes = 2  # Adjust according to your dataset
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  device = "cuda" if torch.cuda.is_available() else "cpu"

  model.to(device=device)
  start_epoch = 0
  optimizer_sgd = optim.SGD(model.parameters(), lr=0.0093, momentum=0.9)
  optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
  scheduler_sgd = ReduceLROnPlateau(optimizer=optimizer_sgd,mode="min",factor=0.98,patience=3)
  scheduler_adam = ReduceLROnPlateau(optimizer=optimizer_adam,mode="min",factor=0.8,patience=3)

  if checkpoint_path:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer_sgd.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]

  
  for epoch in range(start_epoch, num_epochs):
    model.train()    
    train_losses = 0
    start_time = timeToString()
    for idx, (images, targets) in enumerate(train_loader):
      if idx >= 100:
        break
      print(f"Epoch:{epoch} - Train batch index: {idx} of {100}.", end="\r") 
      optimizer_sgd.zero_grad()

      images = list(image.to(device) for image in images)
      targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
      loss_dict = model(images, targets)
      losses = sum(loss for loss in loss_dict.values())

      losses.backward()
      optimizer_sgd.step()
    
      train_losses += losses.item()
    avg_train_loss = train_losses / 100
    
    scheduler_sgd.step(avg_train_loss)
    
    model.eval()
    with torch.no_grad():
      p = []
      for idx, (images, _) in enumerate(test_loader):
        images = list(image.to(device) for image in images)
        pred = model(images)
      print(f"Epoch {epoch}:", pred)  
      if pred[0]["scores"].tolist() != 0:
        pe = pred[0]["scores"].tolist()
        p.append(pe[0])
      else:
        p.append(0)

    model_name = f"v8_{epoch}.pth"
    save_path = os.path.join("saved_models", model_name)
    torch.save({
      "epoch": epoch + 1,
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer_sgd.state_dict(),
      "loss":avg_train_loss
    }, save_path)

    end_time = timeToString()
    
    data_print = f"Epoch: {epoch}, Loss: {avg_train_loss}, Learning rate: {scheduler_sgd.get_last_lr()}, Start:{start_time} - End:{end_time}, Prediction: {p}.\n" 
    with open("data.txt", "a") as f:
      f.write(data_print)     
    

def train():
  cudnn.benchmark = True
  checkDirectoryForModel()
  train_data, test_data = processData()
  trainAndEvaluateRCNN(train_loader=train_data, test_loader=test_data, checkpoint_path="saved_models/v8_169.pth")

def timeToString():
  time_now = str(datetime.datetime.now())
  date, time = time_now.split(" ")
  return time

if __name__ == "__main__":
  train() 