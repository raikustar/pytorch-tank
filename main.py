import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from torch.nn.functional import softmax
import torch.optim as optim
import cv2 as cv
import numpy as np
from model import NeuralNetwork
import os
from PIL import Image 

from methods import loadModelResultCustomCNN, openImageWithOpenCV, checkForAnnotation, checkDirectoryForModel



""" 
# Neural Network
Convolutional Neural Network consists of multiple layers like:
1. Input layer (Image)
2. Convolutional layer (applies filters to extract features) 
  Activation layer
3. Pooling layer (downsizes samples to reduce computation)
  Max pooling(Try first) and average pooling(Try second)
  Might have to be reserved with pooling layers due to information loss from downscaling

  Flattening after Conv. layer and pooling layer.
4. Fully connected layers (makes final prediction)

output from the fully connected layers is then fed into a logistic function for classification tasks like sigmoid or softmax 
which converts the output of each class into the probability score of each class.


# Using two classes(positive and negative images) for image recog.???
# What are all the different models to use in pytorch for image recognition?
model = SimpleNet()

"""

# return Dataset with image and label first
class CustomDataSet(Dataset):
  train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])

  test_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.ToTensor()])

  

  def __init__(self,path,csv_file, train=True):
    self.path = path
    self.csv_file = csv_file
    if train:
      self.data = datasets.ImageFolder(self.path, transform=self.train_transforms)
    else:
      self.data = datasets.ImageFolder(self.path, transform=self.test_transforms)
    self.classes = {0:"notatank", 1:"tank"}
    self.samples = self.data.samples

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    image, class_id = self.data[idx]
    bbox = torch.zeros(4, dtype=torch.float32)
    if class_id == 1:
      res = str(self.samples[idx][:1])[20:]
      bbox_data = checkForAnnotation(searchtag=res, csv_file=self.csv_file)
      if bbox_data is not None:
        bbox = torch.tensor(bbox_data, dtype=torch.float32)
        
    class_name = torch.as_tensor(class_id, dtype=torch.float32)
    return image, {"label":class_name, "boxes":bbox}

# Load data, Prepare data
def processData():
  train_data_src = "train_data"
  test_data_src = "test_data"

  train_set = CustomDataSet(path=train_data_src, csv_file="annotations.csv", train=True)
  test_set = CustomDataSet(path=test_data_src, csv_file="annotations.csv", train=False)

  train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
  test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True)

  return train_loader, test_loader

# Training the neural network
def trainAndEvaluate(model, device, train_loader, test_loader, num_epochs:int=10):
  accuracies = []
  losses = []

  val_accuracies = []
  val_losses = []

  criterion_label = CrossEntropyLoss()
  criterion_bbox = SmoothL1Loss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  for epoch in range(num_epochs): 
    model.train()
    for i, (images, labels) in enumerate(train_loader):     
      optimizer.zero_grad()
      # if images.size(0) != labels.size(0):
      #   raise ValueError(f"Expected input batch_size ({images.size(0)}) to match target batch_size ({labels.size(0)})")
    
      # Forward pass 
      images = images.to(device)
      class_name = labels["label"].to(device).long()
      bbox = labels["boxes"].to(device)
      output_label, output_bbox = model(images)
      
      positive_anno = torch.where(class_name == 1)
      out_bbox_filter = output_bbox[positive_anno]
      box_anno_filter = bbox[positive_anno]
      
      loss_label = criterion_label(output_label, class_name)
      
      if len(positive_anno[0]) > 0:
        loss_bbox = criterion_bbox(out_bbox_filter, box_anno_filter)
      else:
        loss_bbox = torch.tensor(0.0).to(device)

      loss = loss_label * (loss_bbox*0.001)
      loss.backward()
      optimizer.step()

      _, predicted = torch.max(output_label.data, 1)
    acc = (predicted == class_name).sum().item() / class_name.size(0)
    accuracies.append(acc)
    losses.append(loss.item())

    with torch.no_grad():
      for images, labels in test_loader:
        images = images.to(device)
        class_name = labels["label"].to(device).long()
        bbox = labels["boxes"].to(device)
        output_label, output_bbox = model(images)

        positive_anno = torch.where(class_name == 1)
        out_bbox_filter = output_bbox[positive_anno]
        box_anno_filter = bbox[positive_anno]
        
        loss_label = criterion_label(output_label, class_name)
        
        if len(positive_anno[0]) > 0:
          loss_bbox = criterion_bbox(out_bbox_filter, box_anno_filter)
        else:
          loss_bbox = torch.tensor(0.0).to(device)

        val_loss = loss_label * (loss_bbox*0.001)
        optimizer.step()

        _, predicted = torch.max(output_label.data, 1)
      correct = (predicted == class_name).sum().item()

      val_acc = correct / class_name.size(0)
      val_accuracies.append(val_acc)
      val_losses.append(val_loss.item())

    print('Epoch [{}/{}],Loss:{:.4f},Validation Loss:{:.4f},Accuracy:{:.2f},Validation Accuracy:{:.2f}'.format( 
           epoch, num_epochs, loss.item(), val_loss.item(), acc, val_acc))

    model_name = f"v5_better_bb_{epoch}.pth"
    save_path = os.path.join("saved_models", model_name)
    torch.save(model.state_dict(), save_path)

#Opencv, prepare image, prediction
def train(model, device, num_epochs):
  checkDirectoryForModel()
  train_data, test_data = processData()
  trainAndEvaluate(model=model, device=device, train_loader=train_data, test_loader=test_data, num_epochs=num_epochs)

def testing():
  file = "randomimages/positive604.jpg"
  x,y,w,h = [220,76,646,470]
  img = cv.imread(file)
  cv.rectangle(img, (x,y), (w,h), (250,0,0), 1)
  cv.imshow("image", img)
  if cv.waitKey(0) == ord("q"):
    cv.destroyAllWindows()

def main():
  # Input data
  real_image = "randomimages/t90.jpg"
  model_file = "v5_better_bb_25.pth"

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = NeuralNetwork()
  model.to(device)

  # Train
  #train(model=model, device=device, num_epochs=75)
  
  tensor_image, prediction, bbox, w, h = loadModelResultCustomCNN(model=model, image_path=real_image, model_file=model_file)
  openImageWithOpenCV(tensor_image, width=w, height=h, prediction=prediction, bounding_box=bbox)


if __name__ == "__main__":
  main()



