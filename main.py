import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.ops import nms
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import fasterrcnn_resnet50_fpn  
import torch.nn as nn
from torch.nn.functional import softmax
import torch.optim as optim
import cv2 as cv
import numpy as np
from model import NeuralNetwork
from torchinfo import summary
import os
from PIL import Image 




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

def checkDirectoryForModel():
  dir = "saved_models"
  if os.path.exists(dir):
    print(f">>> Folder {dir} already exists.")
  else:
    os.makedirs(dir, exist_ok=True)
    print(f">>> Created {dir} folder.")

# Load data, Prepare data
def processData(batch):
  train_data_src = "train_data"
  test_data_src = "test_data"

  train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])

  test_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.ToTensor()])


  train_data = datasets.ImageFolder(train_data_src, transform=train_transforms)
  test_data = datasets.ImageFolder(test_data_src, transform=test_transforms)

  train_loader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True)
  test_loader = DataLoader(dataset=test_data, batch_size=batch, shuffle=True)

  return train_loader, test_loader, train_data.classes

# Training the neural network
def trainAndEvaluate(model, device, train_loader, test_loader):
  num_epochs = 30

  accuracies = []
  losses = []

  val_accuracies = []
  val_losses = []

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  for epoch in range(num_epochs): 
    model.train()
    for i, (images, labels) in enumerate(train_loader):     
      
      if images.size(0) != labels.size(0):
        raise ValueError(f"Expected input batch_size ({images.size(0)}) to match target batch_size ({labels.size(0)})")
      
      # Forward pass 
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      loss = criterion(outputs, labels)
      
      # Backward pass 
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      
    _, predicted = torch.max(outputs.data, 1)
    acc = (predicted == labels).sum().item() / labels.size(0)
    accuracies.append(acc)
    losses.append(loss.item())

    model.eval()
    val_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
      for images, labels in test_loader:
        labels = labels.to(device)
        images = images.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

      _, predicted = torch.max(outputs.data, 1)
      total = labels.size(0)
      correct = (predicted == labels).sum().item()
      val_acc += correct / total

      val_accuracies.append(acc)
      val_losses.append(loss.item())

    print('Epoch [{}/{}],Loss:{:.4f},Validation Loss:{:.4f},Accuracy:{:.2f},Validation Accuracy:{:.2f}'.format( 
          epoch+1, num_epochs, loss.item(), val_loss, acc ,val_acc))

    model_name = f"test_epoch_{epoch}.pth"
    save_path = os.path.join("saved_models", model_name)
    torch.save(model.state_dict(), save_path)

def loadModelResult(model, image_path, model_file):
  model = NeuralNetwork()
  model.load_state_dict(torch.load(model_file))
  tensor_image, w, h = prepareImage(image_path=image_path)
  model.eval()
  output = model(tensor_image)
  prediction = softmax(output, dim=1)
  return tensor_image, prediction, w, h

def prepareImage(image_path):
  original_image = cv.imread(image_path)
  h,w,_ = original_image.shape
  rgb_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
  pil_image = Image.fromarray(rgb_image)
  transform = transforms.Compose([
      transforms.Resize((224,224)),
      transforms.ToTensor()
    ])
  tensor_image = transform(pil_image)
  image = torch.unsqueeze(tensor_image, 0)
  return image, w, h

def openImageWithOpenCV(tensor_image, width, height, prediction):
  small_image = tensor_image[0].permute(1,2,0).numpy()
  transform = transforms.Compose([
    transforms.Resize((width,height)),
    transforms.ToTensor()
  ])

  pred_val = round(prediction[0][1].item(), 3) * 100
  normal_image = cv.resize(small_image, (width, height))
  cv.putText(normal_image, f"{pred_val}%", (10,50), 2, 2, (255,0,0), 2)
  cv.imshow("Image", normal_image)
  if cv.waitKey(0) == ord("q"):
      cv.destroyAllWindows()





def main():
  # Input data
  real_image = "image.jpg"
  model_file = "first_attempt.pth"

  device = "cuda" if torch.cuda.is_available() else "cpu"
  #model = NeuralNetwork()
  model = fasterrcnn_resnet50_fpn()
  model.to(device)

  # Train
  #checkDirectoryForModel()
  #train_data, test_data, classes = processData(batch=32)
  #trainAndEvaluate(model=model, device=device, train_loader=train_data, test_loader=test_data)
  
  tensor_image, prediction, w, h = loadModelResult(model=model, image_path=real_image, model_file=model_file)
  openImageWithOpenCV(tensor_image, width=w, height=h, prediction=prediction)
  


if __name__ == "__main__":
  main()

