from torchsummary import summary
from torchgeometry.losses import one_hot
import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import time
import imageio
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Resize, PILToTensor, ToPILImage, Compose, InterpolationMode
from collections import OrderedDict
import wandb

from models import UNET, weights_init, load_model, save_model
from dataset import train_loader, valid_loader





# Defining hyperparameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Number of class in the data set (3: neoplastic, non neoplastic, background)
num_classes = 3



# Hyperparameters for training 
learning_rate = 1e-04
batch_size = 4
display_step = 50
checkpoint_path = 'unet_model.pth'

loss_epoch_array = []
train_accuracy = []
test_accuracy = []
valid_accuracy = []

# Define loss_fn

class CEDiceLoss(nn.Module):
    def __init__(self, weights) -> None:
        super(CEDiceLoss, self).__init__()
        self.eps: float = 1e-6
        self.weights: torch.Tensor = weights

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        if not self.weights.shape[1] == input.shape[1]:
            raise ValueError("The number of weights must equal the number of classes")
        if not torch.sum(self.weights).item() == 1:
            raise ValueError("The sum of all weights must equal 1")
            
        # cross entropy loss
        celoss = nn.CrossEntropyLoss(self.weights)(input, target)
        
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)

        # compute the actual dice score
        dims = (2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        
        dice_score = torch.sum(dice_score * self.weights, dim=1)
        
        return torch.mean(1. - dice_score) + celoss
#         return dice_score


model = UNET(out_channels = num_classes).to(device)
weights = torch.Tensor([[0.4, 0.55, 0.05]]).cuda()
loss_function = CEDiceLoss(weights)
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
learing_rate_scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.6)

# Train function for each epoch
def train(train_dataloader, valid_dataloader,learing_rate_scheduler, epoch, display_step):
    print(f"Start epoch #{epoch+1}, learning rate for this epoch: {learing_rate_scheduler.get_last_lr()}")
    start_time = time.time()
    train_loss_epoch = 0
    test_loss_epoch = 0
    last_loss = 999999999
    model.train()
    for i, (data,targets) in enumerate(train_dataloader):
        
        # Load data into GPU
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)

        # Backpropagation, compute gradients
        loss = loss_function(outputs, targets.long())
        loss.backward()

        # Apply gradients
        optimizer.step()
        
        # Save loss
        train_loss_epoch += loss.item()
        if (i+1) % display_step == 0:
#             accuracy = float(test(test_loader))
            print('Train Epoch: {} [{}/{} ({}%)]\tLoss: {:.4f}'.format(
                epoch + 1, (i+1) * len(data), len(train_dataloader.dataset), 100 * (i+1) * len(data) / len(train_dataloader.dataset), 
                loss.item()))
                  
    print(f"Done epoch #{epoch+1}, time for this epoch: {time.time()-start_time}s")
    train_loss_epoch/= (i + 1)
    
    # Evaluate the validation set
    model.eval()
    with torch.no_grad():
        for data, target in valid_dataloader:
            data, target = data.to(device), target.to(device)
            test_output = model(data)
            test_loss = loss_function(test_output, target)
            test_loss_epoch += test_loss.item()
            
    test_loss_epoch/= (i+1)
    
    return train_loss_epoch , test_loss_epoch


def main():
    # Number of epoch
    epochs = 30
    wandb.login(
        # set the wandb project where this run will be logged
    #     project= "PolypSegment", 
        key = "4a5d5d3b091931e391f5f23b25e6932dd6fdcb63",
    )
    wandb.init(
        project = "IWillKillAll"
    )
    # Training loop
    train_loss_array = []
    test_loss_array = []
    last_loss = 9999999999999
    for epoch in range(epochs):
        train_loss_epoch = 0
        test_loss_epoch = 0
        (train_loss_epoch, test_loss_epoch) = train(train_loader, 
                                                  valid_loader, 
                                                  learing_rate_scheduler, epoch, display_step)

        if test_loss_epoch < last_loss:
            save_model(model, optimizer, checkpoint_path)
            last_loss = test_loss_epoch

        learing_rate_scheduler.step()
        train_loss_array.append(train_loss_epoch)
        test_loss_array.append(test_loss_epoch)
        wandb.log({"Train loss": train_loss_epoch, "Valid loss": test_loss_epoch})
    #     train_accuracy.append(test(train_loader))
    #     valid_accuracy.append(test(test_loader))
    #     print("Epoch {}: loss: {:.4f}, train accuracy: {:.4f}, valid accuracy:{:.4f}".format(epoch + 1, 
    #                                         train_loss_array[-1], train_accuracy[-1], valid_accuracy[-1]))    

    torch.save(model.state_dict(), checkpoint_path)

if __name__ == '__main__':
    main()