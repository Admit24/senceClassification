import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from read import *
#from torchvision.models import resnet101
from model_2 import resnet
from utils_1 import Trainer
BATCH_SIZE=30

TRAIN_LIST="train.csv"
VAL_LIST="test.csv"

SAVE_PATH='checkpoints'
def get_dataloader(batch_size):
    '''mytransform = transforms.Compose([
        transforms.ToTensor()])'''

    # torch.utils.data.DataLoader
    train_loader = torch.utils.data.DataLoader(
        ImageFolder(TRAIN_LIST
                      ),
        batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(
        ImageFolder_val(VAL_LIST
                      ),
        batch_size=batch_size, shuffle=True)
    return train_loader, validation_loader

def main(batch_size):
    train_loader, test_loader = get_dataloader(batch_size)
    model=resnet(num_classes=21)
    #optimizer = optim.SGD(params=model.parameters(), lr=1e-1, momentum=0.9,weight_decay=1e-4)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, 80, 0.1)
    optimizer=optim.Adam(params=model.parameters())
    trainer = Trainer(model, optimizer, nn.CrossEntropyLoss ,save_freq=20,save_dir=SAVE_PATH)
    trainer.loop(2000, train_loader, test_loader)


if __name__ == '__main__':
    main(BATCH_SIZE)