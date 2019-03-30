import torch
from collections import OrderedDict
from resnet101 import se_resnet101_ibn_a

def myOwnLoad(model, check):
    modelState = model.state_dict()
    tempState = OrderedDict()
    for i in range(len(check.keys())):
        print(check[list(check.keys())[i]])
        tempState[list(modelState.keys())[i]] = check[list(check.keys())[i]]
    #temp = [[0.02]*1024 for i in range(200)]  # mean=0, std=0.02
    #tempState['myFc.weight'] = torch.normal(mean=0, std=torch.FloatTensor(temp)).cuda()
    #tempState['myFc.bias']   = torch.normal(mean=0, std=torch.FloatTensor([0]*200)).cuda()
    model.load_state_dict(tempState)
    return model
model=se_resnet101_ibn_a()
model_weight = torch.load(r"C:\Users\Administrator\Downloads\se_resnet101.pth.tar")['state_dict']
myOwnLoad(model, model_weight)