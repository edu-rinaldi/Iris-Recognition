import os

from torch import nn, load
from torchvision import models

class VGGFE(nn.Module):
    def __init__(self, pretrainedName=''):
        super(VGGFE, self).__init__()
        
        model = models.vgg16(pretrained= pretrainedName =='vgg16')
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)

        self.pooling = model.avgpool
        self.flatten = nn.Flatten()
        self.fc = model.classifier[0]

        if pretrainedName != '' and pretrainedName != 'vgg16':
            modelPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pretrained", pretrainedName)
            self.load_state_dict(load(modelPath))

    def forward(self, x):
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out) 
        return out