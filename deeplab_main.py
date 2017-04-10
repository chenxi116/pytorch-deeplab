import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import sys
import deeplab
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(321),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    pascal_dir = '/media/Work_HD/cxliu/datasets/VOCdevkit/VOC2012/JPEGImages/'
    list_dir = '/media/Work_HD/cxliu/projects/deeplab/list/'
    lines = np.loadtxt(list_dir + 'val_id.txt', dtype=str)

    if sys.argv[2] == 'train':
        pass
    elif sys.argv[2] == 'eval':
        model = getattr(deeplab, 'resnet101')()
        model.eval()
        model.load_state_dict(torch.load('model/deeplab101_trainaug.pth'))
        if use_gpu:
            model = model.cuda()

        for i, imname in enumerate(lines):
            im = datasets.folder.default_loader(pascal_dir + imname + '.jpg')
            w, h= np.shape(im)[0], np.shape(im)[1]
            inputs = data_transforms['val'](im)
            if use_gpu:
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)
            outputs = model(inputs.unsqueeze(0))
            outputs_up = nn.UpsamplingBilinear2d((w, h))(outputs)
            _, pred = torch.max(outputs_up, 1)
            pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
            seg = Image.fromarray(pred)
            seg.save('data/val/' + imname + '.png')
            print('processing %d/%d' % (i + 1, len(lines)))