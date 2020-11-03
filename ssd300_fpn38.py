import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os

class FusionLayer_Up(nn.Module):
    def __init__(self, in_channels, out_channels, padding, output_shape):
        super(FusionLayer_Up, self).__init__()
        self.output_shape = output_shape
        self.out_channels = out_channels
        self.conv = torch.nn.ConvTranspose2d(in_channels, out_channels, (3,3), stride=2, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output_size = ( x.size()[0], self.out_channels, self.output_shape[0], self.output_shape[1])
        return F.relu(self.bn(self.conv(x, output_size=output_size)))

class FusionLayer_Left(nn.Module):
    def __init__(self, channels, out_channels=None):
        super(FusionLayer_Left, self).__init__()
        if out_channels is None:
            out_channels = channels
        self.conv = torch.nn.Conv2d(channels, out_channels, (1,1), stride=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class mySSD(nn.Module):
    def __init__(self, phase, size, base, extras, head, num_classes):
        super(mySSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward())
        self.size = size

        # self.SElayers = nn.ModuleList([SElayer(64), SElayer(128), SElayer(256), SElayer(512), SElayer(512)])
        channels = [256, 256 ,256, 512, 1024, 512]
        paddings =  [0, 1, 1, 1, 1]
        shapes = [(3,3), (5,5), (10, 10), (19, 19), (38, 38)]
        self.Fusion_Ups = nn.ModuleList( [FusionLayer_Up(channels[i], channels[i+1], paddings[i], shapes[i]) for i in range(len(channels)-1)] )
        self.Fusion_Lefts = nn.ModuleList( [FusionLayer_Left(channels[i+1]) for i in range(len(channels)-1)])

        # self.Fusion_Up_38 = FusionLayer_Up(512, 256, 1, (75, 75))
        # self.Fusion_Up_75 = FusionLayer_Up(256, 128, 1, (150, 150))

        # self.Fusion_Left_75 = FusionLayer_Left(256)
        # self.Fusion_Left_150 = FusionLayer_Left(128)

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])


        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu

        # Add multiple path
        # layer_to_m = [2, 7, 14, 21, 28]
        # reduction = 16

        for k in range(23):
            x = self.vgg[k](x)
        # Now x : 38x38
        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
            # print(k, self.vgg[k])
            # if k in layer_to_m:
            #     i = layer_to_m.index(k)
            #     x = self.SElayers[i](x)
         # Now x : 19x19
        sources.append(x)
        

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        # Now x : 1x1\

        #self.Fusion_Ups
        for i in range(len(sources)-1):
            s_i = len(sources)-1-i
            s_i_next = len(sources)-1-i-1
            s_down = self.Fusion_Ups[i](sources[s_i])
            s_left = self.Fusion_Lefts[i](sources[s_i_next])
            sources[s_i_next] = s_down + s_left


        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')



# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers



def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}

mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)

def build_ssd300_fpn38(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return mySSD(phase, size, base_, extras_, head_, num_classes) 




# Todo : SSD resnet version 


if __name__ == '__main__':
    # net = build_ssd('train', size=300, num_classes=21)
    net = build_ssd300_fpn38('train', size=300, num_classes=21)
    net = net.Fusion_Ups
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total", pytorch_total_params)
    # # net.eval()
    # # print(net)
    x = torch.rand(1, 3, 300, 300)
    y = net(x)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
    
    # net = FusionLayer_Up(10, 20, 1, (38, 38))
    # x = torch.rand(1, 10, 19, 19)
    # y = net(x)
    # print(y.size())




