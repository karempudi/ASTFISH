# Yolo net for block detection
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


NETWORK_CONFIG = [
        (32, 3, 1),
        (64, 3, 2), # H, W gets halved after this layer
        ["B", 1],
        (128, 3, 2),# H, W gets halved after this layer
        ["B", 2],
        (256, 3, 2),
        ["B", 8], # store features in route connections to add them later, H,W, C remains same
        (512, 3, 2), # H, W gets halved after this layer
        ["B", 8], # Store features in route connections to add them layer, H, W, C remains same
        (1024, 3, 2),# H, W gets halved after this layer
        ["B", 4], # We cut H, W by 16 till here, C scaled from 1 to 1024
        (512, 1, 1), # shrink to 512 channels, keeping H, W same
        (1024, 3, 1), # Scale back channels again to 1024
        "S", # do the bbox predictions at this scale on a
             #side branch and keep the main line with half the channels 
        (256, 1, 1), # cut the number of channels in half, H, W is the same
        "U",
        (256, 1, 1),
        (512, 3, 1),
        "S", # will be 256 channels after this step
        (128, 1, 1), # 128 channels after this step
        "U", #  
        (128, 1, 1),
        (256, 3, 1),
        "S"
    ]


ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
] 

class ConvBlock(nn.Module):
    
    def __init__(self, c_in, c_out, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, bias = not bn_act, **kwargs)
        self.batch_norm = nn.BatchNorm2d(c_out)
        self.nonlinear = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act
    def forward(self, x):
        if self.use_bn_act:
            return self.nonlinear(self.batch_norm(self.conv(x)))
        else:
            return self.conv(x)

class ResBlock(nn.Module):
    
    def __init__(self, c_in, res_connection=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    ConvBlock(c_in, c_in //2, kernel_size=1),
                    ConvBlock(c_in//2, c_in, kernel_size=3, padding=1),
                )
            ]
        self.res_connection = res_connection
        self.num_repeats = num_repeats
    
    def forward(self, x):
        for layer in self.layers:
            if self.res_connection:
                x = x + layer(x)
            else:
                x = layer(x)
        
        return x

class ScalePrediction(nn.Module):
    def __init__(self, c_in, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            ConvBlock(c_in, 2 * c_in, kernel_size=3, padding=1),
            ConvBlock(
                2 * c_in, (num_classes + 5) * 3,  bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes
        
    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes+5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )

class YOLOv3(nn.Module):
    
    def __init__(self, c_in=1, config=NETWORK_CONFIG, num_classes=20):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.c_in = c_in
        self.config = config
        self.layers = self._create_network_layers()
        
    def _create_network_layers(self):
        layers = nn.ModuleList()
        c_in = self.c_in
        for module in self.config:
            if isinstance(module, tuple):
                c_out, kernel_size, stride = module
                layers.append(
                    ConvBlock(
                        c_in,
                        c_out,
                        kernel_size=kernel_size,
                        stride = stride,
                        padding=1 if kernel_size==3 else 0
                    )
                )
                c_in = c_out
            
            # this is for the res blocks, they dont change the scale
            # just does more convolutions with residual connnections
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(
                    ResBlock(
                        c_in,
                        num_repeats=num_repeats,
                    )
                )
            
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResBlock(c_in, res_connection=False, num_repeats=1),
                        ConvBlock(c_in, c_in//2, kernel_size=1),
                        ScalePrediction(c_in//2, num_classes=self.num_classes)
                    ]
                    c_in = c_in // 2
                elif module == "U":
                    layers.append(
                        nn.Upsample(scale_factor=2),
                    )
                    c_in = c_in * 3
        
        return layers
        
    def forward(self, x):
        outputs = []
        route_connections = []
        for layer in self.layers:
            if isinstance(layer,  ScalePrediction):
                outputs.append(layer(x))
                continue
            
            #print(f"----> Before shape: {x.shape}")
            x = layer(x)
            #print(f"----> After shape: {x.shape}")
            
            if isinstance(layer, ResBlock) and layer.num_repeats == 8:
                route_connections.append(x)
                #print(f"Route connections appended with : {route_connections[-1].shape}")
            
            elif isinstance(layer, nn.Upsample):
                # after upsampling, you just concatenate the res blocks from before
                # that are stored in route connections
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
            
        return outputs

def test():
    input_img = torch.zeros((1, 1, 512, 2048))
    config = [
        (32, 3, 1),
        (64, 3, 2), # H, W gets halved after this layer
        ["B", 1],
        (128, 3, 2),# H, W gets halved after this layer
        ["B", 2],
        (256, 3, 2),
        ["B", 8], # store features in route connections to add them later, H,W, C remains same
        (512, 3, 2), # H, W gets halved after this layer
        ["B", 8], # Store features in route connections to add them layer, H, W, C remains same
        (1024, 3, 2),# H, W gets halved after this layer
        ["B", 4], # We cut H, W by 16 till here, C scaled from 1 to 1024
        (512, 1, 1), # shrink to 512 channels, keeping H, W same
        (1024, 3, 1), # Scale back channels again to 1024
        "S", # do the bbox predictions at this scale on a
             #side branch and keep the main line with half the channels 
        (256, 1, 1), # cut the number of channels in half, H, W is the same
        "U",
        (256, 1, 1),
        (512, 3, 1),
        "S", # will be 256 channels after this step
        (128, 1, 1), # 128 channels after this step
        "U", #  
        (128, 1, 1),
        (256, 3, 1),
        "S"
    ]
    net = YOLOv3(c_in=1, config=config, num_classes=2)
    out_net = net(input_img)
    print(f"Input shape: {input_img.shape}")
    for tensor in out_net:
        print(f"Output tensor shapes: {tensor.shape}")
    #print(f"Current head shape: {x.shape}")


model_dict = {
    "YOLOv3": YOLOv3
}