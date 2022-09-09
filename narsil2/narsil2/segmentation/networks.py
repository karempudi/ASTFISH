import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace




class ConvBlock(nn.Module):
    
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.block(x)
        return x


    
class UpsampleBlock(nn.Module):
    r"""
    """
    def __init__(self, c_in, c_out, 
                 upsample_type="transposeConv", feature_fusion_type="concat"):
        super().__init__()
        self.upsample_type = upsample_type
        self.feature_fusion_type = feature_fusion_type
        
        if self.upsample_type == "transposeConv":
            self.upsample_block = nn.ConvTranspose2d(c_in, c_in, kernel_size=2, stride=2)
        elif self.upsample_type == "upsample":
            self.upsample_block = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            
        if self.feature_fusion_type == "concat":
            c_in = c_in + c_out
            self.conv_block = ConvBlock(c_in, c_out)
        elif self.feature_fusion_type == "add":
            self.shrink_block = ConvBlock(c_in, c_out)
            self.conv_block = ConvBlock(c_out, c_out)
        
    def forward(self, x, features=None):
        if features is not None:
            # then you have something coming from the side
            x = self.upsample_block(x)
            if self.feature_fusion_type == "concat":
                x = torch.cat([features, x], dim=1) # concat along the channel "C" dimension
            elif self.feature_fusion_type == "add":
                x = self.shrink_block(x) + features
            x = self.conv_block(x)
            return x
        else:
            x = self.upsample_block(x)
            x = self.conv_block(x)
            return x

class Unet(nn.Module):
    
    def __init__(self, channels_by_scale, num_classes=1,
                 upsample_type="transposeConv", feature_fusion_type="concat"):
        super().__init__()
        
        self.hparams = SimpleNamespace(channels_by_scale=channels_by_scale,
                                      num_classes=num_classes,
                                      upsample_type=upsample_type,
                                      feature_fusion_type=feature_fusion_type)
        #self.channels_by_scale = channels_by_scale
        #self.upsample_type = upsample_type
        #self.feature_fusion_type = feature_fusion_type
        #self.num_classes = num_classes
        self._create_network()
        self._init_params()
        
    def _create_network(self):
        down_layers = []
        for layer_idx in range(len(self.hparams.channels_by_scale)-1):
            down_layers.append(
                ConvBlock(self.hparams.channels_by_scale[layer_idx], 
                          self.hparams.channels_by_scale[layer_idx+1])
            )
            if layer_idx < len(self.hparams.channels_by_scale) - 2:
                down_layers.append(nn.MaxPool2d(2))
            
        self.down_layers = nn.Sequential(*down_layers)
        
        up_layers = []
        reversed_channels = self.hparams.channels_by_scale[::-1]
        for layer_idx in range(len(reversed_channels)-2):
            #print(layer_idx, reversed_channels[layer_idx])
            up_layers.append(
                UpsampleBlock(reversed_channels[layer_idx],
                              reversed_channels[layer_idx+1],
                              upsample_type=self.hparams.upsample_type,
                              feature_fusion_type=self.hparams.feature_fusion_type
                             )
            )
            
        self.up_layers = nn.Sequential(*up_layers)
        
        self.last_conv = nn.Conv2d(self.hparams.channels_by_scale[1],
                                   self.hparams.num_classes, 
                                   kernel_size=1)
    
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # as we have relu's we use kaiming inialization
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                
    
    def forward(self, x):
        # pass down and accumulate the values need to sum
        # for every second layer in down layers accumulate values
        features =[]
        for i, layer in enumerate(self.down_layers):
            x = layer(x)
            if i%2 == 0 and i < len(self.down_layers) - 1:
                features.append(x)
            #print(i, x.shape)
        # now go up from here and add the features from the features array 
        # in reverse
        #print("----------")
        #for i, feature in enumerate(features,0):
        #    print("Features", i, feature.shape)
        #print("-----------")
        for i, layer in enumerate(self.up_layers, 1):
            x = layer(x, features[-i])
            #print(x.shape)
            
        x = self.last_conv(x)

        return x

##########################################################
############### U-net with Res-blocks ####################
##########################################################

class ResidualConvBlock(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        )
        self.skip_connection = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(c_out)
        )
    
    def forward(self, x):
        x = self.block(x) + self.skip_connection(x)
        return x

    
class ResidualUpsampleBlock(nn.Module):
    r"""
    """
    def __init__(self, c_in, c_out, 
                 upsample_type="transposeConv", feature_fusion_type="concat"):
        super().__init__()
        self.upsample_type = upsample_type
        self.feature_fusion_type = feature_fusion_type
        
        if self.upsample_type == "transposeConv":
            self.upsample_block = nn.ConvTranspose2d(c_in, c_in, kernel_size=2, stride=2)
        elif self.upsample_type == "upsample":
            self.upsample_block = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            
        if self.feature_fusion_type == "concat":
            c_in = c_in + c_out
            self.conv_block = ResidualConvBlock(c_in, c_out)
        elif self.feature_fusion_type == "add":
            self.shrink_block = ResidualConvBlock(c_in, c_out)
            self.conv_block = ResidualConvBlock(c_out, c_out)
        
    def forward(self, x, features=None):
        if features is not None:
            # then you have something coming from the side
            x = self.upsample_block(x)
            if self.feature_fusion_type == "concat":
                x = torch.cat([features, x], dim=1) # concat along the channel "C" dimension
            elif self.feature_fusion_type == "add":
                x = self.shrink_block(x) + features
            x = self.conv_block(x)
            return x
        else:
            x = self.upsample_block(x)
            return x


class ResUnet(nn.Module):

    def __init__(self, channels_by_scale, num_classes=1,
                upsample_type="transposeConv", feature_fusion_type="concat"):
        super().__init__()
        self.hparams = SimpleNamespace(channels_by_scale=channels_by_scale,
                                      num_classes=num_classes,
                                      upsample_type=upsample_type,
                                      feature_fusion_type=feature_fusion_type,
                                      residual_net=True)
        self._create_network()
        self._init_params()
        
    def _create_network(self):
        down_layers = []
        for layer_idx in range(len(self.hparams.channels_by_scale)-1):
            down_layers.append(
                ResidualConvBlock(self.hparams.channels_by_scale[layer_idx], 
                          self.hparams.channels_by_scale[layer_idx+1])
            )
            if layer_idx < len(self.hparams.channels_by_scale) - 2:
                down_layers.append(nn.MaxPool2d(2))
            
        self.down_layers = nn.Sequential(*down_layers)
        
        up_layers = []
        reversed_channels = self.hparams.channels_by_scale[::-1]
        for layer_idx in range(len(reversed_channels)-2):
            #print(layer_idx, reversed_channels[layer_idx])
            up_layers.append(
                ResidualUpsampleBlock(reversed_channels[layer_idx],
                              reversed_channels[layer_idx+1],
                              upsample_type=self.hparams.upsample_type,
                              feature_fusion_type=self.hparams.feature_fusion_type
                             )
            )
            
        self.up_layers = nn.Sequential(*up_layers)
        
        self.last_conv = nn.Conv2d(self.hparams.channels_by_scale[1],
                                   self.hparams.num_classes, 
                                   kernel_size=1)
    
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # as we have relu's we use kaiming inialization
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                
    
    def forward(self, x):
        # pass down and accumulate the values need to sum
        # for every second layer in down layers accumulate values
        features =[]
        for i, layer in enumerate(self.down_layers):
            x = layer(x)
            if i%2 == 0 and i < len(self.down_layers) - 1:
                features.append(x)
            #print(i, x.shape)
        # now go up from here and add the features from the features array 
        # in reverse
        #print("----------")
        #for i, feature in enumerate(features,0):
        #    print("Features", i, feature.shape)
        #print("-----------")
        for i, layer in enumerate(self.up_layers, 1):
            x = layer(x, features[-i])
            #print(x.shape)
            
        x = self.last_conv(x)

        return x


##########################################################
############### CellPose Network #########################
##########################################################

class CellPoseNet(nn.Module):

    def __init__(self):
        pass

    def forward(self, x):
        return x

##########################################################
################## UNeXt/ConvNeXt Netowrk ################
##########################################################


class DropPath(nn.Module):
    r""" Drop paths (Stochastic Depth) per sample 
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1, ) * (x.ndim - 1)
        # basically dropping batch dimensions with some bernoilli 
        # trail based coin flips and scaling the other appropriately
        # as you randomly nulled voi
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

class LayerNorm(nn.Module):
    r""" Layernorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (N, H, W, C) while channels_first corresponds to (N, C, H , W)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # just doing mean/std dev of each layer ie dimension "C"
            u = x.mean(1, keepdim=True)
            s = (x - u)**pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. Two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    
    They seem to use the (2) option as they noticed that it was slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels
        drop_path (float) : Stochastic depth rate, Default: 0.0 . Don't know what this is yet
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, c_in, c_out, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(c_in, c_out, kernel_size=7, padding=3)
        self.norm = LayerNorm(c_out, eps=1e-6)
        self.pwconv1 = nn.Linear(c_out, 4 * c_out)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * c_out, c_out)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((c_out)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.skip_connection = nn.Conv2d(c_in, c_out, kernel_size=7, padding=3)
        
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        
        x = self.skip_connection(input) + self.drop_path(x)
        return x


   
class neXtUpsampleBlock(nn.Module):
    r"""
    """
    def __init__(self, c_in, c_out, 
                 upsample_type="transposeConv", feature_fusion_type="concat"):
        super().__init__()
        self.upsample_type = upsample_type
        self.feature_fusion_type = feature_fusion_type
        
        if self.upsample_type == "transposeConv":
            self.upsample_block = nn.ConvTranspose2d(c_in, c_in, kernel_size=2, stride=2)
        elif self.upsample_type == "upsample":
            self.upsample_block = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            
        if self.feature_fusion_type == "concat":
            c_in = c_in + c_out
            self.conv_block = ConvNeXtBlock(c_in, c_out)
        elif self.feature_fusion_type == "add":
            self.shrink_block = ConvNeXtBlock(c_in, c_out)
            self.conv_block = ConvNeXtBlock(c_out, c_out)
        
    def forward(self, x, features=None):
        if features is not None:
            # then you have something coming from the side
            x = self.upsample_block(x)
            if self.feature_fusion_type == "concat":
                x = torch.cat([features, x], dim=1) # concat along the channel "C" dimension
            elif self.feature_fusion_type == "add":
                x = self.shrink_block(x) + features
            x = self.conv_block(x)
            return x
        else:
            x = self.upsample_block(x)
            return x


class UNeXt(nn.Module):
    r""" ConvNeXt, 
        Implementation from, testing out the net to understand how it works
        https://github.com/facebookresearch/ConvNeXt/blob/main/semantic_segmentation/backbone/convnext.py
    
    Arguments:
    ----------
    channels_by_scale (list): list of channels by scale [1, 8, 16, ..]
                                from top to bottom of the U-part of the network

    num_classes (int): number of channels in the output 1 for normal, 4 for Omni-method

    upsample_type (string): "transposeConv" or "upsample" different types of upsampling
                            methods
    
    feature_fusion_type (string): "concat" or "add", how are the feature fusion
    
    """
    def __init__(self, channels_by_scale, num_classes=1,
            upsample_type="transposeConv", feature_fusion_type="concat"):
        super().__init__()

        self.hparams = SimpleNamespace(channels_by_scale=channels_by_scale,
                                       num_classes=num_classes,
                                       upsample_type=upsample_type,
                                       feature_fusion_type=feature_fusion_type,
                                       unext=True)
        
        self._create_network()
        self._init_params()
    
    def _create_network(self):
        down_layers = []
        for layer_idx in range(len(self.hparams.channels_by_scale)-1):
            down_layers.append(
                ConvNeXtBlock(self.hparams.channels_by_scale[layer_idx], 
                          self.hparams.channels_by_scale[layer_idx+1])
            )
            if layer_idx < len(self.hparams.channels_by_scale) - 2:
                down_layers.append(nn.MaxPool2d(2))
            
        self.down_layers = nn.Sequential(*down_layers)
        
        up_layers = []
        reversed_channels = self.hparams.channels_by_scale[::-1]
        for layer_idx in range(len(reversed_channels)-2):
            #print(layer_idx, reversed_channels[layer_idx])
            up_layers.append(
                neXtUpsampleBlock(reversed_channels[layer_idx],
                              reversed_channels[layer_idx+1],
                              upsample_type=self.hparams.upsample_type,
                              feature_fusion_type=self.hparams.feature_fusion_type
                             )
            )
            
        self.up_layers = nn.Sequential(*up_layers)
        
        self.last_conv = nn.Conv2d(self.hparams.channels_by_scale[1],
                                   self.hparams.num_classes, 
                                   kernel_size=1)
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # as we have relu's we use kaiming inialization
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                #nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # pass down and accumulate the values need to sum
        # for every second layer in down layers accumulate values
        features =[]
        for i, layer in enumerate(self.down_layers):
            x = layer(x)
            if i%2 == 0 and i < len(self.down_layers) - 1:
                features.append(x)
            #print(i, x.shape)
        # now go up from here and add the features from the features array 
        # in reverse
        #print("----------")
        #for i, feature in enumerate(features,0):
        #    print("Features", i, feature.shape)
        #print("-----------")
        for i, layer in enumerate(self.up_layers, 1):
            x = layer(x, features[-i])
            #print(x.shape)
            
        x = self.last_conv(x)

        return x
#####################################################################
################### PLUS Networks ###################################
#####################################################################

class UNetPlus(nn.Module):

    def __init__(self):
        pass


    def _create_network(self):
        pass

    
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # as we have relu's we use kaiming inialization
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    
    def forward(self, x):
        return x



class basicUnet(nn.Module):
    '''
    Basic Unet stuff, simply copied the architecture with
    changes to sizes and layers
    
    '''

    def __init__(self, transposeConv = True):
        # initialize the superclass initializer from the library modules
        super(basicUnet, self).__init__()
        self.transposeConv = transposeConv
        # 1 because of the number of input channels is 1
        self.initial = double_conv(1, 64)
        self.down1 = double_conv(64, 128)
        self.down2 = double_conv(128, 256)
        self.down3 = double_conv(256, 512)
        self.down4 = double_conv(512, 512)

        self.up1 = up_conv_cat(1024, 256, self.transposeConv)
        self.up2 = up_conv_cat(512, 128, self.transposeConv)
        self.up3 = up_conv_cat(256, 64, self.transposeConv)
        self.up4 = up_conv_cat(128, 64, self.transposeConv)
        self.out = nn.Conv2d(64, 1, 1) # 2 because of no_classes

    def forward(self, x):
        # x will be the image batch tensor, that will be propagated across
        # toward the end and the 
        x1 = self.initial(x)
        x2 = self.down1(F.max_pool2d(x1, (2, 2)))
        x3 = self.down2(F.max_pool2d(x2, (2, 2)))
        x4 = self.down3(F.max_pool2d(x3, (2, 2)))
        x5 = self.down4(F.max_pool2d(x4, (2, 2)))

        # copied code
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)


        return x
        #return torch.sigmoid(x)


class double_conv(nn.Module):
    '''
    Combining conv1- batch - relu - conv batch relu parts of UNet
    
    Features: No change in size of the image (maintained using padding)
    '''
    
    def __init__(self, input_channels, output_channels):
        ''' Just initializes the conv - batch - relu layers twice 
            with no reduction in image size (don't change padding , stride)
        '''
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(output_channels, output_channels, 3, stride = 1, padding =1),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x



class up_conv_cat(nn.Module):
    '''
    UpConv + concatenation of features during downscaling
    '''

    def __init__(self, input_channels, output_channels, transposeConv):
        super(up_conv_cat, self).__init__()

        if transposeConv:
            self.up = nn.ConvTranspose2d(input_channels//2, input_channels//2, 2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)

        self.conv = double_conv(input_channels, output_channels)

    def forward(self, from_bottom, from_side):

        x1 = self.up(from_bottom)

        x2 = torch.cat([from_side, x1], dim = 1)

        x2 = self.conv(x2)
 
        return x2

class smallerUnet(nn.Module):
    '''
    Basic Unet stuff, simply copied the architecture with
    changes to sizes and layers, this one wont get 
    inputs from the side
    I am guessing they act a gradient highways preventing
    the net from learning features
    
    '''

    def __init__(self, transposeConv):
        # initialize the superclass initializer from the library modules
        super(smallerUnet, self).__init__()
        self.transposeConv = transposeConv
        # 1 because of the number of input channels is 1
        self.initial = double_conv(1, 8)
        self.down1 = double_conv(8, 16)
        self.down2 = double_conv(16, 32)
        self.down3 = double_conv(32, 64)
        self.down4 = double_conv(64, 64)

        self.up1 = up_conv(128, 32, self.transposeConv)
        self.up2 = up_conv(64, 16, self.transposeConv)
        self.up3 = up_conv(32, 8, self.transposeConv)
        self.up4 = up_conv(16, 8, self.transposeConv)
        self.out = nn.Conv2d(8, 1, 1) # 2 because of no_classes

    def forward(self, x):
        # x will be the image batch tensor, that will be propagated across
        # toward the end and the 
        x1 = self.initial(x)
        x2 = self.down1(F.max_pool2d(x1, (2, 2)))
        x3 = self.down2(F.max_pool2d(x2, (2, 2)))
        x4 = self.down3(F.max_pool2d(x3, (2, 2)))
        x5 = self.down4(F.max_pool2d(x4, (2, 2)))

        # copied code
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)


        return x # sigmoid is done in the loss function

class up_conv(nn.Module):
    '''
    UpConv and no concatenation from the side, pure down convs and up-convs
    '''

    def __init__(self, input_channels, output_channels, transposeConv):
        super(up_conv, self).__init__()

        if transposeConv:
            self.up = nn.ConvTranspose2d(input_channels//2, input_channels//2, 2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = double_conv(input_channels, output_channels)

    def forward(self, from_bottom, from_side):

        x1 = self.up(from_bottom)

        x2 = torch.cat([from_side, x1], dim = 1)

        x2 = self.conv(x2)
 
        return x2

model_dict = {
    "Unet": Unet,
    "UNeXt": UNeXt,
    "UNetPlus": UNetPlus,
    "CellPose": CellPoseNet,
    "ResUnet": ResUnet
}
    