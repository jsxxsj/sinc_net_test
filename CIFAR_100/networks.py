"""the models used for experiment"""
"""
References:
Resnet:()
Birealnet:()
ReActnet:()
"""
from audioop import bias
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from cmath import pi

"""basic operations"""
def conv3x3(in_channels,out_channels,stride=1):
    """3x3 conv with padding"""
    return nn.Conv2d(in_channels,out_channels, 
                    kernel_size=3,stride=stride,
                    padding=1,bias=False )

def conv1x1(in_channels,out_channels,stride=1):
    """1x1 conv with padding"""
    return nn.Conv2d(in_channels,out_channels, 
                    kernel_size=1,stride=stride,
                    padding=0,bias=False )

"""nn layers and functions"""

class BinaryActivation(nn.Module):
    """Binarize input of a network, using quadratic functions"""
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

class LearnableBias(nn.Module):
    """bias for Prelu and activations"""
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class HardBinaryConv(nn.Module):
    """conv between binarized weights and inputs
    using traditional straight throught estimator
    using full precision scaling factor"""
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        #self.weight = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)

    def forward(self, x):
        #real_weights = self.weights.view(self.shape)
        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y

"""different resblocks"""
class resnet_block(nn.Module):
    """without any binary oprations"""
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(resnet_block, self).__init__()

        self.move0 = LearnableBias(inplanes)
        #self.binary_activation = BinaryActivation()
        self.binary_conv = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.move1 = LearnableBias(planes)
        self.prelu = nn.PReLU(planes)
        self.move2 = LearnableBias(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.move0(x)
        #print(out.shape)
        #out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        #print(out.shape,residual.shape)
        out += residual
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)

        return out

class ReActnet_block(nn.Module):
    """activation and weight binarized with simple and tanh like function"""
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ReActnet_block, self).__init__()

        self.move0 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.move1 = LearnableBias(planes)
        self.prelu = nn.PReLU(planes)
        self.move2 = LearnableBias(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)
        return out

###########################################################################
cut = 20
T0 = 3
'the number remained in the Fourier scales, must be even'
delta_x = T0/(2*cut)
omega_c = 2*pi/T0

def bn(n):
    'the Fourier number of sign function'
    return 2*(1-(-1)**n)/(pi*n)
      
class tough_quant(nn.Module):
    """quantizing the input x to the hypothetical sampling point
    begining with +- delta_x/2"""
    def __init__(self) -> None:
        super(tough_quant,self).__init__()
    
    def forward(self,x):
        #quantize x to n*delta_x
        out0 = (x-delta_x/2)/delta_x
        out1 = torch.round(out0)
        out2 = out1*delta_x + delta_x/2
        #output, using the basic STE(y=q(x) forward, y=x backward)
        out = out2.detach() - x.detach() + x
        return out

class sinc_sign_func(nn.Module):
    """using a cluster of sin fuctions to simulate the sign function"""
    def __init__(self) -> None:
        super(sinc_sign_func,self).__init__()

    def forward(self,x):
        output = x*0
        for i in range(1,cut + 1):
            output += bn(i)*torch.sin(x*2*pi*x/T0)

        return output

class sinc_sign_activation(nn.Module):
    def __init__(self):
        super(sinc_sign_activation, self).__init__()
        self.tough_quant = tough_quant()
        self.fine_quant = sinc_sign_func()

    def forward(self, x):
        cliped_x = torch.clamp(x,-1.0,1.0)
        tough_quanted_x = self.tough_quant(cliped_x)
        fine_quanted_x = self.fine_quant(tough_quanted_x)

        return fine_quanted_x

class sinc_binary_conv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(sinc_binary_conv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        #self.weight = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)
        
        self.tough_quant = tough_quant()
        self.fine_quant = sinc_sign_func()

    def forward(self, x):
        #real_weights = self.weights.view(self.shape)
        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()

        tough_quanted_weights = self.tough_quant(real_weights)
        fine_quanted_weights = self.fine_quant(tough_quanted_weights) 
        scaled_binary_weights = fine_quanted_weights * scaling_factor
        y = F.conv2d(x, scaled_binary_weights, stride=self.stride, padding=self.padding)

        return y

class sincnet_block(nn.Module):
    """activation and weight binarized with simple and tanh like function"""
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(sincnet_block, self).__init__()

        self.move0 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.binary_conv = sinc_binary_conv(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.move1 = LearnableBias(planes)
        self.prelu = nn.PReLU(planes)
        self.move2 = LearnableBias(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)

        return out
##############################################################################

class grad_bias(torch.autograd.Function):
    'return a constant as grad'
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x
    @staticmethod
    def backward(ctx,grad_output):
        x, = ctx.saved_tensors
        return torch.sign(x)

class learnable_grad_bias(nn.Module):
    def __init__(self,out_chn) -> None:
        super(learnable_grad_bias,self).__init__()
        #self.grad_bias = grad_bias()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self,x):
        bias_out = self.bias*(grad_bias.apply(x))
        out = x + bias_out - bias_out.detach() 

        return out

class gradshift_react_block(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(gradshift_react_block, self).__init__()

        self.move0 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(inplanes, planes, stride=stride)
        self.gradmove = learnable_grad_bias(planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.move1 = LearnableBias(planes)
        self.prelu = nn.PReLU(planes)
        self.move2 = LearnableBias(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.gradmove(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)
        return out

"""resnet structure
simplified for MNIST"""
class resnet(nn.Module):
    """modified resnet, for minst data test"""
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False):
        super(resnet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

"""functions calling resnet"""
def resnet5(pretrained=False, **kwargs):
    """Constructs a full-precision resnet model. """
    model = resnet(resnet_block, [4], **kwargs)
    return model

def block_selector(name:str):
    if name == 'fp_block':
        return resnet(resnet_block, [4,4,4,4],num_classes=100)
    if name == 'reactnet_block':
        return resnet(ReActnet_block,[4,4,4,4],num_classes=100)
    if name == 'sincnet_block':
        return resnet(sincnet_block,[4,4,4,4],num_classes=100)
    if name == 'gsreactnet_block':
        return resnet(gradshift_react_block,[4,4,4,4],num_classes=100)

    