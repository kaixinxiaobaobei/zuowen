# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Experimental modules
"""
import math

import numpy as np
import torch
import torch.nn as nn

from models.common import Conv
from utils.downloads import attempt_download

def attempt_load(weights, device=None, inplace=True, fuse=True):
    """
    Loads and fuses an ensemble or single YOLOv5 model from weights, handling device placement and model adjustments.

    Example inputs: weights=[a,b,c] or a single model weights=[a] or weights=a.
    """
    from models.yolo import Detect, Model

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location="cpu")  # load
        ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, "stride"):
            ckpt.stride = torch.tensor([32.0])
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval())  # model in eval mode

    # Module updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f"Models have different class counts: {[m.nc for m in model]}"
    return model
class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))



# ///////////////////////////////////////////////////////////////////////////////////////////////////
class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            y.append(module(x, augment, profile, visualize)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None, inplace=True, fuse=True):
    from models.yolo import Detect, Model

    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location=map_location)  # load
        if fuse:
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
        else:
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().eval())  # without layer fuse

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = inplace  # pytorch 1.7.0 compatibility
            if type(m) is Detect:
                if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model  # return ensemble  #
# ////////////////////////////////////////////////////////////////////////////////////////////////////



# Adaptive Structure Feature Fusion  /////////////////////////////////////////////////////////////////

# SimSPPF ////////////////////////////////////////////////////////////////////////////////////////////
class SimConv(nn.Module):
    '''Normal Conv with ReLU activation'''

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))



# ////////////////////////////////////////////////////////////////////////////////////////////////////


# CSP-stage //////////////////////////////////////////////////////////////////////////////////////////
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module(
        'conv',
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  groups=groups,
                  bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepConv(nn.Module):
    '''RepConv is a basic rep-style block, including training and deploy status
    Code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 deploy=False,
                 act='relu',
                 norm=None):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        if isinstance(act, str):
            self.nonlinearity = get_activation(act)
        else:
            self.nonlinearity = act

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = None
            self.rbr_dense = conv_bn(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=stride,
                                   padding=padding_11,
                                   groups=groups)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(
            self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(
            kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True
# swish激活函数///////////////////////////////////////////////////////
class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)
#/////////////////////////////////////////////////////////////////////
def get_activation(name='silu', inplace=True):
    if name is None:
        return nn.Identity()

    if isinstance(name, str):
        if name == 'silu':
            module = nn.SiLU(inplace=inplace)
        elif name == 'relu':
            module = nn.ReLU(inplace=inplace)
        elif name == 'lrelu':
            module = nn.LeakyReLU(0.1, inplace=inplace)
        elif name == 'swish':
            module = Swish(inplace=inplace)
        elif name == 'hardsigmoid':
            module = nn.Hardsigmoid(inplace=inplace)
        elif name == 'identity':
            module = nn.Identity()
        else:
            raise AttributeError('Unsupported act type: {}'.format(name))
        return module

    elif isinstance(name, nn.Module):
        return name

    else:
        raise AttributeError('Unsupported act type: {}'.format(name))

def get_norm(name, out_channels, inplace=True):
    if name == 'bn':
        module = nn.BatchNorm2d(out_channels)
    else:
        raise NotImplementedError
    return module

class ConvBNAct(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""
    def __init__(
        self,
        in_channels,
        out_channels,
        ksize,
        stride=1,
        groups=1,
        bias=False,
        act='silu',
        norm='bn',
        reparam=False,
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        if norm is not None:
            self.bn = get_norm(norm, out_channels, inplace=True)
        if act is not None:
            self.act = get_activation(act, inplace=True)
        self.with_norm = norm is not None
        self.with_act = act is not None

    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            x = self.bn(x)
        if self.with_act:
            x = self.act(x)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))

class BasicBlock_3x3_Reverse(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_hidden_ratio,
                 ch_out,
                 act='relu',
                 shortcut=True):
        super(BasicBlock_3x3_Reverse, self).__init__()
        assert ch_in == ch_out
        ch_hidden = int(ch_in * ch_hidden_ratio)
        self.conv1 = ConvBNAct(ch_hidden, ch_out, 3, stride=1, act=act)
        self.conv2 = RepConv(ch_in, ch_hidden, 3, stride=1, act=act)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv2(x)
        y = self.conv1(y)
        if self.shortcut:
            return x + y
        else:
            return y

class SPP(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        k,
        pool_size,
        act='swish',
    ):
        super(SPP, self).__init__()
        self.pool = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(kernel_size=size,
                                stride=1,
                                padding=size // 2,
                                ceil_mode=False)
            self.add_module('pool{}'.format(i), pool)
            self.pool.append(pool)
        self.conv = ConvBNAct(ch_in, ch_out, k, act=act)

    def forward(self, x):
        outs = [x]

        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, axis=1)

        y = self.conv(y)
        return y


# BlurPool 一种改进的下采样方法，考虑了下采样前使用低通滤波器来抗混叠，很大程度上缓解了CNNs平移不变性的丢失
class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)),
                          int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        if self.filt_size == 1:
            a = np.array([1., ])
        elif self.filt_size == 2:
            a = np.array([1., 1.])
        elif self.filt_size == 3:
            a = np.array([1., 2., 1.])
        elif self.filt_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif self.filt_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif self.filt_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif self.filt_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer(pad_type):
    if pad_type in ['refl', 'reflect']:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == 'zero':
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer
# /////////////////////////////////////////////////////////////////////////////////////////////////////
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
class PSA_p(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_p, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size-1)//2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1,2))
        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)
        # [N, 1, H*W]
        context = self.softmax_left(context)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x):
        # [N, C, H, W]
        context_channel = self.spatial_pool(x)
        # [N, C, H, W]
        context_spatial = self.channel_pool(x)
        # [N, C, H, W]
        out = context_spatial + context_channel
        return out

class PSA_s(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_s, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        ratio = 4

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)
        # self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_up = nn.Sequential(
            nn.Conv2d(self.inter_planes, self.inter_planes // ratio, kernel_size=1),
            nn.LayerNorm([self.inter_planes // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_planes // ratio, self.planes, kernel_size=1)
        )
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1, 2))

        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        # [N, IC, H*W]
        theta_x = self.softmax_left(theta_x)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x):
        # [N, C, H, W]
        out = self.spatial_pool(x)

        # [N, C, H, W]
        out = self.channel_pool(out)

        # [N, C, H, W]
        # out = context_spatial + context_channel

        return out




class C3_PSA_p(nn.Module):
    # CSP Bottleneck with PSA
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)

        # Replace Bottleneck with PSA
        self.m = nn.Sequential(*(PSA_p(c_, c_) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class C3_PSA_s(nn.Module):
    # CSP Bottleneck with PSA
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)

        # Replace Bottleneck with PSA
        self.m = nn.Sequential(*(PSA_s(c_, c_) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
class SimSPPF(nn.Module):
    '''Simplified SPPF with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = SimConv(in_channels, c_, 1, 1)
        self.cv2 = SimConv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
