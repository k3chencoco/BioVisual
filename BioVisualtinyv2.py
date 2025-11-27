import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch

from utils import InitWeights_He


#ODS 0.809
class RGB2Gray(nn.Module):
    def __init__(self, in_ch =3, learnable=True):
        super().__init__()
        # 使用 1x1 卷积实现线性加权
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, bias=False)
        # 初始化为 Y 通道权重
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[[[0.299]], [[0.587]], [[0.114]]]]))
        # 是否可学习
        if not learnable:
            self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride=stride,
                                   padding=padding, dilation=dilation, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.relu(self.depthwise(x))
        return self.pointwise(x)

class Inhibition(nn.Module):
    def __init__(self, in_ch, out_ch, ckernel=1, skernel=3, cdilation=1, sdilation=1, mode = 'on'):
        super(Inhibition, self).__init__()
        self.cpadding = (ckernel - 1) // 2 * cdilation
        self.spadding = (skernel-1) // 2 * sdilation
        self.mode = mode
        self.alpha = nn.Parameter(torch.ones(1, out_ch, 1, 1))
        if ckernel == 1:
            self.Xcenter = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=ckernel, padding=self.cpadding),
                nn.LeakyReLU(0.1, inplace=True)
            )
        else:
            self.Xcenter = DepthwiseSeparableConv(in_ch, out_ch, kernel_size=ckernel, padding=self.cpadding, dilation=cdilation)

        self.Xsurround = DepthwiseSeparableConv(in_ch, out_ch, kernel_size=skernel, padding=self.spadding, dilation=sdilation)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        alpha =self.alpha
        xcenter = self.Xcenter(x)
        xsurround = self.Xsurround(x)

        if self.mode == 'off':
            x_out = xsurround - alpha * xcenter
        elif self.mode == 'on':
            x_out = xcenter - alpha * xsurround
        else:
            x_out = xcenter + alpha * xsurround
        return x_out


def gabor_kernel(kernel_size=7, sigma=2.0, theta=0, Lambda=3.0, psi=0, gamma=0.5):
    half_size = kernel_size // 2
    y, x = np.meshgrid(np.arange(-half_size, half_size + 1), np.arange(-half_size, half_size + 1))
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb = np.exp(- (x_theta ** 2 + (gamma ** 2) * y_theta ** 2) / (2 * sigma ** 2)) * np.cos(
        2 * np.pi * x_theta / Lambda + psi)
    return gb.astype(np.float32)


def init_gabor_weights(conv, theta, in_channels, kernel_size, phase=0):
    weights = np.zeros((in_channels, 1, kernel_size, kernel_size), dtype=np.float32)
    gabor = gabor_kernel(kernel_size=kernel_size, theta=theta, psi=phase)
    for c in range(in_channels):
        weights[c, 0, :, :] = gabor
    conv.weight.data = torch.from_numpy(weights)
    conv.weight.requires_grad = True  # 如果想冻结 Gabor 则设为 False

class DirectionalConv(nn.Module):
    def __init__(self, in_ch, out_ch, directions=8, kernel_size=7):
        super().__init__()
        self.directions = directions
        self.kernel_size = kernel_size
        self.in_ch = in_ch
        self.out_ch_per_dir = out_ch // directions
        padding = (kernel_size - 1) // 2

        self.conv_phase0 = nn.ModuleList()
        self.conv_phase90 = nn.ModuleList()
        self.conv0p_list = nn.ModuleList()
        self.conv90p_list = nn.ModuleList()

        for i in range(directions):
            theta = i * np.pi / directions
            # depthwise conv for phase0
            # print(f"in_ch={self.in_ch}")
            conv0 = nn.Conv2d(self.in_ch, self.in_ch, kernel_size=kernel_size, padding=padding,groups=self.in_ch)
            init_gabor_weights(conv0, theta, self.in_ch, kernel_size, phase=0)
            conv0p = nn.Conv2d(self.in_ch, self.out_ch_per_dir, kernel_size=1)
            # phase90 conv
            conv90 = nn.Conv2d(self.in_ch, self.in_ch, kernel_size=kernel_size, padding=padding,groups=self.in_ch)
            init_gabor_weights(conv90, theta, self.in_ch, kernel_size, phase=np.pi/2)
            conv90p = nn.Conv2d(self.in_ch, self.out_ch_per_dir, kernel_size=1)

            self.conv_phase0.append(conv0)
            self.conv0p_list.append(conv0p)
            self.conv_phase90.append(conv90)
            self.conv90p_list.append(conv90p)

        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        outs0 = []
        outs90 = []
        for conv0, conv0p, conv90, conv90p in zip(
                self.conv_phase0, self.conv0p_list, self.conv_phase90, self.conv90p_list):
            # print(f"x shape={x.shape}")
            out0 = conv0(x)
            out0 = conv0p(out0)
            out90 = conv90(x)
            out90 = conv90p(out90)
            outs0.append(out0)
            outs90.append(out90)
        # concat 所有方向
        outs0 = torch.cat(outs0, dim=1)  # [B, directions*out_ch_per_dir, H, W]
        outs90 = torch.cat(outs90, dim=1)  # [B, directions*out_ch_per_dir, H, W]
        return outs0, outs90

class RodCell(nn.Module):
    def __init__(self, in_ch=3, out_ch=32, kernel_size=7, dilation=2):
        super().__init__()
        # Rod path (gray-scale only, larger kernel)
        self.padding = (kernel_size-1) // 2 * dilation
        self.rgb2gray = RGB2Gray(in_ch, learnable=True)
        # 直接把 1 通道灰度 → out_ch 通道
        self.expand = nn.Conv2d(1, out_ch, kernel_size=1, bias=False)
        self.rod_ini = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=self.padding, dilation=dilation)
        self.rod_conv = Inhibition(out_ch, out_ch, ckernel=1, skernel=kernel_size-2, sdilation=dilation)
    def forward(self, x):
        # Convert RGB to grayscale luminance (Y channel)
        x_gray = self.rgb2gray(x)
        x_gray = self.expand(x_gray)
        rod_feat = self.rod_ini(x_gray)
        rod_feat = self.rod_conv(rod_feat)
        return rod_feat

class ConeCell(nn.Module):
    def __init__(self, in_ch=3, out_ch=32, kernel_size=7, dilation=2):
        super().__init__()
        self.padding = (kernel_size-1) // 2 * dilation
        # skernel = kernel_size + 2
        self.cone_ini = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=self.padding, dilation=dilation)
        self.cone_conv = Inhibition(out_ch, out_ch, ckernel=1, skernel=(kernel_size -1)//2)

    def forward(self, x):
        cone_feat = self.cone_ini(x)
        return cone_feat

class BipolarCell(nn.Module):
    def __init__(self, rod_in_ch, cone_in_ch, out_ch):
        super().__init__()
        self.rod_inh_on = Inhibition(rod_in_ch, out_ch, ckernel=1, skernel=5, sdilation=2, mode='on')
        self.cone_inh_on = Inhibition(cone_in_ch, out_ch, ckernel=1, skernel=3, sdilation=1, mode='on')
        self.cone_inh_off = Inhibition(cone_in_ch, out_ch, ckernel=1, skernel=3, sdilation=1, mode='off')

    def forward(self, rod_x, cone_x):
        rod_feat_on = self.rod_inh_on(rod_x)
        cone_feat_on = self.cone_inh_on(cone_x)
        cone_feat_off = self.cone_inh_off(cone_x)
        return rod_feat_on, cone_feat_on, cone_feat_off

class GanglionCell(nn.Module):
    def __init__(self, rod_on_ch, cone_on_ch, cone_off_ch, out_ch, cell_type='M'):
        super().__init__()
        self.cell_type = cell_type
        self.out_ch = out_ch
        in_ch_on = cone_on_ch
        in_ch_off = cone_off_ch
        skernel = 3
        ckernel = 1
        sdilation = 1
        # ON/OFF Inhibition
        if cell_type == 'M':
            in_ch_on = rod_on_ch + cone_on_ch
            skernel = 5
            sdilation = 2


        # Inhibition 模块
        self.on_inh = Inhibition(in_ch_on, out_ch, ckernel=ckernel, skernel=skernel, sdilation=sdilation, mode='on')
        self.off_inh = Inhibition(in_ch_off, out_ch, ckernel=ckernel, skernel=skernel, sdilation=sdilation, mode='off')
        # 可学习加权参数 α, β
        self.alpha = nn.Parameter(torch.ones(1, out_ch, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, out_ch, 1, 1))

    def forward(self, rod_on, cone_on, cone_off):
        on_input = cone_on
        off_input = cone_off
        if self.cell_type == 'M':
            on_input = torch.cat([rod_on, cone_on], dim=1)
            # ON/OFF Inhibition
        out_on = self.on_inh(on_input)
        out_off = self.off_inh(off_input)
        # 可学习加权融合
        out_feat = self.alpha * out_on + self.beta * out_off
        return out_feat

class RetinaLayer(nn.Module):
    def __init__(self, in_ch=3, out_ch=32):
        super().__init__()
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.rod = RodCell(in_ch, out_ch)
        self.cone = ConeCell(in_ch, out_ch)
        self.bipolar = BipolarCell(out_ch ,out_ch, out_ch)
        self.ganglion_M = GanglionCell(out_ch, out_ch, out_ch, out_ch,  cell_type='M')
        self.ganglion_P = GanglionCell(out_ch, out_ch, out_ch, out_ch,  cell_type='P')
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        rod_feat = self.rod(x)
        cone_feat = self.cone(x)
        rod_on, cone_on, cone_off = self.bipolar(rod_feat, cone_feat)
        # Ganglion 输出单一路径特征（ON/OFF 融合）
        M_feat = self.ganglion_M(rod_on, cone_on, cone_off)
        P_feat = self.ganglion_P(rod_on, cone_on, cone_off)
        # 残差连接 + 激活
        M_out = M_feat + shortcut
        P_out = P_feat + shortcut

        return M_out, P_out

class LGNLayer(nn.Module):
    def __init__(self, in_ch,  out_ch, type ='P'):
        super(LGNLayer, self).__init__()
        self.skernel_size = 5
        self.sdilation = 2
        self.ckernel_size = 1
        if type == 'P':
            self.skernel_size = 3
            self.sdilation = 1

        self.inh_on = nn.Sequential(
            Inhibition(in_ch, out_ch, ckernel=self.ckernel_size, skernel=self.skernel_size,
                                  sdilation=self.sdilation, mode='on'),
            Inhibition(out_ch, out_ch, ckernel=self.ckernel_size, skernel=self.skernel_size,
                       sdilation=self.sdilation, mode='on'),
            Inhibition(out_ch, out_ch, ckernel=self.ckernel_size, skernel=self.skernel_size,
                       sdilation=self.sdilation, mode='on')
        )
        self.inh_off = nn.Sequential(
            Inhibition(in_ch, out_ch, ckernel=self.ckernel_size, skernel=self.skernel_size,
                                  sdilation=self.sdilation, mode='off'),
            Inhibition(out_ch, out_ch, ckernel=self.ckernel_size, skernel=self.skernel_size,
                       sdilation=self.sdilation, mode='off'),
            Inhibition(out_ch, out_ch, ckernel=self.ckernel_size, skernel=self.skernel_size,
                       sdilation=self.sdilation, mode='off')
        )
        # 可学习加权参数 α, β
        self.alpha = nn.Parameter(torch.ones(1, out_ch, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, out_ch, 1, 1))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.pool(x)
        shortcut = self.shortcut(x)
        # Inhibition stack
        out = self.alpha * self.inh_on(x) + self.beta * self.inh_off(x)

        # 残差连接
        out = out + shortcut
        return out

class V1SimpleCell(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, n_directions=8, use_surround=True):
        super().__init__()
        self.out_ch = out_ch//2
        dilation = 2
        self.direction_conv = DirectionalConv(in_ch, self.out_ch, n_directions, kernel_size)
        self.use_surround = use_surround
        if use_surround:
            self.surround = DepthwiseSeparableConv(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2*dilation, dilation=dilation)
            self.gamma = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        outs0, outs90 = self.direction_conv(x)  # [B, out_ch, H, W]
        outs = torch.cat([outs0, outs90], dim=1) # [B, 2*d*out_ch_per_dir, H, W]

        if self.use_surround:
            outs = outs - self.gamma * self.surround(x)

        return outs

# -------------------------------
# V1 Complex Cell
# -------------------------------
class V1ComplexCell(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch // 2
        skernel = 5
        dilation = 2
        self.crf_ncrf = nn.Sequential(
            Inhibition(self.in_ch, in_ch, ckernel=1, skernel=skernel, sdilation=dilation),
            Inhibition(in_ch, out_ch, ckernel=1, skernel=skernel, sdilation=dilation)
            )
        self.gamma = nn.Parameter(torch.tensor(0.2))
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, simple_out):
        # simple_out shape: [B, 2*C, H, W] -> 2 相位
        outs0, outs90 = simple_out.chunk(2, dim=1)
        energy = torch.sqrt(outs0**2 + outs90**2 + 1e-6)
        out = self.crf_ncrf(energy)
        return out


class V1Layer(nn.Module):
    def __init__(self, lgn_m_ch, lgn_p_ch, out_ch, n_directions=8):
        super().__init__()
        # M/P 通道独立处理
        self.simple_M = V1SimpleCell(lgn_m_ch, 2 * lgn_m_ch, n_directions=n_directions)
        self.complex_M = V1ComplexCell(2 * lgn_m_ch, lgn_m_ch)

        self.simple_P = V1SimpleCell(lgn_p_ch, 2 * lgn_p_ch, n_directions=n_directions)
        self.complex_P = V1ComplexCell(2 * lgn_p_ch, lgn_p_ch)

        # 融合 M/P 输出送 V2
        # self.deep_conv = nn.Sequential(
        #     nn.Conv2d(lgn_m_ch, out_ch, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.1, inplace=True)
        # )
        self.deep_conv = nn.Conv2d(lgn_m_ch + lgn_p_ch, out_ch, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(lgn_m_ch, lgn_m_ch, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, lgn_m, lgn_p):
        lgn_m = self.pool(lgn_m)
        lgn_p = self.pool(lgn_p)
        shortcut_m = self.shortcut(lgn_m)
        shortcut_p = self.shortcut(lgn_p)
        m_complex = self.complex_M(self.simple_M(lgn_m))  + shortcut_m
        p_complex = self.complex_P(self.simple_P(lgn_p))  + shortcut_p
        fused = torch.cat([m_complex, p_complex], dim=1)
        # deep_conv 输出给 V2
        v1_out = self.deep_conv(fused)
        return v1_out

class V2Layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 抑制分支（CRF 抑制作用）
        self.inhibit_branch = nn.Sequential(
            Inhibition(in_ch, out_ch, ckernel=1, skernel=3, cdilation=1, sdilation=1, mode='on'),
            Inhibition(out_ch, out_ch, ckernel=1, skernel=3, cdilation=1, sdilation=2, mode='on'),
            Inhibition(out_ch, out_ch, ckernel=1, skernel=5, cdilation=1, sdilation=2, mode='on')
        )
        # 增强分支（nCRF 增强作用）
        self.enhance_branch = nn.Sequential(
            Inhibition(in_ch, out_ch, ckernel=1, skernel=3, cdilation=1, sdilation=1, mode='add'),
            Inhibition(out_ch, out_ch, ckernel=1, skernel=3, cdilation=1, sdilation=2, mode='add'),
            Inhibition(out_ch, out_ch, ckernel=1, skernel=5, cdilation=1, sdilation=2, mode='add')
        )
        # 可学习融合权重
        self.alpha = nn.Parameter(torch.ones(1, out_ch, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, out_ch, 1, 1))
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.pool(x)
        shortcut = self.shortcut(x)
        x_inhibit = self.inhibit_branch(x)
        x_enhance = self.enhance_branch(x)

        fused = self.alpha * x_inhibit + self.beta * x_enhance
        return self.relu(fused + shortcut)


class MPFusion(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 1. 1x1 Conv 降维 / 融合通道
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.norm1 = nn.InstanceNorm2d(out_ch)
        self.relu = nn.LeakyReLU(0.1,inplace=True)

        # 2. 轻量卷积分支 (1x3 + 3x1) 捕获局部条纹特征
        self.conv_h = nn.Conv2d(out_ch, out_ch, kernel_size=(1, 3), padding=(0, 1))
        self.conv_v = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 1), padding=(1, 0))
        self.norm2 = nn.InstanceNorm2d(out_ch)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.conv_h(x) + self.conv_v(x)
        x = self.norm2(x)
        return x

class UpFuse(nn.Module):
    def __init__(self, in_c, out_c, dp=0, up=False):
        super(UpFuse, self).__init__()
        self.pre_conv1 = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=3, padding=1),
            # MultiScaleConv(in_c[0], out_c),
            nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.pre_conv2 = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=3, padding=1),
            # MultiScaleConv(in_c[1], out_c),
            nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.up = up
        self.deconv_weight = nn.Parameter(bilinear_upsample_weights(2, out_c), requires_grad=False)
        self.merge_conv = nn.Conv2d(out_c, out_c, kernel_size=1)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        # 可学习融合权重 α, β
        self.alpha = nn.Parameter(torch.ones(1, out_c, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, out_c, 1, 1))

    def forward(self, *x):
        x1 = self.pre_conv1(x[0])
        x2 = self.pre_conv2(x[1])
        if self.up:
            # x1 = F.interpolate(x1, size=(x2.size(2), x2.size(3)), mode='bilinear', align_corners=False)
            # x1 = self.up_conv(x1)
            x1 = F.conv_transpose2d(x1, self.deconv_weight, padding=1, stride=2,
                                    output_padding=(x2.size(2) - x1.size(2) * 2, x2.size(3) - x1.size(3) * 2))
        out = self.merge_conv(self.alpha * x1 + self.beta * x2)
        # out = x1 + x2
        return out


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = 2 * factor - factor % 2
    weights = np.zeros((number_of_classes,
                        number_of_classes,
                        filter_size,
                        filter_size,), dtype=np.float32)
    upsample_kernel = upsample_filt(filter_size)

    for i in range(number_of_classes):
        weights[i, i, :, :] = upsample_kernel
    return torch.Tensor(weights)

class BioVisual(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, feature_scale=2, dropout=0):
        super().__init__()
        filters = [ 64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Retina stage
        self.retina = RetinaLayer(num_channels, filters[0])
        # self.retina_fused = nn.Conv2d(filters[0] * 2, filters[0], 1)
        self.retina_fused = MPFusion(filters[0] * 2, filters[0])
        # LGN stage

        self.lgn_m = LGNLayer(filters[0], filters[1], type = 'M')
        self.lgn_p = LGNLayer(filters[0], filters[1], type = 'P')
        # self.lgn_fused = nn.Conv2d(filters[1] * 2, filters[1], 1)
        self.lgn_fused = MPFusion(filters[1] * 2, filters[1])
        # V1 stage
        self.v1 = V1Layer(filters[1], filters[1], filters[2])
        # V2 stage
        self.v2 = V2Layer(filters[2], filters[2])

        # Decoder: 上采样 + skip connection融合
        self.fusionconv1 = UpFuse(in_c=(filters[2], filters[2]), out_c=filters[1], dp=dropout, up=True)

        self.fusionconv2 = UpFuse(in_c=(filters[1], filters[1]), out_c=filters[0], dp=dropout, up=True)

        self.fusionconv3 = UpFuse(in_c=(filters[0], filters[0]), out_c=filters[0], dp=dropout, up=True)
        # 输出头
        self.out_head = nn.Conv2d(filters[0], num_classes, 1)
        # self.apply(InitWeights_He)

    def forward(self, x):
        retina_m,retina_p = self.retina(x)
        retina_skip = self.retina_fused(torch.cat((retina_m,retina_p), 1))

        lgn_m = self.lgn_m(retina_m)
        lgn_p = self.lgn_p(retina_p)
        lgn_skip = self.lgn_fused(torch.cat([lgn_m,lgn_p], dim=1))

        v1_skip = self.v1(lgn_m,lgn_p)  # base V1 enhanced

        v2_skip = self.v2(v1_skip)

        # 上采样 + skip融合
        u1 = self.fusionconv1(v2_skip, v1_skip)
        # print("v2:",v2_skip.shape, "v1:",v1_skip.shape,"u1:", u1.shape, 'lgn_skip:', lgn_skip.shape)
        u2 = self.fusionconv2(u1, lgn_skip)
        u3 = self.fusionconv3(u2, retina_skip)
        out = self.out_head(u3)
        return torch.sigmoid(out)
