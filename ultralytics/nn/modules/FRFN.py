import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class ECA(nn.Module):
    """Efficient Channel Attention (ECA) module with improved numerical stability."""
    def __init__(self, channels, k_size=3, gamma=2, b=1):
        super().__init__()
        # 自适应卷积核大小计算
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        k_size = max(k, k_size)  # 使用计算得到的或指定的核大小
        
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # 改进的权重初始化
        nn.init.xavier_uniform_(self.conv.weight)
        
    def forward(self, x):
        # 数值稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            
        # 全局平均池化
        y = x.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        # 1D卷积操作
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # Sigmoid激活
        y = self.sigmoid(y)
        
        # 通道注意力加权
        return x * y

class FRFN(nn.Module):
    """Fast Receptive Field Network with enhanced numerical stability and performance."""
    def __init__(self, dim, hidden_dim=None, act_layer=nn.GELU, drop=0., use_eca=True):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # 动态分组策略 - 根据通道数自适应调整
        groups = max(1, min(dim // 8, 32))  # 限制最大组数为32
        
        # 部分卷积 - 使用更好的分组策略
        self.partial_conv = nn.Conv2d(
            dim, dim, 
            kernel_size=3, 
            padding=1, 
            groups=groups, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim)
        
        # 门控投影 - 分解为两个较小的卷积以减少参数
        self.gate_proj1 = nn.Conv2d(dim, hidden_dim // 2, kernel_size=1, bias=False)
        self.gate_proj2 = nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        
        # 深度可分离卷积 - 添加批归一化
        dw_groups = hidden_dim // 2
        self.dwconv = nn.Conv2d(
            hidden_dim // 2, 
            hidden_dim // 2, 
            kernel_size=3, 
            padding=1, 
            groups=dw_groups,
            bias=False
        )
        self.bn_dw = nn.BatchNorm2d(hidden_dim // 2)
        
        # 输出投影
        self.output_proj = nn.Conv2d(hidden_dim // 2, dim, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm2d(dim)
        
        # Dropout和激活
        self.dropout = nn.Dropout(drop)
        self.act = act_layer()
        
        # ECA注意力机制
        self.use_eca = use_eca
        if use_eca:
            self.eca = ECA(dim)
            
        # 可学习的残差权重
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.5)
        
        # 梯度裁剪值
        self.grad_clip_value = 1.0
        
        self._init_weights()
        
    def _init_weights(self):
        """改进的权重初始化策略"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用He初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 数值稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 残差连接
        residual = x
        
        # 部分卷积路径
        conv_path = self.partial_conv(x)
        conv_path = self.bn1(conv_path)
        conv_path = self.act(conv_path)
        
        # 门控路径 - 分解投影
        gate = self.gate_proj1(x)
        gate = self.bn2(gate)
        gate = self.act(gate)
        gate = self.gate_proj2(gate)
        gate = self.bn3(gate)
        
        # 分割门控特征
        gate1, gate2 = torch.chunk(gate, 2, dim=1)
        
        # 深度卷积和激活
        gate1 = self.dwconv(gate1)
        gate1 = self.bn_dw(gate1)
        gate1 = self.act(gate1)
        
        # 门控机制
        gate2_sigmoid = torch.sigmoid(gate2)
        gate_out = gate1 * gate2_sigmoid
        
        # 输出投影
        gate_out = self.output_proj(gate_out)
        gate_out = self.bn_out(gate_out)
        gate_out = self.dropout(gate_out)
        
        # 残差连接与自适应缩放
        scale = torch.clamp(self.residual_scale, min=0.1, max=2.0)
        out = conv_path + scale * gate_out
        
        # ECA注意力
        if self.use_eca:
            out = self.eca(out)
            
        # 最终残差连接
        out = residual + out
        
        # 梯度裁剪
        if self.training and out.requires_grad:
            def grad_clip_hook(grad):
                return torch.clamp(grad, -self.grad_clip_value, self.grad_clip_value)
            out.register_hook(grad_clip_hook)
        
        return out


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Enhanced standard convolution with improved initialization and stability."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with improved activation and batch normalization."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-5, momentum=0.1)  # 调整eps和momentum
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        
        # 改进的权重初始化
        self._init_weights()
        
    def _init_weights(self):
        """改进的权重初始化"""
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)

    def forward_fuse(self, x):
        """Perform fused convolution and activation (for inference optimization)."""
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """Enhanced standard bottleneck with improved residual connections."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes an enhanced bottleneck module with adaptive scaling."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        
        # 可学习的残差权重
        if self.add:
            self.residual_scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        """Enhanced forward pass with adaptive residual scaling."""
        if self.add:
            # 自适应残差缩放
            scale = torch.clamp(self.residual_scale, min=0.1, max=2.0)
            return x + scale * self.cv2(self.cv1(x))
        else:
            return self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Enhanced CSP Bottleneck with 2 convolutions and improved feature flow."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes an enhanced CSP bottleneck with improved gradient flow."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        
        # 特征融合权重
        self.fusion_weights = nn.Parameter(torch.ones(n + 2) / (n + 2))
        
        # 梯度累积监控
        self.register_buffer('grad_norm_history', torch.zeros(5))
        self.register_buffer('step_count', torch.zeros(1))

    def forward(self, x):
        """Enhanced forward pass with weighted feature fusion."""
        # 数值稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            
        y = list(self.cv1(x).chunk(2, 1))
        
        # 应用bottleneck模块
        for m in self.m:
            y.append(m(y[-1]))
        
        # 加权特征融合
        weights = F.softmax(self.fusion_weights, dim=0)
        weighted_features = [w * feat for w, feat in zip(weights, y)]
        
        output = self.cv2(torch.cat(weighted_features, 1))
        
        # 梯度监控
        if self.training and output.requires_grad:
            def grad_monitor_hook(grad):
                grad_norm = grad.norm().item()
                self.step_count += 1
                idx = int(self.step_count % 5)
                self.grad_norm_history[idx] = grad_norm
                return grad
            output.register_hook(grad_monitor_hook)
        
        return output

    def forward_split(self, x):
        """Enhanced forward pass using split() with feature weighting."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        
        for m in self.m:
            y.append(m(y[-1]))
            
        # 加权融合
        weights = F.softmax(self.fusion_weights, dim=0)
        weighted_features = [w * feat for w, feat in zip(weights, y)]
        
        return self.cv2(torch.cat(weighted_features, 1))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class Bottleneck_FRFN(nn.Module):
    """Enhanced bottleneck with FRFN module and improved stability."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes an enhanced FRFN bottleneck with adaptive features."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = FRFN(c_)
        self.add = shortcut and c1 == c2
        
        # 可学习的残差权重和梯度缩放
        if self.add:
            self.residual_scale = nn.Parameter(torch.ones(1) * 0.5)
            self.feature_scale = nn.Parameter(torch.ones(1) * 1.0)
        
        # 特征平衡机制
        self.feature_balance = nn.Parameter(torch.ones(c2) * 0.5)

    def forward(self, x):
        """Enhanced forward pass with multi-scale residual connections."""
        features = self.cv1(x)
        enhanced_features = self.cv2(features)
        
        if self.add:
            # 自适应残差和特征缩放
            res_scale = torch.clamp(self.residual_scale, min=0.1, max=2.0)
            feat_scale = torch.clamp(self.feature_scale, min=0.5, max=1.5)
            
            # 特征平衡
            balance = torch.sigmoid(self.feature_balance).view(1, -1, 1, 1)
            
            # 加权融合
            output = res_scale * x + feat_scale * enhanced_features
            output = output * balance
            
            return output
        else:
            return enhanced_features

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_FRFN(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_FRFN(C2f):
    """Enhanced CSP Bottleneck with adaptive FRFN integration and improved stability."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes enhanced C3k2 with adaptive module selection and stability features."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(self.c)
        self.c3k = c3k
        
        # 动态模块选择
        self.m = nn.ModuleList()
        for i in range(n):
            if c3k:
                # 使用C3k模块进行深度特征提取
                module = C3k(self.c, self.c, 2, shortcut, g)
            else:
                # 使用标准Bottleneck
                module = Bottleneck(self.c, self.c, shortcut, g)
            self.m.append(module)
        
        # 自适应特征选择权重
        self.feature_selector = nn.Parameter(torch.ones(n) * (1.0 / n))
        
        # 特征增强模块
        self.feature_enhancer = nn.Sequential(
            nn.Conv2d(self.c, self.c, 1, bias=False),
            nn.BatchNorm2d(self.c),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.c, self.c, 3, padding=1, groups=self.c, bias=False),
            nn.BatchNorm2d(self.c),
            nn.SiLU(inplace=True)
        )
        
        # 性能监控
        self.register_buffer('performance_metric', torch.zeros(1))
        
        self._init_enhanced_weights()
    
    def _init_enhanced_weights(self):
        """增强的权重初始化"""
        for m in self.feature_enhancer:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Enhanced forward pass with adaptive feature selection and enhancement."""
        # 数值稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 初始特征提取
        y = list(self.cv1(x).chunk(2, 1))
        
        # 自适应特征选择权重
        selector_weights = F.softmax(self.feature_selector, dim=0)
        
        # 应用模块并加权
        enhanced_features = []
        for i, m in enumerate(self.m):
            feature = m(y[-1])
            # 特征增强
            enhanced = self.feature_enhancer(feature)
            # 加权融合
            weight = selector_weights[i]
            final_feature = weight * feature + (1 - weight) * enhanced
            enhanced_features.append(final_feature)
            y.append(final_feature)
        
        # 最终特征融合
        output = self.cv2(torch.cat(y, 1))
        
        # 性能度量更新
        if self.training:
            with torch.no_grad():
                # 计算特征多样性作为性能指标
                feature_std = torch.stack([f.std() for f in enhanced_features]).mean()
                self.performance_metric = 0.9 * self.performance_metric + 0.1 * feature_std
        
        return output
