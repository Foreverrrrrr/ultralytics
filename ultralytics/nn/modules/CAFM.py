import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class SimplifiedAttention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 改进的QKV投影，使用更好的初始化
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        # 可学习的温度参数，初始化更稳定
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.1)
        
        # 改进的权重初始化
        self._init_weights()
        
    def _init_weights(self):
        # 使用Kaiming初始化，更适合ReLU类激活函数
        nn.init.kaiming_normal_(self.qkv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
        
        # 如果有bias，初始化为0
        if hasattr(self.qkv, 'bias') and self.qkv.bias is not None:
            nn.init.constant_(self.qkv.bias, 0)
        if hasattr(self.proj, 'bias') and self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 添加数值稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        qkv = self.qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.num_heads), qkv)
        
        # 归一化Q和K以提高稳定性
        q = F.normalize(q, dim=-1) * math.sqrt(self.head_dim)
        k = F.normalize(k, dim=-1) * math.sqrt(self.head_dim)
        
        # 计算注意力权重，添加温度缩放和梯度裁剪
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 温度缩放，限制范围防止梯度爆炸
        temperature = torch.clamp(self.temperature, min=0.01, max=2.0)
        attn = attn * temperature
        
        # 添加数值稳定性
        attn = torch.clamp(attn, min=-50, max=50)  # 防止softmax溢出
        attn = F.softmax(attn, dim=-1)
        
        # 添加dropout以提高泛化能力
        if self.training:
            attn = F.dropout(attn, p=0.1)
        
        out = (attn @ v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=H, y=W)
        
        return self.proj(out)

class EnhancedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 使用分组卷积减少参数量，提高效率
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.bn_conv = nn.BatchNorm2d(dim)
        
        # 改进的门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        
        # 梯度缩放因子
        self.grad_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        self._init_weights()
        
    def _init_weights(self):
        # 改进的权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, attn_out, conv_out):
        # 应用卷积和批归一化
        conv_out = self.bn_conv(self.conv(conv_out))
        
        # 梯度裁剪
        if self.training:
            attn_out = torch.clamp(attn_out, min=-10, max=10)
            conv_out = torch.clamp(conv_out, min=-10, max=10)
        
        # 计算门控权重
        gate_input = torch.cat([attn_out, conv_out], dim=1)
        gate = self.gate(gate_input)
        
        # 使用梯度缩放因子进行融合
        scale = torch.clamp(self.grad_scale, min=0.01, max=1.0)
        fused = scale * (gate * attn_out + (1 - gate) * conv_out)
        
        return fused

class PSABlock_CAFM(nn.Module):
    def __init__(self, c, num_heads=4, expansion_ratio=4, dropout=0.1):
        super().__init__()
        # 使用LayerNorm替代BatchNorm以提高稳定性
        self.norm1 = nn.GroupNorm(min(32, c), c)  # 使用GroupNorm
        self.norm2 = nn.GroupNorm(min(32, c), c)
        
        self.attn = SimplifiedAttention(c, num_heads=num_heads)
        self.fusion = EnhancedFusion(c)
        
        # 改进的FFN设计
        hidden_dim = int(c * expansion_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(c, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, c, 1, bias=False),
            nn.BatchNorm2d(c)
        )
        
        # 可学习的残差连接权重
        self.residual_scale1 = nn.Parameter(torch.ones(1) * 0.5)
        self.residual_scale2 = nn.Parameter(torch.ones(1) * 0.5)
        
        # 梯度累积缓冲区
        self.register_buffer('grad_norm_history', torch.zeros(10))
        self.register_buffer('step_count', torch.zeros(1))
        
        self._init_weights()
        
    def _init_weights(self):
        # 使用更好的初始化策略
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用He初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 第一个残差连接
        residual = x
        x_norm = self.norm1(x)
        
        # 注意力分支
        attn_out = self.attn(x_norm)
        conv_out = self.fusion.conv(x_norm)
        fused = self.fusion(attn_out, conv_out)
        
        # 缩放残差连接
        scale1 = torch.clamp(self.residual_scale1, min=0.1, max=2.0)
        x = residual + scale1 * fused
        
        # 第二个残差连接
        residual2 = x
        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2)
        
        # 梯度监控和自适应缩放
        if self.training:
            # 计算梯度范数
            if ffn_out.requires_grad:
                def grad_hook(grad):
                    grad_norm = grad.norm().item()
                    # 更新梯度历史
                    self.step_count += 1
                    idx = int(self.step_count % 10)
                    self.grad_norm_history[idx] = grad_norm
                    return grad
                ffn_out.register_hook(grad_hook)
        
        # 自适应残差缩放
        scale2 = torch.clamp(self.residual_scale2, min=0.1, max=2.0)
        x = residual2 + scale2 * ffn_out
        
        return x

class C2PSA_CAFM(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, num_heads=4):
        super().__init__()
        self.c = int(c1 * e)
        
        # 输入投影
        self.conv = nn.Conv2d(c1, 2 * self.c, 1, bias=False)
        self.bn = nn.BatchNorm2d(2 * self.c)
        self.act = nn.SiLU()
        
        # PSA块序列
        self.m = nn.Sequential(
            *(PSABlock_CAFM(self.c, num_heads=num_heads) for _ in range(n))
        )
        
        # 改进的SE模块
        se_channels = max(self.c // 4, 8)  # 增加最小通道数
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.c, se_channels, 1, bias=True),  # 使用bias避免BN问题
            nn.ReLU(inplace=True),
            nn.Conv2d(se_channels, self.c, 1, bias=True),  # 使用bias避免BN问题
            nn.Sigmoid()
        )
        
        # 输出投影
        self.conv_out = nn.Conv2d(2 * self.c, c2, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(c2)
        
        # 可学习的特征融合权重
        self.fusion_weight = nn.Parameter(torch.ones(1) * 0.5)
        
        # 梯度裁剪阈值
        self.grad_clip_value = 1.0
        
        self._init_weights()
        
    def _init_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # 输入检查和数值稳定化
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 前向传播
        x = self.act(self.bn(self.conv(x)))
        x1, x2 = x.chunk(2, dim=1)
        
        # 应用PSA块
        x2_processed = self.m(x2)
        
        # SE注意力
        se_weight = self.se(x2_processed)
        x2_processed = x2_processed * se_weight
        
        # 特征融合，使用可学习权重
        fusion_w = torch.clamp(self.fusion_weight, min=0.1, max=0.9)
        x2_final = fusion_w * x2 + (1 - fusion_w) * x2_processed
        
        # 拼接和输出
        x = torch.cat([x1, x2_final], dim=1)
        x = self.bn_out(self.conv_out(x))
        
        # 梯度裁剪
        if self.training and x.requires_grad:
            def grad_clip_hook(grad):
                return torch.clamp(grad, -self.grad_clip_value, self.grad_clip_value)
            x.register_hook(grad_clip_hook)
        
        return x