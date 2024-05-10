import torch
import torch.nn as nn

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1):
        super(Conv, self).__init__()
        self.c = nn.Conv1d(c1, c2, k, s, int(k // 2))
        self.bn = nn.BatchNorm1d(c2)
        self.act = nn.ReLU()
    def forward(self, x):
        return self.act(self.bn(self.c(x)))

class PatchEmbed(nn.Module):
    def __init__(self, c1=256, c2=192, k=8, s=8):
        super(PatchEmbed, self).__init__()
        self.conv = nn.Conv1d(c1, c2, k, s)
    def forward(self, x):
        return self.conv(x).transpose(1, 2)

class ConvEmbedEnd(nn.Module):
    def __init__(self, cs=(128, 128, 192), ks=(9, 5, 3)):
        super(ConvEmbedEnd, self).__init__()
        cin = 64
        layers = []
        for k, c in zip(ks, cs):
            layers.append(Conv(cin, c, k))
            cin = c
        self.m = nn.Sequential(*layers)
    def forward(self, x):
        y = self.m(x)
        x = torch.cat([x, y], 1)
        return x

class Simplified_transformer(nn.Module):
    def __init__(self, dim,  num_heads, qkv_bias=True, attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        qks = []
        for i in range(num_heads):
            qks.append(nn.Linear(dim, dim))
        self.qks = torch.nn.ModuleList(qks)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        B_, N, C = x.shape
        feats_in = x.chunk(self.num_heads, dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qk in enumerate(self.qks):
            v = feat.reshape(B_, N//self.num_heads, self.num_heads, C // self.num_heads).permute( 0, 2, 1, 3)
            if i > 0:  # add the previous output to the input
                feat = feat + feats_in[i]
                v = feat.reshape(B_, N//self.num_heads, self.num_heads, C // self.num_heads).permute( 0, 2, 1, 3)
            q, k = (self.qk(feat).reshape(B_, N//self.num_heads, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)).unbind(0)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            feat = (attn @ v).transpose(1, 2).reshape(B_, N//self.num_heads, C)
            feats_out.append(feat)
        x = torch.cat(feats_out, 1)
        return x

class basicblock(nn.Module):
    def __init__(self, dim=192, depth=2, num_heads=4,  qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.numheads = num_heads
        # build blocks
        self.blocks = nn.ModuleList([
            Simplified_transformer(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias)
            for i in range(depth)])

    def forward(self, x, ):
        for blk in self.blocks:
            x = blk(x)
        return x

class SLCPTN(nn.Module):
    def __init__(self, num_classes=3,act_layer=nn.GELU):
        super(SLCPTN, self).__init__()
        self.dimen_adjust_linear = nn.Linear(1798, 2048)
        self.dimen_adjust_conv1 = Conv(1, 64, 15, 2)
        self.classfication = nn.Linear(192, num_classes)
        self.conv_embend = ConvEmbedEnd()
        self.attention_basic = basicblock()
        self.patch_emb = PatchEmbed()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(192)
        self.drop = nn.Dropout(0.2)
        self.mlp = Mlp(in_features=192, hidden_features=264, act_layer=act_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):#逐层的去初始化每层的参数
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
       # dimension adjustment
        x = x.unsqueeze(1)
        x = self.dimen_adjust_linear(x)
        x = self.dimen_adjust_conv1(x)
        #FEATURE EXTRACTION
        x = self.conv_embend(x)
        x = self.patch_emb(x)
        #SPTN
        atten = self.attention_basic(x)
        x = atten + self.mlp(x)
        #x = x + self.mlp(atten)#good
       # CLASSIFICATION
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.classfication(x)
        return x