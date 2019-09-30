import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()

        self.Encoder = nn.Sequential(
            nn.Conv1d(6, 16, 50, 3),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.Conv1d(16, 32, 30, 2),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Conv1d(32, 32, 10, 2),
            nn.BatchNorm1d(32),
            nn.Tanh()
        )

        self.FC_layer = nn.Sequential(
            nn.Linear(7520, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, vital):   # demo => noise?
        latent = self.Encoder(vital)
        flatten = latent.view(latent.size(0), -1)
        output = self.FC_layer(flatten)
        return output


class ModuleStack(nn.Module):
    def __init__(self, middle, n):
        super(ModuleStack, self).__init__()
        self.middle = middle
        self.n = n
        self.first = nn.Sequential(
            nn.Conv1d(6, middle, 50, 3), nn.BatchNorm1d(middle), nn.Tanh())
        self.last = nn.ModuleList(nn.Sequential(nn.Conv1d(
            middle*i, middle*(i+1), 50, 3), nn.BatchNorm1d(middle), nn.ReLU()) for i in range(1, n))

        self.FC_layer = nn.Sequential(
            nn.Linear(1248, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0)

    def fc_layer(self, x):
        output = self.FC_layer(x)
        return output

    def forward(self, x):
        out = self.first(x)
        out = self.last(out)
        out = out.flatten().view(out.shape[0], -1)
        output = self.fc_layer(out)
        return output


class SpatialNL(nn.Module):
    """Spatial NL block for image classification.
       [https://github.com/facebookresearch/video-nonlocal-net].
       revised to 1d
    """

    def __init__(self, inplanes, planes, use_scale=False):
        self.use_scale = use_scale
        super(SpatialNL, self).__init__()
        self.t = nn.Conv1d(inplanes, planes, kernel_size=1,
                           stride=1, bias=False)
        self.p = nn.Conv1d(inplanes, planes, kernel_size=1,
                           stride=1, bias=False)
        self.g = nn.Conv1d(inplanes, planes, kernel_size=1,
                           stride=1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.z = nn.Conv1d(planes, inplanes, kernel_size=1,
                           stride=1, bias=False)
        self.bn = nn.BatchNorm1d(inplanes)

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)
        b, c, d = t.size()
        
        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)
        g = g.view(b, c, -1).permute(0, 2, 1)
        att = torch.bmm(t, p)
        
        if self.use_scale:
            att = att.div(c**0.5)
        att = self.softmax(att)
        
        x = torch.bmm(att, g)
        x = x.permute(0, 2, 1)
        x = x.contiguous()
        
        x = x.view(b, c, d)
        x = self.z(x)
        x = self.bn(x) + residual

        return x
