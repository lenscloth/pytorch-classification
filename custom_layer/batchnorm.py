import torch
import torch.nn as nn

from torch.nn import Parameter


class _CustomBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_CustomBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

        self.registered_mean = None
        self.registered_var = None
        self.use_registered = False

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def register_stat(self, batch_mean, batch_var):
        self.registered_mean = batch_mean
        self.registered_var = batch_var

    def clear_stat(self):
        self.registered_mean = None
        self.registered_var = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class BatchNorm2d(_CustomBatchNorm):
    def forward(self, input, stat_input=None):
        if self.training:
            if stat_input is None:
                stat_input = input

            # For faster computation do not explicit compute variance
            flattened = stat_input.view(input.size(0), self.num_features, -1)
            batch_mean = flattened.mean(dim=2).mean(dim=0)
            batch_var = (flattened - batch_mean.view(1, self.num_features, 1)).pow(2).mean(dim=2).mean(dim=0)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.detach()
        else:
            batch_mean, batch_var = self.running_mean, self.running_var

        batch_mean, batch_var = batch_mean.view(1, self.num_features, 1, 1), batch_var.view(1, self.num_features, 1, 1)
        weight, bias = self.weight.view(1, self.num_features, 1, 1), self.bias.view(1, self.num_features, 1, 1)

        norm = (input - batch_mean) / (batch_var + self.eps).sqrt()

        if self.affine:
            norm = weight * norm + bias

        return norm


class DoBatchNorm2d(_CustomBatchNorm):
    def forward(self, input):
        if self.training:
            if self.use_registered:
                batch_mean, batch_var = self.registered_mean, self.registered_var
            else:
                # For faster computation do not explicit compute variance
                flattened = input.view(input.size(0), self.num_features, -1)
                batch_mean = flattened.mean(dim=2).mean(dim=0)
                batch_var = (flattened - batch_mean.view(1, self.num_features, 1)).pow(2).mean(dim=2).mean(dim=0)

            self.register_stat(batch_mean.detach(), batch_var.detach())
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.detach()
        else:
            batch_mean, batch_var = self.running_mean, self.running_var

        batch_mean, batch_var = batch_mean.view(1, self.num_features, 1, 1), batch_var.view(1, self.num_features, 1, 1)
        weight, bias = self.weight.view(1, self.num_features, 1, 1), self.bias.view(1, self.num_features, 1, 1)

        norm = (input - batch_mean) / (batch_var + self.eps).sqrt()

        if self.affine:
            norm = weight * norm + bias

        return norm
