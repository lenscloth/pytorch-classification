import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg

from torch.autograd import Variable


class StochasticLinear(nn.Module):
    def __init__(self, in_feature, out_feature,
                       bias=True, n_branch=1, p=1.0, use_bn=False):
        super(StochasticLinear, self).__init__()

        self.use_bn = use_bn
        self.out_feature = out_feature
        self.in_feature = in_feature
        self.n_branch = n_branch
        self.prob = p
        self.linear = nn.Linear(in_feature, out_feature * n_branch, bias=bias)

        if self.use_bn:
            self.batchnorm = nn.BatchNorm1d(out_feature * n_branch)
            self.batchnorm.weight.data.fill_(1)
            self.batchnorm.bias.data.fill_(0)

        self.rescale_parameter()
        self.branching = False
        self.limit = self.n_branch
        self.use_branch = None

    def __str__(self):
        s = "%s(in_feature=%d, out_feature=%d, n_branch=%d, p=%.2f)" %\
               (self.__class__.__name__, self.in_feature, self.out_feature, self.n_branch, self.prob)

        if self.use_bn:
            s += "+ %s" % self.batchnorm

        return s

    def ensemble(self, mode=True):
        self.branching = mode

    def limit_branch(self, n):
        self.limit = n

    def set_use_branch(self, n):
        self.use_branch = n

    def rescale_parameter(self):
        self.linear.weight.data = self.linear.weight.data / (self.prob * self.n_branch)
        self.linear.bias.data = self.linear.bias.data / (self.prob * self.n_branch)

        if self.use_bn:
            self.batchnorm.weight.data = self.batchnorm.weight.data / math.sqrt(self.prob * self.n_branch)
            self.batchnorm.bias.data = self.batchnorm.bias.data / math.sqrt(self.prob * self.n_branch)

    def gen_mask(self, output):
        batch_size = output.size(0)
        mask = output.data.new(batch_size, self.n_branch, 1)
        torch.rand((batch_size, self.n_branch, 1), out=mask)
        mask = (mask <= self.prob).float()

        return Variable(mask)

    def forward(self, input):
        batch_size = input.size(0)
        output = self.linear(input)
        if self.use_bn:
            output = self.batchnorm(output)
        output = output.view(batch_size, self.n_branch, self.out_feature)

        if self.training:
            mask = self.gen_mask(output)
            return (mask * output).sum(dim=1)
        else:
            if self.branching:
                return [self.n_branch * self.prob * output[:, i].contiguous() for i in range(self.n_branch)]
            elif self.use_branch is not None:
                return (self.n_branch * self.prob) * output[:, self.use_branch].contiguous()
            else:
                return (self.n_branch / self.limit) * self.prob * output[:, :self.limit].sum(dim=1)


class StochasticElementLinear(StochasticLinear):
    def gen_mask(self, output):
        batch_size = output.size(0)
        mask = output.data.new(batch_size, self.n_branch, self.out_feature)
        torch.rand((batch_size, self.n_branch, self.out_feature), out=mask)
        mask = (mask <= self.prob).float()

        return Variable(mask)


class StochasticConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                       stride=1, padding=0, dilation=1, groups=1, bias=True, n_branch=1, p=1.0, use_bn=False):
        super(StochasticConv2d, self).__init__()

        self.use_bn = use_bn
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_branch = n_branch
        self.prob = p
        self.conv2d = nn.Conv2d(in_channels, out_channels * n_branch, kernel_size,
                                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        if self.use_bn:
            self.batchnorm2d = nn.BatchNorm2d(out_channels * n_branch)
            self.batchnorm2d.weight.data.fill_(1)
            self.batchnorm2d.bias.data.fill_(0)

        self.branching = False
        self.rescale_parameter()
        self.limit = n_branch

    def __str__(self):
        s = "%s(in_channels=%d, out_channels=%d, kernel_size=(%d, %d), " \
               "stride=(%d, %d), padding=(%d, %d), n_branch=%d, prob=%.2f)" %\
               (self.__class__.__name__, self.in_channels, self.out_channels, self.conv2d.kernel_size[0], self.conv2d.kernel_size[1],
                self.conv2d.stride[0], self.conv2d.stride[1], self.conv2d.padding[0], self.conv2d.padding[1], self.n_branch, self.prob)

        if self.use_bn:
            s += "+ %s" % self.batchnorm2d
        return s

    def ensemble(self, mode=True):
        self.branching = mode

    def limit_branch(self, n):
        self.limit = n

    def rescale_parameter(self):
        self.conv2d.weight.data = self.conv2d.weight.data / (self.prob * self.n_branch)
        self.conv2d.bias.data = self.conv2d.bias.data / (self.prob * self.n_branch)

        if self.use_bn:
            self.batchnorm2d.weight.data = self.batchnorm2d.weight.data / math.sqrt(self.prob * self.n_branch)
            self.batchnorm2d.bias.data = self.batchnorm2d.bias.data / math.sqrt(self.prob * self.n_branch)

    def gen_mask(self, output):
        batch_size = output.size(0)
        mask = output.data.new(batch_size, self.n_branch, 1, 1, 1)
        torch.rand((batch_size, self.n_branch, 1, 1, 1), out=mask)
        mask = (mask <= self.prob).float()

        return Variable(mask)

    def forward(self, input):
        batch_size = input.size(0)
        output = self.conv2d(input)
        if self.use_bn:
            self.batchnorm2d(output)
        h, w = output.size(2), output.size(3)
        output = output.view(batch_size, self.n_branch, self.out_channels, h, w)

        if self.training:
            mask = self.gen_mask(output)
            return (mask * output).sum(dim=1)
        else:
            if self.branching:
                return [self.n_branch * self.prob * output[:, i].contiguous() for i in range(self.n_branch)]
            else:
                return (self.n_branch / self.limit) * self.prob * output[:, :self.limit].sum(dim=1)


class StochasticChannelConv2D(StochasticConv2d):
    def gen_mask(self, output):
        batch_size = output.size(0)

        mask = output.data.new(batch_size, self.n_branch, self.out_channels, 1, 1)
        torch.rand((batch_size, self.n_branch, self.out_channels, 1, 1), out=mask)
        mask = (mask <= self.prob).float()

        return Variable(mask)


class StochasticElementConv2d(StochasticConv2d):
    def gen_mask(self, output):
        batch_size = output.size(0)
        h, w = output.size(3), output.size(4)

        mask = output.data.new(batch_size, self.n_branch, self.out_channels, h, w)
        torch.rand((batch_size, self.n_branch, self.out_channels, h, w), out=mask)
        mask = (mask <= self.prob).float()

        return Variable(mask)
