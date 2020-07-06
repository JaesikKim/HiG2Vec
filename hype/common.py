#!/usr/bin/env python3
# The original source code of Poincare Embedding can be found in  https://github.com/facebookresearch/poincare-embeddings
# This source code is partially modified for the application to HiG2Vec.

import torch as th
from torch.autograd import Function


class Acosh(Function):
    @staticmethod
    def forward(ctx, x, eps):
        z = th.sqrt(x * x - 1)
        ctx.save_for_backward(z)
        ctx.eps = eps
        return th.log(x + z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z = th.clamp(z, min=ctx.eps)
        z = g / z
        return z, None


acosh = Acosh.apply
