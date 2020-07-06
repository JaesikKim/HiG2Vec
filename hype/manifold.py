#!/usr/bin/env python3
# The original source code of Poincare Embedding can be found in  https://github.com/facebookresearch/poincare-embeddings
# This source code is partially modified for the application to HiG2Vec.

from abc import abstractmethod


class Manifold(object):
    def __init__(self, *args, **kwargs):
        pass

    def init_weights(self, w, scale=1e-4):
        w.data.uniform_(-scale, scale)

    @staticmethod
    def dim(dim):
        return dim

    def normalize(self, u):
        return u

    @abstractmethod
    def distance(self, u, v):
        """
        Distance function
        """
        raise NotImplementedError

    @abstractmethod
    def expm(self, p, d_p, lr=None, out=None):
        """
        Exponential map
        """
        raise NotImplementedError

    @abstractmethod
    def logm(self, x, y):
        """
        Logarithmic map
        """
        raise NotImplementedError

    @abstractmethod
    def ptransp(self, x, y, v, ix=None, out=None):
        """
        Parallel transport
        """
        raise NotImplementedError
