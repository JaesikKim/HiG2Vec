#!/usr/bin/env python3
# The original source code of Poincare Embedding can be found in  https://github.com/facebookresearch/poincare-embeddings
# This source code is partially modified for the application to HiG2Vec.

import os
from os.path import join as pjoin
import time
import torch


class LocalCheckpoint(object):
    def __init__(self, path, include_in_all=None, start_fresh=False):
        self.path = path
        self.start_fresh = start_fresh
        self.include_in_all = {} if include_in_all is None else include_in_all

    def initialize(self, params):
        if not self.start_fresh and os.path.isfile(self.path):
            print(f'Loading checkpoint from {self.path}')
            return torch.load(self.path)
        else:
            return params

    def save(self, params, tries=10):
        try:
            torch.save({**self.include_in_all, **params}, self.path)
        except Exception as err:
            if tries > 0:
                print(f'Exception while saving ({err})\nRetrying ({tries})')
                time.sleep(60)
                self.save(params, tries=(tries - 1))
            else:
                print("Giving up on saving...")
