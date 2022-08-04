import os, time
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Coder():
    """在model.encode和decode的基础上封装了时间测试模块等"""
    def __init__(self, model, filename):
        self.model = model 
        self.filename = filename

    @torch.no_grad()
    def encode(self, x, postfix=''):
        start_time = time.time()
        # 因为ImageCoder部分和Model的结构相关，应该将ImageCoder负责的功能转移到model内部
        self.model.encode(x,self.filename,postfix)
        print('Encode Time:\t', round(time.time() - start_time, 3), 's')

    @torch.no_grad()
    def decode(self, postfix=''):
        start_time = time.time()
        out = self.model.decode(self.filename,postfix=postfix)
        print('Decode Time:\t', round(time.time() - start_time, 3), 's')
        return out