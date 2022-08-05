#熵模型
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torchac
from range_coder import prob_to_cum_freq, cum_freq_to_prob
import numpy as np
import scipy.stats
import time
import torch.nn.functional as F

import os

torch.use_deterministic_algorithms(True)
torch.set_deterministic(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#覆写自动求导的类实现反传梯度为1
class RoundNoGradient(torch.autograd.Function):
    """
    Another implementaion in compressai:
        return torch.round(x) - x.detach() + x """
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g

#覆写自动求导类实现对x的下界约束
class Low_bound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,lower_bound=1e-9):
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=lower_bound)
        return x

    @staticmethod
    def backward(ctx, g,lower_bound=1e-9):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        try:
            grad1[x<lower_bound] = 0
        except RuntimeError:
            print("ERROR! grad1[x<%.3e] = 0"%lower_bound)
            grad1 = g.clone()
        pass_through_if = np.logical_or(x.cpu().detach().numpy() >= lower_bound, g.cpu().detach().numpy()<0.0)
        t = torch.Tensor(pass_through_if+0.0).to(grad1.device)

        return grad1*t,None

class EntropyBase(nn.Module):
    """实现熵编码共有的方法, 避免造轮子"""
    def __init__(self):
        super().__init__()
    
    def quantize(self,inputs,mode,means=None):
        if mode not in ("noise","quantize","symbols"):
            raise ValueError(f"Invalid quantization mode: ", mode)
        # noise
        if mode == "noise":
            noise = torch.empty_like(inputs).uniform_(-0.5,0.5)
            return inputs + noise
        # quantize: return  round(x - means) + means
        outputs = inputs.clone()  # Deep copy
        # 量化这一步如果有均值就把均值减掉，当然后面估计的就是减掉均值的outputs，反量化的时候再加回来就好了
        if means is not None:
            outputs -= means
        outputs = torch.round(outputs)
        if mode == "quantize":
            if means is not None:
                outputs += means
            return outputs
        # symbol: return int(x - means)
        outputs = outputs.int()
        return outputs

    def dequantize(self,inputs,means,dtype=torch.float):
        if means is not None:
            outputs = inputs.type_as(means)
            outputs += means
        else:
            outputs = inputs.type(dtype)
        return outputs
    

class EntropyBottleneck(EntropyBase):
    """The layer implements a flexible probability density model to estimate
    entropy of its input tensor, which is described in this paper:
    >"Variational image compression with a scale hyperprior"
    > J. Balle, D. Minnen, S. Singh, S. J. Hwang, N. Johnston
    > https://arxiv.org/abs/1802.01436"""
    
    def __init__(self, channels, init_scale=8, filters=(3,3,3,3)):
        """create parameters.
        """
        super(EntropyBottleneck, self).__init__()
        self._likelihood_bound = 1e-9
        self._init_scale = float(init_scale)
        self._filters = tuple(int(f) for f in filters)
        self._channels = channels
        self.ASSERT = False
        # build.
        filters = (1,) + self._filters + (1,)
        scale = self._init_scale ** (1 / (len(self._filters) + 1))
        # 使用parameterlist在多卡训练时会报错：
        # UserWarning: nn.ParameterList is being used with DataParallel but this is not supported. 
        # This list will appear empty for the models replicated on each GPU except the original one.
        # # Create variables.
        # self._matrices = nn.ParameterList([])
        # self._biases = nn.ParameterList([])
        # self._factors = nn.ParameterList([])

        for i in range(len(self._filters) + 1):
            #
            matrix = torch.Tensor(channels, filters[i+1], filters[i])
            init_matrix = np.log(np.expm1(1.0 / scale / filters[i + 1]))
            matrix.data.fill_(init_matrix)
            self.register_parameter(f"_matrix{i:d}", Parameter(matrix))
            #
            bias = torch.Tensor(channels, filters[i+1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.register_parameter(f"_bias{i:d}", Parameter(bias))
            # 
            if i < len(self._filters):    
                factor = torch.Tensor(channels, filters[i+1], 1)
                factor.data.fill_(0.0)
                self.register_parameter(f"_factor{i:d}", nn.Parameter(factor))

    def _logits_cumulative(self, inputs):
        """Evaluate logits of the cumulative densities.
        
        Arguments:
        inputs: The values at which to evaluate the cumulative densities,
            expected to have shape `(channels, 1, batch*H*W)`.

        Returns:
        A tensor of the same shape as inputs, containing the logits of the
        cumulatice densities evaluated at the the given inputs.
        """
        logits = inputs
        for i in range(len(self._filters) + 1):
            _matrix = getattr(self, f"_matrix{i:d}")
            _bias = getattr(self, f"_bias{i:d}")
            matrix = torch.nn.functional.softplus(_matrix)
            logits = torch.matmul(matrix, logits)
            logits += _bias
            if i < len(self._filters):
                _factor = getattr(self, f"_factor{i:d}")
                factor = torch.tanh(_factor)
                logits += factor * torch.tanh(logits)
        return logits

    def _likelihood(self, inputs):
        #Input:[channels, 1, points] Output:[channels, 1, points]
        lower = self._logits_cumulative(inputs - 0.5)
        upper = self._logits_cumulative(inputs + 0.5)
        sign = -torch.sign(torch.add(lower, upper)).detach()
        likelihood = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        return likelihood

    def forward(self, inputs, quantize_mode="noise"):
        #Input:[B,C,H,W] --> [C, 1, B*H*W] -->Output:[B,C,H,W]
        #维度转换
        perm = np.arange(len(inputs.shape))
        perm[0], perm[1] = perm[1], perm[0]
        inv_perm = np.arange(len(inputs.shape))[np.argsort(perm)]
        inputs = inputs.permute(*perm).contiguous()
        shape = inputs.size()
        inputs = inputs.reshape(inputs.size(0), 1, -1)
        #量化
        if quantize_mode is None: 
            outputs = inputs
        else: 
            outputs = self.quantize(inputs, mode=quantize_mode)
        
        likelihood = self._likelihood(outputs)
        likelihood = Low_bound.apply(likelihood)
        #转化为输入维度
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()
        return outputs, likelihood

    #将概率转换为累积概率
    def _pmf_to_cdf(self, pmf):
        cdf = pmf.cumsum(dim=-1)
        #第一维添加0，最后一维变成1
        spatial_dimensions = pmf.shape[:-1]+(1,)
        zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
        #cdf第一个元素设置为0
        cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
        cdf_max_1 = torch.clamp(cdf_with_0, max=1.)
        return cdf_max_1

    @torch.no_grad()
    def compress(self, inputs, device='cuda'):
        #Input:[B,C,H,W]
        #量化
        values = self.quantize(inputs, mode="symbols")
        batch_size, channels, H, W = values.shape[:]
        #获得编码范围内所有字符的概率值
        minima = values.min().detach().float()
        maxima = values.max().detach().float()
        symbols = torch.arange(minima, maxima+1)
        # 对编码元素做一个偏移，从0开始
        values_norm = (values-minima).to(torch.int16)
        minima, maxima = torch.tensor([minima]), torch.tensor([maxima])  #将minima和maxima重新封装为tensor
        
        #[channels, 1, points],points为编码值范围
        symbols = symbols.reshape(1, 1, -1).repeat(channels, 1, 1).to(device)
        print(f'entropybottleneck encode device: {symbols.device}')
        pmf = self._likelihood(symbols)
        pmf = torch.clamp(pmf, min=self._likelihood_bound)
        cdf = self._pmf_to_cdf(pmf)
        cdf_expand = cdf.unsqueeze(0).unsqueeze(2)
        out_cdf = cdf_expand.repeat(batch_size, 1, H, W, 1).to('cpu')
        values_norm = values_norm.to('cpu')#.cpu() # torchac需要传入元素在CPU上
        #torchac传入两个主要参数，概率表和待编码元素，概率表out_cdf是按顺序排列的概率，格式[B,C,H,W,point]，待编码元素格式[B,C,H,W]
        strings = torchac.encode_float_cdf(out_cdf, values_norm)
        return strings, minima.cpu().numpy(), maxima.cpu().numpy()

    @torch.no_grad()
    def decompress(self, strings, minima, maxima, shape, device='cuda'):
        batch_size, channels, H, W = shape[:]
        symbols = torch.arange(int(minima), int(maxima+1))#类型转换numpy.int32->int32
        symbols = symbols.reshape(1, 1, -1).repeat(channels, 1, 1).to(device)
        print(f'entropybottleneck decode device: {symbols.device}')
        pmf = self._likelihood(symbols)
        pmf = torch.clamp(pmf, min=self._likelihood_bound)

        cdf = self._pmf_to_cdf(pmf)
        out_cdf = cdf.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, H, W, 1).cpu()
        values = torchac.decode_float_cdf(out_cdf, strings)
        values = values.float()
        values += minima

        return values


class GaussianConditional(EntropyBase):
    r"""GaussianConditional Layer是用于估计`y_hat`码率的模块
    在设计思想上和EntropyBottleneck并无大的差别,这里假设y_hat能够较好的满足高斯独立分布,
    因此使用高斯正态分布表示CDF
    """
    def __init__(self,scale_table,scale_bound=0.11,tail_mass=1e-9,entropy_coder="torchac"):
        super(GaussianConditional,self).__init__()
        # 检查scale_table是否规范
        if scale_table and (
            scale_table !=sorted(scale_table) or any(s <=0 for s in scale_table)
        ):
            raise ValueError(f"Invalid scale_table: '({scale_table})'")
        # 保存scal_table
        if scale_table:
            self.scale_table = self._prepare_scale_table(scale_table)
        else:
            self.scale_table = torch.Tensor()
        # 更新scal_bound, 1) 直接传参 2) 传None,将scale_table的第一项(最小值)作为scale_bound
        if scale_bound:
            self.scale_bound = scale_bound
        elif scale_table:
            self.scale_bound = scale_table[0]
        if self.scale_bound <= 0:
            raise ValueError("Invalid parameters")
        # other
        self.tail_mass = tail_mass  # -infinite,[tail_mass/2],PMF or CDF,[tail_mass/2],+infinite
        assert self.tail_mass < 0.5
        self.entropy_coder = entropy_coder # 默认熵编码器是torchac, 在compressai中是内置的ans, 两者是否存在差异
        self._likelihood_bound = 1e-9

    def forward(self,inputs,scales, means=0):
        # 1. 量化（加噪声）
        outputs = self.quantize(inputs,mode="noise", means=means)
        # 2. 计算似然,并保证似然大于1e-9(lower bound)
        likelihood = self._likelihood(outputs,scales, means)
        likelihood = Low_bound.apply(likelihood)
        return outputs,likelihood    
                      
    def _standardized_cumulative(self,inputs):
        # 正态分布的累积分布函数
        half = float(0.5)
        const = float(-(2**-0.5))
        # 这里就是利用erfc的形式来计算高斯分布累积分布，整理后得到：1/2-F(x)
        return half * torch.erfc(const * inputs)

    def _likelihood(self, inputs, scales, means=None):
        # 实际编码时:inputs.size [B,C,H,W,points] scales.size(B,C,H,W,1) means.size(B,C,H,W,1) likelihood.size(B,C,H,W,points)
        scales = Low_bound.apply(scales,0.11) 
        values = inputs
        if means is not None:
            values = inputs-means
        else:
            values = inputs
        values = torch.abs(values)
        # 将普通的高斯分布归一化到正态分布
        upper = self._standardized_cumulative((0.5 - values)/scales)
        lower = self._standardized_cumulative((-0.5 - values)/scales)
        likelihood = upper - lower
        return likelihood
    
    def update(self):
        """更新CDF"""
        multiplier = - self._standardized_quantile(self.tail_mass / 2)
        pmf_center = torch.ceil(self.scale_table * multiplier).int()  # tail_mass所对应的变量值
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.abs(torch.arange(max_length,device=device)).float()
        samples_scale = self.scale_table.unsqueeze(1).float()  # (2,4,5,..) -> ((2),(4),(5),...)
        upper = self._standardized_cumulative((-samples + 0.5)/samples_scale)  
        # samples = (0,1,2,3,4....), 和samples_scale维度不一致
        # 但是, 根据pytorch张量运算的广播机制, samples/samples_scale 最终得到 
        # ( ( 0/2, 1/2, 2/2, 3/2, ...),
        #   ( 0/4, 1/4, 2/4, 3/4, ...),
        #   ( 0/5, 1/5, 2/5, 3/5, ...),
        #   ...)
        lower = self._standardized_cumulative((-samples - 0.5)/samples_scale) # e.g.  tensor([0.4602, 0.3821, 0.3085, 0.2420],[...],...)
        pmf = upper - lower

        # 根据实际分布更新tail_mass
        # TODO: check
        tail_mass = 2 * lower[:,:1] #  compressai
        tail_mass = 2 * lower[:,-1:]  # corrected version

        quantized_cdf = torch.Tensor(len(pmf_length),max_length + 2) 
        quantized_cdf = self._pmf_to_cdf(pmf,tail_mass,pmf_length,max_length)
        self._quantized_cdf = quantized_cdf # 量化后的CDF + tail_mass + entropy_coder_precision
        self._offset = - pmf_center # tail_mass所对应的变量值，即cdf开始计数的变量值 
        self._cdf_length = pmf_length + 2 # 每个通道的CDF长度，多出的两个变量是 tail_mass + entropy_coder_precision
    
    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    #将概率转换为累积概率
    def _pmf_to_cdf(self, pmf):
        cdf = pmf.cumsum(dim=-1)
        #第一维添加0，最后一维变成1
        spatial_dimensions = pmf.shape[:-1]+(1,)
        zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
        #cdf第一个元素设置为0
        cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
        #cdf最大值即最后一个值设置为1
        # cdf_with_0[:, :, :, :, l].fill_(1.)
        cdf_max_1 = torch.clamp(cdf_with_0, max=1.)
        return cdf_max_1

    """
    # Present version in compressai
    # 不同之处是CDF参数不同
    @torch.no_grad()
    def compress(self,inputs,indexes):
        ""
        Compress input tensor to char strings
        ""
        # 1. 量化
        symbols = self.quantize(inputs,mode="symbols")
        # 2. 类型检查
        if len(inputs.size()) < 2:
            raise ValueError(
                "invalid `inputs` size: Expected a tensor with at least 2 dimensions."
            )
        if inputs.size() != indexes.size():
            raise ValueError("`inputs` and `indexes` should hava same size.")

        self._check_cdf_length()
        self._check_cdf_size()
        self._check_offsets_size()

        # 3. 熵编码
        strings = []
        for i in range(symbols.size(0)):
            rv = self.entropy_coder.encode_with_indexes(
                symbols[i].reshape(-1).int().tolist(),
                indexes[i].reshape(-1).int().tolist(),
                self._quantized_cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
        return strings
    
    def _check_cdf_size(self):
        if self._quantized_cdf.numel() == 0:
            raise ValueError('Uninitialized CDFs. Run update() First' )
    
    def _check_cdf_length(self):
        if self._cdf_length.numel() == 0:
            raise ValueError("Unitialized CDF lengths. Run update() First.")
        if len(self._cdf_length.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._cdf_length.size()}")
    
    def _check_offsets_size(self):
        if self._offset.numel() == 0:
            raise ValueError("Unintialized offsets. Run update() firstly.")
    """
    @torch.no_grad()
    def compress(self,inputs,scales, means=None, minima=None, maxima=None, model_name='JointAutoregressive', encoder=None):
        """new version
        return: strings, minima, maxima 
        """
        if model_name == 'JointAutoregressive':
            device = 'cpu'
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 量化
        values = self.quantize(inputs,mode="symbols")
        batch_size, channels,H,W = values.shape
        #获得编码范围内所有字符的概率值，如果给定了就不用统计了
        # if minima == None and maxima == None:
        #     minima = values.min().detach().float()
        #     maxima = values.max().detach().float()
        #     minima, maxima = torch.int([minima]), torch.int([maxima])
        minima = values.min().detach().float()
        maxima = values.max().detach().float()
        symbols = torch.arange(minima, maxima+1).to(device)
        # 对编码元素做一个偏移，使其最小值为0
        values = (values-minima).to(torch.int16)
        # symbols: torch.Size([points]) -> torch.Size([1, 1, 1, 1, points]) -> (batch_size,C,H,W,points)
        # scales: torch.Size([batch_size, C, H, W])  -> (batch_size, C,H,W,1)，对应unsqueeze(-1)操作
        symbols = symbols.reshape(1, 1, 1, 1, -1).repeat(batch_size,channels, H, W, 1).to(device)
        scales = scales.to(device)
        pmf = self._likelihood(symbols, scales.unsqueeze(-1), means=means.unsqueeze(-1))
        pmf = torch.clamp(pmf, min=self._likelihood_bound)
        #pmf = F.softmax(pmf, dim=4)
        if model_name == 'JointAutoregressive':
        # 对于自回归模型使用range-coder编码器编解码，每次编码张量格式为[1, C, 1, 1]
            assert encoder is not None
            resolution = 1e+5
            values = values.squeeze(3).squeeze(2).squeeze(0).cpu()
            cdf = self._pmf_to_cdf(pmf).squeeze(3).squeeze(2).squeeze(0).cpu()
            cdf = torch.round(cdf*resolution).int()
            for i in range(values.size(0)):
                seq_list = []
                seq_list.append(int(values[i]))
                cdf_list = cdf[i].tolist()
                encoder.encode(seq_list, cdf_list)
        else:
        # 对于hyper模型使用常规的torchac编码器编解码
            cdf = self._pmf_to_cdf(pmf).to('cpu')
            values = values.to('cpu')
            #torchac传入两个主要参数，概率表和待编码元素，概率表out_cdf是按顺序排列的概率，格式[B,C,H,W,point]，待编码元素格式[B,C,H,W]
            strings = torchac.encode_float_cdf(cdf, values)
            return strings, minima.cpu().numpy(), maxima.cpu().numpy()

    @torch.no_grad()
    def decompress(self, strings, minima, maxima, shape,scales, means=None, dequantize_dtype=torch.float, model_name='JointAutoregressive', decoder=None):
        """new version
        return: features
        """
        if model_name == 'JointAutoregressive':
            device = 'cpu'    
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size, channels, H, W = shape[:]
        symbols = torch.arange(minima, maxima+1).to(device)
        symbols = symbols.reshape(1, 1, 1, 1, -1).repeat(batch_size, channels, H, W, 1)

        pmf = self._likelihood(symbols,scales.unsqueeze(-1), means.unsqueeze(-1))
        pmf = torch.clamp(pmf, min=self._likelihood_bound)
        #pmf = F.softmax(pmf, dim=4)
        if model_name == 'JointAutoregressive':
            assert decoder is not None
            seq_list = []
            resolution = 1e+5
            cdf = self._pmf_to_cdf(pmf).squeeze(3).squeeze(2).squeeze(0).cpu()
            cdf = torch.round(cdf*resolution).int()
            for i in range(channels):
                cdf_list = cdf[i].tolist()
                values = decoder.decode(1, cdf_list)[0]+minima
                seq_list.append(values)
            # 返回Tensor shape:[1, M]
            return torch.tensor(seq_list).unsqueeze(0)
        else:
            cdf = self._pmf_to_cdf(pmf).to('cpu')
            values = torchac.decode_float_cdf(cdf, strings)
            values = values.to(dequantize_dtype)
            values += minima
            return values
