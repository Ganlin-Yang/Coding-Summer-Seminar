import warnings
from argon2 import Parameters
from range_coder import RangeEncoder, RangeDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from zmq import device
from gdn import GDN, GDN1
from entropy_model import EntropyBottleneck, GaussianConditional
from torch import Tensor
import os

torch.use_deterministic_algorithms(True)
torch.set_deterministic(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Factorized(nn.Module):
    def __init__(self, N=128, M=192):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(M)
        self.g_a = nn.Sequential(
            nn.Conv2d(3,N,stride=2,kernel_size=5,padding=2),
            GDN(N),
            nn.Conv2d(N,N,stride=2,kernel_size=5,padding=2),
            GDN(N),
            nn.Conv2d(N,N,stride=2,kernel_size=5,padding=2),
            GDN(N),
            nn.Conv2d(N,M,kernel_size=5,stride=2,padding=2),
        )

        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(M,N,kernel_size=5,padding=2,output_padding=1,stride=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N,N,kernel_size=5,padding=2,output_padding=1,stride=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N,N,kernel_size=5,padding=2,output_padding=1,stride=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N,3,kernel_size=5,stride=2,output_padding=1,padding=2),
        )
        
    def forward(self, x):
       y = self.g_a(x)
       y_hat, y_likelihoods = self.entropy_bottleneck(y)
       x_hat = self.g_s(y_hat)
       return {'x_hat':x_hat, 'likelihood':y_likelihoods}
    
    def compress(self, x):
        y = self.g_a(x)
        y_strings,y_minima,y_maxima = self.entropy_bottleneck.compress(y)
        return {"strings": y_strings, "shape": y.shape,"scope":[y_minima,y_maxima]}

    def decompress(self, strings, minima,maxima,shape):
        y_hat = self.entropy_bottleneck.decompress(strings, minima, maxima, shape).to(device)
        x_hat = self.g_s(y_hat)
        return x_hat

    def encode(self,inputs,file_name,postfix=''):
        out = self.compress(inputs)
        strings = out['strings']
        shape = out['shape']
        minima,maxima = out['scope']
        
        with open(file_name+postfix+'_F.bin', 'wb') as fout:
            fout.write(strings)

        with open(file_name+postfix+'_H.bin', 'wb') as fout:
            fout.write(np.array(shape, dtype=np.int32).tobytes())
            fout.write(np.array(len(minima), dtype=np.int8).tobytes())
            #记录编码元素的属性：最大值，最小值，存储一个float32需要4Bytes
            fout.write(np.array(minima, dtype=np.float32).tobytes())
            fout.write(np.array(maxima, dtype=np.float32).tobytes())


    def decode(self,filename,postfix=''):

        with open(filename+postfix+'_F.bin', 'rb') as fin:
            strings = fin.read()
        with open(filename+postfix+'_H.bin', 'rb') as fin:
            shape = np.frombuffer(fin.read(4*4), dtype=np.int32)
            len_minima = np.frombuffer(fin.read(1), dtype=np.int8)[0]
            minima = np.frombuffer(fin.read(4*len_minima), dtype=np.float32)[0]
            maxima = np.frombuffer(fin.read(4*len_minima), dtype=np.float32)[0]

        x_hat = self.decompress(strings,minima,maxima,shape)
        return x_hat


class Hyperprior(nn.Module):
    """
    hyperprior版本编解码器网络结构部分, 主要是g_a,g_s,h_a,h_s四种变换，以及forward
    参考CompressAI-master/compressai/models/google.py
    """
    def __init__(self,entropy_bottleneck_channels=128,M=192,init_weights=None):
        super().__init__()
        self.entropy_bottleneck=EntropyBottleneck(entropy_bottleneck_channels)

        N=entropy_bottleneck_channels
        #M:last layer of the encoder and last layer of the hyperprior decoder

        self.g_a=nn.Sequential(
            #in_channels=3,out_channels=N,kernel_size=5,stride=2,padding=kernel_size//2
            nn.Conv2d(3,N,kernel_size=5,stride=2,padding=2),
            GDN(N),
            nn.Conv2d(N,N,kernel_size=5,stride=2,padding=2),
            GDN(N),
            nn.Conv2d(N,N,kernel_size=5,stride=2,padding=2),
            GDN(N),
            nn.Conv2d(N,M,kernel_size=5,stride=2,padding=2),
        )

        self.g_s=nn.Sequential(
            #in_channels=M,out_channels=N,kernel_size=5,stride=2,output_padding=stride-1,padding=kernel_size//2
            nn.ConvTranspose2d(M,N,kernel_size=5,stride=2,output_padding=1,padding=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N,N,kernel_size=5,stride=2,output_padding=1,padding=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N,N,kernel_size=5,stride=2,output_padding=1,padding=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N,3,kernel_size=5,stride=2,output_padding=1,padding=2),
        )

        self.h_a=nn.Sequential(
            #in_channels=M,out_channels=N,kernel_size=3,stride=1,padding=kernel_size//2
            nn.Conv2d(M,N,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(N,N,kernel_size=5,stride=2,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(N,N,kernel_size=5,stride=2,padding=2),
        )

        self.h_s=nn.Sequential(
            nn.ConvTranspose2d(N,N,kernel_size=5,stride=2,output_padding=1,padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(N,N,kernel_size=5,stride=2,output_padding=1,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(N,M,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
    
    @property
    def downsampling_factor(self)->int:
        return 2**(4+2)
    
    def forward(self,x):
        y=self.g_a(x)
        z=self.h_a(torch.abs(y))
        z_hat,z_likelihood=self.entropy_bottleneck(z)
        scales_hat=self.h_s(z_hat)
        y_hat,y_likelihood=self.gaussian_conditional(y,scales_hat)
        x_hat=self.g_s(y_hat)

        return {"x_hat":x_hat, "likelihood":{"y":y_likelihood, "z":z_likelihood}}
    
    @torch.no_grad()
    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_strings, z_minima, z_maxima = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z_minima,z_maxima,z.shape).to(x.device)

        scales_hat = self.h_s(z_hat)
        y_strings,y_minima,y_maxima = self.gaussian_conditional.compress(y,scales_hat,model_name='Hyperprior')
        return {"strings": [y_strings, z_strings], "shape":[y.shape,z.shape],"y_scope":[y_minima,y_maxima],"z_scope":[z_minima,z_maxima]}

    @torch.no_grad()
    def decompress(self, strings, shape, y_scope,z_scope):
        assert isinstance(strings, list) and len(strings) == 2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        y_shape,z_shape = shape
        y_minima, y_maxima = y_scope
        z_minima, z_maxima = z_scope
        z_hat = self.entropy_bottleneck.decompress(strings[1], z_minima,z_maxima,z_shape).to(device)
        scales_hat = self.h_s(z_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0],y_minima,y_maxima,y_shape,scales_hat, dequantize_dtype=z_hat.dtype,model_name='Hyperprior').to(device)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return x_hat

    def encode(self,inputs,file_name,postfix=''):
        """
        在compress的基础上调用熵编码器进一步封装额外的参数
        """
        out = self.compress(inputs)
        y_strings,z_strings = out['strings']
        y_shape,z_shape = out['shape']
        z_minima,z_maxima = out['z_scope']
        y_minima,y_maxima = out['y_scope']
        
        with open(file_name+postfix+'_Fz.bin', 'wb') as fout:
            fout.write(z_strings)
        with open(file_name+postfix+'_Fy.bin', 'wb') as fout:
            fout.write(y_strings)

        with open(file_name+postfix+'_H.bin', 'wb') as fout:
            fout.write(np.array(z_shape, dtype=np.int32).tobytes())
            fout.write(np.array(len(z_minima), dtype=np.int8).tobytes())
            #记录编码元素的属性：最大值，最小值，存储一个float32需要4Bytes
            fout.write(np.array(z_minima, dtype=np.float32).tobytes())
            fout.write(np.array(z_maxima, dtype=np.float32).tobytes())

            fout.write(np.array(y_shape, dtype=np.int32).tobytes())
            fout.write(np.array(len(y_minima), dtype=np.int8).tobytes())
            #记录编码元素的属性：最大值，最小值，存储一个float32需要4Bytes
            fout.write(np.array(y_minima, dtype=np.float32).tobytes())
            fout.write(np.array(y_maxima, dtype=np.float32).tobytes())


    def decode(self,file_name,postfix=''):

        with open(file_name+postfix+'_Fz.bin', 'rb') as fin:
            z_strings = fin.read()
        with open(file_name+postfix+'_H.bin', 'rb') as fin:
            z_shape = np.frombuffer(fin.read(4*4), dtype=np.int32)
            z_len_minima = np.frombuffer(fin.read(1), dtype=np.int8)[0]
            z_minima = np.frombuffer(fin.read(4*z_len_minima), dtype=np.float32)[0]
            z_maxima = np.frombuffer(fin.read(4*z_len_minima), dtype=np.float32)[0]

            y_shape = np.frombuffer(fin.read(4*4), dtype=np.int32)
            y_len_minima = np.frombuffer(fin.read(1), dtype=np.int8)[0]
            y_minima = np.frombuffer(fin.read(4*y_len_minima), dtype=np.float32)[0]
            y_maxima = np.frombuffer(fin.read(4*y_len_minima), dtype=np.float32)[0]

        with open(file_name+postfix+'_Fy.bin', 'rb') as fin:
            y_strings = fin.read()

        x_hat = self.decompress([y_strings,z_strings],[y_shape,z_shape],[y_minima,y_maxima],[z_minima,z_maxima])
        return x_hat

class MaskedConv2d(nn.Conv2d):
    r"""Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.
    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.
    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args, mask_type: str = "A", **kwargs):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        # 当前行，中间开始后面所有列
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        # 后面的所有行
        self.mask[:, :, h // 2 + 1 :] = 0

    def forward(self, x: Tensor) -> Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class JointAutoregressiveHierarchicalPriors(nn.Module):
    """
    HierarchicalPriors版本编解码器网络结构部分
    """
    def __init__(self, entropy_bottleneck_channels=192, M=192):
        # M:last layer of the encoder and last layer of the hyperprior decoder
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)
        N = entropy_bottleneck_channels

        self.g_a = nn.Sequential(
            # in_channels=3,out_channels=M,kernel_size=5,stride=2,padding=kernel_size//2
            nn.Conv2d(3, N, kernel_size=5, stride=2, padding=2),
            GDN1(N),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, M, kernel_size=5, stride=2, padding=2),
        )

        self.g_s = nn.Sequential(
            # in_channels=M,out_channels=N,kernel_size=5,stride=2,output_padding=stride-1,padding=kernel_size//2
            nn.ConvTranspose2d(M, N, kernel_size=5, stride=2, output_padding=1, padding=2),
            GDN1(N, inverse=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, output_padding=1, padding=2),
            GDN1(N, inverse=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, output_padding=1, padding=2),
            GDN1(N, inverse=True),
            nn.ConvTranspose2d(N, 3, kernel_size=5, stride=2, output_padding=1, padding=2),
        )

        self.h_a = nn.Sequential(
            # in_channels=M,out_channels=N,kernel_size=3,stride=1,padding=kernel_size//2
            nn.Conv2d(M, N, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
        )
        # 相较于Hyperprior网络h_s输出通道数增加2倍
        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(N, M, kernel_size=5, stride=2, output_padding=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(M, M * 3//2, kernel_size=5, stride=2, output_padding=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 3//2, M * 2, kernel_size=3, stride=1, padding=1),
        )


        self.context_prediction  = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )
        self.entropy_parameters  =nn.Sequential(
            nn.Conv2d(M * 12//3, M * 10 // 3, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, kernel_size=1),
        )

        self.gaussian_conditional = GaussianConditional(None)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihood = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "quantize"
        )
        #上下文模型返回值
        ctx_params = self.context_prediction(y_hat)
        # params.shape = ctx_params.shape = [B,2M,H,W]
        #将上下文模型返回值与超先验层返回值送入熵模型
        # gaussian_params.shape=[B,2M,H,W]
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        # 参数沿dim=1(channels)分成两组
        # scales_hat.shape[B,C,H,W]means_hat.shape[B,C,H,W]，对应y_hat的每一个元素
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihood = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihood": {"y": y_likelihood, "z": z_likelihood},
        }

    #context_model模型要在cup上测试。解码要求边解码边测试，所以更换算术编码器为range-coder，并且在编码时也需要串行操作。
    def compress(self, x, filepath):
        if next(self.parameters()).device != torch.device('cpu'):
            warnings.warn("Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).")
        # y.shape=[B,M,H,W]
        y = self.g_a(x)
        z = self.h_a(y)
        # 编解码z
        z_strings, z_minima, z_maxima = self.entropy_bottleneck.compress(z)
        #z_hat = self.entropy_bottleneck.decompress(z_strings, z_minima, z_maxima, z.size())
        z_hat = torch.round(z)
        y_q = self.gaussian_conditional.quantize(y, mode='quantize')
        # 编码y
        # params.shape=[B, 2M, H, W]
        params = self.h_s(z_hat)
        s = 4  #y,z下采样倍数4
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s
        # 在H和W这两个维度上做填充，上下左右各填充paddings，后面掩码卷积从第一个元素开始估计均值（不填充行和列的前后kernal_size/2个元素没法被估计）
        y_hat = F.pad(y, (padding, padding, padding, padding))
        # y.size(0) = B， test: default=1
        # y_minima, y_maxima = self.compress_serial(
        #     y_hat,
        #     params,
        #     y_height,
        #     y_width,
        #     kernel_size,
        #     padding,
        #     filepath,
        #  )
        y_minima, y_maxima = self.compress_pa(y_q, params, filepath)
        return {"strings": z_strings, "shape": [y.shape, z.shape], "y_scope":[y_minima, y_maxima], "z_scope":[z_minima, z_maxima],
        }
    
    # 并行编码
    def compress_pa(self, y, params, filepath):
        y_minima = []
        y_maxima = []
        height, width = y.shape[2:]
        ctx_params = self.context_prediction(y)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat = self.gaussian_conditional.quantize( y,mode="quantize")
        encoder = RangeEncoder(filepath)
        for i in tqdm(range(height)):
            for j in range(width):
                minima, maxima = self.gaussian_conditional.compress(y_hat[:, :, i:i+1, j:j+1], 
                scales=scales_hat[:, :, i:i+1, j:j+1], means=means_hat[:, :, i:i+1, j:j+1],
                model_name='JointAutoregressive',
                encoder=encoder)
                y_minima.append(minima)
                y_maxima.append(maxima)
        encoder.close()
        return y_minima, y_maxima
    
    # 串行编码
    def compress_serial(self, y_hat, params, height, width, kernel_size, padding, filepath):
        # 初始化一些参数
        # masked_weight.shape[2M, M, kernal_size, kernal_size]
        masked_weight = self.context_prediction.weight*self.context_prediction.mask
        y_minima = []
        y_maxima = []
        # params.shape=[1, 2M, H, W] y_hat.shape=[1, M, H, W]
        # 初始化range_coder编码器
        encoder = RangeEncoder(filepath)
        # 按顺序得到每个元素的概率分布
        for i in tqdm(range(height)):
            for j in range(width):
                # 每次取y_crop.shape=[1, M, kernal_size, kernal_size], kernal_size=5
                y_crop = y_hat[:, :, i : i + kernel_size, j : j + kernel_size]
                # 掩码卷积，让被估计的元素处在卷积核中间位置
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )
                # ctx_p.shape=[1, 2M, 1, 1], p.shape=[1, 2M, 1, 1]
                p = params[:, :, i:i+1, j:j+1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                # y_crop.shape [1, M, 1, 1]即通过上下文模型预测了在高和宽特定位置上所有通道的元素的均值
                y_crop = y_crop[:, :, padding, padding].unsqueeze(2).unsqueeze(3)

                # 对待编码元素每次只编码H,W某一特定位置上的所有通道
                minima, maxima = self.gaussian_conditional.compress(y_crop, scales=scales_hat, means=means_hat, 
                model_name='JointAutoregressive', encoder=encoder)
                y_minima.append(minima)
                y_maxima.append(maxima)
        encoder.close()
        # y_minima,y_maxinma是H*W的列表
        return y_minima, y_maxima

    def decompress(self, strings, shape, y_scope, z_scope, filepath):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        z_hat = self.entropy_bottleneck.decompress(strings, z_scope[0], z_scope[1] ,shape[1])
        params = self.h_s(z_hat)

        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2
        y_shape = shape[0] 
        # y_hat.shape=[1, M, H+2*padding, W+2*padding]
        y_hat = torch.zeros((1, y_shape[1], y_shape[2]+2*padding, y_shape[3]+2*padding), device=z_hat.device)
        y_hat = self.decompress_serial(
            y_hat,
            params,
            y_shape,
            kernel_size,
            padding,
            y_scope,
            filepath,
        )
        # 去除为了使用掩码卷积而增加的padding
        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return x_hat 

    def decompress_serial(self, y_hat, params, shape, kernel_size, padding, y_scope, filepath):
        masked_weight = self.context_prediction.weight*self.context_prediction.mask
        # print(masked_weight)
        y_minima = y_scope[0]
        y_maxima = y_scope[1]
        batch_size, M, height, width = shape[:]
        channel_shape = [batch_size, M, 1, 1]
        # 初始化range_coder解码器
        decoder = RangeDecoder(filepath)
        for i in tqdm(range(height)):
            for j in range(width):
                y_crop = y_hat[:, :, i: i+kernel_size, j: j+kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias = self.context_prediction.bias,
                )
                p = params[:, :, i:i+1, j:j+1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                channel_value = self.gaussian_conditional.decompress(
                    strings=None, 
                    minima=y_minima[i*width+j], maxima=y_maxima[i*width+j], shape=channel_shape, 
                    scales=scales_hat, means=means_hat, 
                    decoder=decoder)
                y_hat[:, :, i+padding, j+padding] = channel_value
        decoder.close()   
        return y_hat

    def encode(self,inputs,file_name,postfix=''):
        """
        在compress的基础上调用熵编码器进一步封装额外的参数
        """
        # 由于range_coder编码需要文件名，首先创建编码y的文件
        y_file_name = file_name+postfix+'_Fy.bin'
        with open(y_file_name, 'w'):
            pass
        out = self.compress(inputs, filepath=y_file_name)
        z_strings = out['strings']
        y_shape,z_shape = out['shape']
        z_minima,z_maxima = out['z_scope']
        y_minima,y_maxima = out['y_scope']
        with open(file_name+postfix+'_Fz.bin', 'wb') as fout:
            fout.write(z_strings)

        with open(file_name+postfix+'_H.bin', 'wb') as fout:
            fout.write(np.array(z_shape, dtype=np.int32).tobytes())
            fout.write(np.array(len(z_minima), dtype=np.int8).tobytes())
            #记录编码元素的属性：最大值，最小值，存储一个int8需要1Bytes
            fout.write(np.array(z_minima, dtype=np.int8).tobytes())
            fout.write(np.array(z_maxima, dtype=np.int8).tobytes())

            fout.write(np.array(y_shape, dtype=np.int32).tobytes())
            fout.write(np.array(len(y_minima), dtype=np.int16).tobytes())
            #记录编码元素的属性：最大值，最小值，存储一个int16需要2Bytes
            fout.write(np.array(y_minima, dtype=np.int16).tobytes())
            fout.write(np.array(y_maxima, dtype=np.int16).tobytes())  
    
    def decode(self, file_name, postfix=''):
        with open(file_name+postfix+'_Fz.bin', 'rb') as fin:
            z_strings = fin.read()
        with open(file_name+postfix+'_H.bin', 'rb') as fin:
            z_shape = np.frombuffer(fin.read(4*4), dtype=np.int32)
            z_len_minima = np.frombuffer(fin.read(1), dtype=np.int8)[0]
            z_minima = np.frombuffer(fin.read(1*z_len_minima), dtype=np.int8)[0]
            z_maxima = np.frombuffer(fin.read(1*z_len_minima), dtype=np.int8)[0]
            y_shape = np.frombuffer(fin.read(4*4), dtype=np.int32)
            y_len_minima = np.frombuffer(fin.read(2), dtype=np.int16)[0]
            y_minima = np.frombuffer(fin.read(2*y_len_minima), dtype=np.int16)
            y_maxima = np.frombuffer(fin.read(2*y_len_minima), dtype=np.int16)
            # print(f'z_shape: {z_shape}')
            # print(f'z_len_minima: {z_len_minima}')
            # print(f'z_minima: {z_minima}')
            # print(f'z_maxima: {z_maxima}')
            # print(f'y_len_minima: {y_len_minima}')
            # print(f'y_minima: {y_minima}')
            # print(f'y_maxima: {y_maxima}')
            y_file_name = file_name+postfix+'_Fy.bin'
            with open(y_file_name, 'rb'):
                pass
        x_hat = self.decompress(z_strings,[y_shape,z_shape],[y_minima,y_maxima],[z_minima,z_maxima], filepath=y_file_name)
        return x_hat
    
class CheckerboardContext(nn.Conv2d):
    """
    if kernel_size == (5, 5)
    then mask:
        [[0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.]]
    0: non-anchor
    1: anchor
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        self.mask[:, :, 0::2, 1::2] = 1
        self.mask[:, :, 1::2, 0::2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        out = super().forward(x)

        return out

class CheckerboardAutogressive(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N, M, **kwargs)

        self.context_prediction = CheckerboardContext(M, M * 2, 5, 1, 2)
        self.M = M

    def forward(self, x):
        """
        anchor :
            0 1 0 1 0
            1 0 1 0 1
            0 1 0 1 0
            1 0 1 0 1
            0 1 0 1 0
        non-anchor (use anchor as context):
            1 0 1 0 1
            0 1 0 1 0
            1 0 1 0 1
            0 1 0 1 0
            1 0 1 0 1
        """
        batch_size, channel, x_height, x_width = x.shape
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihood = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "quantize"
        )

        anchor = torch.zeros_like(y_hat).to(x.device)
        non_anchor = torch.zeros_like(y_hat).to(x.device)

        anchor[:, :, 0::2, 1::2] = y_hat[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, 0::2] = y_hat[:, :, 1::2, 0::2]
        non_anchor[:, :, 0::2, 0::2] = y_hat[:, :, 0::2, 0::2]
        non_anchor[:, :, 1::2, 1::2] = y_hat[:, :, 1::2, 1::2]

        # print(anchor)
        # print(non_anchor)

        # compress anchor，类似hyperprior的处理，ctx为0
        ctx_params_anchor = torch.zeros([batch_size, self.M * 2, x_height // 16, x_width // 16]).to(x.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        # compress non-anchor
        ctx_params_non_anchor = self.context_prediction(anchor)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)

        scales_hat = torch.zeros([batch_size, self.M, x_height // 16, x_width // 16]).to(x.device)
        means_hat = torch.zeros([batch_size, self.M, x_height // 16, x_width // 16]).to(x.device)

        scales_hat[:, :, 0::2, 1::2] = scales_anchor[:, :, 0::2, 1::2]
        scales_hat[:, :, 1::2, 0::2] = scales_anchor[:, :, 1::2, 0::2]
        scales_hat[:, :, 0::2, 0::2] = scales_non_anchor[:, :, 0::2, 0::2]
        scales_hat[:, :, 1::2, 1::2] = scales_non_anchor[:, :, 1::2, 1::2]
        means_hat[:, :, 0::2, 1::2] = means_anchor[:, :, 0::2, 1::2]
        means_hat[:, :, 1::2, 0::2] = means_anchor[:, :, 1::2, 0::2]
        means_hat[:, :, 0::2, 0::2] = means_non_anchor[:, :, 0::2, 0::2]
        means_hat[:, :, 1::2, 1::2] = means_non_anchor[:, :, 1::2, 1::2]

        # print(scales_hat - scales_anchor)
        # print(scales_hat - scales_non_anchor)

        _, y_likelihood = self.gaussian_conditional(y, scales=scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihood": {"y": y_likelihood, "z": z_likelihood}
        }

    @torch.no_grad()
    def compress(self, x):
        """
        if y[i, :, j, k] == 0 
        then bpp = 0
        Not recommend
        """
        batch_size, channel, x_height, x_width = x.shape
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings,z_minima,z_maxima = self.entropy_bottleneck.compress(z)
        #z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = torch.round(z)

        params = self.h_s(z_hat)
        anchor = torch.zeros_like(y).to(x.device)
        non_anchor = torch.zeros_like(y).to(x.device)

        anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]
        non_anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        non_anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]

        # compress anchor
        ctx_params_anchor = torch.zeros([batch_size, self.M * 2, x_height // 16, x_width // 16]).to(x.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        anchor_strings,an_minima,an_maxima = self.gaussian_conditional.compress(anchor, 
            scales_anchor, 
            means_anchor, 
            model_name='CheckerboardAutogressive')
        anchor_quantized = self.gaussian_conditional.quantize(anchor,"quantize")

        # compress non-anchor
        ctx_params_non_anchor = self.context_prediction(anchor_quantized)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)
        non_anchor_strings,non_minima,non_maxima = self.gaussian_conditional.compress(non_anchor, 
            scales_non_anchor, 
            means=means_non_anchor, 
            model_name='CheckerboardAutogressive'
        )

        return {
            "strings": [anchor_strings, non_anchor_strings, z_strings],
            "shape": [y.shape,z.shape],
            "anchor_scope":[an_minima,an_maxima],
            "non_anchor_scope":[non_minima,non_maxima],
            "z_scope":[z_minima,z_maxima]
        }

    @torch.no_grad()
    def decompress(self, strings, shape,anchor_scope,non_anchor_scope,z_scope):
        """
        if y[i, :, j, k] == 0 
        then bpp = 0
        Not recommend
        """
        assert isinstance(strings, list) and len(strings) == 3
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        y_shape,z_shape = shape
        an_minima, an_maxima = anchor_scope
        non_minima, non_maxima = non_anchor_scope
        z_minima, z_maxima = z_scope
        z_hat = self.entropy_bottleneck.decompress(strings[2],z_minima, z_maxima,z_shape).to(device)
        params = self.h_s(z_hat)

        batch_size, channel, z_height, z_width = z_hat.shape

        # decompress anchor
        ctx_params_anchor = torch.zeros([batch_size, 2 * self.M, z_height * 4, z_width * 4]).to(z_hat.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        anchor_quantized = self.gaussian_conditional.decompress(strings[0], 
            an_minima, 
            an_maxima, 
            y_shape, 
            scales_anchor, 
            means_anchor, 
            model_name='CheckerboardAutogressive'
        ).to("cuda")

        # decompress non-anchor
        ctx_params_non_anchor = self.context_prediction(anchor_quantized)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)
        non_anchor_quantized = self.gaussian_conditional.decompress(strings[1], 
            non_minima, 
            non_maxima, 
            y_shape, 
            scales_non_anchor, 
            means_non_anchor, 
            model_name='CheckerboardAutogressive'
        ).to("cuda")

        y_hat = anchor_quantized + non_anchor_quantized
        x_hat = self.g_s(y_hat)

        return x_hat

    def encode(self,inputs,file_name,postfix=''):
        """
        在compress的基础上调用熵编码器进一步封装额外的参数
        """
        out = self.compress(inputs)
        anchor_strings,non_anchor_strings,z_strings = out['strings']
        y_shape,z_shape = out['shape']
        z_minima,z_maxima = out['z_scope']
        an_minima,an_maxima = out['anchor_scope']
        non_minima,non_maxima = out['non_anchor_scope']
        
        with open(file_name+postfix+'_Fz.bin', 'wb') as fout:
            fout.write(z_strings)
        with open(file_name+postfix+'_Fanchor.bin', 'wb') as fout:
            fout.write(anchor_strings)
        with open(file_name+postfix+'_Fnon_anchor.bin', 'wb') as fout:
            fout.write(non_anchor_strings)

        with open(file_name+postfix+'_H.bin', 'wb') as fout:
            fout.write(np.array(z_shape, dtype=np.int32).tobytes())
            fout.write(np.array(len(z_minima), dtype=np.int8).tobytes())
            #记录编码元素的属性：最大值，最小值，存储一个float32需要4Bytes
            fout.write(np.array(z_minima, dtype=np.float32).tobytes())
            fout.write(np.array(z_maxima, dtype=np.float32).tobytes())

            fout.write(np.array(y_shape, dtype=np.int32).tobytes())
            fout.write(np.array(len(an_minima), dtype=np.int8).tobytes())
            #记录编码元素的属性：最大值，最小值，存储一个float32需要4Bytes
            fout.write(np.array(an_minima, dtype=np.float32).tobytes())
            fout.write(np.array(an_maxima, dtype=np.float32).tobytes())
            fout.write(np.array(non_minima, dtype=np.float32).tobytes())
            fout.write(np.array(non_maxima, dtype=np.float32).tobytes())


    def decode(self,file_name,postfix=''):

        with open(file_name+postfix+'_Fz.bin', 'rb') as fin:
            z_strings = fin.read()
        with open(file_name+postfix+'_H.bin', 'rb') as fin:
            z_shape = np.frombuffer(fin.read(4*4), dtype=np.int32)
            z_len_minima = np.frombuffer(fin.read(1), dtype=np.int8)[0]
            z_minima = np.frombuffer(fin.read(4*z_len_minima), dtype=np.float32)[0]
            z_maxima = np.frombuffer(fin.read(4*z_len_minima), dtype=np.float32)[0]

            y_shape = np.frombuffer(fin.read(4*4), dtype=np.int32)
            y_len_minima = np.frombuffer(fin.read(1), dtype=np.int8)[0]
            an_minima = np.frombuffer(fin.read(4*y_len_minima), dtype=np.float32)[0]
            an_maxima = np.frombuffer(fin.read(4*y_len_minima), dtype=np.float32)[0]
            non_minima = np.frombuffer(fin.read(4*y_len_minima), dtype=np.float32)[0]
            non_maxima = np.frombuffer(fin.read(4*y_len_minima), dtype=np.float32)[0]

        with open(file_name+postfix+'_Fanchor.bin', 'rb') as fin:
            anchor_strings = fin.read()
        with open(file_name+postfix+'_Fnon_anchor.bin', 'rb') as fin:
            non_anchor_strings = fin.read()

        x_hat = self.decompress([anchor_strings,non_anchor_strings,z_strings],[y_shape,z_shape],[an_minima,an_maxima],[non_minima,non_maxima],[z_minima,z_maxima])
        return x_hat

