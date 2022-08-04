import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


#GDN变换部分
#参考CompressAI-master/compressai/layers/gdn.py
class GDN(nn.Module):
    #y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}
    def __init__(
        self,
        in_channels:int,
        inverse:bool=False,
        beta_min:float=1e-6,
        gamma_init:float=0.1,
    ):
        super().__init__()

        #初始化beta_min和gamma，以及逆变换标志inverse
        beta_min=float(beta_min)
        gamma_init=float(gamma_init)
        self.inverse=bool(inverse)

        #根据in_channels大小对beta初始化，并将不可训练的类型的Tensor转换成可以训练类型parameter
        self.beta_reparam=NonNegativeParametrizer(minimum=beta_min)
        beta=torch.ones(in_channels)
        beta=self.beta_reparam.init(beta)
        self.beta=nn.Parameter(beta)

        #同理，对gamma初始化
        self.gamma_reparam=NonNegativeParametrizer()
        gamma=gamma_init*torch.eye(in_channels)
        gamma=self.gamma_reparam.init(gamma)
        self.gamma=nn.Parameter(gamma)

    def forward(self,x:Tensor)->Tensor:
        #获取通道数
        _,C,_,_=x.size()

        #每一步后对beta和gamma都进行Non negative reparametrization
        #论文中解释为消除每个线性变换和其后面的非线性之间的缩放模糊性
        beta=self.beta_reparam(self.beta)
        gamma=self.gamma_reparam(self.gamma)
        #对gamma进行转置变换使其对称，达到上述相同的目的
        gamma=gamma.reshape(C,C,1,1)

        #计算GDN or IGDN
        norm=F.conv2d(x**2,gamma,beta)
        if self.inverse:
            norm=torch.sqrt(norm)
        else:
            norm=torch.rsqrt(norm)

        out=x*norm

        return out

class GDN1(GDN):
    """
    Simplified GDN layer.
    math:
    y[i] = \frac{x[i]}{\beta[i] + \sum_j(\gamma[j, i] * |x[j]|}
    
    """

    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(torch.abs(x), gamma, beta)

        if not self.inverse:
            norm = 1.0 / norm

        out = x * norm

        return out

#重参数化部分，包括NonNegativeParametrizer类、LowerBound类以及一些相关函数
#参考CompressAI-master/compressai/ops/parametrizers.py, CompressAI-master/compressai/ops/bound_ops.py
class NonNegativeParametrizer(nn.Module):  # Non negative reparametrization
    pedestal:Tensor

    def __init__(self,minimum:float=0,reparam_offset:float=2**-18):
        super().__init__()

        self.minimum=float(minimum)
        self.reparam_offset=float(reparam_offset)

        #定义一个缓冲区buffer："pedestal"，无梯度，优化GDN参数时使用
        pedestal=self.reparam_offset**2
        self.register_buffer("pedestal",torch.Tensor([pedestal]))
        bound=(self.minimum+self.reparam_offset**2) ** 0.5
        self.lower_bound=LowerBound(bound)
    
    #初始化参数方法
    def init(self,x:Tensor)->Tensor:
        return torch.sqrt(torch.max(x+self.pedestal,self.pedestal))

    def forward(self,x:Tensor)->Tensor:
        out=self.lower_bound(x)
        #进行参数优化
        #论文中解释为平方确保在接近0的参数值周围梯度较小，否则优化可能变得不稳定。
        out=out**2-self.pedestal
        return out


#forward操作，计算torch.max(x, bound)
def lower_bound_fwd(x:Tensor,bound:Tensor)->Tensor:
    return torch.max(x,bound)


#backward操作
def lower_bound_bwd(x:Tensor,bound:Tensor,grad_output:Tensor):
    pass_through_if=(x>=bound)|(grad_output<0)
    return pass_through_if * grad_output,None


#LowerBound操作的自动求导function
class LowerBoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,bound):
        #在正向传播中，我们接收到一个上下文对象和一个包含输入的Tensor，
        #我们必须返回一个包含输出的Tensor，并且我们可以使用上下文对象来缓存对象，以便在反向传播中使用。
        ctx.save_for_backward(x,bound)
        return lower_bound_fwd(x,bound)

    @staticmethod
    def backward(ctx,grad_output):
        #在反向传播中，我们接收到上下文对象和一个Tensor，其包含了相对于正向传播过程中产生的输出的损失的梯度。
        #我们可以从上下文对象中检索缓存的数据，并且必须计算并返回与正向传播的输入相关的损失的梯度。
        x,bound = ctx.saved_tensors
        return lower_bound_bwd(x,bound,grad_output)


#Lower bound运算符，使用自定义梯度计算torch.max(x, bound)
#当x向"bound"移动时，导数将被特定的函数取代，否则梯度将保持为零。
class LowerBound(nn.Module):
    bound:Tensor

    def __init__(self,bound:float):
        super().__init__()
        #定义一个缓冲区buffer："bound"
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    @torch.jit.unused
    #官方文档解释：这个装饰器向编译器表明，一个函数或方法应该被忽略，并以引发一个异常来代替。
    #这允许在模型中留下尚未与TorchScript兼容的代码，并仍然导出模型。
    def lower_bound(self,x):
        return LowerBoundFunction.apply(x,self.bound)

    def forward(self,x):
        #与@unused修饰符配合使用，在模型中留下尚未与TorchScript兼容的代码
        if torch.jit.is_scripting():
            return torch.max(x,self.bound)
        return self.lower_bound(x)

