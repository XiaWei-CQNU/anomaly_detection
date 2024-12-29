import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """实现Positional Encoding功能"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        位置编码器的初始化函数
        :param d_model: 词向量的维度,与输入序列的特征维度相同
        :param dropout: 置零比率
        :param max_len: 句子最大长度,5000
        """
        super(PositionalEncoding, self).__init__()
        # 初始化一个nn.Dropout层,设置给定的dropout比例
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个位置编码矩阵
        # (5000,512)矩阵,保持每个位置的位置编码,一共5000个位置,每个位置用一个512维度向量来表示其位置编码
        pe = torch.zeros(max_len, d_model)
        # 偶数和奇数在公式上有一个共同部分,使用log函数把次方拿下来,方便计算
        # position表示的是字词在句子中的索引,如max_len是128,那么索引就是从0,1,2,...,127
        # 论文中d_model是512,2i符号中i从0取到255,那么2i对应取值就是0,2,4...510
        # (5000) -> (5000,1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算用于控制正余弦的系数,确保不同频率成分在d_model维空间内均匀分布
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 根据位置和div_term计算正弦和余弦值,分别赋值给pe的偶数列和奇数列
        pe[:, 0::2] = torch.sin(position * div_term)   # 从0开始到最后面,补长为2,其实代表的就是偶数位置
        pe[:, 1::2] = torch.cos(position * div_term[0:int(d_model/2)])   # 从1开始到最后面,补长为2,其实代表的就是奇数位置
        # 上面代码获取之后得到的pe:[max_len * d_model]
        # 下面这个代码之后得到的pe形状是：[1 * max_len * d_model]
        # 多增加1维,是为了适应batch_size
        # (5000, 512) -> (1, 5000, 512)
        # 将计算好的位置编码矩阵注册为模块缓冲区（buffer）,这意味着它将成为模块的一部分并随模型保存与加载,但不会被视为模型参数参与反向传播
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]  经过词向量的输入
        """
        x = x + self.pe[:x.size(0), :x.size(1)].clone().detach()   # 经过词向量的输入与位置编码相加
        # Dropout层会按照设定的比例随机“丢弃”（置零）一部分位置编码与词向量相加后的元素,
        # 以此引入正则化效果,防止模型过拟合
        return self.dropout(x)

class TranAD_PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TranAD_PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos+x.size(0), :]
        return self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, scale_factor, dropout=0.0):
        super().__init__()
        self.scale_factor = scale_factor
        #dropout用于防止过拟合,在前向传播的过程中,让某个神经元的激活值以一定的概率停止工作
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # batch_size: 批量大小
        # len_q,len_k,len_v: 序列长度 在这里他们都相等
        # n_head: 多头注意力,论文中默认为8
        # d_k,d_v: k v 的dim(维度) 默认都是64
        # 此时q的shape为(batch_size, n_head, len_q, d_k) (batch_size, 8, len_q, 64)
        # 此时k的shape为(batch_size, n_head, len_k, d_k) (batch_size, 8, len_k, 64)
        # 此时v的shape为(batch_size, n_head, len_k, d_v) (batch_size, 8, len_k, 64)
        # q先除以self.scale_factor,再乘以k的转置(交换最后两个维度(这样才可以进行矩阵相乘))。
        # attn的shape为(batch_size, n_head, len_q, len_k)

        attn = torch.matmul(q / self.scale_factor, k.transpose(1, 2))
        
        # 先在attn的最后一个维度做softmax 再dropout 得到注意力分数
        attn = self.dropout(torch.softmax(attn, dim=-1))
        # 最后attn与v矩阵相乘
        # output的shape为(batch_size, 8, len_q, 64)
        output = torch.matmul(attn, v)
        # 返回 output和注意力分数
        return output, attn
    
class In_channel_anomaly(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, batch=128, scale_factor = 64, dim_feedforward=512, dropout=0.1):
        '''
        # q k v先经过不同的线性层,再用ScaledDotProductAttention,最后再经过一个线性层
        '''
        super().__init__()


        self.w_qs = nn.Linear(batch, batch, bias=False)
        self.w_ks = nn.Linear(batch, batch, bias=False)
        self.w_vs = nn.Linear(batch, batch, bias=False)
        self.fc = nn.Linear(batch, batch, bias=False)

        self.attention = ScaledDotProductAttention(scale_factor=scale_factor ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(batch, eps=1e-6)  # 默认对最后一个维度归一化
        
        self.linear1 = nn.Linear(batch, dim_feedforward)
        self.activation = nn.LeakyReLU(True)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, batch)
        self.dropout2 = nn.Dropout(dropout)

        self.linear = nn.Linear(batch, batch)


    def forward(self, q, k, v):

        q = q.transpose(0, 1);k = k.transpose(0, 1);v = v.transpose(0, 1)
        # 用作残差连接

        residual = q

        q = self.layer_norm(q);k = self.layer_norm(k);v = self.layer_norm(v)


        q = self.w_qs(q).unsqueeze(-1);k = self.w_ks(k).unsqueeze(-1);v = self.w_vs(v).unsqueeze(-1)


        src, attn = self.attention(q, k, v)
        src = src.squeeze()

        src = self.dropout(self.fc(src))
        #Add & Norm
        src += residual
        src = self.layer_norm(src)
        
        #FCN
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        #Add & Norm
        src = self.layer_norm(src + self.dropout2(src2))

        src = self.linear(src).transpose(0, 1)
        return src,attn

class Between_channel_anomaly(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, d_model=128, scale_factor = 64, dim_feedforward=512, dropout=0.1):
        # 论文中这里的n_head, d_model, d_k, d_v分别默认为8, 512, 64, 64
        '''
        # q k v先经过不同的线性层,再用ScaledDotProductAttention,最后再经过一个线性层
        '''
        super().__init__()

        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(scale_factor=scale_factor ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)  # 默认对最后一个维度归一化
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.LeakyReLU(True)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.linear = nn.Linear(d_model, d_model)


    def forward(self, q, k, v):
        # 用作残差连接
        residual = q
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # q k v 分别经过一个线性层再改变维度
        # 由(batch_size, len_q, n_head*d_k) => (batch_size, len_q, n_head, d_k)
        # (batch_size, len_q, 8*64) => (batch_size, len_q, 8, 64)
        q = self.layer_norm(q);k = self.layer_norm(k);v = self.layer_norm(v)

        # 与q,k,v相关矩阵相乘,得到相应的q,k,v向量,d_model=n_head * d_k
        q = self.w_qs(q).unsqueeze(-1);k = self.w_ks(k).unsqueeze(-1);v = self.w_vs(v).unsqueeze(-1)

        # 输出的q为Softmax(QK/d)V, attn 为QK/D
        src, attn = self.attention(q, k, v)
        src = src.squeeze()
        # 经过fc和dropout
        src = self.dropout(self.fc(src))
        #Add & Norm中的Add
        src += residual
        src = self.layer_norm(src)
        # q的shape为(batch_size, len_q, 512)
        # attn的shape为(batch_size, n_head, len_q, len_k)
        
        #FCN
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        #Add & Norm
        src = self.layer_norm(src + self.dropout2(src2))

        src = self.linear(src)
        return src,attn

#TranAD
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src,src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class ComputeLoss:
    def __init__(self, model, lambda_energy, lambda_cov, device, n_gmm):
        self.model = model
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.device = device
        self.n_gmm = n_gmm
    
    def forward(self, x, x_hat, z, gamma):
        """Computing the loss function for DAGMM."""
        reconst_loss = torch.mean((x-x_hat).pow(2))

        sample_energy, cov_diag = self.compute_energy(z, gamma)

        loss = reconst_loss + self.lambda_energy * sample_energy + self.lambda_cov * cov_diag
        return Variable(loss, requires_grad=True)
    
    def compute_energy(self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True):
        """Computing the sample energy function"""
        if (phi is None) or (mu is None) or (cov is None):
            phi, mu, cov = self.compute_params(z, gamma)

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        eps = 1e-12
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for k in range(self.n_gmm):
            cov_k = cov[k] + (torch.eye(cov[k].size(-1))*eps).to(self.device)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag += torch.sum(1 / cov_k.diag())
        
        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).to(self.device)

        E_z = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        E_z = torch.exp(E_z)
        E_z = -torch.log(torch.sum(phi.unsqueeze(0)*E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)
        if sample_mean==True:
            E_z = torch.mean(E_z)            
        return E_z, cov_diag

    def compute_params(self, z, gamma):
        """Computing the parameters phi, mu and gamma for sample energy function """ 
        # K: number of Gaussian mixture components
        # N: Number of samples
        # D: Latent dimension
        # z = NxD
        # gamma = NxK

        #phi = D
        phi = torch.sum(gamma, dim=0)/gamma.size(0) 

        #mu = KxD
        mu = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        mu /= torch.sum(gamma, dim=0).unsqueeze(-1)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        
        #cov = K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0)
        cov /= torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov
        
class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.cholesky(a, False)
        ctx.save_for_backward(l)
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s