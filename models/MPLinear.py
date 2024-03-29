import torch
from torch import nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class PatchEmbedding(nn.Module):
    def __init__(self,configs, patch_num,c_in,d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.d_model=d_model

        #MLP_Time
        self.MLP_Time = nn.Sequential(
            nn.BatchNorm1d(c_in),
            nn.Linear(patch_len, patch_len),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(patch_len, patch_len),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # MLP_Feature
        self.MLP_Feature = nn.Sequential(
            nn.BatchNorm1d(patch_len),
            nn.Linear(c_in, c_in),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(c_in, c_in),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(d_model, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        B=x.shape[0]
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.permute(0, 2, 3, 1)
        patch_num=x.shape[1]
        x = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))
        x = x.permute(0, 2, 1)  # b*patch_num x C x patch_len
        x = x+self.MLP_Time(x)
        x = x.permute(0, 2, 1)  # b*patch_num x patch_len x C
        x = x + self.MLP_Feature(x)
        x = torch.reshape(x, (B, patch_num,-1))
        return x, n_vars
    
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Multiple Series decomposition block from FEDformer
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.series_decomp = [series_decomp(kernel) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.series_decomp:
            sea, moving_avg = func(x)
            moving_mean.append(moving_avg)
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class PatchBlock(nn.Module):
    def __init__(self, configs,patch_len=10):
        super(PatchBlock, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = configs.stride

        self.patch_num = int((configs.seq_len - patch_len) / configs.stride + 2)
        self.patch_embedding = PatchEmbedding(configs,self.patch_num,configs.enc_in,
            configs.d_model, patch_len, configs.stride, padding, configs.dropout)
        decomp_kernel = []  # kernel of decomposition operation
        for ii in configs.conv_kernel:
            if ii % 2 == 0:  # the kernel of decomposition operation must be odd
                decomp_kernel.append(ii + 1)
            else:
                decomp_kernel.append(ii)
        self.decomp_multi = series_decomp_multi(decomp_kernel)
        self.Linear_Seasonal=nn.Sequential(
            nn.Linear(self.patch_num, self.patch_num)
        )

        self.Linear_Trend=nn.Sequential(
            nn.Linear(self.patch_num, self.patch_num)
        )

        self.Linear_Seasonal.weight = nn.Parameter(
            (1 / self.patch_num) * torch.ones([self.patch_num, self.patch_num]))

        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.patch_num) * torch.ones([self.patch_num, self.patch_num]))

        #MLP
        self.MLP_Seq=nn.Sequential(
            nn.Linear(self.patch_num * patch_len, configs.d_model),
            nn.Linear(configs.d_model, configs.d_model),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(configs.dropout1),
            nn.Linear(configs.d_model, configs.d_model),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(configs.dropout1),
            nn.Linear(configs.d_model, self.pred_len)
        )

    def forward(self, x):
        enc_out, n_vars = self.patch_embedding(x)
        enc_out=self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (enc_out.shape[0], -1,n_vars)) #bxpatch_num*patch_len*c
        enc_out=enc_out.permute(0, 2, 1)
        dec_out=self.MLP_Seq(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        return dec_out

    def encoder(self, x):
        seasonal_init, trend_init = self.decomp_multi(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)



class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.top_k=configs.top_k

        self.model = nn.ModuleList([PatchBlock(configs,patch_len=patch)for patch in configs.patch_list])
        self.weights = nn.Parameter(torch.ones(self.top_k), requires_grad=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x_enc = x_enc.permute(0, 2, 1)
        res=[]
        for i in range(self.top_k):
            out=self.model[i](x_enc)
            res.append(out)
        res = torch.stack(res, dim=-1)
        res = torch.sum(res * self.weights.view(1,1,1,self.top_k),dim=-1)

        return res[:, -self.pred_len:, :]  # [B, L, D]