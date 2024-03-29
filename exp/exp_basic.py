import os
import torch
from models import MPLinear
from data_provider.data_factory import data_provider
import numpy as np
from collections import Counter

def top_k_frequent_elements(nums, k):
    count = Counter(nums)
    top_k = count.most_common(k)
    top_k_elements = [x[0] for x in top_k]

    return top_k_elements

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'MPLinear': MPLinear
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:


            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices

            # os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
            # os.environ["CUDA_VISIBLE_DEVICES"]='1'
            # device = torch.device('cuda:{}'.format(self.args.gpu))
            device = torch.device('cuda')
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    def get_patch(self):
        train_data, train_loader = self._get_data(flag='train')
        res_list = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            # 傅里叶变换
            ft = np.fft.rfft(batch_x, axis=1)
            # 频率f
            freqs = np.fft.rfftfreq(batch_x.shape[1], 1)
            mags = abs(ft).mean(0).mean(-1)
            # 寻找极大值对应的尖峰
            inflection = np.diff(np.sign(np.diff(mags)))
            peaks = (inflection < 0).nonzero()[0] + 1
            for _ in range(self.args.top_k):
                if len(peaks) > 0:
                    max_index = np.argmax(mags[peaks])
                    max_peak = peaks[max_index]
                    # 寻找peak对应的频率signal_freq
                    signal_freq = freqs[max_peak]
                    # 周期与频率的关系：period=1/signal_freq
                    period = int(1 / signal_freq)
                    res_list.append(period)
                    #top_k_peaks.append(max_peak)
                    peaks = np.delete(peaks, max_index)# 从 peaks 列表中删除已经找到的最大值

        self.args.patch_list = top_k_frequent_elements(res_list, self.args.top_k)
        if (len(self.args.patch_list)<self.args.top_k):
            self.args.top_k=len(self.args.patch_list)
        print(self.args.patch_list)

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
