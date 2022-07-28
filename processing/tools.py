import os
import json
import pickle
import torch
from tqdm import tqdm
import numpy as np
import random
import re
import torch.nn.functional as F

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def clean(text):
    text = text.replace('\n', '').replace('\r', '').replace('\t', '').replace('\\', '')

    # remove [...]
    text = re.sub(r'\[.*?\]', '', text)
    # remove 【...】
    text = re.sub(r'【.*?】', '', text)
    # remove （...）
    text = re.sub(r'（.*?）', '', text)
    # remove (...)
    text = re.sub(r'\(.*?\)', '', text)

    return text

def top_k_top_p_filtering(logits, top_k, top_p, filter_value=-float("Inf")):
    """
    top_k或top_p解码策略，仅保留top_k个或累积概率到达top_p的标记，其他标记设为filter_value，后续在选取标记的过程中会取不到值设为无穷小。
    Args:
        logits: 预测结果，即预测成为词典中每个词的分数
        top_k: 只保留概率最高的top_k个标记
        top_p: 只保留概率累积达到top_p的标记
        filter_value: 过滤标记值

    Returns:

    """
    # logits的维度必须为2，即size:[batch_size, vocab_size]
    assert logits.dim() == 2
    # 获取top_k和字典大小中较小的一个，也就是说，如果top_k大于字典大小，则取字典大小个标记
    top_k = min(top_k, logits[0].size(-1))
    # 如果top_k不为0，则将在logits中保留top_k个标记
    if top_k > 0:
        # 由于有batch_size个预测结果，因此对其遍历，选取每个预测结果的top_k标记
        for logit in logits:
            indices_to_remove = logit < torch.topk(logit, top_k)[0][..., -1, None]
            logit[indices_to_remove] = filter_value
    # 如果top_p不为0，则将在logits中保留概率值累积达到top_p的标记
    if top_p > 0.0:
        # 对logits进行递减排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        # 对排序后的结果使用softmax归一化，再获取累积概率序列
        # 例如：原始序列[0.1, 0.2, 0.3, 0.4]，则变为：[0.1, 0.3, 0.6, 1.0]
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # 删除累积概率高于top_p的标记
        sorted_indices_to_remove = cumulative_probs > top_p
        # 将索引向右移动，使第一个标记也保持在top_p之上
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for index, logit in enumerate(logits):
            # 由于有batch_size个预测结果，因此对其遍历，选取每个预测结果的累积概率达到top_p的标记
            indices_to_remove = sorted_indices[index][sorted_indices_to_remove[index]]
            logit[indices_to_remove] = filter_value
    return logits

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, fmt=':f'):
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.mle_loss = 0
        self.cl_loss = 0
        self.loss = 0
        self.count = 0

    def update(self, mle_loss, cl_loss, loss, n=1):
        self.mle_loss += mle_loss
        self.cl_loss += cl_loss
        self.loss += loss
        self.count += n

    def get_loss(self):
        return self.mle_loss / self.count, self.cl_loss / self.count, self.loss / self.count

    def __str__(self):
        fmtstr = '{val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

