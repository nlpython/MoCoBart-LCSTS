# -*- coding: utf-8 -*-
"""
   @Author: YRUNS
   @Time:   2022/7/18
"""
from torch import nn
import torch
from transformers import BartForConditionalGeneration

class MoCoBart(nn.Module):

    def __init__(self, checkpoint, K=1024, m=0.999, T=0.07, mlp=False, pooling='last-avg'):
        super(MoCoBart, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.mlp = mlp
        self.pooling = pooling

        self.bart_q = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.bart_k = BartForConditionalGeneration.from_pretrained(checkpoint)

        if mlp:
            dim_mlp = 768
            self.mlp_q = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_mlp))
            self.mlp_k = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_mlp))

        for param_q, param_k in zip(self.bart_q.parameters(), self.bart_k.parameters()):
            param_k.data.copy_(param_q.data)    # initialize
            param_k.requires_grad = False       # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(768, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.bart_q.parameters(), self.bart_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr+batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K   # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, past_key_values=None):
        return self.bart_q(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                           decoder_attention_mask=decoder_attention_mask, past_key_values=past_key_values)

    def save_pretrained(self, output_dir):
        model_to_save = self.bart_q.module if hasattr(self.bart_q, 'module') else self.bart_q
        return model_to_save.save_pretrained(output_dir)


    def forward(self, X, y_q):
        """
        X: [B, C_L]
        y_q: [B, L]
        y_k: [B, L]
        """
        q_output = self.bart_q(X, attention_mask=X > 0, decoder_input_ids=y_q, decoder_attention_mask=y_q > 0,
                        output_hidden_states=True)

        q = q_output['decoder_hidden_states'][-1]
        mle_logits = q_output['logits']

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.bart_k(X, attention_mask=X > 0, decoder_input_ids=y_q, decoder_attention_mask=y_q > 0,
                            output_hidden_states=True)['encoder_last_hidden_state']   # [B, L, H]

        # average the vectors
        if self.pooling == 'last-avg':   # average pooling
            q = q.transpose(1, 2)    # [batch, 768, seqlen]
            q = torch.avg_pool1d(q, kernel_size=q.shape[-1]).squeeze(-1) # [batch, 768]
            k = k.transpose(1, 2)
            k = torch.avg_pool1d(k, kernel_size=k.shape[-1]).squeeze(-1)
        elif self.pooling == 'last-max': # max pooling
            q = q.transpose(1, 2)
            q = torch.max_pool1d(q, kernel_size=q.shape[-1]).squeeze(-1)
            k = k.transpose(1, 2)
            k = torch.max_pool1d(k, kernel_size=k.shape[-1]).squeeze(-1)


        if self.mlp:
            q = self.mlp_q(q)
            k = self.mlp_k(k)

        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Bx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: BxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        cl_logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        cl_logits /= self.T

        # labels: positive key indicators
        cl_labels = torch.zeros(cl_logits.shape[0], dtype=torch.long, device=X.device)

        # dequeue and enqueue
        if self.training:
            self._dequeue_and_enqueue(k)

        return mle_logits, cl_logits, cl_labels





