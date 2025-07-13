# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch


class MyAutoModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(MyAutoModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.loss_weight = args.loss_weight

    def forward(self, input_ids=None, labels=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]

        logits = outputs
        prob = torch.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = self.loss_weight * torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (
                    1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob


class MyT5Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(MyT5Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.loss_weight = args.loss_weight

    def forward(self, input_ids=None, labels=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(0))[0]

        logits = outputs
        prob = torch.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = self.loss_weight * torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (
                    1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob


class MyBGEModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(MyBGEModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.loss_weight = args.loss_weight

    def forward(self, input_ids=None, labels=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(0))[0]

        logits = outputs
        prob = torch.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = self.loss_weight * torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (
                    1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob
