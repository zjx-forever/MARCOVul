import os
import sys
from typing import List

from torch import nn
from omegaconf import DictConfig
import torch
import numpy
from gensim.models import KeyedVectors

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vocabulary import Vocabulary
from os.path import exists
from torch.nn.utils.rnn import pad_sequence

def memory_allocated_decorator(func):
    def wrapper(*args, **kwargs):
        before = torch.cuda.memory_allocated()
        result = func(*args, **kwargs)
        after = torch.cuda.memory_allocated()
        mem = (after - before) / 1024 / 1024
        print(f'{func.__name__}: {mem} MB')
        return result
    return wrapper


class RNNLayer(torch.nn.Module):
    """

    """
    __negative_value = -numpy.inf

    def __init__(self, config: DictConfig, pad_idx: int):
        super(RNNLayer, self).__init__()
        self.__pad_idx = pad_idx
        self.__config = config
        self.__rnn = nn.GRU(
            input_size=config.RNN.input_size,
            hidden_size=config.RNN.output_size,
            num_layers=config.RNN.num_layers,
            bidirectional=config.RNN.use_bi,
            dropout=config.RNN.drop_out if config.RNN.num_layers > 1 else 0,
            batch_first=True)
        
    
    def forward(self, subtokens_embed: torch.Tensor, node_ids: torch.Tensor):
        """

        Args:
            subtokens_embed: [n nodes; max len; embed dim]
            node_ids: [n nodes; max len]

        Returns:

        """
        with torch.no_grad():
            
            is_contain_pad_id, first_pad_pos = torch.max(node_ids == self.__pad_idx, dim=1)
            
            first_pad_pos[~is_contain_pad_id] = node_ids.shape[1]  
            
            sorted_path_lengths, sort_indices = torch.sort(first_pad_pos,descending=True)
            
            _, reverse_sort_indices = torch.sort(sort_indices)
            sorted_path_lengths = sorted_path_lengths.to(torch.device("cpu"))
        
        subtokens_embed = subtokens_embed[sort_indices]
        
        
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            subtokens_embed, sorted_path_lengths, batch_first=True, enforce_sorted=False)

        _, node_embedding = self.__rnn(packed_embeddings)

        node_embedding = node_embedding.sum(dim=0)

        node_embedding = node_embedding[reverse_sort_indices]

        return node_embedding


class Word2vecEmbedding(torch.nn.Module):
    def __init__(self, config: DictConfig):
        super(Word2vecEmbedding, self).__init__()
        self.__config = config

        self.vocab = Vocabulary.build_from_w2v(config.w2v.w2v_path)
        self.vocabulary_size = self.vocab.get_vocab_size()
        self.__pad_idx = self.vocab.get_pad_id()

        
        self.__wd_embedding = nn.Embedding(self.vocabulary_size,
                                           config.w2v.vector_size,
                                           padding_idx=self.__pad_idx)

        torch.nn.init.xavier_uniform_(self.__wd_embedding.weight.data)
        if exists(config.w2v.w2v_path):
            self.__add_w2v_weights(config.w2v.w2v_path, self.vocab)

        self.__rnn_attn = RNNLayer(config, self.__pad_idx)

    
    def __add_w2v_weights(self, w2v_path: str, vocab: Vocabulary):
        """
        add pretrained word embedding to embedding layer

        Args:
            w2v_path: path to the word2vec model

        Returns:

        """
        model = KeyedVectors.load(w2v_path, mmap="r")
        w2v_weights = self.__wd_embedding.weight.data
        for wd in model.index2word:
            w2v_weights[vocab.convert_token_to_id(wd)] = torch.from_numpy(model[wd])
        self.__wd_embedding.weight.data.copy_(w2v_weights)

    
    
    def forward(self, seq: torch.Tensor):
        """

        Args:
            seq: [n nodes (seqs); max len (seq len)]

        Returns:

        """
        
        
        wd_embedding = self.__wd_embedding(seq)
        
        node_embedding = self.__rnn_attn(wd_embedding, seq)
        return node_embedding
