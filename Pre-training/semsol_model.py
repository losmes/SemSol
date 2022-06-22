from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
from tqdm import tqdm, trange


import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from transformers import BertTokenizer,BertConfig
from transformers import BertForPreTraining
from transformers import AdamW
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import pickle

from torch.utils.data import Dataset
import random
from setproctitle import setproctitle

class GetTopic(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.bilstm = nn.LSTM(
                            input_size = config.hidden_size,
                            hidden_size = int(config.hidden_size / 2),
                            num_layers = 1,
                            dropout = 0,
                            batch_first = True,
                            bidirectional = True,
                            )
        self.topic_weights = nn.Parameter(torch.Tensor(np.random.normal(size=(args.topic_num, config.hidden_size))))
        self.topic_bias = nn.Parameter(torch.Tensor(np.zeros(args.topic_num)))
        self.topic_table = nn.Parameter(torch.Tensor(np.random.normal(size=(args.topic_num, config.hidden_size))))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs_embeds, utter_ids):
        batch = np.shape(inputs_embeds)[0]
        seq_length = np.shape(inputs_embeds)[1]
        width = np.shape(inputs_embeds)[2]

        hidden_tensor, (h_0, c_0) = self.bilstm(inputs_embeds)
        
        context_max = torch.max(utter_ids.reshape([-1]))
        topic_data = []
        for i in range(1, context_max + 1):
            utterance_mask = torch.eq(utter_ids, -i)
            utterance_mask = utterance_mask.unsqueeze(-1)
            utterance_mask = torch.tile(utterance_mask, [1, 1, width])

            token_mask = torch.eq(utter_ids, i)
            token_mask = token_mask.unsqueeze(-1)
            token_mask = torch.tile(token_mask, [1, 1, width])

            t_input = torch.where(token_mask, hidden_tensor, torch.zeros_like(hidden_tensor))
            mp, _ = torch.max(t_input, dim=1, keepdim=False)

            logits = torch.matmul(mp, self.topic_weights.permute(1, 0))
            logits = logits + self.topic_bias
            probs = self.softmax(logits)

            utterance_embedding = torch.matmul(probs, self.topic_table)
            utterance_embedding = utterance_embedding.unsqueeze(1)
            utterance_embedding = torch.tile(utterance_embedding, [1, seq_length, 1])

            token_mp = torch.where(token_mask, 
                                   utterance_embedding, 
                                   torch.zeros_like(utterance_embedding))
            utterance_embedding = torch.where(utterance_mask, 
                                              utterance_embedding, 
                                              torch.zeros_like(utterance_embedding))
            topic_data.append(token_mp + 2 * utterance_embedding)
        topic_data = torch.stack(topic_data)
        topic_embedding = torch.sum(topic_data, dim=0)

        return topic_embedding

class PassageRankerNoun(nn.Module):
    def __init__(self, example_len, width):
        super().__init__()

        self.dense_0 = nn.Linear(width, 1)
        self.dense_1 = nn.Linear(example_len, 1)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, embeds_list, mask):
        out_put_list = []
        for embeds in embeds_list:
            example_list = []
            semantic_mp_list = []
            for emb in embeds:
                batch_first_embeds = emb.permute(1, 0, 2)
                b_f_embeds = self.dense_0(batch_first_embeds)
                b_f_embeds = self.dense_1(b_f_embeds.permute(0, 2, 1))
                b_f_embeds = torch.squeeze(b_f_embeds, dim=-1)
                example_list.append(b_f_embeds)

                mp, _ = torch.max(batch_first_embeds, dim=1, keepdim=False)
                semantic_mp_list.append(mp)
            examples = torch.stack(example_list)
            examples = examples.permute(1, 0, 2)
            examples = self.sigmoid(examples)
            examples = self.softmax(examples) # [batch, example_num, 1]

            semantic_mp = torch.stack(semantic_mp_list)
            semantic_mp = semantic_mp.permute(1, 0, 2) # [batch, example_num, width]

            semantic_emd = torch.mul(semantic_mp, examples)

            out_put_list.append(semantic_emd)
        out_put = torch.stack(out_put_list)
        out_put = out_put.permute(1, 0, 2, 3)

        o_p_s = np.shape(out_put)
        m_s = np.shape(mask)

        out_put = torch.reshape(out_put, (o_p_s[0], o_p_s[1], o_p_s[2], 1, o_p_s[3]))
        out_put = torch.tile(out_put, [1, 1, 1, m_s[2], 1])

        mask_re = torch.reshape(mask, (m_s[0], m_s[1], 1, m_s[2], 1))
        mask_re = torch.tile(mask_re, [1, 1, o_p_s[2], 1, 1])

        out_put = torch.mul(out_put, mask_re)
        out_put = torch.sum(out_put, dim=2)
        out_put = torch.sum(out_put, dim=1) # [batch, len, width]
        return out_put

class PassageRankerAdj(nn.Module):
    def __init__(self, example_len, width):
        super().__init__()

        self.dense_0 = nn.Linear(width, 1)
        self.dense_1 = nn.Linear(example_len, 1)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, embeds_list, mask):
        out_put_list = []
        for embeds in embeds_list:
            example_list = []
            semantic_mp_list = []
            for emb in embeds:
                batch_first_embeds = emb.permute(1, 0, 2)
                b_f_embeds = self.dense_0(batch_first_embeds)
                b_f_embeds = self.dense_1(b_f_embeds.permute(0, 2, 1))
                b_f_embeds = torch.squeeze(b_f_embeds, dim=-1)
                example_list.append(b_f_embeds)

                mp, _ = torch.max(batch_first_embeds, dim=1, keepdim=False)
                semantic_mp_list.append(mp)
            examples = torch.stack(example_list)
            examples = examples.permute(1, 0, 2)
            examples = self.sigmoid(examples)
            examples = self.softmax(examples) # [batch, example_num, 1]

            semantic_mp = torch.stack(semantic_mp_list)
            semantic_mp = semantic_mp.permute(1, 0, 2) # [batch, example_num, width]

            semantic_emd = torch.mul(semantic_mp, examples)

            out_put_list.append(semantic_emd)
        out_put = torch.stack(out_put_list)
        out_put = out_put.permute(1, 0, 2, 3)

        o_p_s = np.shape(out_put)
        m_s = np.shape(mask)

        out_put = torch.reshape(out_put, (o_p_s[0], o_p_s[1], o_p_s[2], 1, o_p_s[3]))
        out_put = torch.tile(out_put, [1, 1, 1, m_s[2], 1])

        mask_re = torch.reshape(mask, (m_s[0], m_s[1], 1, m_s[2], 1))
        mask_re = torch.tile(mask_re, [1, 1, o_p_s[2], 1, 1])

        out_put = torch.mul(out_put, mask_re)
        out_put = torch.sum(out_put, dim=2)
        out_put = torch.sum(out_put, dim=1) # [batch, len, width]
        return out_put

class SemSol(nn.Module):
    def __init__(self, args, args_bert_model, tokenizer_len=0):
        super().__init__()

        bertconfig = BertConfig.from_pretrained(args_bert_model)
        bertconfig.output_hidden_states = True
        self.bert = BertForPreTraining.from_pretrained(args_bert_model, config=bertconfig)
        self.bert.resize_token_embeddings(tokenizer_len)
        self.bert.cls.seq_relationship = nn.Linear(bertconfig.hidden_size, 3)
        self.word_embedding = nn.Parameter(self.bert.get_input_embeddings().weight.clone())

        bertconfig_1 = BertConfig.from_pretrained(args_bert_model)
        self.bert_1 = BertForPreTraining.from_pretrained(args_bert_model, config=bertconfig_1)
        self.bert_1.resize_token_embeddings(tokenizer_len)
        self.bert_1.cls.seq_relationship = nn.Linear(bertconfig_1.hidden_size, 3)
        self.word_embedding_1 = nn.Parameter(self.bert_1.get_input_embeddings().weight.clone())

        self.topic = GetTopic(args, bertconfig)
        # TODO batch_firstはtorch=1.9.0からしか使えない
        self.self_attention_common = nn.MultiheadAttention(embed_dim=bertconfig.hidden_size,
                                                           num_heads=1,
                                                           bias=False)

        self.sourcetarget_attention_noun = nn.MultiheadAttention(embed_dim=bertconfig.hidden_size,
                                                                 num_heads=1,
                                                                 bias=False)

        self.sourcetarget_attention_adj = nn.MultiheadAttention(embed_dim=bertconfig.hidden_size,
                                                                num_heads=1,
                                                                bias=False)

        
        self.passage_ranker_noun = PassageRankerNoun(args.max_semantic_word_length, bertconfig.hidden_size)
        self.passage_ranker_adj = PassageRankerAdj(args.max_semantic_word_length, bertconfig.hidden_size)

        self.args = args

    def get_word_embedding(self, input_ids, word_type=0):
        shape = np.shape(input_ids)
        batch = shape[0]
        seq_length = shape[1]

        if word_type == 0:
          embedding_table = self.word_embedding
        else:
          embedding_table = self.word_embedding_1

        width = np.shape(embedding_table)[-1]

        flat_input_ids = input_ids.reshape([-1])
        inputs_embeds = embedding_table[flat_input_ids]

        inputs_embeds = inputs_embeds.reshape([batch, seq_length, width])
        return inputs_embeds

    def forward(self, input_ids, input_mask, segment_ids, lm_label_ids, is_next, utter_ids,
                mask_noun, masks_noun, ids_noun,
                mask_adj, masks_adj, ids_adj,
                masks_noun_ids, masks_adj_ids):

        inputs_embeds = self.get_word_embedding(input_ids)
        if self.args.use_topic:
            topic_embeds = self.topic(inputs_embeds, utter_ids)
            inputs_embeds = inputs_embeds + topic_embeds

        _, _, context_output_hidden = self.bert(inputs_embeds=inputs_embeds, 
                                                attention_mask=input_mask, 
                                                token_type_ids=segment_ids)
        last_heads = context_output_hidden[-1]

        if self.args.use_semantic:
            # NOUN self attention
            if 0 != np.shape(ids_noun)[1]:
                ids_noun = ids_noun.permute(1, 2, 0, 3) #[num, example_num, batch, semantic_len]
                noun_embeds_list = []
                for i in range(np.shape(ids_noun)[0]):
                    example_list = []
                    for j in range(np.shape(ids_noun[i])[0]):
                        example_embeds = self.get_word_embedding(ids_noun[i][j], word_type=1) #[batch, semantic_len, width]
    
                        example_embeds = example_embeds.permute(1,0,2) #[semantic_len, batch, width]
                        self_attn_output, attn_weights = self.self_attention_common(#[semantic_len, batch, width]
                                                                               example_embeds,
                                                                               example_embeds, 
                                                                               example_embeds)
                        sourcetarget_attn_output, _ = self.sourcetarget_attention_noun( #[semantic_len, batch, width]
                                                                                       self_attn_output,
                                                                                       context_output_hidden[-1],
                                                                                       context_output_hidden[-1])
                        example_list.append(sourcetarget_attn_output)
                    noun_embeds_list.append(example_list)
                passage_ranker_noun_embeds = self.passage_ranker_noun(noun_embeds_list, mask_noun)
                last_heads = last_heads + passage_ranker_noun_embeds
    
    
            # ADJ self attention
            if 0 != np.shape(ids_adj)[1]:
                ids_adj = ids_adj.permute(1, 2, 0, 3) #[num, example_num, batch, semantic_len]
                adj_embeds_list = []
                for i in range(np.shape(ids_adj)[0]):
                    example_list = []
                    for j in range(np.shape(ids_adj[i])[0]):
                        example_embeds = self.get_word_embedding(ids_adj[i][j], word_type=1) #[batch, semantic_len, width]
    
                        example_embeds = example_embeds.permute(1,0,2) #[semantic_len, batch, width]
                        self_attn_output, _ = self.self_attention_common(example_embeds, #[semantic_len, batch, width]
                                                                         example_embeds, 
                                                                         example_embeds)
                        sourcetarget_attn_output, _ = self.sourcetarget_attention_adj( #[semantic_len, batch, width]
                                                                                      self_attn_output,
                                                                                      context_output_hidden[-1],
                                                                                      context_output_hidden[-1])
        
                        example_list.append(example_embeds)
                    adj_embeds_list.append(example_list)
                passage_ranker_adj_embeds = self.passage_ranker_adj(adj_embeds_list, mask_adj)
                last_heads = last_heads + passage_ranker_adj_embeds

        prediction_scores, seq_relationship_score, _ = self.bert(inputs_embeds=last_heads, 
                                                                 attention_mask=input_mask, 
                                                                 token_type_ids=segment_ids)
        return prediction_scores, seq_relationship_score

