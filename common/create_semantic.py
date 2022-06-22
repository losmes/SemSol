# coding=utf-8
import re
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn
import sys
import spacy


class CreateSemantic(object):
    def __init__(self, args):
        # semantic lang
        # English : eng
        # Chinese : cmn
        self.semantic_lang = args.semantic_lang

        if self.semantic_lang == "eng":
            self.nlp = spacy.load('en_core_web_sm')
        elif self.semantic_lang == "cmn":
            self.nlp = spacy.load('zh_core_web_sm')
        else:
            exit()

        # semantic type
        # 0:definition               40
        # 1:definition + example     70
        # 2:definition + #           55
        # 3:definition + example + # 85
        self.semantic_type = args.semantic_type
        self.max_semantic_word_length = args.max_semantic_word_length

        self.noun_num = args.noun_num
        self.noun_example_num = args.noun_example_num
        self.adj_num = args.adj_num
        self.adj_example_num = args.adj_example_num

    def make_sentence(self, tokens, token_utter_ids):
        new_sentence = []
        token_ids = []
        token_idx_dict = {}
        # 1:[']flag
        # 2:[##]flag
        flag1 = 0
        flag2 = 0
    
        for idx, token in enumerate(tokens):
            if re.match(r'\[.*\]$', token):
                flag1 = 0
                flag2 = 0
                continue
    
            # [']処理
            # 連結
            if flag1 == 1:
                new_sentence[-1] = new_sentence[-1] + "'" + token
                tmp_ids.append(idx)
                token_ids[-1] = tmp_ids
                tmp = []
                for i in token_ids[-1]:
                    if type(i) == list:
                        for j in i:
                            tmp.append(j)
                    else:
                        tmp.append(i)
                token_ids[-1] = tmp
                if new_sentence[-1] in token_idx_dict:
                    tmp = []
                    if type(token_idx_dict[new_sentence[-1]]) == list:
                        for j in token_idx_dict[new_sentence[-1]]:
                            tmp.append(j)
                    else:
                        tmp = [token_idx_dict[new_sentence[-1]], idx]
                    token_idx_dict[new_sentence[-1]] = tmp
                else:
                    token_idx_dict[new_sentence[-1]] = tmp
                flag1 = 0
                flag2 = 0
            # [']の前が[MASK]の時の処理
            elif flag1 == 2:
                flag1 = 0
                continue
            # [PAD] 処理
            elif token == "[PAD]":
                if token_utter_ids[idx] < 0 and len(new_sentence):
                    if new_sentence[-1] != '.':
                        new_sentence.append('.')
                        token_ids.append(idx)
                        if '.' in token_idx_dict:
                            tmp = []
                            if type(token_idx_dict['.']) == list:
                                for j in token_idx_dict['.']:
                                    tmp.append(j)
                            else:
                                tmp = [token_idx_dict['.'], idx]
                            token_idx_dict['.'] = tmp
                        else:
                            token_idx_dict['.'] = idx
                        flag2 = 0
            # 前方が[MASK][PAD]の場合はスルー
            elif re.match("##", token):
                tmp_token = token[2:]
                if tokens[idx-1] != "[MASK]" and tokens[idx-1] != "[PAD]" and len(new_sentence) and flag2 == 0:
                    new_sentence[-1] = new_sentence[-1] + token[2:]
                    token_ids[-1] = [token_ids[-1], idx]
                    tmp = []
                    for i in token_ids[-1]:
                        if type(i) == list:
                            for j in i:
                                tmp.append(j)
                        else:
                            tmp.append(i)
                    token_ids[-1] = tmp
                    if new_sentence[-1] in token_idx_dict:
                            tmp = []
                            if type(token_idx_dict[new_sentence[-1]]) == list:
                                for j in token_idx_dict[new_sentence[-1]]:
                                    tmp.append(j)
                            else:
                                tmp = [token_idx_dict[new_sentence[-1]], idx]
                            token_idx_dict[new_sentence[-1]] = tmp
                    else:
                        token_idx_dict[new_sentence[-1]] = tmp
                else:
                    flag2 = 1
            # [']処理
            # flag切り替え
            elif token == '\'':
                if tokens[idx-1] != "[MASK]" and tokens[idx-1] != "[PAD]" and len(new_sentence):
                    tmp_ids = [token_ids[-1], idx]
                    flag1 = 1
                    flag2 = 0
                else:
                    flag1 = 2
            # 通常処理
            else:
                new_sentence.append(token)
                token_ids.append(idx)
                if token in token_idx_dict:
                    tmp = []
                    if type(token_idx_dict[token]) == list:
                        for j in token_idx_dict[token]:
                            tmp.append(j)
                    else:
                        tmp = [token_idx_dict[token], idx]
                    token_idx_dict[token] = tmp
                else:
                    token_idx_dict[token] = idx
                flag2 = 0
    
        return ' '.join(new_sentence), token_idx_dict


    def semantic_proc(self, tokenizer, sentence, lmms_tokens_dict, tokens, token_len):
        mask_noun = [[0] * token_len] * self.noun_num
        ids_noun = [[[0] * self.max_semantic_word_length] * self.noun_example_num] * self.noun_num
        masks_noun = [[[0] * self.max_semantic_word_length] * self.noun_example_num] * self.noun_num
        semantic_masks_noun_ids = [[[0] * self.max_semantic_word_length] * self.noun_example_num] * self.noun_num
        mask_adj = [[0] * token_len] * self.adj_num
        ids_adj = [[[0] * self.max_semantic_word_length] * self.adj_example_num] * self.adj_num
        masks_adj = [[[0] * self.max_semantic_word_length] * self.adj_example_num] * self.adj_num
        semantic_masks_adj_ids = [[[0] * self.max_semantic_word_length] * self.adj_example_num] * self.adj_num
    
        if not len(sentence):
            return mask_noun, ids_noun, masks_noun, \
                   mask_adj, ids_adj, masks_adj, \
                   semantic_masks_noun_ids, semantic_masks_adj_ids
    
        noun_token_list, noun_list, noun_idx_list, \
                adj_token_list, adj_list, adj_idx_list = self.wordnet_proc(sentence)
    
        tokens_noun_list, ids_noun_list, \
                semantic_masks_noun_ids = self.semantic_tokenizer(tokenizer, noun_list, 
                                                             self.noun_num, self.noun_example_num)

        tokens_adj_list, ids_adj_list, \
                semantic_masks_adj_ids = self.semantic_tokenizer(tokenizer, adj_list, 
                                                            self.adj_num, self.adj_example_num)
    
        mask_noun, ids_noun, masks_noun = self.make_semantic_mask(lmms_tokens_dict, noun_token_list, 
                                                             tokens, tokens_noun_list, ids_noun_list, 
                                                             token_len, self.noun_num, self.noun_example_num)

        mask_adj, ids_adj, masks_adj = self.make_semantic_mask(lmms_tokens_dict, adj_token_list, 
                                                          tokens, tokens_adj_list, ids_adj_list, 
                                                          token_len, self.adj_num, self.adj_example_num)
    
        return mask_noun, ids_noun, masks_noun, \
               mask_adj, ids_adj, masks_adj, semantic_masks_noun_ids, semantic_masks_adj_ids

    def calc_histogram(self, freq_list, threshold):
        if threshold < 1:
            return -1
    
        sorted_list = sorted(freq_list, reverse=True)
        cut = sum(freq_list) * float((100 - threshold) / 100)
    
        sum_ = 0
        for s in sorted_list:
            sum_ += s
            if sum_ > cut:
                break
    
        return s

    def wordnet_proc(self, sentence):
        noun_token_list = []
        noun_list = []
        noun_idx_list = []
        adj_token_list = []
        adj_list = []
        adj_idx_list = []
    
        doc = self.nlp(sentence)
        sent_info = {'tokens': [], 'pos': []}
        for tok in doc:
            sent_info['tokens'].append(tok.text.replace(' ', '_'))
            sent_info['pos'].append(tok.pos_)
    
        for idx, (tok, pos) in enumerate(zip(sent_info['tokens'], sent_info['pos'])):
            if pos == 'NOUN' or pos == 'ADJ':
                if pos == 'NOUN':
                    pos_str = wn.NOUN
                elif pos == 'ADJ':
                    pos_str = wn.ADJ
    
                try:
                    syns = wn.synsets(tok, pos=pos_str, lang=self.semantic_lang)
                except:
                    continue
    
                tmp_list = []
                tmp_dict = {'lemma': '', 'definition': '', 'examples': '', 'max_count': -1}
                if pos == 'NOUN':
                    example_num = self.noun_example_num
                elif pos == 'ADJ':
                    example_num = self.adj_example_num
                for _ in range(example_num):
                    tmp_list.append(tmp_dict)
    
                if len(syns) < 1:
                    continue
    
                for syn in syns:
                    max_count = 0
                    for sl in syn.lemmas():
                        if max_count < sl.count():
                            max_count = sl.count()
    
                    tmp_dict = {}
                    tmp_dict['lemma'] = syn.lemmas()[0].key()
                    tmp_dict['definition'] = syn.definition()
                    tmp_dict['examples'] = (', '.join(syn.examples()))
                    tmp_dict['max_count'] = max_count
                    tmp_list.append(tmp_dict)
    
                tmp_list_sorted = sorted(tmp_list, reverse=True, key=lambda x:x['max_count'])
                del tmp_list_sorted[example_num:]
    
                if pos == 'NOUN':
                    noun_token_list.append(tok)
                    noun_list.append(tmp_list_sorted)
                    noun_idx_list.append(idx)
                elif pos == 'ADJ':
                    adj_token_list.append(tok)
                    adj_list.append(tmp_list_sorted)
                    adj_idx_list.append(idx)
    
        # 削除処理
        # noun_num:10
        # adj_num:5
        while len(noun_idx_list) > self.noun_num:
            del noun_token_list[0]
            del noun_idx_list[0]
            del noun_list[0]
    
        while len(adj_idx_list) > self.adj_num:
            del adj_token_list[0]
            del adj_idx_list[0]
            del adj_list[0]
    
        return noun_token_list, noun_list, noun_idx_list, adj_token_list, adj_list, adj_idx_list

    def semantic_tokenizer(self, tokenizer, wordnet_list, word_num, example_num):
        tokens_list = []
        masks_ids = []
    
        # semantic_mask_ids
        # 0 [UNK]
        # 1 [CLS]
        # 2 [sep]
        # 3 定義のID
        # 4 例文のID
        # 5 ＃のID
        for wordnet in wordnet_list:
            tokens = []
            mask_ids = []
            for word in wordnet:
                tmp_tokens = []
                tmp_ids = []
                if word['max_count'] == -1:
                    for _ in range(self.max_semantic_word_length):
                        tmp_tokens.append("[UNK]")
                        tmp_ids.append(0)
                else:
                    tmp_tokens.append("[CLS]")
                    tmp_ids.append(1)
    
                    lemma = tokenizer.tokenize(word['lemma'])
                    definition = tokenizer.tokenize(word['definition'])
                    examples = tokenizer.tokenize(word['examples'])
    
                    if self.semantic_type == 0:
                        for token in definition:
                            tmp_tokens.append(token)
                            tmp_ids.append(3)
                    elif self.semantic_type == 1:
                        for token in definition:
                            tmp_tokens.append(token)
                            tmp_ids.append(3)
                        tmp_tokens.append("[SEP]")
                        tmp_ids.append(2)
                        for token in examples:
                            tmp_tokens.append(token)
                            tmp_ids.append(4)
                    elif self.semantic_type == 2:
                        for token in definition:
                            tmp_tokens.append(token)
                            tmp_ids.append(3)
                        tmp_tokens.append("[SEP]")
                        tmp_ids.append(2)
                        for token in lemma:
                            tmp_tokens.append(token)
                            tmp_ids.append(5)
                    elif self.semantic_type == 3:
                        for token in definition:
                            tmp_tokens.append(token)
                            tmp_ids.append(3)
                        tmp_tokens.append("[SEP]")
                        tmp_ids.append(2)
                        for token in examples:
                            tmp_tokens.append(token)
                            tmp_ids.append(4)
                        tmp_tokens.append("[SEP]")
                        tmp_ids.append(2)
                        for token in lemma:
                            tmp_tokens.append(token)
                            tmp_ids.append(5)
    
                    while len(tmp_tokens) >= self.max_semantic_word_length:
                        del tmp_tokens[-1]
                        del tmp_ids[-1]
                    tmp_tokens.append("[SEP]")
                    tmp_ids.append(2)
                    for _ in range(self.max_semantic_word_length - len(tmp_tokens)):
                        tmp_tokens.append("[UNK]")
                        tmp_ids.append(0)
    
                    tmp_ids = []
                    for _ in range(self.max_semantic_word_length):
                        tmp_ids.append(0)
    
                tokens.append(tmp_tokens)
                mask_ids.append(tmp_ids)
            tokens_list.append(tokens)
            masks_ids.append(mask_ids)
        tmps = []
        tmp = [0] * self.max_semantic_word_length
        for _ in range(example_num):
            tmps.append(tmp)
        for _ in range(word_num - len(masks_ids)):
            masks_ids.append(tmps)
    
        tokens_ids = []
        for tokens in tokens_list:
            tmp_id = []
            for token in tokens:
                tmp_id.append(tokenizer.convert_tokens_to_ids(token))
            tokens_ids.append(tmp_id)
    
        return tokens_list, tokens_ids, masks_ids
   
    def make_semantic_mask(self, lmms_tokens_dict, token_list, tokens, 
                           tokens_semantic_list, ids_list, token_len, word_num, example_num):
        # make word mask
        # noun:mask_[0-9]
        # asj :mask_[0-4]
        word_mask = []
    
        for t in token_list:
            if not t in lmms_tokens_dict:
                continue
            token_idx = lmms_tokens_dict[t]
            tmp_mask = [0] * token_len
            if type(token_idx) == list:
                for i in token_idx:
                    tmp_mask[i] = 1
            else:
                tmp_mask[token_idx] = 1
            word_mask.append(tmp_mask)
    
        # 使わない部分を0埋め
        tmp_mask = [0] * token_len
        for _ in range(word_num - len(word_mask)):
            word_mask.append(tmp_mask)
    
        # make individual word mask
        # noun:mask_[0-9]_[0-3]
        # adj :mask_[0-4]_[0-3]
        # [UNK] 以外は1
        masks_list = []
        for words in tokens_semantic_list:
            masks = []
            for tokens in words:
                tmp_mask = [0] * self.max_semantic_word_length
                for idx, token in enumerate(tokens):
                    if token == '[UNK]':
                        break
                    tmp_mask[idx] = 1
                masks.append(tmp_mask)
            masks_list.append(masks)
    
        # 使わない部分を0埋め
        tmps = []
        tmp = [0] * self.max_semantic_word_length
        for _ in range(example_num):
            tmps.append(tmp)
        for _ in range(word_num - len(masks_list)):
            masks_list.append(tmps)
            ids_list.append(tmps)
    
        return word_mask, masks_list, ids_list
