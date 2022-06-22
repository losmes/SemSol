import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from Metrics import Metrics
import logging
from torch.utils.data import  RandomSampler
from transformers import AdamW
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.functional as F
from collections import OrderedDict
import fine_semsol_model

import sys
sys.path.append('./')
import common.create_semantic as create_semantic

FT_model={
    'ubuntu': 'bert-base-uncased',
    'douban': 'bert-base-chinese',
    'e_commerce': 'bert-base-chinese',
    'reddit10': 'bert-base-uncased'
}

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label, lenidx, utter_ids,
                 mask_noun, masks_noun, ids_noun, mask_adj, masks_adj, ids_adj, masks_noun_ids, masks_adj_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label
        self.lenidx = lenidx
        self.utter_ids = utter_ids

        self.mask_noun=mask_noun
        self.masks_noun=masks_noun
        self.ids_noun=ids_noun
        self.mask_adj=mask_adj
        self.masks_adj=masks_adj
        self.ids_adj=ids_adj
        self.masks_noun_ids=masks_noun_ids
        self.masks_adj_ids=masks_adj_ids


class BERTDataset(Dataset):
    def __init__(self, args, train, tokenizer, semantic):
        self.train = train
        self.args = args
        self.bert_tokenizer = tokenizer

        self.semantic = semantic

    def __len__(self):
        return len(self.train['cr'])

    def __getitem__(self, item):
        cur_features = convert_examples_to_features(item, self.train, self.bert_tokenizer, self.semantic)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.label, dtype=torch.float),
                       torch.tensor(cur_features.lenidx),
                       torch.tensor(cur_features.utter_ids),
                       torch.tensor(cur_features.mask_noun),
                       torch.tensor(cur_features.masks_noun),
                       torch.tensor(cur_features.ids_noun),
                       torch.tensor(cur_features.mask_adj),
                       torch.tensor(cur_features.masks_adj),
                       torch.tensor(cur_features.ids_adj),
                       torch.tensor(cur_features.masks_noun_ids),
                       torch.tensor(cur_features.masks_adj_ids)
                       )

        return cur_tensors


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop()


def convert_examples_to_features(item, train, bert_tokenizer, semantic):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    ex_index = item
    input_ids = train['cr'][item]


    sep = input_ids.index(bert_tokenizer.sep_token_id)
    context = input_ids[:sep]
    response = input_ids[sep + 1:]
    _truncate_seq_pair(context, response, 253)

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    context_len = len(context)

    input_ids = [bert_tokenizer.cls_token_id] + context + [bert_tokenizer.sep_token_id] + response + [
        bert_tokenizer.sep_token_id]

    utter_count = 1
    utter_ids = []

    a_tokens = bert_tokenizer.convert_ids_to_tokens(
            [bert_tokenizer.cls_token_id] + context + [bert_tokenizer.sep_token_id])
    for token in a_tokens:
        if "[eos]" == token:
            utter_ids.append(-utter_count)
            utter_count += 1
        else:
            utter_ids.append(utter_count)
    utter_ids[-1] = utter_ids[-1] - 1

    b_tokens = bert_tokenizer.convert_ids_to_tokens(
            response + [bert_tokenizer.sep_token_id])
    for token in b_tokens:
        if "[SEP]" == token:
            utter_ids.append(-utter_count)
            utter_count += 1
        else:
            utter_ids.append(utter_count)

    segment_ids = [0] * (context_len + 2)  # context
    segment_ids += [1] * (len(input_ids) - context_len - 2)  # #response

    lenidx = [1 + context_len, len(input_ids) - 1]

    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = 256 - len(input_ids)

    tokens = bert_tokenizer.convert_ids_to_tokens(input_ids)

    if (padding_length > 0):
        input_ids = input_ids + ([0] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([0] * padding_length)
        utter_ids = utter_ids + ([0] * padding_length)

    lmms_sentence, lmms_tokens_dict = semantic.make_sentence(tokens, utter_ids)
    mask_noun, masks_noun, ids_noun, \
    mask_adj, masks_adj, ids_adj, \
    masks_noun_ids, masks_adj_ids = semantic.semantic_proc(
                                                          bert_tokenizer,
                                                          lmms_sentence,
                                                          lmms_tokens_dict,
                                                          tokens,
                                                          len(utter_ids))

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             label=train['y'][item],
                             lenidx=lenidx,
                             utter_ids=utter_ids,
                             mask_noun=mask_noun,
                             masks_noun=masks_noun,
                             ids_noun=ids_noun,
                             mask_adj=mask_adj,
                             masks_adj=masks_adj,
                             ids_adj=ids_adj,
                             masks_noun_ids=masks_noun_ids,
                             masks_adj_ids=masks_adj_ids
                             )
    return features


class NeuralNetwork(nn.Module):

    def __init__(self, args):
        super(NeuralNetwork, self).__init__()
        self.args = args
        self.patience = 0
        self.init_clip_max_norm = 5.0
        self.optimizer = None
        self.best_result = [0, 0, 0, 0, 0, 0]
        self.metrics = Metrics(self.args.score_file_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']

        self.bert_tokenizer = BertTokenizer.from_pretrained(FT_model[args.task],do_lower_case=args.do_lower_case)
        special_tokens_dict = {'eos_token': '[eos]'}
        num_added_toks = self.bert_tokenizer.add_special_tokens(special_tokens_dict)

        self.semantic = create_semantic.CreateSemantic(self.args)

        self.model = fine_semsol_model.FineSemSol(self.args, config_class, model_class, FT_model[args.task], len(self.bert_tokenizer))

        load_parameter = torch.load(self.args.load_checkpoint, map_location=self.device)

        new_state_dict = OrderedDict()
        for k, v in load_parameter.items():
            if k.startswith('bert.'):
                k1 = k[5:]
                k1 = "bert_1." + k1
                new_state_dict[k1] = v
            new_state_dict[k] = v

        self.model.load_state_dict(state_dict=new_state_dict, strict=False)
        
        for param in self.model.bert.parameters():
            param.requires_grad = False
        #for param in self.model.bert_1.parameters():
        #    param.requires_grad = False
        for param in self.model.topic.parameters():
            param.requires_grad = False
        for param in self.model.self_attention_common.parameters():
            param.requires_grad = False
        for param in self.model.sourcetarget_attention_noun.parameters():
            param.requires_grad = False
        for param in self.model.sourcetarget_attention_adj.parameters():
            param.requires_grad = False
        for param in self.model.passage_ranker_noun.parameters():
            param.requires_grad = False
        for param in self.model.passage_ranker_adj.parameters():
            param.requires_grad = False

        self.model.word_embedding.requires_grad = False
        self.model.word_embedding_1.requires_grad = False

        self.model = self.model.cuda()

    def forward(self):
        raise NotImplementedError

    def train_step(self, i, data):
        with torch.no_grad():
            batch_ids, batch_mask, batch_seg, batch_y, batch_len, batch_u, \
            mask_noun, masks_noun, ids_noun, \
            mask_adj, masks_adj, ids_adj, \
            masks_noun_ids, masks_adj_ids = (item.cuda(device=self.device) for item in data)

        self.optimizer.zero_grad()

        output = self.model(batch_ids, batch_mask, batch_seg, batch_u,
                mask_noun, masks_noun, ids_noun, mask_adj, masks_adj, ids_adj, masks_noun_ids, masks_adj_ids)

        logits = torch.sigmoid(output[0])
        loss = self.loss_func(logits.squeeze(), target=batch_y)
        loss.backward()

        self.optimizer.step()
        if i % 100 == 0:
            print('Batch[{}] - loss: {:.6f}  batch_size:{}'.format(i, loss.item(),
                                                                   batch_y.size(0)))  
        return loss

    def fit(self, train, dev):  

        if torch.cuda.is_available(): self.cuda()

        dataset = BERTDataset(self.args, train, self.bert_tokenizer, self.semantic)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, sampler=sampler,num_workers=2)

        self.loss_func = nn.BCELoss()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate,correct_bias=True)  

        for epoch in range(self.args.epochs):
            print("\nEpoch ", epoch + 1, "/", self.args.epochs)
            avg_loss = 0

            self.train()

            for i, data in tqdm(enumerate(dataloader)): 
                if epoch >= 2 and self.patience >= 3:
                    print("Reload the best model...")
                    self.load_state_dict(torch.load(self.args.save_path))
                    self.adjust_learning_rate()
                    self.patience = 0

                loss = self.train_step(i, data)

                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)

                avg_loss += loss.item()
            cnt = len(train['y']) // self.args.batch_size + 1
            print("Average loss:{:.6f} ".format(avg_loss / cnt))

            self.evaluate(dev)

    def adjust_learning_rate(self, decay_rate=.5):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            self.args.learning_rate = param_group['lr']
        print("Decay learning rate to: ", self.args.learning_rate)

    def evaluate(self, dev, is_test=False):
        y_pred = self.predict(dev)
        with open(self.args.score_file_path, 'w') as output:
            for score, label in zip(y_pred, dev['y']):
                output.write(
                    str(score) + '\t' +
                    str(label) + '\n'
                )
        if is_test == False and not(self.args.task =='ubuntu' or self.args.task =='reddit10'):
            self.metrics.segment = 2
        else:
            self.metrics.segment = 10
        result = self.metrics.evaluate_all_metrics()
        print("Evaluation Result: \n",
              "MAP:", result[0], "\t",
              "MRR:", result[1], "\t",
              "P@1:", result[2], "\t",
              "R1:", result[3], "\t",
              "R2:", result[4], "\t",
              "R5:", result[5])

        if not is_test and result[3] + result[4] + result[5] > self.best_result[3] + self.best_result[4] + \
                self.best_result[5]:
            print("Best Result: \n",
                  "MAP:", self.best_result[0], "\t",
                  "MRR:", self.best_result[1], "\t",
                  "P@1:", self.best_result[2], "\t",
                  "R1:", self.best_result[3], "\t",
                  "R2:", self.best_result[4], "\t",
                  "R5:", self.best_result[5])
            self.patience = 0
            self.best_result = result
            torch.save(self.state_dict(), self.args.save_path)
            print("save model!!!\n")
        else:
            self.patience += 1

    def predict(self, dev):
        self.eval()
        y_pred = []
        dataset = BERTDataset(self.args, dev, self.bert_tokenizer, self.semantic)
        dataloader = DataLoader(dataset, batch_size=400)

        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_ids, batch_mask, batch_seg, batch_y, batch_len, batch_u, \
                mask_noun, masks_noun, ids_noun, \
                mask_adj, masks_adj, ids_adj, \
                masks_noun_ids, masks_adj_ids = (item.cuda() for item in data)
            with torch.no_grad():
                output = self.model(batch_ids, batch_mask, batch_seg, batch_u, 
                                    mask_noun, masks_noun, ids_noun,
                                    mask_adj, masks_adj, ids_adj,
                                    masks_noun_ids, masks_adj_ids)
                logits = torch.sigmoid(output[0]).squeeze()

            if i % 100 == 0:
                print('Batch[{}] batch_size:{}'.format(i, batch_ids.size(0))) 
            y_pred += logits.data.cpu().numpy().tolist()
        return y_pred

    def load_model(self, path):
        self.load_state_dict(state_dict=torch.load(path))
        if torch.cuda.is_available(): self.cuda()

