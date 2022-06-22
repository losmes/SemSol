import time
import argparse
import pickle
import os
from BERT_finetuning import NeuralNetwork
from setproctitle import setproctitle

setproctitle('SemSol Fine-Tuning')

#Dataset path.
FT_data={
    'ubuntu': 'ubuntu_data/ubuntu_dataset_1M.pkl',
    'douban': 'douban_data/dataset_1M.pkl',
    'e_commerce': 'e_commerce_data/e_commerce_dataset_1M.pkl',
    'reddit10': 'reddit10_data/dataset_1M.pkl'
}
print(os.getcwd())
## Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--task",
                    default='ubuntu',
                    type=str,
                    help="The dataset used for training and test.")

parser.add_argument("--is_training",
                    action='store_true',
                    help="Training model or testing model?")

parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="The batch size.")
parser.add_argument("--learning_rate",
                    default=1e-5,
                    type=float,
                    help="The initial learning rate for Adamw.")
parser.add_argument("--epochs",
                    default=2,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--save_path",
                    default="./Fine-Tuning/FT_checkpoint/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--score_file_path",
                    default="./Fine-Tuning/scorefile.txt",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--do_lower_case", action='store_true', default=True,
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--load_checkpoint",
                    default="error",
                    type=str,
                    help="The path to load checkpoint.")

parser.add_argument("--topic_num", type=int, default=50, help="topic num.")

parser.add_argument('--semantic_lang', type=str, default="cmn", help="")
parser.add_argument('--semantic_type', type=int, default=2, help="")
parser.add_argument('--max_semantic_word_length', type=int, default=55, help="")

parser.add_argument('--noun_num', type=int, default=10, help="")
parser.add_argument('--noun_example_num', type=int, default=4, help="")
parser.add_argument('--adj_num', type=int, default=0, help="")
parser.add_argument('--adj_example_num', type=int, default=0, help="")

parser.add_argument('--use_topic', action='store_true', help="")
parser.add_argument('--use_semantic', action='store_true', help="")

args = parser.parse_args()
args.save_path += '.' + "0.pt"
args.score_file_path = args.score_file_path
# load bert


print(args)
print("Task: ", args.task)


def train_model(train, dev):
    model = NeuralNetwork(args=args)
    model.fit(train, dev)


def test_model(test):
    model = NeuralNetwork(args=args)
    #model.load_model(args.save_path)
    model.evaluate(test, is_test=True)


if __name__ == '__main__':
    start = time.time()
    with open(FT_data[args.task], 'rb') as f:
        train, dev, test = pickle.load(f, encoding='ISO-8859-1')

    if args.is_training==True:
        train_model(train,dev)
        test_model(test)
    else:
        test_model(test)

    end = time.time()
    print("use time: ", (end - start) / 60, " min")




