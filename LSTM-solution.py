import os, re, random, argparse
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metric(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='macro')

    print('Accuracy: {:.1%}    Precision: {:.1%}    Recall: {:.1%}    F1: {:.1%}'
                .format(accuracy, macro_precision, macro_recall, weighted_f1))
    # target_names = ['Negative', 'Positive']
    # print(classification_report(y_true, y_pred, target_names=target_names))


def load_file(path):
    with open(os.path.join(path, 'positive.txt'), 'r') as f:
        pos = np.array([re.sub('[^\u4e00-\u9fa5 ]+', '', line).split() for line in f.readlines()], dtype=object)
    with open(os.path.join(path, 'negative.txt'), 'r') as f:
        neg = np.array([re.sub('[^\u4e00-\u9fa5 ]+', '', line).split() for line in f.readlines()], dtype=object)
    all_txt = np.concatenate((pos, neg), axis=0)
    labels = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)), axis=0)
    return all_txt, labels


def Word2VecTrain(text, save_path, n_dim):
    w2v_model = Word2Vec(text, vector_size=n_dim, min_count=1, alpha=0.05, epochs=50)
    w2v_model.save(save_path)


def load_model(model_path):
    return Word2Vec.load(model_path)


class MyDataset(Dataset):
    def __init__(self, text, label, word2idx, maxL=100):
        super().__init__()
        self.text = text
        self.label = label
        self.maxL = maxL
        self.tokenizer = word2idx

    def get_tokenize(self, word_list):
        vec = (len(self.tokenizer) - 1) * np.ones(self.maxL, dtype='int32')
        for i, word in enumerate(word_list):
            if word in self.tokenizer:
                vec[self.maxL - len(word_list) + i] = self.tokenizer[word]
        return vec

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return torch.from_numpy(self.get_tokenize(self.text[index])), self.label[index]


class LSTM4Classify(nn.Module):
    def __init__(self, vocab_size, n_dim, hidden_size, num_layers, dropout_rate, embedding_weights=None):
        super().__init__()
        self.n_dim = n_dim
        if embedding_weights is None:
            self.embedding = nn.Embedding(vocab_size, n_dim, padding_idx=-1)
        else:
            self.embedding = nn.Embedding.from_pretrained(embedding_weights, padding_idx=-1)

        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(n_dim, hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout_rate)
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        output, _ = self.lstm(embeddings)
        output = output[:, -1]
        output = torch.sigmoid(self.linear(output))
        return output.squeeze(1)


def train_epoch(model, optimizer, loss_func, epoch, device, train_data, valid_data=None):
    model.train()
    pbar = tqdm(train_data)
    pbar.set_description('Training Epoch {}'.format(epoch))
    for id, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        preds = model(inputs)
        loss = loss_func(preds, labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        pbar.set_postfix(loss=loss.item())
    if valid_data is not None:
        acc = evaluate(model, device, valid_data, 'Validation')
        return acc


def evaluate(model, device, eval_data, prefix):
    model.eval()
    y_pred, y_true = [], []
    pbar = tqdm(eval_data)
    pbar.set_description('[Evaluating in {} set]'.format(prefix))
    with torch.no_grad():
        for id, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).detach().cpu().numpy()
            y_pred.extend(preds.round().astype(int))
            y_true.extend(labels.detach().cpu().numpy())
    y_true, y_pred = np.array(y_true, dtype=int), np.array(y_pred, dtype=int)
    metric(y_true, y_pred)
    return (y_true == y_pred).sum() / len(y_true)


# ============================== hyper-parameter =================================== #

os.environ["CUDA_VISIBLE_DEVICES"] = '7'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()

# Seed
parser.add_argument('--seed', type=int, default=8)

# word2vec param
parser.add_argument('--n_dim', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--w2vmodel_path', type=str, default=None)

# LSTM training param
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--dropout_rate', type=float, default=0.1)
parser.add_argument('--model_save_path', type=str, default='./LSTM-model.pt')

# train or eval
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--do_test', action='store_true')

args = parser.parse_args()
setup_seed(args.seed)


# ============================== word2vec train ==================================== #

all_txt, labels = load_file('./dataset')
if args.w2vmodel_path is None:
    Word2VecTrain(all_txt, 'w2vmodel', args.n_dim)
    w2vmodel = load_model('w2vmodel')
else:
    w2vmodel = load_model(args.w2vmodel_path)

word2idx, embedding_weights = w2vmodel.wv.key_to_index, w2vmodel.wv.vectors
embedding_weights = np.insert(embedding_weights, len(word2idx), 0, axis=0)
word2idx['[pad]'] = len(word2idx)


# ============================ split the dataset =================================== #

rand_perm = np.random.permutation(len(all_txt))
all_txt, labels = all_txt[rand_perm], labels[rand_perm]
sz = len(all_txt)
train_X, train_y = all_txt[:int(0.7 * sz)], labels[:int(0.7 * sz)]
valid_X, valid_y = all_txt[int(0.7 * sz):int(0.8 * sz)], labels[int(0.7 * sz):int(0.8 * sz)]
test_X, test_y = all_txt[int(0.8 * sz):], labels[int(0.8 * sz):]


# =================== construct the dataset and dataloader ========================= #

train_dataset = MyDataset(train_X, train_y, word2idx)
valid_dataset = MyDataset(valid_X, valid_y, word2idx)
test_dataset = MyDataset(test_X, test_y, word2idx)
print('Train Data {}, Valid Data {}, Test Data {}'.format(len(train_dataset), len(valid_dataset), len(test_dataset)))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


# ============= initialize LSTM model with pretrained word embedding =============== #

embedding_weights = torch.from_numpy(embedding_weights).to(device)
model = LSTM4Classify(len(word2idx), args.n_dim, args.hidden_size, args.num_layers, args.dropout_rate, embedding_weights).to(device)


# ============================== Train and Evaluate ================================ #

if args.do_train:
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_valid_acc = 0
    for epoch in range(1, args.epochs + 1):
        now_valid_acc = train_epoch(model, optimizer, loss_func, epoch, device, train_loader, valid_loader)
        test_acc = evaluate(model, device, test_loader, 'Test')
        if now_valid_acc > best_valid_acc:
            best_valid_acc = now_valid_acc
            torch.save({
                'epoch': epoch,
                'valid_acc': now_valid_acc,
                'test_acc': test_acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, args.model_save_path)
        print()


# ==================================== Test ======================================== #

if args.do_test:
    ckpt = torch.load(args.model_save_path)
    model.load_state_dict(ckpt['model_state_dict'])
    evaluate(model, device, test_loader, 'Test')
