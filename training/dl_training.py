import numpy as np
import pandas as pd

import tqdm
import time

import matplotlib.pyplot as plt
# %matplotlib inline

import os




# DATA_PATH = ...
positive_df = pd.read_csv(os.path.join(DATA_PATH, 'positive_df.tsv'), sep='\t')
negative_df = pd.read_csv(os.path.join(DATA_PATH, 'negative_df.tsv'), sep='\t')

import collections
cancer_type_n = collections.Counter(list(positive_df['cancer_type']))

import itertools
import nltk

def make_ngram_matrix(seqs, max_n):
    all_comb = dict()
    for n in range(1, max_n + 1):
        all_comb[n] = sorted(itertools.product(list('ACGT'), repeat=n))

    n_features = sum([len(all_comb[x]) for x in all_comb.keys()])
    n_items = len(seqs)

    X = np.zeros(shape=(n_items, n_features), dtype='float32')

    i = 0
    for s in tqdm.tqdm(seqs):
        j = 0
        for n in range(1, max_n + 1):
            my_ngrams = nltk.ngrams(list(s), n)
            my_ngram_count = collections.Counter(my_ngrams)
            for k in all_comb[n]:
                X[i, j] = my_ngram_count[k] / (len(s) - n + 1) * len(all_comb[n])
                j += 1
        i += 1
    return X


from sklearn.model_selection import train_test_split
positive_partition_train_ids, positive_partition_test_ids = train_test_split(range(len(positive_df)), test_size=2560/len(positive_df))
negative_partition_train_ids, negative_partition_test_ids = train_test_split(range(len(negative_df)), test_size=2560/len(negative_df))
negative_partition_train_ids, _ = train_test_split(negative_partition_train_ids, train_size=7*len(positive_partition_train_ids)/len(negative_partition_train_ids))



train_seqs_positive = np.array(
    [
     [x[50:850], x[100:900], x[150:950],
      x[200:1000],
      x[250:1050], x[300:1100], x[350:1150]
     ]for x in
     np.array(list(positive_df['seq']))[positive_partition_train_ids]
    ]
).flatten()
train_seqs_negative = np.array(
    [
     x for x in
     np.array(list(negative_df['seq']))[negative_partition_train_ids]
    ]
)
train_seqs = np.concatenate([train_seqs_positive, train_seqs_negative])

train_labels = np.concatenate(
    [
     np.array([1 for x in train_seqs_positive]),
     np.array([0 for x in train_seqs_negative])
    ]
)

train_ngrams = make_ngram_matrix(train_seqs, max_n=4)



test_seqs = np.concatenate(
    [np.array([x[200:1000] for x in positive_df['seq']])[positive_partition_test_ids],
     np.array([x for x in negative_df['seq']])[negative_partition_test_ids]
    ]
)

test_labels = np.concatenate(
    [np.array(list(positive_df['chromo']))[positive_partition_test_ids],
     np.array(list(negative_df['chromo']))[negative_partition_test_ids]
    ]
)

test_ngrams = make_ngram_matrix(test_seqs, max_n=4)



nitrobase_dict = { 'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3 }
def input_encoding(x_str):
    values = [nitrobase_dict[b] for b in x_str]
    sz, max_value = len(values), len(nitrobase_dict)
    matrix = np.zeros([sz, max_value], dtype='float32')
    matrix[np.arange(sz), values] = 1
    return matrix

import torch
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, data_seq, data_ngram, labels):
        self.data_seq = data_seq # list of X seqs
        self.data_ngram = data_ngram # list of X ngrams
        self.labels = labels # list of y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, ID):
        # Select sample
        X_seq = self.data_seq[ID]
        X_seq = input_encoding(X_seq)
        X_seq = torch.tensor(X_seq)

        X_ngram = self.data_ngram[ID]
        X_ngram = torch.tensor(X_ngram)

        y = self.labels[ID]

        return (X_seq, X_ngram), y

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Parameters
params = {
    'batch_size': 256,
    'shuffle': True,
    'num_workers': 6
}


# Generators
training_set = Dataset(train_seqs, train_ngrams, train_labels)
training_generator = data.DataLoader(training_set, **params)

test_set = Dataset(test_seqs, test_ngrams, test_labels)
test_generator = data.DataLoader(test_set, **params)




def bce_loss(y, z, eps=1e-20):
    return -np.mean(y * np.log(z + eps) + (1 - y) * np.log(1 - z + eps))

def mse_loss(y, z):
    return np.mean((y - z)**2)

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def calculate_metrics(y, z_proba):
    z_class = np.round(z_proba)
    acc = np.mean(y == z_class)
    prec = precision_score(y, z_class)
    recall = recall_score(y, z_class)
    f1 = f1_score(y, z_class)
    roc = roc_auc_score(y, z_proba)
    return acc, prec, recall, f1, roc

def print_loss_and_metrics(name, bce, mse, acc, prec, recall, f1, roc):
    print('{}:'.format(name))
    print('BCE loss: {:.4f}, MSE loss: {:.4f}'.format(bce, mse))
    print('Accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, F1: {:.4f}, ROC-AUC: {:.4f}'.format(
        acc, prec, recall, f1, roc))

def quality_by_cancer_type(pos_df, pos_part, cancer_type_count, test_pred):
    df = pd.DataFrame()
    ct_ct = []
    ct_items = []
    ct_bce = []
    ct_mse = []
    ct_acc = []
    test_pos_cancer_types = np.array(pos_df['cancer_type'])[pos_part]
    for ct in sorted(cancer_type_count.keys()):
        ct_ct.append(ct)
        ct_mask = test_pos_cancer_types == ct
        ct_pred = test_pred[:len(test_pos_cancer_types)][ct_mask]
        ct_y = np.ones_like(ct_pred)
        ct_items.append(len(ct_pred))
        ct_bce.append(bce_loss(ct_y, ct_pred))
        ct_mse.append(mse_loss(ct_y, ct_pred))
        ct_acc.append(np.mean(ct_y == np.round(ct_pred)))
    return pd.DataFrame({'Cancer' : ct_ct,
                         'Items' : ct_items,
                         'BCE' : ct_bce,
                         'MSE' : ct_mse,
                         'Accuracy' : ct_acc})




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from dl_models import <model> as NeededModel
model = NeededModel().to(device)
LR = 3e-4
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()



def train_model(model, opt, criterion, train_gen, test_gen, max_epochs, verbose):
    stats = {
        'BCE' : {'train' : [], 'test' : []},
        'MSE' : {'train' : [], 'test' : []},
        'Accuracy' : {'train' : [], 'test' : []},
        'Precision' : {'train' : [], 'test' : []},
        'Recall' : {'train' : [], 'test' : []},
        'F1' : {'train' : [], 'test' : []},
        'ROC-AUC' : {'train' : [], 'test' : []}
    }

    # Loop over epochs
    for epoch in range(max_epochs):
        t = time.time()
        # Training
        train_loss = 0
        train_steps = 0
        y, z = [], []
        for local_batch, local_labels in train_gen:
            # Transfer to GPU
            local_batch = (local_batch[0].to(device), local_batch[1].to(device))
            local_labels = local_labels.float().to(device)

            # Model computations
            opt.zero_grad()

            output = model(local_batch)
            loss = criterion(output, local_labels)

            loss.backward()
            train_loss += loss.data.item()
            train_steps += 1
            opt.step()

            y.extend(local_labels.int().cpu().numpy())
            z.extend(torch.sigmoid(output).cpu().detach().numpy())

        train_loss /= train_steps

        y, z = np.array(y), np.array(z)
        bce_loss_train = train_loss
        mse_loss_train = mse_loss(y, z)
        accuracy_value_train, precision_value_train, recall_value_train, \
        f1_value_train, roc_auc_value_train = calculate_metrics(y, z)

        stats['BCE']['train'].append(bce_loss_train)
        stats['MSE']['train'].append(mse_loss_train)
        stats['Accuracy']['train'].append(accuracy_value_train)
        stats['Precision']['train'].append(precision_value_train)
        stats['Recall']['train'].append(recall_value_train)
        stats['F1']['train'].append(f1_value_train)
        stats['ROC-AUC']['train'].append(roc_auc_value_train)


        # Validation
        validation_loss = 0
        validation_steps = 0
        y, z = [], []
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in test_gen:
                # Transfer to GPU
                local_batch = (local_batch[0].to(device), local_batch[1].to(device))
                local_labels = local_labels.float().to(device)

                # Model computations
                output = model(local_batch)
                loss = criterion(output, local_labels)

                y.extend(local_labels.int().cpu().numpy())
                z.extend(torch.sigmoid(output).cpu().numpy())

                validation_loss += loss.item()
                validation_steps += 1

        validation_loss /= validation_steps

        y, z = np.array(y), np.array(z)
        bce_loss_test = validation_loss
        mse_loss_test = mse_loss(y, z)
        accuracy_value_test, precision_value_test, recall_value_test, \
        f1_value_test, roc_auc_value_test = calculate_metrics(y, z)

        stats['BCE']['test'].append(bce_loss_test)
        stats['MSE']['test'].append(mse_loss_test)
        stats['Accuracy']['test'].append(accuracy_value_test)
        stats['Precision']['test'].append(precision_value_test)
        stats['Recall']['test'].append(recall_value_test)
        stats['F1']['test'].append(f1_value_test)
        stats['ROC-AUC']['test'].append(roc_auc_value_test)

        if verbose > 0:
            print('\n Epoch {}/{} finished in {:.1f} seconds'.format(epoch+1, max_epochs,
                                                              time.time() - t))
            if epoch % print_each == 0:
                print()
                print_loss_and_metrics('Train',
                                      bce_loss_train, mse_loss_train,
                                      accuracy_value_train,
                                      precision_value_train,
                                      recall_value_train,
                                      f1_value_train,
                                      roc_auc_value_train)
                print()
                print_loss_and_metrics('Validation',
                                      bce_loss_test, mse_loss_test,
                                      accuracy_value_test,
                                      precision_value_test,
                                      recall_value_test,
                                      f1_value_test,
                                      roc_auc_value_test)
                print('=' * 90, '\n')

    return model, opt, stats



model, optimizer, stats = train_model(model, optimizer, criterion,
                                      training_generator, test_generator,
                                      max_epochs=20, verbose=1)





# prediction quality by cancer type

n_batches = int(np.ceil(len(positive_partition_test_ids) / params['batch_size']))
model_preds = []
with torch.set_grad_enabled(False):
    for i in tqdm.tnrange(n_batches):
        start_i = i * params['batch_size']
        end_i = start_i + params['batch_size']
        seq_i = test_seqs[start_i : end_i]
        seq_i = np.array([input_encoding(x) for x in seq_i])
        seq_i = torch.tensor(seq_i)
        ngrams_i = test_ngrams[start_i : end_i]
        ngrams_i = torch.tensor(ngrams_i)

        seq_i, ngrams_i = seq_i.to(device), ngrams_i.to(device)
        pred_i = model1((seq_i, ngrams_i))

        pred_i = torch.sigmoid(pred_i)
        pred_i = pred_i.cpu().numpy()

        model_preds.extend(pred_i)
model_preds = np.array(preds)

model_by_ct = quality_by_cancer_type(positive_df, positive_partition_test_ids, cancer_type_n, model_preds)



