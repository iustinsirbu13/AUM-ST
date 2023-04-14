import pandas as pd
import torch
import random
from collections import defaultdict
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, text_list, labels, tokenizer, seq_len=128):
        self.text_list = text_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __getitem__(self, idx):
        inputs = self.text_list[idx]
        if isinstance(inputs, list) and len(inputs) == 2:
            tok = self.tokenizer(*self.text_list[idx], padding='max_length', max_length=self.seq_len, truncation=True, return_token_type_ids=True)
        else:
            tok = self.tokenizer(self.text_list[idx], padding='max_length', max_length=self.seq_len, truncation=True, return_token_type_ids=True)

        item = {key: torch.tensor(tok[key]) for key in tok}
        item['lbl'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.text_list)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, augmentation_dict, labels_dict, ids, tokenizer, pseudo_labels=None, seq_len=128):
        self.ids = ids
        self.augmentation_dict = augmentation_dict
        self.labels_dict = labels_dict
        self.tokenizer = tokenizer
        self.pseudo_labels = pseudo_labels
        self.seq_len = seq_len

    def __getitem__(self, idx):
        inputs = random.choice(self.augmentation_dict[self.ids[idx]])
        if isinstance(inputs, list) and len(inputs) == 2:
            tok = self.tokenizer(*inputs, padding='max_length', max_length=self.seq_len, truncation=True, return_token_type_ids=True)
        else:
            tok = self.tokenizer(inputs, padding='max_length', max_length=self.seq_len, truncation=True, return_token_type_ids=True)

        item = {key: torch.tensor(tok[key]) for key in tok}
        item['lbl'] = torch.tensor(
            self.labels_dict[self.ids[idx]], dtype=torch.long)
        if self.pseudo_labels != None:
            item['lbl'] = torch.tensor(
                self.pseudo_labels[self.ids[idx]], dtype=torch.long)
        item['idx'] = self.ids[idx]
        return item

    def __len__(self):
        return len(self.ids)


class TensorboardLog:
    def __init__(self, tb_dir, file_name):
        self.tb_dir = tb_dir
        self.writer = SummaryWriter(os.path.join(self.tb_dir, file_name))

    def update(self, tb_dict, it):
        for key, value in tb_dict.items():
            self.writer.add_scalar(key, value, it)


def sample_split(initial_set, FLAGS):
    random.seed(FLAGS.seed)
    label_frequencies_train = defaultdict(int)
    df = pd.read_csv(os.path.join(FLAGS.augmentation_dir, initial_set))
    for i, row in df.iterrows():
        label_frequencies_train[row['Label']] += 1

    train_label_to_set = {}
    for i, row in df.iterrows():
        if row['Label'] not in train_label_to_set:
            train_label_to_set[row['Label']] = set()
        train_label_to_set[row['Label']].add(row['Text'])

    training_set = set()
    unlabeled_set = set()

    for cls in train_label_to_set:
        crt_texts = train_label_to_set[cls]
        remaining = []
        if len(crt_texts) < FLAGS.num_labels:
            num_repeating = FLAGS.num_labels // len(crt_texts) + 1
            resulting_text = list(crt_texts) * num_repeating
            resulting_text = resulting_text[:FLAGS.num_labels]
        elif len(crt_texts) > FLAGS.num_labels:
            sampled = random.sample(crt_texts, FLAGS.num_labels)
            remaining = list(
                crt_texts.difference(set(sampled)))
            resulting_text = list(sampled)
        else:
            resulting_text = list(crt_texts)

        training_set = training_set.union(resulting_text)
        unlabeled_set = unlabeled_set.union(remaining)

    ids_train = []
    labels_train = {}
    ids_unlabeled = []
    labels_unlabeled = {}

    for i, row in df.iterrows():
        if row['Text'] in training_set:
            ids_train.append(row['Id'])
            labels_train[row['Id']] = row['Label']
        elif row['Text'] in unlabeled_set:
            ids_unlabeled.append(row['Id'])
            labels_unlabeled[row['Id']] = row['Label']

    return ids_train, labels_train, ids_unlabeled, labels_unlabeled


def read_split(FLAGS):
    df_train = pd.read_csv(os.path.join(FLAGS.input_dir, FLAGS.train_file))
    ids_train = df_train.Id.tolist()
    labels_train = df_train[['Id', 'Label']].set_index('Id').to_dict()['Label']

    df_unlabeled = pd.read_csv(os.path.join(FLAGS.input_dir, 'unlabeled.csv'))
    ids_unlabeled = df_unlabeled.Id.tolist()
    labels_unlabeled = df_unlabeled[['Id', 'Label']].set_index('Id').to_dict()['Label']

    return ids_train, labels_train, ids_unlabeled, labels_unlabeled 


def get_eval_dataset(path, tokenizer, sampling_strategy=-1):

    df = pd.read_csv(path)
    text_list = []
    labels_list = []
    for i, row in df.iterrows():
        text_list.append(row['Text'])
        labels_list.append(row['Label'])

    tl = []
    ll = []
    if sampling_strategy != -1:
        zipped = list(zip(text_list, labels_list))
        random.shuffle(zipped)
        num_per_label = defaultdict(int)
        for elem in zipped:
            num_per_label[elem[1]] += 1
            if num_per_label[elem[1]] >= sampling_strategy:
                continue
            else:
                tl.append(elem[0])
                ll.append(elem[1])
        text_list = tl
        labels_list = ll
    dataset = TestDataset(text_list, labels_list, tokenizer)

    return dataset


def get_eval_dataset_v2(path, tokenizer, sampling_strategy=-1):

    df = pd.read_csv(path)
    text_list = []
    labels_list = []
    for i, row in df.iterrows():
        text_list.append([row['Text'], row['Query']])
        labels_list.append(row['Label'])

    if sampling_strategy != -1:
        raise NotImplementedError()

    dataset = TestDataset(text_list, labels_list, tokenizer)

    return dataset


def retrieve_simple(ids, paths, min_str, max_str):
    id_set = set(ids)
    aug_dict = defaultdict(list)
    for augmented_file in tqdm(paths):
        df = pd.read_csv(augmented_file)
        if df['Strength'][0] >= min_str and df['Strength'][0] <= max_str:
            for i, row in df.iterrows():
                if row['Id'] in id_set:
                    aug_dict[row['Id']].append(row['Text'])

    return aug_dict


def retrieve_with_query(ids, paths, min_str, max_str):
    id_set = set(ids)
    aug_dict = defaultdict(list)
    for augmented_file in tqdm(paths):
        df = pd.read_csv(augmented_file)
        if df['Strength'][0] >= min_str and df['Strength'][0] <= max_str:
            for i, row in df.iterrows():
                if row['Id'] in id_set:
                    aug_dict[row['Id']].append([row['Text'], row['Query']])

    return aug_dict


def retrieve_augmentations(available_augmentations, weak_aug_min, weak_aug_max, strong_aug_min, strong_aug_max, ids_train, ids_unlabeled, augmentation_dir, with_query=False):
    aug_paths = []
    retrieve_fn = retrieve_with_query if with_query else retrieve_simple

    for e in available_augmentations:
        aug_paths.append(os.path.join(augmentation_dir, e))

    print('Collecting weak augmentations for training set.')
    weak_train_dict = retrieve_fn(
        ids_train, aug_paths, weak_aug_min, weak_aug_max)
    print('Collecting weak augmentations for unlabeled set.')
    weak_unlabeled_dict = retrieve_fn(
        ids_unlabeled, aug_paths, weak_aug_min, weak_aug_max)
    print('Collecting strong augmentations for unlabeled set.')
    strong_unlabeled_dict = retrieve_fn(
        ids_unlabeled, aug_paths, strong_aug_min, strong_aug_max)

    return weak_train_dict, weak_unlabeled_dict, strong_unlabeled_dict
