from distutils.command.config import config
from sklearn.utils import shuffle

import argparse
import logging
import numpy as np
import os
import pandas as pd
import random
import sys
import transformers

from absl import app
from absl import flags
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from collections import defaultdict
from copy import deepcopy
from scipy.special import softmax
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from transformers import *
from tqdm import tqdm

import logging
import math
import numpy as np
import os
import torch
from utils import all_metrics, _get_output_dicts
from calibration_error import calculate_calibration_error
import json
from data import TrainDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_unlabeled(FLAGS, unlabeled_weak_dataloader):

    model = AutoModelForSequenceClassification.from_pretrained(
        FLAGS.intermediate_model_path, num_labels=FLAGS.num_classes)
    model.to(device)
    model.eval()

    y_predictions_unlabeled = []
    y_ids_unlabeled = []
    y_ids_hidden_gold_labels = []

    with torch.no_grad():
        for elem in tqdm(unlabeled_weak_dataloader):
            x = {key: elem[key].to(device)
                 for key in elem if key not in ['idx', 'lbl']}
            pred = model(
                input_ids=x['input_ids'], token_type_ids=x['token_type_ids'], attention_mask=x['attention_mask'])
            y_predictions_unlabeled.extend(pred.logits.cpu().numpy())
            if isinstance(elem['idx'], list):
                y_ids_unlabeled.extend(elem['idx'])
            else:
                y_ids_unlabeled.extend(elem['idx'].cpu().numpy())
            y_ids_hidden_gold_labels.extend(elem['lbl'].numpy())

    pseudo_labels = np.argmax(y_predictions_unlabeled, axis=-1).flatten()
    y_predictions_unlabeled = softmax(
        np.array(y_predictions_unlabeled), axis=1)

    mask = y_predictions_unlabeled > FLAGS.threshold
    mask = np.array(np.sum(mask, axis=1), dtype=bool)

    pseudo_labels_dict = {}
    indices = np.where(mask)[0]
    total = pseudo_labels.shape[0]

    ind = []
    num_correct = 0.0
    for elem in indices:
        ind.append(y_ids_unlabeled[elem])
        pseudo_labels_dict[y_ids_unlabeled[elem]] = pseudo_labels[elem]
        num_correct += int(pseudo_labels[elem]
                           == y_ids_hidden_gold_labels[elem])

    return ind, total, num_correct, pseudo_labels_dict


def train_supervised(FLAGS, best_f1_overall, labeled_dataloader, validation_dataloader, test_dataloader):

    best_f1_overall = 0
    for _ in range(FLAGS.initial_num):
        f1_validation, loss_validation, model = train_supervised_once(FLAGS, labeled_dataloader, validation_dataloader, FLAGS.intermediate_model_path,
                                                                      best_f1_overall, FLAGS.pt_teacher_checkpoint, FLAGS.inference_batch_size)

        if f1_validation > best_f1_overall:
            model.save_pretrained(FLAGS.intermediate_model_path)
            loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
            f1_test, loss_test = evaluate(
                model, test_dataloader, loss_fn, FLAGS.inference_batch_size)
            supervised_validation_f1 = f1_validation
            supervised_validation_loss = loss_validation
            supervised_test_f1 = f1_test
            supervised_test_loss = loss_test
            best_f1_overall = f1_validation

    return best_f1_overall, supervised_validation_f1, supervised_validation_loss, supervised_test_f1, supervised_test_loss


def train_supervised_once(FLAGS, labeled_dataloader, validation_dataloader, intermediate_model_path, best_f1_overall, pt_teacher_checkpoint, inference_batch_size, patience=10):

    model = AutoModelForSequenceClassification.from_pretrained(
        pt_teacher_checkpoint, num_labels=FLAGS.num_classes)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    best_f1 = 0
    for epoch in range(15):
        for data in tqdm(labeled_dataloader):
            # print(data)
            # print(data['input_ids'].shape)
            cuda_tensors = {key: data[key].to(
                device) for key in data if key not in ['idx']}
            optimizer.zero_grad()
            logits = model(
                input_ids=cuda_tensors['input_ids'], token_type_ids=cuda_tensors['token_type_ids'], attention_mask=cuda_tensors['attention_mask'])
            loss = loss_fn(logits.logits, cuda_tensors['lbl'])
            loss.backward()
            optimizer.step()

        f1_macro_validation, loss_validation = evaluate(
            model, validation_dataloader, loss_fn, inference_batch_size)

        if f1_macro_validation >= best_f1:
            crt_patience = 0
            best_f1 = f1_macro_validation
            corresponding_loss = loss_validation
            print('New best macro validation', best_f1, 'Epoch', epoch)
            continue

        if crt_patience == patience:
            crt_patience = 0
            print('Exceeding max patience; Exiting..')
            break

        crt_patience += 1

    return best_f1, corresponding_loss, model

    del model


def train_ssl(FLAGS, best_f1_overall, train_dataset, strong_unlabeled_dict, labels_unlabeled, ind, tokenizer, threshold_pseudo_labels_dict, validation_dataloader, test_dataloader, ulb_epochs, use_aum=False, aum_calculator=None):

    strongly_augmented_unlabeled_dataset = TrainDataset(
        strong_unlabeled_dict, labels_unlabeled, ind, tokenizer, threshold_pseudo_labels_dict)

    unlabeled_strong_dataloader = torch.utils.data.DataLoader(
        strongly_augmented_unlabeled_dataset, batch_size=FLAGS.unsup_batch_size, shuffle=True, drop_last=True)

    if use_aum:
        model = AutoModelForSequenceClassification.from_pretrained(
            FLAGS.pt_teacher_checkpoint, num_labels=FLAGS.num_classes + 1)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            FLAGS.pt_teacher_checkpoint, num_labels=FLAGS.num_classes)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()

    loss_fn_supervised = torch.nn.CrossEntropyLoss(reduction='mean')
    loss_fn_unsupervised = torch.nn.CrossEntropyLoss(reduction='none')

    data_sampler = torch.utils.data.RandomSampler(
        train_dataset, num_samples=10**5)
    batch_sampler = torch.utils.data.BatchSampler(
        data_sampler, FLAGS.sup_batch_size, drop_last=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=batch_sampler)

    crt_patience = 0
    best_f1 = 0
    for epoch in range(ulb_epochs):
        for data_supervised, data_unsupervised in tqdm(zip(train_dataloader, unlabeled_strong_dataloader)):
            cuda_tensors_supervised = {key: data_supervised[key].to(
                device) for key in data_supervised if key not in ['idx']}

            cuda_tensors_unsupervised = {key: data_unsupervised[key].to(
                device) for key in data_unsupervised if key not in ['idx']}

            merged_tensors = {}
            for k in cuda_tensors_supervised:
                merged_tensors[k] = torch.cat(
                    (cuda_tensors_supervised[k], cuda_tensors_unsupervised[k]))

            num_lb = cuda_tensors_supervised['input_ids'].shape[0]

            optimizer.zero_grad()
            logits = model(input_ids=merged_tensors['input_ids'], token_type_ids=merged_tensors[
                'token_type_ids'], attention_mask=merged_tensors['attention_mask'])

            logits_lbls = logits.logits[:num_lb]
            logits_ulbl = logits.logits[num_lb:]

            if use_aum:
                aum_calculator.update(
                    logits_ulbl.detach(), cuda_tensors_unsupervised['lbl'], np.array(data_unsupervised['idx']))

            loss_sup = loss_fn_supervised(
                logits_lbls, cuda_tensors_supervised['lbl'])
            loss_unsup = loss_fn_unsupervised(
                logits_ulbl, cuda_tensors_unsupervised['lbl'])
            loss = 0.5 * loss_sup + 0.5 * torch.mean(loss_unsup)
            loss.backward()
            optimizer.step()

        if not use_aum:
            loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
            f1_macro_test, loss_test = evaluate(
                model, test_dataloader, loss_fn, FLAGS.inference_batch_size)
            f1_macro_validation, loss_validation = evaluate(
                model, validation_dataloader, loss_fn, FLAGS.inference_batch_size)

            if f1_macro_validation >= best_f1:
                crt_patience = 0
                best_f1 = f1_macro_validation
                corresponding_test = f1_macro_test
                best_loss_validation = loss_validation
                best_loss_test = loss_test

                if best_f1 > best_f1_overall:
                    model.save_pretrained(FLAGS.intermediate_model_path)
                    best_f1_overall = best_f1
                print('New best macro validation', best_f1, 'Epoch', epoch)
                continue

            if crt_patience == FLAGS.unsupervised_patience:
                crt_patience = 0
                print('Exceeding max patience; Exiting..')
                break

        crt_patience += 1

    if not use_aum:
        return best_f1_overall, best_f1, corresponding_test, best_loss_validation, best_loss_test


def evaluate(model, test_dataloader, criterion, batch_size):
    full_predictions = []
    true_labels = []

    model.eval()
    crt_loss = 0

    with torch.no_grad():
        for elem in tqdm(test_dataloader):
            x = {key: elem[key].to(device)
                 for key in elem if key not in ['idx']}
            logits = model(
                input_ids=x['input_ids'], token_type_ids=x['token_type_ids'], attention_mask=x['attention_mask'])
            results = torch.argmax(logits.logits, dim=1)

            crt_loss += criterion(logits.logits, x['lbl']
                                  ).cpu().detach().numpy()
            full_predictions = full_predictions + \
                list(results.cpu().detach().numpy())
            true_labels = true_labels + \
                list(elem['lbl'].cpu().detach().numpy())

    model.train()

    return f1_score(true_labels, full_predictions, average='macro'), crt_loss / len(test_dataloader)


def evaluate_full(model, test_dataloader, criterion, FLAGS, tst_name='Valid', add_calibrated_ece=False):
    full_predictions = []
    full_logits = []
    true_labels = []
    output_dicts = []

    model.eval()
    crt_loss = 0

    with torch.no_grad():
        for idx, elem in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            x = {key: elem[key].to(device)
                 for key in elem if key not in ['idx']}
            logits = model(
                input_ids=x['input_ids'], token_type_ids=x['token_type_ids'], attention_mask=x['attention_mask'])
            results = torch.argmax(logits.logits, dim=1)

            crt_loss += criterion(logits.logits, x['lbl']
                                  ).cpu().detach().numpy()
            # full_logits = full_logits + list(logits.logits.cpu().detach().numpy())
            full_logits = full_logits + list(logits.logits.cpu().detach().numpy().tolist())
            full_predictions = full_predictions + \
                list(results.cpu().detach().numpy().tolist())
            true_labels = true_labels + \
                list(elem['lbl'].cpu().detach().numpy().tolist())
            output_dicts += _get_output_dicts(FLAGS.inference_batch_size, idx, elem['lbl'].cpu().detach(), logits.logits.cpu().detach())

    model.train()

    scores = all_metrics(full_predictions, true_labels, use_logits=False)
    ece_outputs = calculate_calibration_error(output_dicts, FLAGS.num_classes)
    scores['ECE/uncalibrated'] = ece_outputs['expected error']

    if add_calibrated_ece:
        if tst_name.startswith('Valid'):
            temperature = None # optimal temperature will be computed
        elif tst_name.startswith('Test'):
            valid_name = tst_name.replace('Test', 'Valid', 1)
            with open(os.path.join(FLAGS.intermediate_model_path, f'stats_{valid_name}.json')) as fp:
                valid_stats = json.load(fp)
                temperature = valid_stats['ece_outputs_calibrated']['temperature']
                print(f"Calibrating with temperature {temperature}")

        ece_outputs_calibrated = calculate_calibration_error(output_dicts, FLAGS.num_classes, temperature)
        scores['ECE/calibrated'] = ece_outputs_calibrated['expected error']
    else:
        ece_outputs_calibrated = {}

    stats = {
        'outputs': full_logits,
        'preds': full_predictions,
        'targets': true_labels,
        'scores': scores,
        'ece_outputs_uncalibrated': ece_outputs,
        'ece_outputs_calibrated': ece_outputs_calibrated
    }
    # print(stats)

    stats_file = os.path.join(FLAGS.intermediate_model_path, f'stats_{tst_name}.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f)

    return f1_score(true_labels, full_predictions, average='macro'), crt_loss / len(test_dataloader)
