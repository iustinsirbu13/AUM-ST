import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def all_metrics(output, target, use_logits=True):
    if use_logits:
        output = output.argmax(dim=1)
    # output = output
    # target = target.cpu()
    all_metrics = {}
    eval_methods = {
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
    }
    
    for average in [None, 'micro', 'macro', 'weighted']:
        for name, func in eval_methods.items():
            score = func(target, output, average=average)
            if isinstance(score, np.ndarray):
                score = score.tolist()
            all_metrics[f'{average}/{name}'] = score
    all_metrics['accuracy'] = accuracy_score(target, output)
    return all_metrics


def _get_output_dicts(batch_size, batch_idx, labels, logits):
    output_dicts = []
    for j in range(logits.size(0)):
        probs = F.softmax(logits[j], -1)
        output_dict = {
            'index': batch_size * batch_idx + j,
            'true': labels[j].item(),
            'pred': logits[j].argmax().item(),
            'conf': probs.max().item(),
            'logits': logits[j].cpu().numpy().tolist(),
            'probs': probs.cpu().numpy().tolist(),
        }
        output_dicts.append(output_dict)
    return output_dicts
