import torch
import math
import torch.nn.functional as F
import numpy as np

from train_utils import ce_loss


class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value


def TSA(schedule, cur_iter, total_iter, num_classes):
    training_progress = cur_iter / total_iter

    if schedule == 'linear':
        threshold = training_progress
    elif schedule == 'exp':
        scale = 5
        threshold = math.exp((training_progress - 1) * scale)
    elif schedule == 'log':
        scale = 5
        threshold = 1 - math.exp((-training_progress) * scale)
    elif schedule == 'none':
        return 1
    tsa = threshold * (1 - 1 / num_classes) + 1 / num_classes
    return tsa


def consistency_loss_uda(logits_s, logits_w, class_acc, it, ds, name='ce', T=1.0, p_cutoff=0.0, use_flex=False):
    logits_w = logits_w.detach()

    if name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        if use_flex:
            mask = max_probs.ge(p_cutoff * (class_acc[max_idx] / (2. - class_acc[max_idx]))).float()
        else:
            mask = max_probs.ge(p_cutoff).float()
        select = max_probs.ge(p_cutoff).long()

        pseudo_label = torch.softmax(logits_w / T, dim=-1)
        masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels=False) * mask
        return masked_loss.mean(), mask.mean(), select, max_idx.long()

    if name == 'kld_tf':
        # The implementation of loss_unsup for Google's TF
        # logits_tgt by sharpening
        logits_tgt = (logits_w / T).detach()

        # pseudo_labels_mask by confidence masking
        pseudo_labels = torch.softmax(logits_w, dim=-1).detach()
        p_class_max = torch.max(pseudo_labels, dim=-1, keepdim=False)[0]
        loss_mask = p_class_max.ge(p_cutoff).float().detach()

        kld = F.kl_div(torch.log_softmax(logits_s, dim=-1), torch.softmax(logits_tgt, dim=-1), reduction='none')
        masked_loss = kld * loss_mask.unsqueeze(dim=-1).repeat(1, pseudo_labels.shape[1])
        masked_loss = torch.sum(masked_loss, dim=1)
        return masked_loss.mean(), loss_mask.mean()


    else:
        assert Exception('Not Implemented consistency_loss')
        
def consistency_loss_flex(logits_s, logits_w, class_acc, p_target, p_model, name='ce',
                     T=1.0, p_cutoff=0.0, use_hard_labels=True, use_DA=False):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        if use_DA:
            if p_model == None:
                p_model = torch.mean(pseudo_label.detach(), dim=0)
            else:
                p_model = p_model * 0.999 + torch.mean(pseudo_label.detach(), dim=0) * 0.001
            pseudo_label = pseudo_label * p_target / p_model
            pseudo_label = (pseudo_label / pseudo_label.sum(dim=-1, keepdim=True))

        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        # mask = max_probs.ge(p_cutoff * (class_acc[max_idx] + 1.) / 2).float()  # linear
        # mask = max_probs.ge(p_cutoff * (1 / (2. - class_acc[max_idx]))).float()  # low_limit
        mask = max_probs.ge(p_cutoff * (class_acc[max_idx] / (2. - class_acc[max_idx]))).float()  # convex
        # mask = max_probs.ge(p_cutoff * (torch.log(class_acc[max_idx] + 1.) + 0.5)/(math.log(2) + 0.5)).float()  # concave
        select = max_probs.ge(p_cutoff).long()
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean(), select, max_idx.long(), p_model

    else:
        assert Exception('Not Implemented consistency_loss')


def torch_device_one():
    return torch.tensor(1.)
