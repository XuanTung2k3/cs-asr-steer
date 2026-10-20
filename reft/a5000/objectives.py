"""Token-global objectives. Logical-batch denominators can be supplied for exact accumulation."""
import unicodedata

import torch
from torch import nn
from torch.nn import functional as F

IGNORE = -100
UNKNOWN, ZH, EN = -100, 0, 1


def script_group(text):
    """Conservative script proxy: mixed/partial-byte, numeric, controls and punctuation are unknown.

    This cannot identify English names/romanized Mandarin linguistically. It is explicitly script-derived.
    """
    letters = [c for c in text.strip() if c.isalpha()]
    if not letters or any(c.isdigit() or c == '\ufffd' for c in text):
        return UNKNOWN
    han = [('\u3400' <= c <= '\u9fff') for c in letters]
    latin = ['LATIN' in unicodedata.name(c, '') for c in letters]
    if all(han):
        return ZH
    if all(latin):
        return EN
    return UNKNOWN


def token_groups(tokenizer, target_ids):
    from reft.text import tok_bytes_fn
    tb = tok_bytes_fn(tokenizer)
    groups = []
    for i in target_ids:
        try:
            surface = tb(int(i)).decode('utf-8', errors='strict')
        except UnicodeDecodeError:
            surface = ''
        groups.append(script_group(surface) if int(i) not in tokenizer.all_special_ids else UNKNOWN)
    return torch.tensor(groups, dtype=torch.long)


def weighted_ce(logits, labels, groups=None, english_weight=1.0, denominator=None):
    if english_weight < 1:
        raise ValueError('English-targeted weight must be >= 1')
    valid = labels != IGNORE
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           ignore_index=IGNORE, reduction='none').reshape(labels.shape)
    weights = valid.to(loss.dtype)
    if english_weight != 1:
        if groups is None:
            raise ValueError('weighted CE requires audited group labels')
        weights = weights * torch.where(groups == EN, english_weight, 1.0)
    den = weights.sum() if denominator is None else denominator
    return (loss * weights).sum() / den


class LIDHead(nn.Module):
    """Downstream decoder-final state at the position PREDICTING the target, not its input embedding."""
    def __init__(self, d):
        super().__init__()
        self.classifier = nn.Linear(d, 2)

    def forward(self, states, groups):
        valid = groups != UNKNOWN
        z = self.classifier(states.float())
        if not bool(valid.any()):
            return z.sum() * 0
        return F.cross_entropy(z[valid], groups[valid])


def objective(logits, labels, groups=None, states=None, head=None, lid_lambda=0.0, english_weight=1.0):
    ce = weighted_ce(logits, labels, groups, english_weight)
    if lid_lambda == 0:
        return ce
    if head is None or states is None or groups is None:
        raise ValueError('O2 requires downstream states/head/audited groups')
    return ce + lid_lambda * head(states, groups)


def goldkl(teacher_logits, student_logits, labels):
    """Separate future bridge; correct-gold-prefix positions only; no duplicate teacher instance needed."""
    valid = (labels != IGNORE) & (teacher_logits.argmax(-1) == labels)
    if not bool(valid.any()):
        return student_logits.sum() * 0
    t = F.log_softmax(teacher_logits.detach().float()[valid], -1)
    s = F.log_softmax(student_logits.float()[valid], -1)
    return (t.exp() * (t-s)).sum(-1).mean()
