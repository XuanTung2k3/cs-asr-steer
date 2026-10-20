"""D1/D2 foundations; script groups are proxies and probe accuracy is not causal evidence."""
import torch


def decompose(logits, labels, vocab_groups):
    """Exact NLL = group NLL + within-group NLL at correctly shifted target positions."""
    valid = labels != -100
    z, y = logits.float()[valid], labels[valid]
    groups = vocab_groups.to(y.device)
    g = groups[y]
    lp = z.log_softmax(-1)
    gold = -lp.gather(-1, y[:, None]).squeeze(-1)
    group_nll = torch.empty_like(gold)
    for tag in torch.unique(g):
        rows = g == tag
        group_nll[rows] = -torch.logsumexp(lp[rows][:, groups == tag], -1)
    return {'gold_nll': gold, 'group_nll': group_nll, 'within_group_nll': gold-group_nll,
            'target_group': g}


def fit_probe(states, labels, seed=42):
    from sklearn.linear_model import LogisticRegression
    x, y = states.detach().cpu().numpy(), labels.detach().cpu().numpy()
    keep = y >= 0
    if len(set(y[keep])) < 2:
        raise ValueError('probe fitting needs both groups')
    # Fit scaling only on training data; pipeline carries the frozen scaler into held-out evaluation.
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    p = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=seed))
    return p.fit(x[keep], y[keep])


def controlled_readout(train_base, train_adapted, train_y, held_base, held_adapted, held_y,
                       train_prefix, held_prefix, train_dialogues, held_dialogues):
    if set(train_dialogues) & set(held_dialogues):
        raise ValueError('probe train/evaluation dialogues overlap')
    base = fit_probe(train_base, train_y)
    refit = fit_probe(train_adapted, train_y)
    prefix = fit_probe(train_prefix, train_y)
    y = held_y.cpu().numpy()
    keep = y >= 0
    def acc(p, x):
        return float(p.score(x.detach().cpu().numpy()[keep], y[keep]))
    return {'base_heldout': acc(base, held_base), 'fixed_probe_on_adapted': acc(base, held_adapted),
            'refitted_probe_on_adapted': acc(refit, held_adapted), 'prefix_only': acc(prefix, held_prefix),
            'claim': 'held-out script readout; not causality'}
