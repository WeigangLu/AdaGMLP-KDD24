import copy
import torch
import numpy as np
# from pathlib import Path
import torch.nn.functional as F
from utils import *
import time

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Training for teacher GNNs
def train(model, g, feats, labels, criterion, optimizer, idx):
    model.train()

    _, logits = model(g, feats)
    out = logits.log_softmax(dim=1)
    loss = criterion(out[idx], labels[idx])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# Testing for teacher GNNs
def evaluate(model, g, feats):
    model.eval()

    with torch.no_grad():
        _, logits = model(g, feats)
        out = logits.log_softmax(dim=1)

    return logits, out


# Training for student MLPs
def train_mini_batch(model, feats, labels, out_t_all, criterion_l, criterion_t, optimizer, idx, param):
    model.train()

    _, logits = model(None, feats)
    out = logits.log_softmax(dim=1)
    loss_l = criterion_l(out[idx], labels[idx])
    loss_t = 0
    loss_t = criterion_t((logits / param['tau']).log_softmax(dim=1),
                             (out_t_all / param['tau']).log_softmax(dim=1))

    loss = loss_l * param['lamb'] + loss_t * (1 - param['lamb'])

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    return loss_l.item() * param['lamb'], loss_t.item() * (1 - param['lamb'])


def adagmlp_train_mini_batch(model, feats, labels, out_t_all, criterion_l, criterion_t, optimizer, idx, param,
                             node_weights, alpha_vals):
    model.train()
    kl_loss = torch.nn.KLDivLoss(reduction="none", log_target=True)
    K = param['K']

    l_size = idx.shape[0]
    subset_size = l_size // K
    rand_idx = idx

    loss_l = 0
    loss_t = 0
    loss_a_out = 0
    loss_a_hid = 0
    if param['selective'] == 1:
        rand_idx = torch.randperm(l_size)

    for k in range(K):
        inputs, target_l, target_t = feats, labels, out_t_all
        idx_l = idx

        # random classification
        if param['selective'] == 1:
            start_idx = k * subset_size
            end_idx = start_idx + subset_size
            sub_idx = rand_idx[rand_idx[start_idx:end_idx]]
            idx_l = idx[sub_idx]

        masked_inputs = None
        if param['aug_feat_missing_rate'] > 0:
            masked_inputs = inputs.clone()
            masked_inputs = mask_features(masked_inputs[idx_l], param['aug_feat_missing_rate'])
            inputs = torch.cat([inputs, masked_inputs], dim=0)  # concat the raw feats and masked feats

        hidden, logits = model(None, inputs, k)

        # classification loss
        out_l = logits.log_softmax(dim=1)
        loss_l += criterion_l(out_l[idx_l], target_l[idx_l])

        # alignment loss
        if param['aug_feat_missing_rate'] > 0:
            loss_a_out += F.mse_loss(out_l[idx_l], out_l[-idx_l.shape[0]:])
            for h in hidden[:-1]:
                loss_a_hid += F.mse_loss(h[idx_l], h[-idx_l.shape[0]:]) / len(hidden[:-1])

        # KD loss
        kd_logits = logits[:-idx_l.shape[0]] if param['aug_feat_missing_rate'] > 0 else logits
        soften_logits = kd_logits / param['tau']
        soften_target = target_t / param['tau']
        P = (soften_logits).log_softmax(dim=1)
        Q = (soften_target).log_softmax(dim=1)
        loss_t += (kl_loss(P, Q).sum(1) * node_weights).sum()

        node_weights, alpha = adagmlp_undate_weights(kd_logits, target_t, node_weights, param)
        alpha_vals[k] = alpha

    loss_l /= K
    loss_t /= K
    loss_a_out /= K
    loss_a_hid /= K
    alpha_vals /= alpha_vals.sum()

    loss = (loss_l * param['lamb'] + loss_t * (1 - param['lamb']) +
            loss_a_out * param['lamb_a'] + loss_a_hid * (1 - param['lamb_a']))

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    return loss_l.item() * param['lamb'], loss_t.item() * (1 - param['lamb']), node_weights, alpha_vals


# Testing for student MLPs
def evaluate_mini_batch(model, feats):
    model.eval()

    with torch.no_grad():
        _, logits = model(None, feats)
        out = logits.log_softmax(dim=1)

    return logits, out


def adagmlp_evaluate_mini_batch(model, feats, alpha_vals, K):
    model.eval()
    # criterion = torch.nn.KLDivLoss(reduction="bacthmean", log_target=True)
    with torch.no_grad():
        pred_list = []
        for k in range(0, K):
            _, logits = model(None, feats, k)
            out = logits.log_softmax(dim=1)
            pred_list.append(out.softmax(dim=1))

        pred_all = torch.stack(pred_list)
        alpha = alpha_vals.unsqueeze(1).unsqueeze(1)

        logits = (pred_all * alpha).sum(dim=0)
        out = torch.log(logits + 1e-16)

    return logits, out


def adagmlp_undate_weights(
        logits_s, logits_t, node_weights, param
):
    beta = param["beta"]
    criterion = torch.nn.KLDivLoss(reduction="none", log_target=True)
    with torch.no_grad():
        out_s = logits_s.log_softmax(dim=1)
        out_t = logits_t.log_softmax(dim=-1)
        loss = criterion(out_s, out_t).sum(1)
        errors = 1 - torch.exp(-beta * loss)  # torch.sigmoid(loss)
        error = torch.sum(node_weights * errors) / torch.sum(node_weights)
        error = error + 1e-16
        alpha = max(torch.log((1 - error) / error + 1e-16), 1e-16)
        node_weights = node_weights * torch.exp(alpha * errors)
        node_weights /= node_weights.sum()

    return node_weights, alpha


def train_teacher(param, model, g, feats, labels, indices, criterion, evaluator, optimizer):
    device = get_device(param)
    if param['exp_setting'] == 'tran':
        idx_train, idx_val, idx_test = indices
    else:
        obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices
        obs_feats = feats[idx_obs]
        obs_labels = labels[idx_obs]
        obs_g = g.subgraph(idx_obs).to(device)

    g = g.to(device)

    es = 0
    val_best = 0
    test_val = 0
    test_best = 0

    for epoch in range(1, param["max_epoch"] + 1):
        if param['exp_setting'] == 'tran':
            train_loss = train(model, g, feats, labels, criterion, optimizer, idx_train)
            _, out = evaluate(model, g, feats)
            train_acc = evaluator(out[idx_train], labels[idx_train])
            val_acc = evaluator(out[idx_val], labels[idx_val])
            test_acc = evaluator(out[idx_test], labels[idx_test])
        else:
            train_loss = train(model, obs_g, obs_feats, obs_labels, criterion, optimizer, obs_idx_train)
            _, obs_out = evaluate(model, obs_g, obs_feats)
            train_acc = evaluator(obs_out[obs_idx_train], obs_labels[obs_idx_train])
            val_acc = evaluator(obs_out[obs_idx_val], obs_labels[obs_idx_val])
            _, out = evaluate(model, g, feats)
            test_acc = evaluator(out[idx_test_ind], labels[idx_test_ind])

        if epoch % 1 == 0:
            print("\033[0;30;46m [{}] CLA: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, "
                  "Test Val: {:.4f}, Test Best: {:.4f}\033[0m".format(
                epoch, train_loss, train_acc, val_acc, test_acc, val_best, test_val, test_best))

        if test_acc > test_best:
            test_best = test_acc

        if val_acc >= val_best:
            val_best = val_acc
            test_val = test_acc
            state = copy.deepcopy(model.state_dict())
            es = 0
        else:
            es += 1

        if es == 50:
            print("Early stopping!")
            break

    model.load_state_dict(state)
    model.eval()
    if param['exp_setting'] == 'tran':
        out, _ = evaluate(model, g, feats)
    else:
        obs_out, _ = evaluate(model, obs_g, obs_feats)
        out, _ = evaluate(model, g, feats)
        out[idx_obs] = obs_out

    return out, test_acc, test_val, test_best, state


def train_student(param, model, g, feats, labels, out_t_all, indices, criterion_l, criterion_t, evaluator, optimizer):
    device = get_device(param)
    if param['exp_setting'] == 'tran':
        idx_train, idx_val, idx_test = indices
    else:
        obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices
        obs_feats = feats[idx_obs]
        obs_labels = labels[idx_obs]
        obs_out_t = out_t_all[idx_obs]

    es = 0
    val_best = 0
    test_val = 0
    test_best = 0

    node_weights = torch.ones(feats.shape[0])
    if param['exp_setting'] == 'ind':
        node_weights = node_weights[idx_obs]
    node_weights /= node_weights.sum()
    node_weights = node_weights.to(device)
    alpha_vals = torch.ones(param["K"]).to(device)

    for epoch in range(1, param["max_epoch"] + 1):
        if param['exp_setting'] == 'tran':
            if param["student"] == "AdaGMLP":
                loss_l, loss_t, node_weights, alpha_vals = adagmlp_train_mini_batch(model, feats, labels, out_t_all,
                                                                                    criterion_l, criterion_t,
                                                                                    optimizer, idx_train, param,
                                                                                    node_weights, alpha_vals)

                logits_s, out = adagmlp_evaluate_mini_batch(model, feats, alpha_vals, param["K"])

            else:
                loss_l, loss_t = train_mini_batch(model, feats, labels, out_t_all, criterion_l, criterion_t,
                                                  optimizer, idx_train, param)
                logits_s, out = evaluate_mini_batch(model, feats)
            train_acc = evaluator(out[idx_train], labels[idx_train])
            val_acc = evaluator(out[idx_val], labels[idx_val])
            test_acc = evaluator(out[idx_test], labels[idx_test])

        else:
            if param["student"] == "AdaGMLP":
                loss_l, loss_t, node_weights, alpha_vals = adagmlp_train_mini_batch(model, obs_feats, obs_labels,
                                                                                    obs_out_t,
                                                                                    criterion_l, criterion_t,
                                                                                    optimizer, obs_idx_train, param,
                                                                                    node_weights, alpha_vals)

                logits_s, obs_out = adagmlp_evaluate_mini_batch(model, feats, alpha_vals, param["K"])
            else:
                loss_l, loss_t = train_mini_batch(model, obs_feats, obs_labels, obs_out_t, criterion_l,
                                                  criterion_t, optimizer, obs_idx_train, param)
                logits_s, obs_out = evaluate_mini_batch(model, obs_feats)
            train_acc = evaluator(obs_out[obs_idx_train], obs_labels[obs_idx_train])
            val_acc = evaluator(obs_out[obs_idx_val], obs_labels[obs_idx_val])
            _, out = evaluate_mini_batch(model, feats)
            test_acc = evaluator(out[idx_test_ind], labels[idx_test_ind])

        if epoch % 1 == 0:
            print("\033[0;30;43m [{}] CLA: {:.5f}, KD: {:.5f}, Total: {:.5f} | Train: {:.4f}, Val: {:.4f}, "
                  "Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f} \033[0m".format(
                epoch, loss_l, loss_t, loss_l + loss_t, train_acc, val_acc, test_acc, val_best, test_val, test_best,
            ))

        if test_acc > test_best:
            test_best = test_acc

        if val_acc >= val_best:
            val_best = val_acc
            test_val = test_acc
            state = copy.deepcopy(model.state_dict())
            es = 0
        else:
            es += 1

        if es == 50:
            print("Early stopping!")
            break

    model.load_state_dict(state)
    model.eval()
    inference_time = 9999
    if param['exp_setting'] == 'tran':
        start_time = time.time()
        if param["student"] == "AdaGMLP":
            out, _ = adagmlp_evaluate_mini_batch(model, feats, alpha_vals, param["K"])
        else:
            out, _ = evaluate_mini_batch(model, feats)
        end_time = time.time()
        inference_time = end_time - start_time
    else:
        start_time = time.time()
        obs_out, _ = evaluate_mini_batch(model, obs_feats)
        out, _ = evaluate_mini_batch(model, feats)
        out[idx_obs] = obs_out
        end_time = time.time()
        inference_time = end_time - start_time

    print(f"Inference time: {inference_time* 1000} ms")

    return out, test_acc, test_val, test_best, state
