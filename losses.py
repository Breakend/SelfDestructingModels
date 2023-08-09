import torch
import torch.nn.functional as F
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np



def cls_loss(target, pred):
    if pred.dim() > target.dim() + 1:
        target = target.unsqueeze(0).repeat(pred.shape[0], 1)
    return -pred.log_softmax(-1).gather(-1, target.unsqueeze(-1)).squeeze(-1).mean()


def cls_acc(target, pred):
    if pred.dim() > target.dim() + 1:
        pred = pred[-1]
    return (target == pred.argmax(-1)).float().mean()


def entropy(p):
    return -(p.log_softmax(-1) * p.softmax(-1)).sum(-1).mean()


def confusion_loss(_, pred):
    ent = entropy(pred)
    max_entropy = torch.tensor(pred.shape[-1]).log()
    # To maximize entropy and give a non-negative loss
    return max_entropy - ent


def kl_penalty(target, pred):
    return (target.softmax(-1) * (target.log_softmax(-1) - pred.log_softmax(-1))).sum(-1).mean()


def get_cvxlayer(batch_size, n_classes, labels):
    w = cp.Variable((n_classes, n_classes))
    cp_logits = cp.Parameter((batch_size, n_classes))

    adversary_preds = cp_logits @ w
    obj_value = cp.sum(cp.log_sum_exp(adversary_preds, axis=1) - adversary_preds[np.arange(labels.shape[0]), labels.cpu().numpy()])
    problem = cp.Problem(cp.Minimize(obj_value), constraints=[w <= 1, w >= -1])
    return CvxpyLayer(problem, parameters=[cp_logits], variables=[w])


def linear_adversary_loss(logits, labels):
    logits = logits.view(-1, logits.shape[-1])

    assert labels.dim() == logits.dim() - 1  # labels should be a single integer (class) for each batch element

    cvxpylayer = get_cvxlayer(*logits.shape, labels)

    solution, = cvxpylayer(logits)
    pred = logits @ solution
    loss = F.cross_entropy(pred, labels)
    acc = (pred.argmax(-1) == labels).float().mean()

    return {
        "loss": loss,
        "solution": solution.detach(),
        "acc": acc.item()
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    import time
    pos_neg = torch.cat([torch.ones(3), -torch.ones(3)])
    labels = (pos_neg > 0).long()

    logits = torch.stack([pos_neg.flip(0), pos_neg], -1)
    logits += torch.randn_like(logits)
    logits = torch.nn.Parameter(logits)

    print(labels)
    print(logits)

    opt = torch.optim.Adam([logits], lr=1e-2)
    for idx in range(100):
        loss_dict = linear_adversary_loss(logits, labels)
        loss, solution = loss_dict["loss"], loss_dict["solution"]
        (-loss).backward()
        print(idx, loss.item(), end="\r")
        opt.step()
        opt.zero_grad()
    print(solution)
    print(logits)
    print(logits.softmax(-1))
