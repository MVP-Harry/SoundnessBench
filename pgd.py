import torch
import torch.nn.functional as F
from utils import clamp


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               upper_limit=0.1, lower_limit=-0.1, loss_type='margin'):
    grad_status = {p: p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)

    batch_size = X.size(0)
    delta_lower_bound = torch.max(torch.full_like(X, -epsilon), lower_limit - X)
    delta_upper_bound = torch.min(torch.full_like(X, epsilon), upper_limit - X)

    # Adding a restart dimension
    delta = torch.rand(batch_size, restarts, *X.shape[1:], device=X.device)
    delta = delta * (delta_upper_bound - delta_lower_bound).unsqueeze(1) + delta_lower_bound.unsqueeze(1)
    delta = delta.requires_grad_()

    max_loss = torch.zeros_like(y, dtype=torch.float32)
    max_delta = torch.empty_like(X).uniform_() * (delta_upper_bound - delta_lower_bound) + delta_lower_bound

    for _ in range(attack_iters):
        X_perturbed = X.unsqueeze(1) + delta  # (batch_size, restarts, ...)

        output = model(X_perturbed.view(-1, *X.shape[1:]))  # (batch_size * restarts, num_classes)
        output = output.view(batch_size, restarts, -1)  # (batch_size, restarts, num_classes)

        if loss_type == 'margin':
            logits_y = torch.gather(output, index=y.unsqueeze(1).unsqueeze(2).expand(-1, restarts, -1), dim=-1)
            margin_loss = (output - logits_y).sum(dim=-1)  # (batch_size, restarts)
            loss = margin_loss
        elif loss_type == 'ce':
            loss = F.cross_entropy(output.view(-1, output.size(-1)), y.repeat_interleave(restarts), reduction='none')
            loss = loss.view(batch_size, restarts)
        else:
            raise NotImplementedError(loss_type)

        all_loss = loss.max(1).values
        improved = all_loss >= max_loss
        max_loss[improved] = all_loss[improved]
        best_restarts = loss.argmax(dim=1)  # (batch_size,)
        max_delta[improved] = delta[improved, best_restarts[improved], ...]

        loss.backward(torch.ones_like(loss))
        grad = delta.grad.detach()
        d = clamp(delta + alpha * torch.sign(grad), delta_lower_bound.unsqueeze(1), delta_upper_bound.unsqueeze(1))
        delta.data.copy_(d)

        delta.grad.zero_()

    for p in model.parameters():
        p.requires_grad_(grad_status[p])

    return max_delta
