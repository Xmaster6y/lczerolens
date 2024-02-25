"""
Training dictionaries
"""

import os

import torch as t
from sklearn.metrics import explained_variance_score, r2_score

from lczerolens.xai.helpers.sae import AutoEncoder

EPS = 1e-8


class ConstrainedAdam(t.optim.Adam):
    def __init__(self, params, constrained_params, **kwargs):
        super().__init__(params, **kwargs)
        self.constrained_params = list(constrained_params)

    def step(self, closure=None):
        with t.no_grad():
            for p in self.constrained_params:
                normed_p = p / p.norm(dim=1, keepdim=True)
                # project away the parallel component of the gradient
                p.grad -= (p.grad * normed_p).sum(
                    dim=0, keepdim=True
                ) * normed_p
        super().step(closure=closure)
        with t.no_grad():
            for p in self.constrained_params:
                # renormalize the constrained parameters
                p /= p.norm(dim=1, keepdim=True)


def entropy(p, eps=1e-8):
    p_sum = p.sum(dim=-1, keepdim=True)
    # epsilons for numerical stability
    p_normed = p / (p_sum + eps)
    p_log = t.log(p_normed + eps)
    ent = -(p_normed * p_log)

    # Zero out the entropy where p_sum is zero
    ent = t.where(p_sum > 0, ent, t.zeros_like(ent))

    return ent.sum(dim=-1).mean()


def sae_loss(
    activations,
    ae,
    sparsity_penalty,
    use_entropy=False,
    num_samples_since_activated=None,
    ghost_threshold=None,
    explained_variance=False,
    r2=False,
    use_constraint_loss=False,
    constraint_penalty=0.1,
):
    """
    Compute the loss of an autoencoder on some activations
    If num_samples_since_activated is not None, update it in place
    If ghost_threshold is not None, use it to do ghost grads
    """
    if isinstance(activations, tuple):
        in_acts, out_acts = activations
    else:
        in_acts = out_acts = activations

    ghost_grads = False
    if ghost_threshold is not None:
        if num_samples_since_activated is None:
            raise ValueError(
                "num_samples_since_activated must be provided for ghost grads"
            )
        ghost_mask = num_samples_since_activated > ghost_threshold
        if ghost_mask.sum() > 0:  # if there are dead neurons
            ghost_grads = True
        else:
            ghost_loss = None
    else:
        ghost_loss = None

    if not ghost_grads:  # if we're not doing ghost grads
        out = ae(in_acts, output_features=True)
        x_hat = out["x_hat"]
        f = out["features"]
        mse_loss = t.nn.MSELoss()(out_acts, x_hat).sqrt()

    else:  # if we're doing ghost grads
        out = ae(in_acts, output_features=True, ghost_mask=ghost_mask)
        x_hat = out["x_hat"]
        f = out["features"]
        x_ghost = out["x_ghost"]
        residual = out_acts - x_hat
        mse_loss = t.sqrt((residual**2).mean())
        x_ghost = (
            x_ghost
            * residual.norm(dim=-1, keepdim=True).detach()
            / (2 * x_ghost.norm(dim=-1, keepdim=True).detach() + EPS)
        )
        ghost_loss = t.nn.MSELoss()(residual.detach(), x_ghost).sqrt()

    if (
        num_samples_since_activated is not None
    ):  # update the number of samples since each neuron was last activated
        deads = (f == 0).all(dim=0)
        num_samples_since_activated.copy_(
            t.where(deads, num_samples_since_activated + 1, 0)
        )

    if use_entropy:
        sparsity_loss = entropy(f)
    else:
        sparsity_loss = f.norm(p=1, dim=-1).mean()

    out_losses = {"mse_loss": mse_loss, "sparsity_loss": sparsity_loss}
    classical_loss = mse_loss + sparsity_penalty * sparsity_loss
    out_losses["ghost_loss"] = ghost_loss
    if use_constraint_loss:
        # constraint less than 1
        constraint_loss = (f.norm(p=2, dim=-1) - 1).clamp(min=0).mean()
        out_losses["constraint_loss"] = constraint_loss
        classical_loss += constraint_penalty * constraint_loss
    else:
        out_losses["constraint_loss"] = 0
    if ghost_loss is None:
        out_losses["total_loss"] = classical_loss
    else:
        out_losses["total_loss"] = classical_loss + ghost_loss * (
            mse_loss.detach() / (ghost_loss.detach() + EPS)
        )
    if explained_variance:
        out_losses["explained_variance"] = explained_variance_score(
            out_acts.detach().cpu(), x_hat.detach().cpu()
        )
    if r2:
        out_losses["r2_score"] = r2_score(
            out_acts.detach().cpu(), x_hat.detach().cpu()
        )
    return out_losses


@t.no_grad
def resample_neurons(deads, activations, ae, optimizer):
    """
    resample dead neurons according to the following scheme:
    Reinitialize the decoder vector for each dead neuron to be an activation
    vector v from the dataset with probability proportional to ae's loss on v.
    Reinitialize all dead encoder vectors to be the mean alive encoder.
    Reset the bias vectors for dead neurons to 0.
    Reset the Adam parameters for the dead neurons to their default values.
    """
    if deads.sum() == 0:
        return
    if isinstance(activations, tuple):
        in_acts, out_acts = activations
    else:
        in_acts = out_acts = activations
    in_acts = in_acts.reshape(-1, in_acts.shape[-1])
    out_acts = out_acts.reshape(-1, out_acts.shape[-1])

    # compute the loss for each activation vector
    losses = (out_acts - ae(in_acts)["x_hat"]).norm(dim=-1)

    # resample decoder vectors for dead neurons
    indices = t.multinomial(losses, num_samples=deads.sum(), replacement=True)
    ae.W_dec[deads] = out_acts[indices]
    ae.W_dec /= ae.W_dec.norm(dim=1, keepdim=True)

    # resample encoder vectors for dead neurons
    ae.W_enc[:, deads] = ae.W_enc[:, ~deads].mean(dim=0) * 0.2

    # reset bias vectors for dead neurons
    ae.b_enc[:, deads] = 0.0

    # reset Adam parameters for dead neurons
    state_dict = optimizer.state_dict()["state"]
    # # encoder weight
    state_dict[1]["exp_avg"][deads] = 0.0
    state_dict[1]["exp_avg_sq"][deads] = 0.0
    # # encoder bias
    state_dict[2]["exp_avg"][deads] = 0.0
    state_dict[2]["exp_avg_sq"][deads] = 0.0


def trainSAE(
    train_dataloader,
    activation_dim,
    dictionary_size,
    lr,
    sparsity_penalty,
    *,
    beta1=0.9,
    beta2=0.999,
    n_epochs=1,
    pre_bias=False,
    val_dataloader=None,
    entropy=False,
    warmup_steps=1000,
    cooldown_steps=1000,
    resample_steps=None,
    ghost_threshold=None,
    save_steps=None,
    val_steps=None,
    save_dir=None,
    log_steps=1000,
    device="cpu",
    from_checkpoint=None,
    wandb=None,
    do_print=True,
    use_constraint_optim=False,
    use_constraint_loss=False,
    constraint_penalty=0.1,
):
    """
    Train and return a sparse autoencoder
    """
    ae = AutoEncoder(activation_dim, dictionary_size, pre_bias=pre_bias).to(
        device
    )
    if from_checkpoint is not None:
        loaded = t.load(from_checkpoint)
        if isinstance(loaded, t.nn.Module):
            ae.load_state_dict(loaded.state_dict())
        else:
            ae.load_state_dict(loaded)

    num_samples_since_activated = t.zeros(dictionary_size, dtype=int).to(
        device
    )  # how many samples since each neuron was last activated?

    # set up optimizer and scheduler
    optimizer = ConstrainedAdam(
        ae.parameters(),
        [ae.W_dec] if use_constraint_optim else [],
        lr=lr,
        betas=(beta1, beta2),
    )
    total_steps = n_epochs * len(train_dataloader)

    def lr_fn(step):
        # cooldown
        cooldown_ratio = min(1.0, (total_steps - step) / cooldown_steps)

        # warmup
        if resample_steps is not None:
            ini_step = step % resample_steps
        else:
            ini_step = step
        return min(ini_step / warmup_steps, 1.0) * cooldown_ratio

    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)
    step = 0
    for _ in range(n_epochs):
        for acts in train_dataloader:
            step += 1

            if isinstance(acts, t.Tensor):  # typical casse
                acts = acts.to(device)
            elif isinstance(acts, tuple):
                acts = tuple(a.to(device) for a in acts)

            optimizer.zero_grad()
            # updates num_samples_since_activated in place
            loss = sae_loss(
                acts,
                ae,
                sparsity_penalty,
                use_constraint_loss=use_constraint_loss,
                constraint_penalty=constraint_penalty,
                use_entropy=entropy,
                num_samples_since_activated=num_samples_since_activated,
                ghost_threshold=ghost_threshold,
            )
            loss["total_loss"].backward()
            optimizer.step()
            scheduler.step()

            # deal with resampling neurons
            if resample_steps is not None and step % resample_steps == 0:
                resample_neurons(
                    num_samples_since_activated > resample_steps / 2,
                    acts,
                    ae,
                    optimizer,
                )

            # logging
            if log_steps is not None and step % log_steps == 0:
                with t.no_grad():
                    losses = sae_loss(
                        acts,
                        ae,
                        sparsity_penalty,
                        entropy,
                        use_constraint_loss=use_constraint_loss,
                        constraint_penalty=constraint_penalty,
                        num_samples_since_activated=(
                            num_samples_since_activated
                        ),
                        ghost_threshold=ghost_threshold,
                    )
                    if wandb is not None:
                        wandb.log({f"train/{k}": l for k, l in losses.items()})
                    if do_print:
                        print(f"[INFO] Train step {step}: {losses}")
            if (
                save_steps is not None
                and save_dir is not None
                and step % save_steps == 0
            ):
                if not os.path.exists(os.path.join(save_dir, "checkpoints")):
                    os.mkdir(os.path.join(save_dir, "checkpoints"))
                t.save(
                    ae.state_dict(),
                    os.path.join(save_dir, "checkpoints", f"ae_{step}.pt"),
                )
            if val_steps is not None and val_dataloader is not None:
                with t.no_grad():
                    if step % val_steps == 0:
                        val_losses = {
                            "total_loss": 0,
                            "mse_loss": 0,
                            "sparsity_loss": 0,
                            "constraint_loss": 0,
                            "explained_variance": 0,
                            "r2_score": 0,
                        }
                        for val_acts in val_dataloader:
                            losses = sae_loss(
                                val_acts.to(device),
                                ae,
                                sparsity_penalty,
                                use_entropy=entropy,
                                use_constraint_loss=use_constraint_loss,
                                constraint_penalty=constraint_penalty,
                                num_samples_since_activated=(
                                    num_samples_since_activated
                                ),
                                ghost_threshold=ghost_threshold,
                                explained_variance=True,
                                r2=True,
                            )
                            for k, _ in val_losses.items():
                                val_losses[k] += losses[k]

                        for k, v in val_losses.items():
                            val_losses[k] /= len(val_dataloader)
                        if wandb is not None:
                            wandb.log(
                                {f"val/{k}": l for k, l in val_losses.items()}
                            )
                        if do_print:
                            print(f"[INFO] Val step {step}: {val_losses}")

    return ae
