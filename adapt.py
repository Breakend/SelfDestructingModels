from torch.optim import Adam, RMSprop, SGD, Adadelta
import higher
from higher.patch import monkeypatch as make_functional
# from model import LinearPredictor, PredictorAndModel
from losses import cls_loss, cls_acc
from utils import ragged_average, wandb_plot_list, average_dicts, moving_average
import numpy as np
import copy
import hyperopt
from hyperopt import hp, tpe, fmin, rand
import logging
import random
import torch
from utils import CustomGradientCheckpointMAML
import os
import torch_maml

LOG = logging.getLogger(__name__)


GRAD_NORM = None


def get_clip_grads_fn(config):
    def _clip(grads):
        global GRAD_NORM
        GRAD_NORM = (1e-8 + sum([g.pow(2).sum() for g in grads])).sqrt()
        if GRAD_NORM > config.maml_max_grad_norm:
            return [(g / GRAD_NORM.detach()) * config.maml_max_grad_norm for g in grads]
        else:
            return grads
    return _clip


def fine_tuning_train(model, sampler, steps, lr, config, output_key, outer_batch, adversarial_params=None):
    if config.bad_adapt_mode == "maml":
        return fine_tuning_train_maml(model, sampler, steps, lr, config, output_key, outer_batch, adversarial_params=adversarial_params)
    elif config.bad_adapt_mode == "torch_maml":
        return fine_tuning_train_mem_efficient_maml(model, sampler, steps, lr, config, output_key, outer_batch, adversarial_params=adversarial_params)
    # elif config.bad_adapt_mode == "concept":
    #     return concept_fine_tuning(model, sampler, steps, lr, config, output_key, outer_x=outer_x)
    else:
        raise ValueError(f"{config.bad_adapt_mode} not supported")

def fine_tuning_train_mem_efficient_maml(model, sampler, steps, lr, config, output_key, outer_batch, adversarial_params=None):
    outer_x = outer_batch["inputs"]
    info = {"loss": [], "acc": []}
    # We must preload the batches
    sampler = sampler if config.adversary.n_examples == -1 else sampler.random_subset(config.adversary.n_examples)
    batches = []
    for i in range(steps):
        batch = sampler.sample("cpu")
        x, y = batch["inputs"], batch[output_key]
        batches.append([x,y])
    
    def compute_loss(model_inner, data):
        x, y = data
        # TODO: do we need to be moving x,y to device?
        preds = model_inner(**x.to(config.device))[output_key]
        loss = cls_loss(y.to(config.device), preds)
        return loss
    
    # TODO: make this all meta-learned so we don't have a fixed thing
    optimizer = torch_maml.IngraphRMSProp(learning_rate=lr, beta=0.9, epsilon=1e-5)

    inner_model = model
        
    efficient_maml = CustomGradientCheckpointMAML(
        inner_model, compute_loss, optimizer=optimizer, checkpoint_steps=5)
    
    fm, loss_history, final_optimizer_state, output = efficient_maml(batches, max_grad_grad_norm=config.maml_max_grad_norm, outer_x=outer_x)

    info = {"loss": loss_history, "acc": []}

    for batch in batches:
        x, y = batch
        preds = fm(**x.to(config.device))[output_key]
        acc = cls_acc(y.to(config.device), preds)
        info["acc"].append(acc.item())

    inner_preds = output[output_key]
    return fm, inner_preds, output["reps"], info

def fine_tuning_train_maml(model, sampler, steps, lr, config, output_key, outer_batch, adversarial_params=None):
    global GRAD_NORM
    opts = {"reset_head": False, "freeze_base": False}
    outer_x, outer_y = outer_batch["inputs"], outer_batch[output_key]

    if config.inner_loop_learn_adv_head:
        inner_model = model.with_linear_reset(adversarial_params["linear_head"])
        fm0 = make_functional(inner_model, copy_initial_weights=False) # why do we need this??
        opts["reset_head"] = True
    else:
        inner_model = model

    if outer_x is not None:
        outer_base_outputs = fm0(**outer_x)
        outer_preds = [outer_base_outputs[output_key]]
        outer_reps = [outer_base_outputs["reps"]]
    else:
        outer_preds = []
        outer_reps = []

    inner_grad_norms = []

    if steps > 0:    
        if config.inner_loop_freeze_base and random.choice([True, False]):
            inner_params = inner_model.linear.parameters()
            opts["freeze_base"] = True
        else:
            inner_params = inner_model.parameters()

        batch_size = random.choice(list(config.batch_size_search_space))

        optimizer = Adam(inner_params, lr=1e9 if config.inner_loop_learn_lr else lr)  # we override the learning rate selected here in the innerloop_ctx line
        override = {"lr": lr} if config.inner_loop_learn_lr else None
        sampler = sampler if config.adversary.n_examples_inner == -1 else sampler.random_subset(config.adversary.n_examples_inner)
        with higher.innerloop_ctx(inner_model, optimizer, copy_initial_weights=False, override=override) as (fm, do):
            for _ in range(steps):
                if config.param_noise > 0:
                    fm.update_params([p + torch.randn_like(p) * config.param_noise for p in fm.parameters()])
                batch = sampler.sample(config.device, batch_size=batch_size)
                x, y = batch["inputs"], batch[output_key]
                preds = fm(**x)[output_key]
                loss = cls_loss(y, preds)
                do.step(loss, grad_callback=get_clip_grads_fn(config))
                inner_grad_norms.append(GRAD_NORM)
                GRAD_NORM = None # needed so we don't have a dangling pointer to the gradient history for the gradient norm?

                if outer_x is not None:
                    outer_outputs = fm(**outer_x)
                    outer_preds.append(outer_outputs[output_key])
                    outer_reps.append(outer_outputs["reps"])
    else:
        fm = fm0
    
    last_adapted_bad_loss = cls_loss(outer_y, outer_preds[-1])
    grads = torch.autograd.grad(last_adapted_bad_loss, fm.parameters(), create_graph=True, retain_graph=True)
    last_grad_norm = torch.cat([g.view(-1) for g in grads]).norm(2)
    inner_grad_norms.append(last_grad_norm)

    return fm, torch.stack(outer_preds), torch.stack(outer_reps), torch.stack(inner_grad_norms), opts


def fine_tune_evaluation(model, max_steps, val_sampler, test_sampler, lr, config, eval_key, optimizer="adam", 
                         freeze_intermediate=False, reset_head=False, deterministic=False, batch_size=None,
                         return_model=False, seed=0, full_test_set=False):
    if deterministic:
        rstate = random.getstate()
        npstate = np.random.get_state()
        tstate = torch.random.get_rng_state() # WARNING: ONLY WORKS FOR CPU TENSORS, not GPU!
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    max_steps = int(max_steps)  # because of hyperopt quniform gives floats
    best_test_acc = None
    best_test_loss = None
    LOG.info(f"lr: {lr} steps: {max_steps} reset_head: {reset_head} freeze: {freeze_intermediate} batch_size: {batch_size} optimizer: {optimizer}")
    model_copy = copy.deepcopy(model)
    
    if reset_head:
        model_copy = model_copy.with_linear_reset()

    if freeze_intermediate:
        ft_params = model_copy.linear.parameters()
    else:
        ft_params = model_copy.parameters()

    if optimizer == "adam":    
        optimizer = Adam(ft_params, lr=lr)
    elif optimizer == "rmsprop":
        optimizer = RMSprop(ft_params, lr=lr)
    elif optimizer == "sgd":
        optimizer = SGD(ft_params, lr=lr)
    elif optimizer == "sgdm":
        optimizer = SGD(ft_params, lr=lr, momentum=0.9)
    elif optimizer == "adadelta":
        optimizer = Adadelta(ft_params, lr=lr)
    else:
        raise ValueError(f"Optimizer {str(optimizer)} not supported.")

    info = {"loss_curve": [], "acc_curve": [], "test_loss_curve": [], "test_acc_curve": []}
    sampler = val_sampler if config.adversary.n_examples == -1 else val_sampler.random_subset(config.adversary.n_examples)
    if batch_size is None:
        batch_size = config.batch_size
    for step in range(max_steps):
        batch = sampler.sample(config.device, batch_size=batch_size)
        x, y = batch["inputs"], batch[eval_key]
        preds = model_copy(**x)[eval_key]
        loss = cls_loss(y, preds)
        acc = cls_acc(y, preds)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        info["loss_curve"].append(loss.item())
        info["acc_curve"].append(acc.item())

        if step % config.adversary.eval_period == 0 and step != 0:
            new_test_loss, new_test_acc = get_test_loss(model_copy, test_sampler, config.device, eval_key, config.adversary.eval_steps, full_test_set=full_test_set)
            info["test_loss_curve"].append(new_test_loss.item())
            info["test_acc_curve"].append(new_test_acc.item())
            if best_test_acc is None or new_test_acc.item() > best_test_acc:
                best_test_acc = new_test_acc
                best_test_loss = new_test_loss
            elif config.adversary.early_stop:
                LOG.info(f"EARLY STOPPING {step} {new_test_acc.item()} {best_test_acc}")
                break

    if best_test_acc is None:
        best_test_loss, best_test_acc = get_test_loss(model_copy, test_sampler, config.device, eval_key, config.adversary.eval_steps, full_test_set=full_test_set)

    info["real_loss"] = info["loss_curve"][-1]
    info["loss"] = -1 * best_test_acc
    info['status'] = hyperopt.STATUS_OK
    info["best_test_loss"] = best_test_loss
    info["best_test_acc"] = best_test_acc
    info["percentile"] = {}
    info["test_percentile"] = {}
    for acc_target in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        window_size = 10

        met_target = [x >= acc_target for x in moving_average(info["acc_curve"], window_size)]
        test_met_target = [x >= acc_target for x in info["test_acc_curve"]]

        # get first instance where we hit the target
        info["percentile"][f"{acc_target}"] = (met_target.index(True) + (window_size-1)) * batch_size if True in met_target else max_steps * batch_size
        info["test_percentile"][f"{acc_target}"] = (1 + test_met_target.index(True)) * config.adversary.eval_period * batch_size if True in test_met_target else max_steps * batch_size

    if deterministic:
        random.setstate(rstate)
        np.random.set_state(npstate)
        torch.random.set_rng_state(tstate)

    if return_model:
        return model_copy
    else:
        return info

def get_test_loss(fm, sampler, device, key, steps=50, full_test_set=False):
    fixed_sampler = sampler.random_subset(fixed_order=True)
    test_losses = []
    test_accs = []
    for _ in range(steps if not full_test_set else len(fixed_sampler) // fixed_sampler.batch_size):
        batch = fixed_sampler.sample(device)
        if batch is None:
            # did a full epoch
            break
        x, y = batch["inputs"], batch[key]
        preds = fm(**x)[key]
        test_losses.append(cls_loss(y, preds).item())
        test_accs.append(cls_acc(y, preds).item())
    return np.mean(test_losses), np.mean(test_accs)

def adapt_model_evaluation(model, val_sampler, test_sampler, config, key, return_model=False, full_test_set=False):
    OPT_EARLY_STOP = config.adversary.hparam_early_stop_trials
    def stop_fn(trials, best_loss=None, iteration_no_progress=0):
        new_loss = trials.trials[len(trials.trials) - 1]["result"]["loss"]
        if best_loss is None:
            return new_loss <= -0.99, [new_loss, iteration_no_progress + 1]
        if new_loss < best_loss:
            best_loss = new_loss
            iteration_no_progress = 0
        else:
            iteration_no_progress += 1

        return iteration_no_progress >= OPT_EARLY_STOP or new_loss == -1.0, [best_loss, iteration_no_progress],

    rstate = np.random.default_rng(config.seed)
    optimizer_search_space = config.eval_optimizer_search_space
    reset_head_search_space = [True] if config.eval_random_head_reset else [False]
    freeze_intermediate_search_space = [True, False] if config.eval_freeze_base else [False]
    batch_size_search_space = list(config.batch_size_search_space)
    log_lr_search_space = [np.log(config.eval_lr_search_range[0]), np.log(config.eval_lr_search_range[1])]

    def opt_fn(x):
        return fine_tune_evaluation(model, x[1], val_sampler, test_sampler, x[0], config, key, optimizer=x[5],
                                    freeze_intermediate = x[4], reset_head=x[2], deterministic=True, batch_size=x[3],
                                    seed=config.seed, full_test_set=full_test_set)

    tuning_info = fmin(fn=opt_fn,
                       space=[hp.loguniform('lr', *log_lr_search_space), 
                              hp.quniform('max_steps', *config.adversary.ft_steps_range), 
                              hp.choice("reset_head", reset_head_search_space), 
                              hp.choice("batch_size", batch_size_search_space), 
                              hp.choice("freeze_intermediate", freeze_intermediate_search_space),
                              hp.choice("optimizer", optimizer_search_space)],
                       algo=tpe.suggest,
                       max_evals=config.hyperopt_max_evals,
                       early_stop_fn=stop_fn,
                       show_progressbar=False, rstate=rstate)

    if return_model:
        return fine_tune_evaluation(model, tuning_info["max_steps"], 
                                        val_sampler, 
                                        test_sampler, 
                                        tuning_info['lr'], 
                                        config, 
                                        key,
                                        optimizer = optimizer_search_space[tuning_info["optimizer"]],
                                        freeze_intermediate=freeze_intermediate_search_space[tuning_info["freeze_intermediate"]],
                                        reset_head=reset_head_search_space[tuning_info["reset_head"]], 
                                        batch_size=batch_size_search_space[tuning_info["batch_size"]],
                                        deterministic=True, seed=0, return_model=True)
    else:
        eval_runs = [fine_tune_evaluation(model, 
                                        tuning_info["max_steps"], 
                                        val_sampler, 
                                        test_sampler, 
                                        tuning_info['lr'], 
                                        config, 
                                        key,
                                        optimizer = optimizer_search_space[int(tuning_info["optimizer"])],
                                        freeze_intermediate=freeze_intermediate_search_space[tuning_info["freeze_intermediate"]],
                                        reset_head=reset_head_search_space[tuning_info["reset_head"]], 
                                        batch_size=batch_size_search_space[tuning_info["batch_size"]],
                                        deterministic=True, seed=seed, full_test_set=full_test_set) 
                                        for seed in range(config.post_hyperopt_eval_runs)]

        info = {
            "loss" : np.mean([x["best_test_loss"] for x in eval_runs]),
            "acc" : np.mean([x["best_test_acc"] for x in eval_runs]),
            "loss_std" : np.std([x["best_test_loss"] for x in eval_runs]),
            "acc_std" : np.std([x["best_test_acc"] for x in eval_runs]),
            "hyperopt/max_steps": tuning_info["max_steps"],
            "hyperopt/lr": tuning_info["lr"],
            "hyperopt/optimizer" : optimizer_search_space[int(tuning_info["optimizer"])],
            "hyperopt/batch_size" : batch_size_search_space[tuning_info["batch_size"]],
            "hyperopt/reset_head" : int(reset_head_search_space[tuning_info["reset_head"]]),
            "hyperopt/freeze_intermediate" : int(freeze_intermediate_search_space[tuning_info["freeze_intermediate"]]),
            **average_dicts([x["percentile"] for x in eval_runs], prefix=f"{key}_solve_datapoints/"),
            **average_dicts([x["test_percentile"] for x in eval_runs], prefix=f"{key}_test_solve_datapoints/"),
        }

        if not config.debug and not config.eval_only:
            info[f"AVG_LOSS/{key}"] = wandb_plot_list(ragged_average([run["loss_curve"] for run in eval_runs]), title=f"AVG_LOSS/{key}")
            info[f"AVG_LOSS/{key}_test"] = wandb_plot_list(ragged_average([run["test_loss_curve"] for run in eval_runs]), title=f"AVG_LOSS/{key}_test")
            info[f"AVG_ACC/{key}"] = wandb_plot_list(ragged_average([run["acc_curve"] for run in eval_runs]), title=f"AVG_ACC/{key}")
            info[f"AVG_ACC/{key}_test"] = wandb_plot_list(ragged_average([run["test_acc_curve"] for run in eval_runs]), title=f"AVG_ACC/{key}_test")

            for idx, run in enumerate(eval_runs):
                info[f"{key}_eval_loss_curves/curve{idx}"] = wandb_plot_list(run["loss_curve"], title=f"eval_loss{idx}_{key}")
                info[f"{key}_eval_acc_curves/curve{idx}"] = wandb_plot_list(run["acc_curve"], title=f"eval_acc{idx}_{key}")
                info[f"{key}_test_eval_loss_curves/curve{idx}"] = wandb_plot_list(run["test_loss_curve"], title=f"test_eval_loss{idx}")
                info[f"{key}_test_eval_acc_curves/curve{idx}"] = wandb_plot_list(run["test_acc_curve"], title=f"test_eval_acc{idx}")

        return info


def get_adapted_predictions(model, inner_sampler, outer_batch, config, output_key, current_step, adversarial_params={}):
    if config.use_inner_loop_schedule:
        # Use a schedule for taking more steps
        top = min(config.max_adapt_steps, int(current_step / config.inner_loop_schedule)) + 1
        bottom = max(0, top - config.inner_loop_lag)
        steps = 1 + np.random.randint(bottom, top)
    else:
        steps = config.max_adapt_steps

    if not config.inner_loop_update_every_step:
        outer_x = None

    # def adapt_model(model, sampler, config, max_steps, output_key, outer_x=None, meta_learned_hyperparams=None):
    if "lr" in adversarial_params:
        lr = adversarial_params["lr"]
    else:
        log_lr_lo, log_lr_hi = np.log(config.train_lr_search_range[0]), np.log(config.train_lr_search_range[1])
        lr = float(np.exp(np.random.uniform(log_lr_lo, log_lr_hi)))

    return fine_tuning_train(model, inner_sampler, steps, lr, config, output_key, outer_batch, adversarial_params=adversarial_params)


