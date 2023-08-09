import typing
import wandb
from collections import namedtuple
from itertools import chain
import torch
import torch.nn as nn
import numpy as np
from torch.utils.checkpoint import checkpoint
from socket import gethostname
import os
import getpass
import struct

from torch_maml.optimizers import IngraphGradientDescent
from torch_maml.utils import copy_and_replace, do_not_copy, disable_batchnorm_stats, nested_flatten, nested_pack
from torch_maml.maml import NaiveMAML


def uuid(digits=8):
    if not hasattr(uuid, "uuid_value"):
        uuid.uuid_value = struct.unpack('I', os.urandom(4))[0] % int(10**digits)

    return uuid.uuid_value


def ragged_average(arrays):
    max_len = max(len(a) for a in arrays)
    stacked = np.stack([np.pad(a, (0, max_len - len(a)), constant_values=float('nan')) for a in arrays])
    return np.nanmean(stacked, axis=0)


def cache_dir():
    hostname = gethostname()

    if hostname.startswith("jag"):
        machine_name = hostname.split(".")[0]
        if not os.path.exists("/" + machine_name):
            raise RuntimeError(f"Expected directory /{machine_name} to exists, but it doesn't...")
        scratch_dirs = os.listdir(f"/{machine_name}")
        scratch_dirs.remove("scr-sync")
        if len(scratch_dirs) == 0:
            raise RuntimeError(f"No scratch directories in /{machine_name}?")
        path = "/" + machine_name + "/" + scratch_dirs[0]
    elif hostname.startswith("iris") or hostname.startswith("sphinx"):
        if os.path.exists("/scr-ssd"):
            path = "/scr-ssd"
        else:
            path = "/scr"
    else:
        raise RuntimeError(f"Couldn't automatically identify local disk for host {hostname}")

    user_cache_dir = path + "/" + getpass.getuser()

    if not os.path.exists(user_cache_dir):
        os.mkdir(user_cache_dir)

    return user_cache_dir

def params_same(params1, params2):
    for p1, p2 in zip(params1, params2):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

def flatten_dict(d):
    to_process = list(d.items())
    output = {}
    while len(to_process):
        k, v = to_process.pop()
        if isinstance(v, typing.MutableMapping):
            to_process.extend([(f"{k}.{k_}", v_) for (k_, v_) in v.items()])
        else:
            assert k not in output.keys(), "Somehow ended up with duplicate keys"
            output[k] = v

    return output

def wandb_plot_list(vals, title="unnamed plot"):
    data = [(idx, vals[idx]) for idx in range(len(vals))]
    table = wandb.Table(data=data, columns = ["x", "y"])
    return wandb.plot.line(table, "x", "y", title=title)


def average_dicts(dicts, prefix=""):
    avg_dict = {}
    for k in dicts[0].keys():
        avg = sum([d[k] for d in dicts]) / len(dicts)
        avg_dict[prefix + k] = avg
    return avg_dict

def dict_map(list_of_dicts, f):
    mapped = {}
    for k in list_of_dicts[0]:
        mapped[k] = f([d[k] for d in list_of_dicts])
    return mapped


def recursive_to(obj, dev):
    if torch.is_tensor(obj):
        return obj.to(dev)
    elif isinstance(obj, dict) or (hasattr(obj, "keys") and hasattr(obj, "values")):
        return { k: recursive_to(v, dev) for k, v in obj.items() }
    elif isinstance(obj, list):
        return [recursive_to(x, dev) for x in obj]
    else:
        return obj


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


class CustomGradientCheckpointMAML(NaiveMAML):
    Result = namedtuple('Result', ['model', 'loss_history', 'optimizer_state', 'output'])
    def __init__(self, *args, checkpoint_steps, **kwargs):
        """
        MAML: attempts to change model by performing gradient descent steps
        :param model: a torch module that will be updated
        :param loss_function: objective function(model(inputs), targets) that is minimized inside MAML
        :param optimizer: in-graph optimizer that creates updated copies of model
        :param checkpoint_steps: uses gradient checkpoints every *this many* steps
            Note: this parameter highly affects the memory footprint
        :param get_parameters: function(model) that returns a list of parameters affected by MAML updates
            Note: this function should always return parameters in the same order
        """
        super().__init__(*args, **kwargs)
        self.checkpoint_steps = checkpoint_steps

    def forward(self, inputs, opt_kwargs=None, loss_kwargs=None, optimizer_state=None, outer_x=None, **kwargs):
        """
        Apply optimizer to the model (out-of-place) and return an updated copy
        :param inputs: data that is fed into the model
        :param opt_kwargs: optional overrides for optimizer.get_initial_state
        :param optimizer_state: if specified, the optimizer starts with this state
        :param kwargs: extra parameters passed to optimizer.step
        :returns: updated_model, loss_history, optimizer_state
            * updated_model: a copy of model that was trained for len(inputs) steps, differentiable w.r.t. original
            * loss_history: a list of loss function values BEFORE each optimizer update; differentiable
            * optimizer_state: final state of the chosen optimizer AFTER the last step; you guessed it, differentiable
        :rtype: GradientCheckpointMAML.Result
        """
        assert len(inputs) > 0, "Non-empty inputs are required"
        opt_kwargs, loss_kwargs = opt_kwargs or {}, loss_kwargs or {}

        parameters_to_copy = list(self.get_parameters(self.model))
        parameters_not_to_copy = [param for param in chain(self.model.parameters(), self.model.buffers())
                                  if param not in set(parameters_to_copy)]

        if optimizer_state is None:
            optimizer_state = self.optimizer.get_initial_state(self.model, parameters=parameters_to_copy, **opt_kwargs)

        # initial maml state
        step_index = torch.zeros(1, requires_grad=True)
        initial_maml_state = (step_index, parameters_to_copy, optimizer_state)
        flat_maml_state = list(nested_flatten(initial_maml_state))

        # WARNING: this code treats parameters_to_copy and parameters_not_to_copy as global
        # variables for _maml_internal. Please DO NOT change or delete them in this function
        def _maml_internal(steps, *flat_maml_state):
            step_index, trainable_parameters, optimizer_state = \
                nested_pack(flat_maml_state, structure=initial_maml_state)
            updated_model = copy_and_replace(
                self.model, dict(zip(parameters_to_copy, trainable_parameters)), parameters_not_to_copy)

            is_first_pass = not torch.is_grad_enabled()
            # Note: since we use gradient checkpoining, this code will be executed two times:
            # (1) initial forward with torch.no_grad(), used to create checkpoints
            # (2) second forward with torch.enable_grad() used to backpropagate from those checkpoints
            # During first pass, we deliberately set detach=True to avoid creating inter-checkpoint graph

            inner_losses = []

            for _ in range(int(steps)):
                with torch.enable_grad(), disable_batchnorm_stats(updated_model), do_not_copy(*parameters_not_to_copy):
                    loss = self.loss_function(updated_model, inputs[int(step_index)], **loss_kwargs)
                    inner_losses.append(loss)
                    optimizer_state, updated_model = self.optimizer.step(
                        optimizer_state, updated_model, loss=loss, detach=is_first_pass,
                        parameters=self.get_parameters(updated_model), **kwargs)

                step_index = step_index + 1
            
            new_maml_state = (step_index, list(self.get_parameters(updated_model)), optimizer_state)
            outputs = (torch.stack(inner_losses), *nested_flatten(new_maml_state))
            return tuple(tensor if tensor.requires_grad else tensor.clone().requires_grad_(True) for tensor in outputs)

        loss_history = []
        # inner_preds1 = []
        # inner_preds2 = []
        # inner_reps = []
        inner_outs = []
        for chunk_start in range(0, len(inputs), self.checkpoint_steps):
            steps = min(self.checkpoint_steps, len(inputs) - chunk_start)
            inner_losses, *flat_maml_state = checkpoint(_maml_internal, torch.as_tensor(steps), *flat_maml_state)
            loss_history.extend(inner_losses.split(1))
            step_index, final_trainable_parameters, final_optimizer_state = \
                nested_pack(flat_maml_state, structure=initial_maml_state)
            intermediate_model = copy_and_replace(
                self.model, dict(zip(parameters_to_copy, final_trainable_parameters)), parameters_not_to_copy)
            inner_out = intermediate_model(**outer_x, output_reps=True)
            inner_outs.append(inner_out)
            # inner_preds1.append(preds1)
            # inner_preds2.append(preds2)
            # inner_reps.append(reps)

        step_index, final_trainable_parameters, final_optimizer_state = \
            nested_pack(flat_maml_state, structure=initial_maml_state)
        final_model = copy_and_replace(
            self.model, dict(zip(parameters_to_copy, final_trainable_parameters)), parameters_not_to_copy)
        return self.Result(final_model, loss_history=loss_history, optimizer_state=final_optimizer_state, output=dict_map(inner_outs, torch.stack))