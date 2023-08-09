import torch
import numpy as np
import random
import transformers
import hydra
import tempfile
import os
from collections import defaultdict
import logging
import wandb
from tqdm import tqdm
import copy
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())


from model import MLMModel
from adapt import get_adapted_predictions, adapt_model_evaluation
from losses import cls_loss, cls_acc, linear_adversary_loss
import utils
import json


logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
                    level=logging.INFO)
LOG = logging.getLogger(__name__)


def append_to_keys(d, tag):
    return {f"{k}_{tag}": v for k, v in d.items()}


def average_dicts(dicts):
    lists = defaultdict(list)

    for d in dicts:
        for k, v in d.items():
            lists[k].append(v)

    return {k: sum(v) / len(v) for k, v in lists.items()}


class Trainer:
    def __init__(self, config, model, train, val1, val2, test, mlm_data_sampler, aux_eval_samplers=[]):
        self.model = model
        if config.l_mlm > 0 and config.mlm.distill == True:
            self.original_model = copy.deepcopy(model)
        self.train_set = train
        self.aux_eval_samplers = aux_eval_samplers
        self.val1_set = val1
        self.val2_set = val2
        self.test_set = test
        self.config = config
        if "mi_loss_predictor" in config and self.config.l_mi > 0:
            self.predictor = hydra.utils.instantiate(config.mi_loss_predictor).to(config.device)
            self.predictor_opt = hydra.utils.instantiate(config.mi_loss_predictor_opt)(self.predictor.parameters())

        self.adversarial_param_list = []
        self.adversarial_params = {}
        if config.inner_loop_learn_lr:
            if config.bad_adapt_mode != "maml":
                raise ValueError(f"We currently don't support meta-learned learning rates with {config.bad_adapt_mode}")
            self.adversarial_params["lr"] = torch.tensor([3e-4], requires_grad=True)
            self.adversarial_param_list.append(self.adversarial_params["lr"])

        if config.inner_loop_learn_adv_head:
            self.adversarial_params["linear_head"] = copy.deepcopy(self.model.linear)
            self.adversarial_param_list.extend(self.adversarial_params["linear_head"].parameters())

        if len(self.adversarial_params):
            self.adversarial_params_opt = torch.optim.Adam(self.adversarial_param_list, config.hyperparams_lr)
        else:
            self.adversarial_params_opt = None

        if config.l_mlm > 0:
            self.mlm_data_sampler = mlm_data_sampler
            self.mlm_head = MLMModel(self.model.trunk, config)

        self.opt = torch.optim.Adam(self.model.parameters(), config.lr)
        if not self.config.debug and not self.config.eval_only:
            wandb_dir = tempfile.mkdtemp()
            LOG.info(f"Writing wandb run to {wandb_dir}")
            wandb.init(
                project="selfdestruct",
                entity="selfdestruct",
                config=utils.flatten_dict(self.config),
                dir=wandb_dir
            )
            if self.config.batch_hash is not None:
                wandb.run.name = wandb.run.name + f" [{self.config.batch_hash}]"
                wandb.run.save()

        self.best_val_loss = 1e9

    def do_step(self, sampler, train=False):
        outer_batch = sampler.sample(self.config.device)

        SHOULD_ADAPT_GOOD = self.config.l_good_adapted > 0 or self.config.l_good_adapted_bad > 0
        good_key, bad_key = self.config.data.good_key, self.config.data.bad_key

        # TODO: should probably meta learn both good and bad
        adapted_model_bad, intermediate_preds_bad, reps_bad, bad_grad_norms, bad_opts = get_adapted_predictions(self.model, sampler, outer_batch,
                                                                                                                self.config, bad_key, self.global_it,
                                                                                                                adversarial_params=self.adversarial_params)
        if SHOULD_ADAPT_GOOD:
            adapted_model_good, intermediate_preds_good, _, _ = get_adapted_predictions(self.model, sampler, outer_batch, self.config, good_key, 
                                                                                        self.global_it, adversarial_params=self.adversarial_params)

        # Get all the outer predictions for the good and bad adapted models as well as the base model
        if SHOULD_ADAPT_GOOD:
            good_adapted_outer_outputs = adapted_model_good(**outer_batch["inputs"])
        bad_adapted_outer_outputs = adapted_model_bad(**outer_batch["inputs"])
        base_outputs = self.model(**outer_batch["inputs"])

        # how good is the good adapted model at the bad task
        if SHOULD_ADAPT_GOOD:
            good_adapted_bad_loss = cls_loss(outer_batch[bad_key], good_adapted_outer_outputs[bad_key])
            acc_good_adapted_bad = cls_acc(outer_batch[bad_key], good_adapted_outer_outputs[bad_key])
        else:
            good_adapted_bad_loss = torch.tensor(0.0)
            acc_good_adapted_bad = torch.tensor(0.0)

        # How good is the base model at both tasks
        good_base_loss = cls_loss(outer_batch[good_key], base_outputs[good_key])
        bad_base_loss = cls_loss(outer_batch[bad_key], base_outputs[bad_key])
        acc_good_base = cls_acc(outer_batch[good_key], base_outputs[good_key])
        acc_bad_base = cls_acc(outer_batch[bad_key], base_outputs[bad_key])            

        # How good are the adapted models at their respective tasks
        acc_bad_adapted = cls_acc(outer_batch[bad_key], bad_adapted_outer_outputs[bad_key])
        if SHOULD_ADAPT_GOOD:
            acc_good_adapted = cls_acc(outer_batch[good_key], good_adapted_outer_outputs[good_key])
        else:
            acc_good_adapted = torch.tensor(0.0)

        # If we want intermediate updates we grab those, otherwise grab the outer loop only to backprop through
        good_adapted_loss = torch.tensor(0.0)
        if self.config.inner_loop_update_every_step:
            if SHOULD_ADAPT_GOOD:
                good_adapted_loss = cls_loss(outer_batch[good_key], intermediate_preds_good)
            bad_adapted_loss = cls_loss(outer_batch[bad_key], intermediate_preds_bad)
        else:
            if SHOULD_ADAPT_GOOD:
                good_adapted_loss = cls_loss(outer_batch[good_key], good_adapted_outer_outputs[good_key])
            bad_adapted_loss = cls_loss(outer_batch[bad_key], bad_adapted_outer_outputs[bad_key])

        total_loss = (
            self.config.l_good_base * good_base_loss +
            -self.config.l_bad_base * bad_base_loss +
            self.config.l_good_adapted * good_adapted_loss + 
            -self.config.l_bad_adapted * bad_adapted_loss +
            -self.config.l_good_adapted_bad * good_adapted_bad_loss
        )

        info = {
            "loss/bad_adapted": bad_adapted_loss.item(),
            "loss/bad_base": bad_base_loss.item(),
            "loss/good_base": good_base_loss.item(),
            "loss/good_adapted": good_adapted_loss.item(),
            "loss/good_adapted_bad": good_adapted_bad_loss.item(),
            "acc/good_adapted_bad": acc_good_adapted_bad.item(),
            "acc/good_adapted": acc_good_adapted.item(),
            "acc/good_base": acc_good_base.item(),
            "acc/bad_base": acc_bad_base.item(),
            "acc/bad_adapted": acc_bad_adapted.item(),
        }

        if self.config.l_mlm > 0:
            mlm_data_batch = self.mlm_data_sampler.sample(self.config.device)
            if self.config.mlm.distill:
                del mlm_data_batch["labels"]
                distill_targets = self.original_model(**mlm_data_batch)["last_hidden_state"]
                distill_pred = self.model(**mlm_data_batch)["last_hidden_state"]
                mlm_loss = (distill_targets - distill_pred).pow(2).mean()
            else:
                mlm_loss = self.mlm_head(**mlm_data_batch)
            total_loss += self.config.l_mlm * mlm_loss
            info["loss/mlm"] = mlm_loss.item()

        if self.config.inner_loop_learn_lr:
            info["meta_hparams/lr_bad"] = self.adversarial_params["lr"].item()

        if self.config.l_bad_base_grad > 0:
            grads = torch.autograd.grad(bad_base_loss, self.model.parameters(), create_graph=True, retain_graph=True)
            bad_base_grad_loss = torch.cat([g.view(-1) for g in grads]).norm(2)
            if self.config.square_grad:
                bad_base_grad_loss = bad_base_grad_loss.pow(2)
            total_loss += self.config.l_bad_base_grad * bad_base_grad_loss
            info["loss/bad_base_grad"] = bad_base_grad_loss.item()

        if self.config.l_bad_adapted_grad > 0:
            if self.config.rgd:
                last_adapted_bad_loss = cls_loss(outer_batch[bad_key], intermediate_preds_bad[-1])
                grads = torch.autograd.grad(last_adapted_bad_loss, adapted_model_bad.parameters(), create_graph=True, retain_graph=True)
                _, random_bad_pred, _ = self.random_model(outer_batch["inputs"])
                random_bad_loss = self.bad_loss(outer_batch[bad_key], random_bad_pred)
                random_grads = torch.autograd.grad(random_bad_loss, self.random_model.parameters())
                bad_adapted_grad_loss = torch.cat([g.view(-1) - rg.view(-1) for g, rg in zip(grads, random_grads)]).norm(2)
            else:
                if self.config.minimize_all_inner_grad_norms:
                    bad_adapted_grad_loss = bad_grad_norms.mean()
                else:
                    bad_adapted_grad_loss = bad_grad_norms[-1]
            if self.config.square_grad:
                bad_adapted_grad_loss = bad_adapted_grad_loss.pow(2)
            total_loss += self.config.l_bad_adapted_grad * bad_adapted_grad_loss
            info["loss/bad_adapted_grad"] = bad_adapted_grad_loss.item()

        if self.config.l_linear_mi > 0:
            linear_mi_losses = []
            linear_mi_accs = []
            for preds in intermediate_preds_bad:
                loss_dict = linear_adversary_loss(preds, outer_batch[bad_key])
                linear_mi_losses.append(loss_dict["loss"])
                linear_mi_accs.append(loss_dict["acc"])
            linear_mi_loss = sum(linear_mi_losses) / len(linear_mi_losses)
            linear_mi_acc = sum(linear_mi_accs) / len(linear_mi_accs)
            info["loss/linear_mi"] = linear_mi_loss.item()
            info["acc/linear_mi"] = linear_mi_acc

            total_loss += -self.config.l_linear_mi * linear_mi_loss

        if self.config.l_mi > 0:
            # First, do a training step on the predictor
            if self.config.predictor_train_steps > 0 and self.config.predictor_train_steps < 1:
                period = int(1./self.config.predictor_train_steps)
                if self.global_it % period == 0:
                    predictor_train_steps = 1
                else:
                    predictor_train_steps = 0
            else:
                predictor_train_steps = self.config.predictor_train_steps

            for _ in range(predictor_train_steps):
                predictor_estimates = self.predictor(reps_bad.detach(), bad_opts)
                predictor_loss = cls_loss(outer_batch[bad_key], predictor_estimates)
                if train:
                    predictor_loss.backward()
                    self.predictor_opt.step()
                    self.predictor_opt.zero_grad()

            # Now do it again (without detaching) to train the blocked model
            with torch.set_grad_enabled(not self.config.no_mi):
                predictor_estimates = self.predictor(reps_bad, bad_opts)
                predictor_loss = -cls_loss(outer_batch[bad_key], predictor_estimates)
                predictor_acc = cls_acc(outer_batch[bad_key], predictor_estimates)
                total_loss += self.config.l_mi * predictor_loss
                info["loss/predictor"] = predictor_loss
                info["acc/predictor"] = predictor_acc

        info["loss/total"] = total_loss.item()

        if train:
            if self.config.inner_loop_learn_lr or self.config.inner_loop_learn_adv_head:
                for p, g in zip(self.adversarial_param_list, torch.autograd.grad((bad_adapted_loss / self.config.accumulate_steps), self.adversarial_param_list, retain_graph=True, allow_unused=True)):
                    if g is not None:
                        if p.grad is None:
                            p.grad = g
                        else:
                            p.grad += g

                torch.nn.utils.clip_grad_norm_(self.adversarial_param_list, self.config.grad_clip)
                if self.global_it % self.config.accumulate_steps == 0:
                    self.adversarial_params_opt.step()
                    self.adversarial_params_opt.zero_grad()

            adv_grads_cache = [p.grad.clone() if p.grad is not None else None for p in self.adversarial_param_list]
            total_loss.backward()
            for p, g in zip(self.adversarial_param_list, adv_grads_cache):
                p.grad = g
            grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip).item()
            info["grad/norm"] = grad

            self.opt.step()
            self.opt.zero_grad()
            if self.config.l_mi > 0:
                self.predictor_opt.zero_grad()
            if self.config.inner_loop_learn_lr:
                with torch.no_grad():
                    self.adversarial_params["lr"].clamp_(1e-6, 1e-2)
        return info

    def train_step(self):
        info = self.do_step(self.train_set, train=True)
        return append_to_keys(info, "train")

    def val_step(self):
        infos = []
        for _ in tqdm(range(self.config.val_steps), desc="Running validation"):
            infos.append(self.do_step(self.val1_set, train=False))

        mean_total_loss = sum(info["loss/total"] for info in infos) / len(infos)
        best_path = os.path.abspath("best_model.pt")
        latest_path = os.path.abspath("latest_model.pt")
        save_dict = {
            "model": self.model.state_dict(),
            "adv": self.adversarial_param_list,
            "opt": self.opt.state_dict(),
            "val_loss": self.best_val_loss,
            "step": self.global_it,
            "config": utils.flatten_dict(self.config)
        }
        if not self.config.debug and not self.config.eval_only:
            LOG.info(f"Saving model to {latest_path}...")
            torch.save(save_dict, latest_path)
            if mean_total_loss < self.best_val_loss:
                self.best_val_loss = mean_total_loss
                LOG.info(f"Saving new best model to {best_path}...")
                torch.save(save_dict, best_path)

        infos[-1]["loss/total"] = mean_total_loss
        return append_to_keys(infos[-1], "val")

    def eval_step(self, only_bad=False):
        LOG.info("Beginning BAD evaluation")
        info = adapt_model_evaluation(self.model, self.val1_set, self.val2_set, self.config, key=self.config.data.bad_key)
        if only_bad:
            return info
        # TODO: make this align with do_step logging
        info1 = append_to_keys(info, "eval_bad")
        LOG.info("Beginning GOOD evaluation")
        info = adapt_model_evaluation(self.model, self.val1_set, self.val2_set, self.config, key=self.config.data.good_key)
        info2 = append_to_keys(info, "eval_good")
        LOG.info("Beginning additional GOOD evaluations")
        _return_dict = {**info1, **info2}
        for (dataset, aux_num_labels, aux_train, aux_val) in self.aux_eval_samplers:
            # For aux tasks we create new model heads.
                        # don't restrict sample size on auxiliary tasks?
            _cfg_copy = copy.deepcopy(self.config)
            _cfg_copy.adversary.n_examples = -1
            _reset_model = self.model.with_linear_reset(n_good=aux_num_labels, n_bad=None, good_key="label", bad_key=None)
            info = adapt_model_evaluation(_reset_model, aux_train, aux_val, _cfg_copy, key="label")
            info2 = append_to_keys(info, f"eval_good_{dataset}")
            _return_dict = {**_return_dict, **info2}
        return _return_dict


    def run(self):
        for it in tqdm(range(self.config.train_steps + 1)):
            self.global_it = it
            train_info = self.train_step()

            if it % 10 == 0:
                if not self.config.debug and not self.config.eval_only:
                    wandb.log(train_info, step=it)

            if it % self.config.val_every == 0:
                LOG.info("Beginning validation")
                val_info = self.val_step()

                if it == 0 and self.config.skip_initial_eval:
                    continue

                if not self.config.no_eval:
                    LOG.info("Beginning evaluation")
                    eval_info = self.eval_step()

                LOG.info(train_info)
                LOG.info(val_info)
                if not self.config.no_eval:
                    LOG.info(eval_info)
                if not self.config.debug and not self.config.eval_only:
                    wandb.log(val_info, step=it)
                    if not self.config.no_eval:
                        wandb.log(eval_info, step=it)


@hydra.main(config_name="config", config_path="config")
def run(cfg):
    os.environ["WANDB_START_METHOD"] = "thread"
    LOG.info(f"\n\n{OmegaConf.to_yaml(cfg)}\n")
    base_dir = hydra.utils.get_original_cwd()
    LOG.info(f"Project base directory: {base_dir}")
    LOG.info(f"Run directory: {os.getcwd()}")
    os.environ["WANDB_CACHE_DIR"] = utils.cache_dir()
    torch.autograd.set_detect_anomaly(cfg.detect_anomaly)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic and not cfg.debug, warn_only=True)

    mlm_train_sampler = None
    if cfg.exp_name == "bios":
        tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.bert_model)

        from data import build_bios_datasets
        train, val1, val2, test = build_bios_datasets(tokenizer, cfg.batch_size, censor_settings=cfg.bios)

        if cfg.l_mlm > 0:
            from data import build_mlm_sampler
            mlm_train_sampler = build_mlm_sampler(tokenizer, cfg)

        from model import GoodBadBERT
        bert = transformers.AutoModel.from_pretrained(cfg.bert_model)
        model = GoodBadBERT(bert, len(train.prof_map), len(train.gender_map), cfg.data.good_key, cfg.data.bad_key).to(cfg.device)
    elif cfg.exp_name == "face":
        if cfg.image_model.tv_name is not None:
            import torchvision
            feature_extractor = None
            trunk = getattr(torchvision.models, cfg.image_model.tv_name)(pretrained=cfg.image_model.tv_pretrained)
            hidden_dim = 1000 # correct for squeezenet1_0() and squeezenet1_1()
        else:
            feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(cfg.image_model.hf_name)
            trunk = transformers.AutoModel.from_pretrained(cfg.image_model.hf_name)
            hidden_dim = trunk.config.hidden_sizes[-1] if hasattr(trunk.config, "hidden_sizes") else trunk.config.hidden_size

        from data import build_celeba_datasets
        train, val1, val2, test = build_celeba_datasets(feature_extractor, cfg, cfg.batch_size, n_workers=cfg.n_workers, queue_size=cfg.max_adapt_steps*4)
        from model import GoodBadRegnet
        model = GoodBadRegnet(trunk, hidden_dim, len(train.good_map), len(train.bad_map), cfg.data.good_key, cfg.data.bad_key).to(cfg.device)
    else:
        from data import build_regression_datasets
        train, val1, val2, test = build_regression_datasets(cfg.batch_size)
        from model import GoodBadMLP
        model = GoodBadMLP(train.data.shape[-1], 2, 2, cfg.mi_loss_predictor.width, cfg.data.good_key, cfg.data.bad_key).to(cfg.device)

    aux_eval_samplers = []

    if len(cfg.aux_eval_tasks) > 0:
        from data import build_hf_classification_datasets
        for (dataset_name, subset_name) in cfg.aux_eval_tasks:
            aux_train_sampler, aux_val_sampler, aux_num_labels = build_hf_classification_datasets(tokenizer, dataset_name, subset_name, batch_size=16)
            aux_eval_samplers.append((f"{dataset_name}_{subset_name}", aux_num_labels, aux_train_sampler, aux_val_sampler))

    if cfg.eval_only:
        loaded_model_conf = None
        if cfg.eval_network_type == "random":
            prev_model= copy.deepcopy(model)
            for module in model._modules.values(): module.apply(model.trunk._init_weights)
            assert not utils.params_same(prev_model.parameters(), model.parameters())
            # Free pointer
            prev_model = None
        elif cfg.eval_network_type == "good-tuned":   
            _updated_cfg = copy.deepcopy(cfg)
            _updated_cfg.adversary.n_examples = -1 # don't restrict our pre-training
            model = adapt_model_evaluation(model, val1, val2, _updated_cfg, _updated_cfg.data.good_key, return_model=True)
        elif cfg.eval_network_type == "pretrained":
            pass # keep the model as the pretrained model
        elif cfg.eval_network_type == "loaded":
            model_path = os.path.join(cfg.eval_loaded_model_dir, "best_model.pt")
            # model = torch.load(model_path)
            archive = torch.load(model_path, map_location=cfg.device)
            model.load_state_dict(archive["model"])
            loaded_model_conf = OmegaConf.load(os.path.join(cfg.eval_loaded_model_dir, ".hydra/overrides.yaml"))
        else:
            raise ValueError(f"Eval network type {cfg.eval_network_type}, not supported. Please select one from: random, good-tuned, bert.")

        if loaded_model_conf is not None:
            OmegaConf.save(config=loaded_model_conf, f="loaded_model_conf.yaml")

        trainer = Trainer(cfg, model, train, val1, val2, test, mlm_train_sampler, aux_eval_samplers)
        
        eval_info = trainer.eval_step(cfg.eval_only_bad)

        LOG.info(eval_info)
        
        if not cfg.debug and not cfg.eval_only:
            wandb.log(eval_info, step=0)
        with open("eval_info.json", "w") as f:
            f.write(json.dumps({ k: v for k, v in eval_info.items() if "curve" not in k}))
    else:
        trainer = Trainer(cfg, model, train, val1, val2, test, mlm_train_sampler, aux_eval_samplers)

        if cfg.dry_run:
            import pdb; pdb.set_trace()

        trainer.run()


if __name__ == "__main__":
    run()
