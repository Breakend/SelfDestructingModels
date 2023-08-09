import queue
import torch
import pickle
import copy
import random
import numpy as np
import hydra
import logging
import torch.nn as nn
from datasets import load_dataset
import threading
import time
import warnings

from utils import cache_dir, recursive_to

LOG = logging.getLogger(__name__)

def remove_pronouns_from_bio(bio):
    # base tokens we want to replace
    replace_map = {
        "She": "They",
        "He": "They",
        "Her": "Their",
        "Him": "Their",
        "His": "Their",
    }

    # transformed versions of the keys above that we want to match
    transforms = [lambda x: x, lambda x: x + ".", lambda x: x + ",", lambda x: x.lower()] 

    # simple tokenization by whitespace
    bio_tokens = [token.strip() for token in bio.split(" ")]

    for idx in range(len(bio_tokens)):
        for t in transforms:
            for k in replace_map.keys():
                if bio_tokens[idx] == t(k):
                    bio_tokens[idx] = t(replace_map[k])

    return " ".join(bio_tokens)

def build_regression_datasets(batch_size=16, train_p=0.85, val_p=0.05):
    N = 1000000
    data = torch.randn(N, 25)
    train = data[:int(N*train_p)]
    val1 = data[int(N*train_p):int(N*(train_p+val_p/2))]
    val2 = data[int(N*(train_p+val_p/2)):int(N*(train_p+val_p))]
    test = data[int(N*(train_p + val_p)):]

    LOG.info(f"Split {len(train)} train items")
    LOG.info(f"Split {len(val1)} val1 items")
    LOG.info(f"Split {len(val2)} val2 items")
    LOG.info(f"Split {len(test)} test items")

    train_sampler = RegressionDataSampler(train, batch_size)
    val1_sampler = RegressionDataSampler(val1, batch_size, train_sampler.good_cls, train_sampler.bad_cls)
    val2_sampler = RegressionDataSampler(val2, batch_size, train_sampler.good_cls, train_sampler.bad_cls)
    test_sampler = RegressionDataSampler(test, batch_size, train_sampler.good_cls, train_sampler.bad_cls)

    return train_sampler, val1_sampler, val2_sampler, test_sampler


def build_bios_datasets(tokenizer, batch_size=16, path="data/cache/BIOS.pkl", train_p=0.85, val_p=0.05, censor_settings=None):
    if censor_settings is None:
        remove_pronouns, degender = False, False
    else:
        remove_pronouns, degender = censor_settings.remove_pronouns, censor_settings.degender

    try:
        full_path = hydra.utils.get_original_cwd() + "/" + path
    except Exception as e:
        full_path = path
    LOG.info(f"Loading data from {full_path}")
    with open(full_path, "rb") as f:
        data = pickle.load(f)

    for d in data:
        if remove_pronouns:
            d["bio"] = remove_pronouns_from_bio(d["raw"])
        elif not degender:
            d["bio"] = d["raw"]

        del d["raw"]

    LOG.info(f"Loaded {len(data)} examples")

    professions = sorted(list(set([d["title"] for d in data])))
    genders = sorted(list(set([d["gender"] for d in data])))

    LOG.info(f"Got {len(professions)} professions")
    LOG.info(f"Got {len(genders)} genders")

    train, val, test = split_data(data, train_p, val_p)
    val1, val2 = split_data(val, 0.5, 0.5, no_test=True)

    LOG.info(f"Split {len(train)} train items")
    LOG.info(f"Split {len(val1)} val1 items")
    LOG.info(f"Split {len(val2)} val2 items")
    LOG.info(f"Split {len(test)} test items")

    train_sampler = BiosDataSampler(train, professions, genders, tokenizer, batch_size)
    val1_sampler = BiosDataSampler(val1, professions, genders, tokenizer, batch_size)
    val2_sampler = BiosDataSampler(val2, professions, genders, tokenizer, batch_size)
    test_sampler = BiosDataSampler(test, professions, genders, tokenizer, batch_size)

    return train_sampler, val1_sampler, val2_sampler, test_sampler


def build_celeba_datasets(tokenizer, config, batch_size=16, train_p=0.85, val_p=0.05, n_workers=16, queue_size=32):
    data = load_dataset("mariosasko/CelebA-faces-with-attributes", "mariosasko--CelebA-faces-with-attributes", 
                        use_auth_token="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", cache_dir=cache_dir())["train"]
    bad_label = config.data.bad_key
    good_label = config.data.good_key

    # TODO: If we want to switch to identities we can download identity mapping and then ensure good/bad holdouts for identities.
    goods = sorted(list(set(data[good_label])))
    bads = sorted(list(set(data[bad_label])))

    LOG.info(f"Got {len(goods)} {good_label}")
    LOG.info(f"Got {len(bads)} {bad_label}")

    _split = data.train_test_split(train_size=train_p, shuffle=True)
    train = _split["train"]
    _split = _split["test"].train_test_split(train_size=0.33)
    val1 = _split["train"]
    _split = _split["test"].train_test_split(train_size=0.5)
    val2 = _split["train"]
    test = _split["test"]

    LOG.info(f"Split {len(train)} train items")
    LOG.info(f"Split {len(val1)} val1 items")
    LOG.info(f"Split {len(val2)} val2 items")
    LOG.info(f"Split {len(test)} test items")

    train_sampler = FaceDataSampler(train, goods, bads, tokenizer, batch_size, good_label=good_label, bad_label=bad_label, n_workers=n_workers, queue_size=queue_size)
    val1_sampler = FaceDataSampler(val1, goods, bads, tokenizer, batch_size, good_label=good_label, bad_label=bad_label, n_workers=n_workers, queue_size=queue_size)
    val2_sampler = FaceDataSampler(val2, goods, bads, tokenizer, batch_size, good_label=good_label, bad_label=bad_label, n_workers=n_workers, queue_size=queue_size)
    test_sampler = FaceDataSampler(test, goods, bads, tokenizer, batch_size, good_label=good_label, bad_label=bad_label, n_workers=n_workers, queue_size=queue_size)

    return train_sampler, val1_sampler, val2_sampler, test_sampler


def split_data(data, train_p, val_p, no_test=False):
    state = random.getstate()

    random.seed(0)

    data = copy.deepcopy(data)
    random.shuffle(data)

    random.setstate(state)

    N = len(data)
    if no_test:
        return (
            data[:int(N * train_p)],
            data[int(N * train_p):]
        )
    else:
        return (
            data[:int(N * train_p)],
            data[int(N * train_p):int(N * (train_p + val_p))],
            data[int(N * (train_p + val_p)):]
        )

class DataSampler(torch.utils.data.IterableDataset):
    def __init__(self, data, batch_size, n_workers=0, queue_size=32):
        self.seed = np.random.randint(int(1e9)) # for if we need to generate a fixed subset
        self.data = data
        self.batch_size = batch_size
        self.perm = self.new_perm()
        self.cycle = True
        self.idx = 0
        self.n_workers = n_workers
        self.queue_size = queue_size
        self.lock = threading.Lock()

        if n_workers > 0:
            self.queue = {}
            self.queue_lens = {}
            self.workers = None            
            self._restart_workers()
        else:
            self.queue = None

    def new_perm(self, seed=None):
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        return torch.randperm(len(self), generator=generator)

    def __len__(self):
        return len(self.data)

    def random_subset_idxs(self, n, fixed_order=False):
        generator = torch.Generator().manual_seed(self.seed) if fixed_order else None
        assert n <= len(self.data)
        return torch.randperm(len(self.data), generator=generator)[:n]

    def select_subset(self, idxs):
        raise NotImplementedError

    def random_subset(self, n=None, fixed_order=False):
        if n is None:
            n = len(self.data)
        new_dataset = self.select_subset(self.random_subset_idxs(n, fixed_order=fixed_order))
        if fixed_order:
            new_dataset.cycle = False
            new_dataset.perm = new_dataset.new_perm(seed=self.seed)
        return new_dataset
    
    def get_next_batch_idxs(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.idx + batch_size > len(self):
            if self.cycle:
                self.idx = 0
                self.perm = self.new_perm()
            else:
                return None

        idxs = self.perm[self.idx:self.idx + batch_size]
        self.idx += batch_size
        return idxs

    def _worker(self, worker_idx):
        # currently there's no way to actually stop the thread; unsure if that's a problem though
        while True:
            time.sleep(1e-2)
            need_load = False
            keys, values = list(self.queue.keys()), list(self.queue.values())
            kv = [(keys[idx], values[idx]) for idx in range(len(values))]
            random.shuffle(kv)
            for k, v in kv:
                if self.queue_lens[k] < self.queue_size:
                    with self.lock:
                        # print(self, f"Adding to queue {k}", self.queue_lens)
                        self.queue_lens[k] += 1
                    need_load = True

                    with self.lock:
                        idxs = self.get_next_batch_idxs(batch_size=k[1])
                    if idxs is None:
                        v.append(None)
                    else:
                        v.append(self.get_idxs(idxs, "cpu"))

            if not need_load: # stop the worker once the queue is full; it's restarted when needed
                # print(f"stopping {worker_idx} because queues are full")
                break

    def _restart_workers(self):
        if self.workers is None:
            self.workers = [threading.Thread(target=self._worker, args=(idx,)) for idx in range(self.n_workers)]
        else:
            self.workers = [self.workers[idx] if self.workers[idx].is_alive() else threading.Thread(target=self._worker, args=(idx,)) for idx in range(self.n_workers)]

        for w in self.workers:
            if not w.is_alive():
                w.start()

    def sample(self, device="cpu", batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if self.queue is not None:
            batch_type = (device, batch_size)
            if (device, batch_size) not in self.queue:
                self.queue[batch_type] = []
                self.queue_lens[batch_type] = 0

            if self.queue_lens[batch_type] < self.queue_size:
                self._restart_workers()

            wait_idx = 0
            wait_time = 0.001
            while len(self.queue[batch_type]) == 0:
                time.sleep(wait_time)
                # print(self, f'sleeping {batch_type}', self.queue.keys(), self.queue_lens, len(self.queue[batch_type]))

                wait_idx += 1

                if wait_idx * wait_time > 1.5:
                    LOG.info("Still waiting for data from the queue... (maybe something is stuck?)")
            self.queue_lens[batch_type] -= 1
            return recursive_to(self.queue[batch_type].pop(0), device)
        else:
            idxs = self.get_next_batch_idxs(batch_size)
            if idxs is None:
                return None # if this is a fixed-size subset and we've reached the end

            return self.get_idxs(idxs, device)

    def get_idxs(self, idxs, device="cpu"):
        raise NotImplementedError

class RegressionDataSampler(DataSampler):
    def __init__(self, data, batch_size=16, good_cls=None, bad_cls=None):
        super().__init__(data, batch_size)

        if good_cls is None:
            self.good_cls = nn.Linear(data.shape[-1], 1)
            self.bad_cls = nn.Linear(data.shape[-1], 1)
            self.good_cls.bias.data[:] = 0
            self.bad_cls.bias.data[:] = 0
        else:
            self.good_cls = good_cls
            self.bad_cls = bad_cls

    def __repr__(self) -> str:
        return "\nGood weight:\n{}\nGood bias:\n{}\nBad weight:\n{}\nBad bias:\n{}".format(
            self.good_cls.weight.data,
            self.good_cls.bias.data,
            self.bad_cls.weight.data,
            self.bad_cls.bias.data
        )

    def select_subset(self, idxs):
        return RegressionDataSampler(self.data[idxs], batch_size=self.batch_size, good_cls=self.good_cls, bad_cls=self.bad_cls)

    def get_idxs(self, idxs, device="cpu"):
        inputs = self.data[idxs]

        with torch.no_grad():
            ygood = (self.good_cls(inputs) > 0).long().squeeze(-1)
            ybad = (self.bad_cls(inputs) > 0).long().squeeze(-1)

        return {
            "inputs": { "input": inputs.to(device) },
            "ygood": ygood.to(device),
            "ybad": ybad.to(device)
        }

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.sample()
        return batch["inputs"][0], batch["ybad"][0]


class FaceDataSampler(DataSampler):
    def __init__(self, data, good_map, bad_map, tok, batch_size, good_label, bad_label, n_workers=0, queue_size=32):
        super().__init__(data, batch_size, n_workers=n_workers, queue_size=queue_size)

        from torchvision import transforms

        self.tok = tok
        self.good_map = good_map
        self.bad_map = bad_map
        self.good_label = good_label
        self.bad_label = bad_label
        self.n_workers = n_workers
        self.tv_preprocess = transforms.Compose([ # from https://pytorch.org/hub/pytorch_vision_squeezenet/
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def select_subset(self, idxs):
        return FaceDataSampler(self.data.select(idxs), self.good_map, self.bad_map, self.tok, self.batch_size, self.good_label, self.bad_label, n_workers=0)

    def preprocess(self, inputs, device):
        if self.tok:
            return self.tok(inputs, return_tensors="pt").to(device)
        else:
            return { "x": torch.stack([self.tv_preprocess(i) for i in inputs]).to(device) }

    def get_idxs(self, idxs, device="cpu"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            sample = self.data[idxs]
        inputs = sample["image"]
        good = torch.tensor([self.good_map.index(s) for s in sample[self.good_label]]).to(device)
        bad = torch.tensor([self.bad_map.index(s) for s in sample[self.bad_label]]).to(device)

        return {
            "inputs": self.preprocess(inputs, device),
            self.good_label : good,
            self.bad_label : bad
        }


class BiosDataSampler(DataSampler):
    def __init__(self, data, profession_map, gender_map, tok, batch_size):
        super().__init__(data, batch_size)

        self.tok = tok
        self.prof_map = profession_map
        self.gender_map = gender_map

    def select_subset(self, idxs):
        return BiosDataSampler([self.data[idx] for idx in idxs], self.prof_map, self.gender_map, self.tok, self.batch_size)

    def get_idxs(self, idxs, device="cpu"):
        sample = [self.data[idx] for idx in idxs]

        inputs = [s["bio"] for s in sample]
        professions = torch.tensor([self.prof_map.index(s["title"]) for s in sample]).to(device)
        genders = torch.tensor([self.gender_map.index(s["gender"]) for s in sample]).to(device)

        return {
            "inputs": self.tok(inputs, return_tensors="pt", padding=True, truncation=True, max_length=200).to(device),
            "professions": professions,
            "genders": genders
        }


def build_hf_classification_datasets(tokenizer, dataset_name, dataset_subset_name, batch_size=16):
    from datasets import load_dataset
    dataset = load_dataset(dataset_name, dataset_subset_name)
    labels = sorted(list(set([d["label"] for d in dataset["train"]])))

    train_sampler = HFTextDataSampler(dataset["train"], labels, tokenizer, batch_size)
    val_sampler = HFTextDataSampler(dataset["validation"], labels, tokenizer, batch_size)

    return train_sampler, val_sampler, len(labels)

class HFTextDataSampler(DataSampler):
    def __init__(self, data, profession_map, tok, batch_size):
        super().__init__(data, batch_size)

        self.tok = tok
        self.prof_map = profession_map

    def select_subset(self, idxs):
        return HFTextDataSampler(self.data.select(idxs), self.prof_map, self.tok, self.batch_size)

    def get_idxs(self, idxs, device="cpu", batch_size=None):
        sample = self.data.select(idxs)

        inputs = [s["sentence"] for s in sample]
        professions = torch.tensor([self.prof_map.index(s["label"]) for s in sample]).to(device)

        return {
            "inputs": self.tok(inputs, return_tensors="pt", padding=True, truncation=True, max_length=200).to(device),
            "label": professions
        }


class MLMDataSampler(DataSampler):
    def __init__(self, data, tok, batch_size, mlm_probability):
        super().__init__(data, batch_size)

        self.tok = tok
        self.mlm_probability = mlm_probability

    def __len__(self):
        return self.data["input_ids"].shape[0]

    def get_idxs(self, idxs, device="cpu"):
        sample = {k: self.data[k][idxs] for k in self.data.keys()}

        inputs, labels = apply_mlm_mask(self.mlm_probability, self.tok, sample["input_ids"], sample["special_tokens_mask"])
        sample["input_ids"] = inputs
        sample["labels"] = labels
        del sample["special_tokens_mask"]
        return {k: v.to(device) for k, v in sample.items() }


def build_mlm_sampler(tokenizer, config):
    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(
        config.mlm.dataset_name,
        config.mlm.subset_name
    )
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    padding = "max_length" 
    max_seq_length = 128

    def tokenize_function(examples):
        # Remove empty lines
        examples[text_column_name] = [
            line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )

    train_dataset = raw_datasets["train"].map(
        tokenize_function,
        batched=True,
        num_proc=8,
        remove_columns=[text_column_name],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset chunk...",
    )

    if config.mlm.max_train_samples is not None and len(train_dataset) > config.mlm.max_train_samples:
        train_dataset = train_dataset.select(range(config.mlm.max_train_samples))
    
    train_data_dict = { k: torch.tensor(train_dataset[k], dtype=torch.long) for k in train_dataset.column_names }
    return MLMDataSampler(train_data_dict, tokenizer, config.batch_size, config.mlm.mlm_probability)


def apply_mlm_mask(mlm_probability, tokenizer, inputs, special_tokens_mask=None):
    """
    Prepare masked tokens inputs/labels for masked language modeling:
        1 - `mlm_probability` are left unchanged, `mlm_probability` * 0.8 are replaced with [MASK], and the rest 
        are replaced with a random token.
    """
    if special_tokens_mask is None:
        special_ids = torch.tensor(tokenizer.all_special_ids, device=inputs.device)
        special_tokens_mask = (inputs.unsqueeze(-1) == special_ids.expand(*inputs.shape, *special_ids.shape)).any(-1)

    # Sample `mlm_probability` random tokens; of these, 80% become [MASK], 20% become a random token
    new_inputs, labels = inputs.clone(), inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability).masked_fill(special_tokens_mask, 0.0) # don't change special tokens
    mask_or_random_mask = torch.bernoulli(probability_matrix).bool() # will get [MASK] or random token
    mask_probabilities = mask_or_random_mask.float().masked_fill(mask_or_random_mask == 1, 0.8)
    mask_mask = torch.bernoulli(mask_probabilities).bool() # will get [MASK]
    random_mask = mask_or_random_mask & ~mask_mask # will get random token

    labels[~mask_or_random_mask] = -100  # We only compute loss on masked tokens
    new_inputs[mask_mask] = tokenizer.mask_token_id
    new_inputs[random_mask] = torch.randint(0, len(tokenizer), (random_mask.count_nonzero(),), device=inputs.device)

    return new_inputs, labels


if __name__ == "__main__":
    from types import SimpleNamespace
    import transformers
    import torchvision
    prob = 0.4
    # inputs = torch.randint(0, len(tokenizer), (100, 100))
    # inputs_, labels = torch_mask_tokens(prob, tokenizer, inputs)
    cfg = SimpleNamespace()
    cfg.data = SimpleNamespace()
    cfg.data.bad_key = "Male"
    cfg.data.good_key = "Wearing_Hat"
    cfg.batch_size = 10
    cfg.image_model = "facebook/regnet-y-040"
    cfg.mlm = SimpleNamespace()
    cfg.mlm.dataset_name = "wikitext"
    cfg.mlm.subset_name = "wikitext-2-raw-v1"
    cfg.mlm.max_train_samples = 100000
    cfg.mlm.mlm_probability = 0.4

    # tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased')
    # sampler = build_mlm_sampler(tokenizer, cfg)
    # sample = sampler.sample()
    # ss = sampler.random_subset()
    # fixed_ss = sampler.random_subset(fixed_order=True)
    # fixed_ss2 = sampler.random_subset(fixed_order=True)

    feature_extractor = None #transformers.AutoFeatureExtractor.from_pretrained(cfg.image_model)
    trunk = torchvision.models.squeezenet1_1() #transformers.AutoModel.from_pretrained(cfg.image_model)
    train, a, a, a = build_celeba_datasets(feature_extractor, cfg, cfg.batch_size, n_workers=16)
    del a
    import pdb; pdb.set_trace()
    import time
    train.sample()
    time.sleep(2)
    start = time.time()
    for _ in range(16):
        train.sample()
    print(time.time() - start)
    import pdb; pdb.set_trace()
    print()

