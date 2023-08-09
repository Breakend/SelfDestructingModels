import torch.nn as nn
from torch.nn import functional as F
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch
import warnings


def tensor_from_opts(opts, device):
    assert all([isinstance(v, bool) for v in opts.values()])
    opts_tensor = torch.tensor([int(opts[k]) for k in sorted(opts.keys())], device=device, dtype=torch.float32)

    return opts_tensor


class StepwiseMLP(nn.Module):
    def __init__(self, n_steps, width, n_out, n_opts, linear, separate, input_dim=None):
        super().__init__()
        if input_dim is None:
            input_dim = width
        self.linear = linear
        self.separate = separate
        if separate:
            if linear:
                linears = [nn.Linear(input_dim, n_out) for _ in range(n_steps + 1)]
                self.weight = nn.Parameter(torch.stack([l.weight for l in linears]))
                self.bias = nn.Parameter(torch.stack([l.bias for l in linears]))
            else:
                linears1 = [nn.Linear(input_dim, input_dim * 2) for _ in range(n_steps + 1)]
                linears2 = [nn.Linear(input_dim * 2, n_out) for _ in range(n_steps + 1)]
                self.weight1 = nn.Parameter(torch.stack([l.weight for l in linears1]))
                self.bias1 = nn.Parameter(torch.stack([l.bias for l in linears1]))
                self.weight2 = nn.Parameter(torch.stack([l.weight for l in linears2]))
                self.bias2 = nn.Parameter(torch.stack([l.bias for l in linears2]))
        else:
            self.weight1 = nn.Linear(input_dim + (n_steps + 1) + n_opts, width * 2)
            self.weight2 = nn.Linear(width * 2, width * 2)
            self.fc3 = nn.Linear(width * 2 + (n_steps + 1) + n_opts, width * 2)
            self.fc4 = nn.Linear(width * 2, n_out)
        self.n_steps = n_steps

    def forward(self, x, opts):
        # disable this assert for rep_base_only mode
        # assert x.shape[0] == self.n_steps + 1
        if x.dim() > 3:
            for _ in range(x.dim() - 3):
                x = x.squeeze(-1)
            assert x.dim() == 3

        if self.separate:
            if self.linear:
                return torch.einsum("sbi,soi->sbo", x, self.weight) + self.bias.unsqueeze(1)
            else:
                fc1_out = torch.einsum("sbi,soi->sbo", x, self.weight1) + self.bias1.unsqueeze(1)
                return torch.einsum("sbi,soi->sbo", fc1_out.relu(), self.weight2) + self.bias2.unsqueeze(1)
        else:
            # assumes we get input of shape (inner_steps, batch_size, hidden_dim)
            timestep_one_hots = torch.eye(self.n_steps + 1, device=x.device).unsqueeze(1).repeat(1, x.shape[1], 1)[:x.shape[0]]
            opts_tensor = tensor_from_opts(opts, x.device).view(1, 1, -1).repeat(*timestep_one_hots.shape[:2], 1)

            out1 = self.weight1(torch.cat((timestep_one_hots, opts_tensor, x), -1)).relu()
            out2 = self.weight2(out1).relu()
            out3 = self.fc3(torch.cat((timestep_one_hots, opts_tensor, out2), -1)).relu()
            return self.fc4(out3)


class GoodBadMLP(nn.Module):
    def __init__(self, n_in, n_good, n_bad, width, good_key, bad_key, trunk=None, linear: nn.Module = None):
        super().__init__()

        if trunk is None:
            self.trunk = nn.Sequential(
                nn.Linear(n_in, width), nn.ReLU(),
                nn.Linear(width, width), nn.ReLU(),
                nn.Linear(width, width), nn.ReLU(),
                nn.Linear(width, width), nn.ReLU()
            )
        else:
            self.trunk = trunk

        if linear is None:
            self.linear = nn.Linear(width, n_good + n_bad, bias=False)
        else:
            self.linear = linear

        self.n_in = n_in
        self.n_good = n_good
        self.n_bad = n_bad
        self.width = width
        self.good_key = good_key
        self.bad_key = bad_key

    def forward(self, *args, **kwargs):
        reps = self.trunk(*args, **kwargs)
        logits = self.linear(reps)
        good_logits = logits[:, :self.n_good]
        bad_logits = logits[:, self.n_good:]

        return { self.good_key: good_logits, self.bad_key: bad_logits, "reps": reps}

    def with_linear_reset(self, linear: nn.Module = None):
        device = list(self.trunk.parameters())[0].device
        return GoodBadMLP(self.n_in, self.n_good, self.n_bad, self.width, self.good_key, self.bad_key, self.trunk, linear=linear).to(device)


class GoodBadBERT(nn.Module):
    def __init__(self, bert, n_good, n_bad, good_key, bad_key, linear: nn.Module = None):
        super().__init__()

        self.trunk = bert
        if linear is None:
            self.linear = nn.Linear(bert.config.hidden_size, n_good + n_bad, bias=False)
        else:
            self.linear = linear

        self.n_good = n_good
        self.n_bad = n_bad
        self.good_key = good_key
        self.bad_key = bad_key

    def forward(self, *args, **kwargs):
        all_outputs = self.trunk(*args, **kwargs)
        outputs = all_outputs.pooler_output
        logits = self.linear(outputs)
        good_logits = logits[:, :self.n_good]
        bad_logits = logits[:, self.n_good:]

        return { self.good_key: good_logits, self.bad_key: bad_logits, "reps": outputs, "last_hidden_state": all_outputs.last_hidden_state}

    def with_linear_reset(self, linear: nn.Module = None, n_good=None, n_bad=None, good_key=None, bad_key=None):
        device = list(self.trunk.parameters())[0].device

        # Allow overrides, sorry it's messy
        if n_good is None: n_good = self.n_good
        if n_bad is None: n_bad = self.n_bad
        if good_key is None: good_key = self.good_key
        if bad_key is None: bad_key = self.bad_key

        return GoodBadBERT(self.trunk, n_good, n_bad, good_key, bad_key).to(device)


class GoodBadRegnet(nn.Module):
    def __init__(self, trunk, hidden_dim, n_good, n_bad, good_key, bad_key, linear: nn.Module = None):
        super().__init__()

        self.trunk = trunk
        if linear is None:
            lin = nn.Linear(hidden_dim, n_good + n_bad) if n_good + n_bad > 0 else nn.Identity()
            self.linear = nn.Sequential(nn.Flatten(), lin)
        else:
            self.linear = linear

        self.n_good = n_good
        self.n_bad = n_bad
        self.good_key = good_key
        self.bad_key = bad_key

    def forward(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outputs = self.trunk(*args, **kwargs)
        if hasattr(outputs, "pooler_output"):
            outputs = outputs.pooler_output

        logits = self.linear(outputs)
        good_logits = logits[:, :self.n_good]
        bad_logits = logits[:, self.n_good:]

        return { self.good_key: good_logits, self.bad_key: bad_logits, "reps": outputs}

    def with_linear_reset(self, linear: nn.Module = None, n_good=None, n_bad=None, good_key=None, bad_key=None):
        device = list(self.trunk.parameters())[0].device

        # Allow overrides, sorry it's messy
        if n_good is None: n_good = self.n_good
        if n_bad is None: n_bad = self.n_bad
        if good_key is None: good_key = self.good_key
        if bad_key is None: bad_key = self.bad_key

        return GoodBadRegnet(self.trunk, n_good, n_bad, good_key, bad_key).to(device)

class MLMModel(nn.Module):
    def __init__(self, bert, config):
        super().__init__()
        self.bert = bert
        self.config = config
        self.cls = BertOnlyMLMHead(bert.config).to(config.device)

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        labels = None,
        output_attentions = None,
        output_hidden_states = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        loss_fct = CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.mlm.vocab_size), labels.view(-1))
        return masked_lm_loss