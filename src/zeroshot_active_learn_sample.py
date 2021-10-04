import argparse
import os

import autonlu
import numpy as np
import torch
from autonlu.core.classifier import Classifier
from autonlu.core.zerofewshot import zero_shot
from IPython import embed
from tqdm.auto import tqdm
from transformers import *

from src.adapet import adapet
from src.data.Batcher import Batcher
from src.eval.eval_model import test_eval
from src.utils.Config import Config
from src.utils.util import device

def entropies(p):
    logp = torch.log2(p)
    return torch.sum(-p*logp, dim=1)


def break_ties(p):
    """The reciprocal of difference of the two top most probabilities
    Reciprocal so we stay consistent with entropy (higher values are more unsure)
    """
    s, _ = torch.sort(p, descending=True)
    return 1/(s[:, 0] - s[:, 1])


def prepare_data(x, tokenizer):
    tokendicts = [tokenizer.encode_plus(sample) for sample in x]
    tokens = Classifier._stack_tokens(tokendicts)
    tokens = {key: torch.tensor(value, dtype=torch.long).to("cuda") for key, value in tokens.items()}
    return tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', "--exp_dir", required=True)
    args = parser.parse_args()

    config_file = os.path.join(args.exp_dir, "config.json")
    config = Config(config_file, mkdir=False)

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_weight)
    batcher = Batcher(config, tokenizer, config.dataset)
    dataset_reader = batcher.get_dataset_reader()

    model = adapet(config, tokenizer, dataset_reader).to(device)
    model.load_state_dict(torch.load(os.path.join(args.exp_dir, "best_model.pt")))

    # Patch dropout
    dropoutlayers = autonlu.core.modules.modelsampler.monte_carlo_dropout_patch(model.model)
    dl = dropoutlayers[0]
    # Send probestring through model
    probestring = "a a a a a a a a a a as sd we sd"
    tokenlength = len(model.tokenizer.encode_plus(probestring)["input_ids"])
    batchsize = 7
    input = prepare_data([probestring]*7, tokenizer=model.tokenizer)
    model.model(**input)
    # Determine dimensions needed for the mask
    for dl in dropoutlayers:
        for dimsize, last_input_shape in dl.last_input_shape.items():
            dl.tokenlengthdims[dimsize] = [i for i, v in enumerate(last_input_shape) if v == tokenlength]
            dl.batchsizedims[dimsize] = [i for i, v in enumerate(last_input_shape) if v == batchsize]

    # Activate Dropout layers
    for dl in dropoutlayers:
        dl.active = True

    H, BT = None, None

    samples = []
    for epoch in tqdm(range(3), desc="Epochs"):
        # Reset dropout
        for d in dropoutlayers:
            d.clear_mask()
        # Get samples
        logits = []
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(list(batcher.get_test_batch()), desc="Samples")):
                pred_lbl, lbl_logits = model.predict(batch)
                logits.append(lbl_logits.cpu().numpy())
        all_logits = np.vstack(logits)
        samples.append(all_logits)

        if H is None:
            H = entropies(torch.tensor(all_logits))
        if BT is None:
            BT = break_ties(torch.tensor(all_logits))
    np.savez("active_learning.py",
             entropies=H.detach().cpu().numpy(),
             breaking_ties=BT.detach().cpu().numpy(),
             samples = np.stack(samples))
    print("Active learning sampling finished")
