

# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert BART checkpoint."""


import argparse
import logging
import os
from pathlib import Path

import fairseq
import torch
from packaging import version
from fairseq.models.bart import BARTModel

from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartForSequenceClassification,
    BartModel,
    BartTokenizer,
)
from transformers.modeling_bart import _make_linear_from_emb


FAIRSEQ_MODELS = ["bart.large", "bart.large.mnli", "bart.large.cnn", "bart_xsum/model.pt"]
extra_arch = {"bart.large": BartModel, "bart.large.mnli": BartForSequenceClassification}
if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_TEXT = " Hello world! cécé herlolip"

mnli_rename_keys = [
    ("model.classification_heads.mnli.dense.weight", "classification_head.dense.weight"),
    ("model.classification_heads.mnli.dense.bias", "classification_head.dense.bias"),
    ("model.classification_heads.mnli.out_proj.weight", "classification_head.out_proj.weight"),
    ("model.classification_heads.mnli.out_proj.bias", "classification_head.out_proj.bias"),
]


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def load_xsum_checkpoint(checkpoint_path):
    """Checkpoint path should end in model.pt"""
    #sd = torch.load(checkpoint_path, map_location="cpu")
    #hub_interface = torch.hub.load("pytorch/fairseq", "bart.large.cnn").eval()
    #hub_interface.model.load_state_dict(sd["model"])
    hub_interface = BARTModel.from_pretrained(
        "/private/home/alexfabbri/convosumm/Argument-Graph-Mining/longformer/bart.large/",
        checkpoint_file='model.pt',
        data_name_or_path='/private/home/alexfabbri/convosumm/Argument-Graph-Mining/acl2018_abssumm/truncated_ratio-bin',
        gpt2_encoder_json='/private/home/alexfabbri/convosumm/Argument-Graph-Mining/val_test/encoder.json',
        gpt2_vocab_bpe='/private/home/alexfabbri/convosumm/Argument-Graph-Mining/val_test/vocab.bpe',
        max_source_positions=2048,
    )
    return hub_interface


def convert_checkpoint_from_disk(checkpoint_path, **config_kwargs):
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    remove_ignore_keys_(state_dict)
    vocab_size = state_dict["encoder.embed_tokens.weight"].shape[0]
    state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
    mbart_config = BartConfig(vocab_size=vocab_size, **config_kwargs)
    model = BartForConditionalGeneration(mbart_config)
    model.model.load_state_dict(state_dict)
    if hasattr(model, "lm_head"):
        model.lm_head = _make_linear_from_emb(model.model.shared)
    return model


@torch.no_grad()
def convert_bart_checkpoint(checkpoint_path, pytorch_dump_folder_path, hf_checkpoint_name=None):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    if not os.path.exists(checkpoint_path):
        bart = torch.hub.load("pytorch/fairseq", checkpoint_path).eval()
    else:
        bart = load_xsum_checkpoint(checkpoint_path)

    bart.model.upgrade_state_dict(bart.model.state_dict())
    if hf_checkpoint_name is None:
        hf_checkpoint_name = checkpoint_path.replace(".", "-")
    config = BartConfig.from_pretrained(hf_checkpoint_name)
    tokens = bart.encode(SAMPLE_TEXT).unsqueeze(0)
    tokens2 = BartTokenizer.from_pretrained(hf_checkpoint_name).encode(SAMPLE_TEXT, return_tensors="pt").unsqueeze(0)
    assert torch.eq(tokens, tokens2).all()

    if checkpoint_path == "bart.large.mnli":
        state_dict = bart.state_dict()
        remove_ignore_keys_(state_dict)
        state_dict["model.shared.weight"] = state_dict["model.decoder.embed_tokens.weight"]
        for src, dest in mnli_rename_keys:
            rename_key(state_dict, src, dest)
        model = BartForSequenceClassification(config).eval()
        model.load_state_dict(state_dict)
        fairseq_output = bart.predict("mnli", tokens, return_logits=True)
        new_model_outputs = model(tokens)[0]  # logits
    else:  # no classification heads to worry about
        state_dict = bart.model.state_dict()
        remove_ignore_keys_(state_dict)
        state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
        fairseq_output = bart.extract_features(tokens)
        if hf_checkpoint_name == "facebook/bart-large":
            import pdb;pdb.set_trace()
            tmp_model = BartModel(config).eval()
            sd = tmp_model.state_dict()
            shorter_pos_embeds = sd['encoder.embed_positions.weight']
            new_config = tmp_model.config
            new_config.max_position_embeddings = 2048
            model = BartModel(new_config).eval()
            import pdb;pdb.set_trace()
            correctly_shaped_pos_weight = model.encoder.embed_positions.weight
            
            correctly_shaped_pos_weight[:shorter_pos_embeds.shape[0]] = shorter_pos_embeds
            
            sd['decoder.embed_positions.weight'] = correctly_shaped_pos_weight
            sd['encoder.embed_positions.weight'] = correctly_shaped_pos_weight
            
            model.load_state_dict(sd, strict=True)
            import pdb;pdb.set_trace()
            #model.load_state_dict(state_dict)
            new_model_outputs = model(tokens)[0]
        else:
            model = BartForConditionalGeneration(config).eval()  # an existing summarization ckpt
            model.model.load_state_dict(state_dict)
            if hasattr(model, "lm_head"):
                model.lm_head = _make_linear_from_emb(model.model.shared)
            new_model_outputs = model.model(tokens)[0]

    # Check results
    assert fairseq_output.shape == new_model_outputs.shape
    assert (fairseq_output == new_model_outputs).all().item()
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--fairseq_path", type=str, help="bart.large, bart.large.cnn or a path to a model.pt on local filesystem."
    )
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--hf_config", default=None, type=str, help="Which huggingface architecture to use: bart-large-xsum"
    )
    args = parser.parse_args()
    convert_bart_checkpoint(args.fairseq_path, args.pytorch_dump_folder_path, hf_checkpoint_name=args.hf_config)

