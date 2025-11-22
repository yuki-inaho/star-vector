import types
import torch

import starvector.model.models.starvector_base as base_module
from starvector.model.models.starvector_base import StarVectorBase


class _DummyConfig:
    def __init__(self, torch_dtype="float32"):
        self.torch_dtype = torch_dtype
        self.image_encoder_type = "clip"
        self.adapter_norm = "layer_norm"
        self.max_length_train = 600
        # Attributes used downstream; set simple placeholders.
        self.starcoder_model_name = "dummy"
        self.max_length = 512


class _DummySVGTransformer:
    def __init__(self):
        # Only hidden_size is used by Adapter creation.
        self.transformer = types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=8))
        # Minimal tokenizer fields used by StarVectorBase helpers.
        self.tokenizer = types.SimpleNamespace(pad_token_id=0)
        self.svg_start_token = "<svg>"
        self.tokenizer.eos_token_id = 1
        self._params = []

    def named_parameters(self):
        return []


class _DummyStarVector(StarVectorBase):
    """Minimal concrete subclass to exercise dtype handling without loading real models."""

    def _get_svg_transformer(self, config, **kwargs):
        return _DummySVGTransformer()

    def _get_embeddings(self, input_ids):
        # Return zeros with the right batch/seq shape and hidden size matching transformer.config.hidden_size.
        hidden_size = self.svg_transformer.transformer.config.hidden_size
        return torch.zeros(input_ids.shape[0], input_ids.shape[1], hidden_size)

    def _get_svg_text(self, svg_list):
        return svg_list


class _StubImageEncoder:
    """Lightweight stub to avoid constructing the full vision backbone."""

    def __init__(self, config, **kwargs):
        # get_hidden_size_and_query_length reads num_features
        self.visual_encoder = types.SimpleNamespace(num_features=4)
        self._params = []

    def named_parameters(self):
        return []


def _make_model(torch_dtype):
    cfg = _DummyConfig(torch_dtype=torch_dtype)
    # Patch ImageEncoder to our lightweight stub.
    original_encoder = base_module.ImageEncoder
    base_module.ImageEncoder = _StubImageEncoder
    try:
        return _DummyStarVector(cfg, task="im2svg", model_precision=torch_dtype)
    finally:
        base_module.ImageEncoder = original_encoder


def test_model_precision_accepts_string_and_converts_to_torch_dtype():
    model = _make_model("float32")
    assert model.model_precision is torch.float32
    # Adapter weights should also be converted.
    assert model.image_projection.c_fc.weight.dtype == torch.float32


def test_model_precision_accepts_torch_dtype_object():
    model = _make_model(torch.float16)
    assert model.model_precision is torch.float16
    assert model.image_projection.c_proj.weight.dtype == torch.float16
