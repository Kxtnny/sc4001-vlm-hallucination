"""Fast tests that need no GPU and no downloads. Run: python -m pytest -q tests

The model test builds a *tiny randomly initialised* Qwen2-VL with the same
architecture as the real one, so we can check our image-token and attention
helpers in seconds. It says nothing about answer quality.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halluc.data import subsample  # noqa: E402
from halluc.metrics import parse_yes_no, pope_metrics  # noqa: E402


@pytest.mark.parametrize("text,expected,fallback", [
    ("Yes", "yes", False),
    ("yes.", "yes", False),
    ("No, there is no dog.", "no", False),
    ("NO", "no", False),
    ("There is not a dog in the image.", "no", True),
    ("There is a dog.", "yes", True),
    ("", "yes", True),
    ("Nope", "yes", True),  # documents a known limitation of the POPE rule
])
def test_parse_yes_no(text, expected, fallback):
    assert parse_yes_no(text) == (expected, fallback)


def test_pope_metrics():
    recs = [{"pred": p, "label": l} for p, l in
            [("yes", "yes"), ("yes", "no"), ("no", "no"), ("no", "yes"), ("yes", "yes")]]
    m = pope_metrics(recs)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 1, 1)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["yes_ratio"] == pytest.approx(3 / 5)


def test_subsample_is_fixed_and_nested():
    rows = list(range(3000))
    a, b = subsample(rows, 50), subsample(rows, 1500)
    assert a == subsample(rows, 50)          # same every time
    assert set(a) <= set(b)                   # small subset is inside the big one (resume works)
    assert subsample(rows, None) == rows


def test_attention_helpers_on_tiny_qwen2vl():
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    from PIL import Image
    from halluc.model import check_image_tokens, image_attention_share, image_token_mask

    cfg = tf.Qwen2VLConfig(
        text_config=dict(hidden_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                         intermediate_size=128, vocab_size=151700,
                         rope_parameters={"rope_type": "default", "rope_theta": 1e6, "mrope_section": [2, 3, 3]}),
        vision_config=dict(depth=2, embed_dim=32, hidden_size=64, num_heads=4, mlp_ratio=2),
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = tf.Qwen2VLForConditionalGeneration(cfg).eval()
    ip = tf.Qwen2VLImageProcessor(min_pixels=16 * 28 * 28, max_pixels=64 * 28 * 28)
    px = ip(images=[Image.new("RGB", (320, 240), (120, 30, 200))], return_tensors="pt")
    n_img = int(px["image_grid_thw"].prod()) // 4
    ids = torch.tensor([[1, 2, 3, cfg.vision_start_token_id] + [cfg.image_token_id] * n_img
                        + [cfg.vision_end_token_id, 5, 6, 7]])

    pos = check_image_tokens(ids, px["image_grid_thw"], cfg)
    assert pos.tolist() == list(range(4, 4 + n_img))
    assert n_img <= 64

    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    mm_token_type_ids=(ids == cfg.image_token_id).long(),
                    pixel_values=px["pixel_values"], image_grid_thw=px["image_grid_thw"],
                    output_attentions=True)
    assert len(out.attentions) == 3
    for a in out.attentions:
        assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-4)

    share = image_attention_share(out.attentions, image_token_mask(ids, cfg.image_token_id))
    assert share.shape == (3, ids.shape[1])
    assert torch.all(share[:, :4] == 0)        # causal: text before the image cannot see it
    assert torch.all((share >= 0) & (share <= 1))

    # A wrong grid must be caught.
    bad = px["image_grid_thw"].clone()
    bad[0, 1] += 2
    with pytest.raises(AssertionError):
        check_image_tokens(ids, bad, cfg)
