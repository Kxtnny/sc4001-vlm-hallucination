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


# ---------------------------------------------------------------------------
# E2: CHAIR
# ---------------------------------------------------------------------------
MINI_SYNONYMS = """person, girl, boy, man, woman, people
dog, puppy
cat, kitten
dining table, table, desk
hot dog
teddy bear, teddybear
chair, seat, stool
toilet, urinal
wine glass
skis, ski
elephant
"""


@pytest.fixture
def chair(tmp_path):
    pytest.importorskip("inflect")
    from halluc.chair import Chair
    p = tmp_path / "syn.txt"
    p.write_text(MINI_SYNONYMS, encoding="utf-8")
    return Chair(p, expected_categories=None)


def test_chair_parse_synonyms_plurals_and_double_words(chair):
    ms = chair.parse("Two men and a Puppy sit at the dining tables near some hot dogs and skis.")
    assert [m["node"] for m in ms] == ["person", "dog", "dining table", "hot dog", "skis"]
    text = "Two men and a Puppy sit at the dining tables near some hot dogs and skis."
    assert [text[m["start"]:m["end"]] for m in ms] == ["men", "Puppy", "tables", "hot dogs", "skis"]  # "table" alone maps to dining table, as in the original


def test_chair_special_cases(chair):
    # "toilet seat" is the toilet, not a chair; "baby elephant" is an elephant.
    assert [m["node"] for m in chair.parse("A toilet seat in a bathroom.")] == ["toilet"]
    assert [m["node"] for m in chair.parse("The seat of the toilet is up.")] == ["toilet"]
    assert [m["node"] for m in chair.parse("A baby elephant and wine glasses.")] == ["elephant", "wine glass"]
    # Non-object words are never turned into objects by singularising.
    assert chair.parse("This is a glass of water.") == []


def test_chair_scores(chair):
    from halluc.chair import chair_metrics
    gt = chair.ground_truth(["dog", "dining table"], ["A man with his dog."])
    assert gt == {"dog", "dining table", "person"}
    s1 = chair.score_caption("A dog and a cat on a table. The dog sleeps.", gt)
    assert s1["n_mentions"] == 4 and s1["n_hallucinated"] == 1 and s1["hallucinated_nodes"] == ["cat"]
    s2 = chair.score_caption("A man.", gt)
    m = chair_metrics([s1, s2])
    assert m["chair_s"] == pytest.approx(0.5)
    assert m["chair_i"] == pytest.approx(1 / 5)
    assert m["recall"] == pytest.approx((2 + 1) / 6)   # {dog, table} + {person} over 3 + 3 gt objects


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def test_auroc_matches_sklearn():
    np = pytest.importorskip("numpy")
    metrics = pytest.importorskip("sklearn.metrics")
    from halluc.stats import auroc, risk_coverage
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.normal(size=300) + y, 1)       # rounded -> ties
    assert auroc(y, s) == pytest.approx(metrics.roc_auc_score(y, s))
    cov, acc = risk_coverage([1, 0, 1, 1], [0.9, 0.1, 0.8, 0.5])
    assert acc.tolist() == [1.0, 1.0, 1.0, 0.75]


def test_parse_layers():
    from halluc.model import parse_layers
    assert parse_layers("none", 28) == []
    assert parse_layers("all", 4) == [0, 1, 2, 3]
    assert parse_layers("8-10", 28) == [8, 9, 10]
    assert parse_layers("0-1,26-27", 28) == [0, 1, 26, 27]
    with pytest.raises(AssertionError):
        parse_layers("20-30", 28)


# ---------------------------------------------------------------------------
# E4: the image-attention boost on a tiny random Qwen2-VL
# ---------------------------------------------------------------------------
def _tiny_model_and_inputs():
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    from PIL import Image
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
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                  mm_token_type_ids=(ids == cfg.image_token_id).long(),
                  pixel_values=px["pixel_values"], image_grid_thw=px["image_grid_thw"])
    return torch, model, cfg, inputs


def test_boost_alpha_zero_is_identical_and_alpha_changes_attention():
    torch, model, cfg, inputs = _tiny_model_and_inputs()
    from halluc.model import (boosted, check_image_tokens, image_attention_share, image_token_mask,
                              install_boost)
    boost = install_boost()
    assert install_boost() is boost                      # installing twice does not double-wrap
    pos = check_image_tokens(inputs["input_ids"], inputs["image_grid_thw"], cfg)
    mask = image_token_mask(inputs["input_ids"], cfg.image_token_id)
    after = int(pos[-1]) + 1                             # first query position after the image

    with torch.no_grad():
        ref = model(**inputs, output_attentions=True)
        with boosted(boost, 0.0, [0, 1, 2], mask):       # alpha = 0: must be bit-identical
            same = model(**inputs, output_attentions=True)
        calls = boost.calls
        with boosted(boost, 1.5, [0], mask):             # boost layer 0 only
            boo = model(**inputs, output_attentions=True)
        assert boost.calls == calls + 1
        assert boost.alpha == 0.0                        # switched off again afterwards
        gen_ref = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        with boosted(boost, 0.0, [0, 1, 2], mask):
            gen_same = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    assert torch.equal(ref.logits, same.logits)
    assert torch.equal(gen_ref, gen_same)
    assert not torch.allclose(ref.logits, boo.logits)

    s0 = image_attention_share(ref.attentions, mask)
    s1 = image_attention_share(boo.attentions, mask)
    for a in boo.attentions:                             # rows still sum to 1
        assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)
    # Layer 0 sees identical inputs, so for every head h: s' = s(1+a) / (1 + a s).
    a0, b0 = ref.attentions[0][0].float(), boo.attentions[0][0].float()   # [heads, Q, K]
    sh0 = a0[..., mask].sum(-1)
    sh1 = b0[..., mask].sum(-1)
    expected = sh0 * 2.5 / (1 + 1.5 * sh0)
    assert torch.allclose(sh1[:, after:], expected[:, after:], atol=1e-5)
    assert torch.allclose(sh1[:, :after], sh0[:, :after])                 # rows up to the image untouched
    assert torch.all(s1[0, after:] > s0[0, after:])


def test_boost_batched_rows_match_single_rows():
    """With left padding, each row of a batch must be boosted exactly like it is alone."""
    torch, model, cfg, inputs = _tiny_model_and_inputs()
    from halluc.model import boosted, image_token_mask, install_boost
    boost = install_boost()
    ids = inputs["input_ids"]
    pad = torch.tensor([[0, 0, 0]])
    ids2 = torch.cat([pad, ids[:, :-3]], dim=1)          # second row: 3 pads + a 3-token-shorter prompt
    ids2[0, 3] = 9                                       # (replace its first token so rows differ)
    batch = dict(inputs)
    batch["input_ids"] = torch.cat([ids, ids2])
    batch["attention_mask"] = torch.cat([torch.ones_like(ids), torch.cat([torch.zeros_like(pad),
                                                                         torch.ones_like(ids[:, :-3])], 1)])
    batch["mm_token_type_ids"] = (batch["input_ids"] == cfg.image_token_id).long()
    batch["pixel_values"] = torch.cat([inputs["pixel_values"]] * 2)
    batch["image_grid_thw"] = torch.cat([inputs["image_grid_thw"]] * 2)
    with torch.no_grad():
        with boosted(boost, 2.0, [0, 1], image_token_mask(ids, cfg.image_token_id)):
            single = model(**inputs).logits[0, -1]
        with boosted(boost, 2.0, [0, 1], batch["input_ids"] == cfg.image_token_id):
            both = model(**batch).logits
    assert torch.allclose(both[0, -1], single, atol=1e-4)
