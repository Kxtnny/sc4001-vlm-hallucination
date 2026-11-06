"""Loading Qwen2-VL-2B-Instruct and running it on one image + one prompt.

Key facts about Qwen2-VL that later experiments rely on:
  * The image becomes a contiguous run of <|image_pad|> tokens in input_ids,
    wrapped by <|vision_start|> ... <|vision_end|>. Their ids are in model.config.
  * The number of image tokens = (t * h * w) of image_grid_thw / merge_size**2,
    where merge_size = 2 (each image token merges a 2x2 block of 14px patches).
  * With attn_implementation="eager", a forward pass with output_attentions=True
    returns one tensor per decoder layer of shape [batch, heads, query, key].
"""
import torch

from . import config


# ---------------------------------------------------------------------------
# Model-independent helpers (also unit-tested on a tiny random model).
# ---------------------------------------------------------------------------
def image_token_mask(input_ids, image_token_id):
    """Bool tensor, True where the token is an image token. input_ids: [1, L] or [L]."""
    return input_ids.reshape(-1) == image_token_id


def expected_num_image_tokens(image_grid_thw, merge_size=2):
    return int(image_grid_thw.prod(dim=-1).sum().item()) // (merge_size ** 2)


def check_image_tokens(input_ids, image_grid_thw, cfg, merge_size=2):
    """Assert the image tokens are where we think they are. Returns their positions.

    Checks: count matches the grid, the run is contiguous, and it is wrapped by
    <|vision_start|> / <|vision_end|>. If any of these fail, every attention
    statistic we compute later would be wrong, so we stop immediately.
    """
    ids = input_ids.reshape(-1)
    pos = torch.nonzero(image_token_mask(ids, cfg.image_token_id)).reshape(-1)
    n_expected = expected_num_image_tokens(image_grid_thw, merge_size)
    assert len(pos) == n_expected, f"found {len(pos)} image tokens, expected {n_expected}"
    assert int(pos[-1] - pos[0]) + 1 == len(pos), "image tokens are not contiguous"
    assert int(ids[pos[0] - 1]) == cfg.vision_start_token_id, "no <|vision_start|> before image"
    assert int(ids[pos[-1] + 1]) == cfg.vision_end_token_id, "no <|vision_end|> after image"
    return pos


def image_attention_share(attentions, img_mask):
    """Share of attention each query token puts on image tokens, per layer.

    attentions: tuple (one per layer) of [1, heads, Q, K] tensors.
    img_mask:   bool [K], True at image-token key positions.
    Returns a float tensor [layers, Q], averaged over heads.
    """
    shares = []
    for att in attentions:
        a = att[0].float()                        # [heads, Q, K]
        s = a[..., img_mask].sum(-1) / a.sum(-1)  # divide by row sum in case of fp16 drift
        shares.append(s.mean(0))                  # average over heads -> [Q]
    return torch.stack(shares)


# ---------------------------------------------------------------------------
# The model wrapper.
# ---------------------------------------------------------------------------
class VLM:
    def __init__(self, model_id=config.MODEL_ID, dtype=config.DTYPE, attn_impl=config.ATTN_IMPL,
                 min_pixels=config.MIN_PIXELS, max_pixels=config.MAX_PIXELS, device=None):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=getattr(torch, dtype),
            attn_implementation=attn_impl,
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        self.cfg = self.model.config
        self.max_image_tokens = max_pixels // (28 * 28)
        self.merge_size = getattr(self.processor.image_processor, "merge_size", 2)
        self.num_layers = self.cfg.text_config.num_hidden_layers

    def build_inputs(self, image, prompt):
        """Chat-format one image + one text prompt and return model inputs on the GPU."""
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        return inputs.to(self.device)

    @torch.no_grad()
    def generate(self, image, prompt, max_new_tokens=config.POPE_MAX_NEW_TOKENS, return_inputs=False):
        """Greedy decoding (deterministic), returns the decoded answer string."""
        inputs = self.build_inputs(image, prompt)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            # Qwen's default generation_config sets these for sampling; unset them
            # so greedy decoding is used and transformers does not warn.
            temperature=None, top_p=None, top_k=None,
        )
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        return (answer, inputs, out) if return_inputs else answer

    @torch.no_grad()
    def forward_attentions(self, inputs):
        """One forward pass returning all decoder-layer attentions."""
        return self.model(**inputs, output_attentions=True)
