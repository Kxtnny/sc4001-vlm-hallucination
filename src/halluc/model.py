"""Loading Qwen2-VL-2B-Instruct and running it on one image + one prompt.

Key facts about Qwen2-VL that later experiments rely on:
  * The image becomes a contiguous run of <|image_pad|> tokens in input_ids,
    wrapped by <|vision_start|> ... <|vision_end|>. Their ids are in model.config.
  * The number of image tokens = (t * h * w) of image_grid_thw / merge_size**2,
    where merge_size = 2 (each image token merges a 2x2 block of 14px patches).
  * With attn_implementation="eager", a forward pass with output_attentions=True
    returns one tensor per decoder layer of shape [batch, heads, query, key].
  * Decoder attention goes through `eager_attention_forward` in transformers'
    modeling_qwen2_vl, which we wrap for the E4 intervention (ImageAttentionBoost).
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


def parse_layers(spec, num_layers):
    """Layer-range string -> sorted list of layer indices.
    "none" -> [], "all" -> every layer, "8-15" -> 8..15 inclusive, "0-3,20-27" -> both ranges."""
    spec = str(spec).strip().lower()
    if spec in ("", "none"):
        return []
    if spec == "all":
        return list(range(num_layers))
    layers = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        lo, hi = int(a), int(b or a)
        assert 0 <= lo <= hi < num_layers, f"bad layer range {part!r} for {num_layers} layers"
        layers.update(range(lo, hi + 1))
    return sorted(layers)


# ---------------------------------------------------------------------------
# E4 intervention: boost attention to image tokens.
# ---------------------------------------------------------------------------
class ImageAttentionBoost:
    """Multiplies attention weights on image-token keys by (1 + alpha) in chosen decoder
    layers, then renormalises every attention row so it sums to 1 again.

    It only changes rows whose query comes *after* the image (prompt text and generated
    tokens), so the image tokens' own representations are left untouched. The boost is
    applied after the softmax, exactly as described in the report:
        a'_qk = a_qk * (1 + alpha * [k is an image token]) / sum_k' (...)
    (This equals adding log(1 + alpha) to the image-key logits before the softmax.)

    Works for a left-padded batch: `img_mask` [batch, prompt_len] marks each row's image
    tokens. Keys beyond the prompt (generated tokens) are never image tokens.

    mode = "pai" instead applies the logit amplification of PAI (Liu et al., ECCV 2024) *before* the
    softmax, to the last query position only (the token being generated), as in PAI's code:
        z'_qk = z_qk + alpha * |z_qk|   for image keys k   (without PAI's image-free contrast term)

    Usage: the VLM wrapper installs one instance as Qwen2-VL's eager attention function;
    the `boosted` context manager switches it on for one input and off afterwards. With
    alpha = 0 the original function runs unchanged, so outputs are bit-identical.
    """

    def __init__(self, original_fn, attention_cls, repeat_kv):
        self.original_fn = original_fn
        self.attention_cls = attention_cls
        self.repeat_kv = repeat_kv
        self.alpha, self.layers, self.mode = 0.0, frozenset(), "post"
        self.img_mask = self.img_end = None
        self.calls = 0   # number of boosted attention calls (lets tests check it really ran)

    def active_for(self, module):
        return (self.alpha != 0 and isinstance(module, self.attention_cls)
                and getattr(module, "layer_idx", None) in self.layers)

    def __call__(self, module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if not self.active_for(module):
            return self.original_fn(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
        # Same computation as transformers' eager_attention_forward, plus the boost.
        key_states = self.repeat_kv(key, module.num_key_value_groups)
        value_states = self.repeat_kv(value, module.num_key_value_groups)
        attn = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn = attn + attention_mask[:, :, :, : key_states.shape[-2]]

        bsz, _, q_len, k_len = attn.shape
        img = self.img_mask.to(attn.device)
        assert img.shape[0] == bsz and img.shape[1] <= k_len, "image positions not set for this input"
        key_img = torch.zeros(bsz, k_len, dtype=torch.bool, device=attn.device)
        key_img[:, :img.shape[1]] = img
        q_pos = torch.arange(k_len - q_len, k_len, device=attn.device)            # absolute query positions
        rows = q_pos[None, :] > self.img_end.to(attn.device)[:, None]             # [batch, Q] after the image

        if self.mode == "pai":
            # Pre-softmax: amplify the image-key logits of the last query (the token being predicted).
            last = rows & (q_pos == k_len - 1)[None, :]
            sel = last[:, None, :, None] & key_img[:, None, None, :]               # [batch, 1, Q, K]
            if bool(sel.any()):
                attn = torch.where(sel, attn + self.alpha * attn.abs(), attn)
                self.calls += 1
            attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32)
            rows = torch.zeros_like(rows)                                           # no post-softmax step
        else:
            attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32)
        if rows.any():
            scale = 1.0 + self.alpha * key_img.to(attn.dtype)                      # [batch, K]
            boosted = attn * scale[:, None, None, :]
            boosted = boosted / boosted.sum(-1, keepdim=True)
            attn = torch.where(rows[:, None, :, None], boosted, attn)
            self.calls += 1

        attn = attn.to(query.dtype)
        attn = torch.nn.functional.dropout(attn, p=dropout, training=module.training)
        out = torch.matmul(attn, value_states).transpose(1, 2).contiguous()
        return out, attn


def install_boost():
    """Install (once) the ImageAttentionBoost as Qwen2-VL's eager attention function.

    Qwen2-VL looks up `eager_attention_forward` in its module namespace on every call,
    so replacing that name is enough. Returns the (shared) boost object.
    """
    from transformers.models.qwen2_vl import modeling_qwen2_vl as mq
    if not isinstance(mq.eager_attention_forward, ImageAttentionBoost):
        mq.eager_attention_forward = ImageAttentionBoost(mq.eager_attention_forward, mq.Qwen2VLAttention,
                                                         mq.repeat_kv)
    return mq.eager_attention_forward


class boosted:
    """Context manager: switch the boost on for one (batched) input, and always off afterwards.

    img_mask: bool [batch, prompt_len] (or [prompt_len]) marking image tokens.
    """

    def __init__(self, boost, alpha, layers, img_mask, mode="post"):
        assert mode in ("post", "pai"), mode
        self.boost, self.alpha, self.layers, self.mode = boost, float(alpha), frozenset(layers), mode
        self.img_mask = None if img_mask is None else img_mask.reshape(-1, img_mask.shape[-1])

    def __enter__(self):
        if self.alpha != 0 and self.layers:
            m = self.img_mask
            pos = torch.arange(m.shape[1], device=m.device)
            self.boost.img_mask = m
            self.boost.img_end = torch.where(m, pos, -1).max(dim=1).values      # last image position per row
            assert bool((self.boost.img_end >= 0).all()), "a row has no image tokens"
            self.boost.alpha, self.boost.layers, self.boost.mode = self.alpha, self.layers, self.mode
        return self.boost

    def __exit__(self, *exc):
        self.boost.alpha, self.boost.layers, self.boost.mode = 0.0, frozenset(), "post"
        self.boost.img_mask = self.boost.img_end = None


def resolve_dtype(dtype):
    """'auto' -> bfloat16 on GPUs with native bf16 (compute capability >= 8.0), else float32.

    float16 is NOT used by default: with transformers 5.22.2, Qwen2-VL-2B overflows in
    float16 (NaN logits from decoder layer 1 onward; caught by scripts/sanity_check.py).
    """
    if dtype != "auto":
        return dtype
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 0):
        return "bfloat16"
    return "float32"


# ---------------------------------------------------------------------------
# The model wrapper.
# ---------------------------------------------------------------------------
class VLM:
    def __init__(self, model_id=config.MODEL_ID, dtype=config.DTYPE, attn_impl=config.ATTN_IMPL,
                 min_pixels=config.MIN_PIXELS, max_pixels=config.MAX_PIXELS, device=None):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = resolve_dtype(dtype)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=getattr(torch, self.dtype),
            attn_implementation=attn_impl,
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        self.processor.tokenizer.padding_side = "left"   # batched generation needs left padding
        self.cfg = self.model.config
        self.max_image_tokens = max_pixels // (28 * 28)
        self.merge_size = getattr(self.processor.image_processor, "merge_size", 2)
        self.num_layers = self.cfg.text_config.num_hidden_layers
        self.boost = install_boost()
        self.boost_mode = "post"   # "post" (our post-softmax boost) or "pai" (PAI-style logit amplification)
        tok = self.processor.tokenizer
        # First token of each answer word, for P(yes) / P(no) at the decision position.
        self.yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", " Yes", " yes")})
        self.no_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no", " No", " no")})
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        print(f"[VLM] {model_id} in {self.dtype} on {self.device}")

    def _image_mask(self, inputs):
        """Bool [batch, L] of image-token positions, after checking them row by row."""
        ids, grids = inputs["input_ids"], inputs["image_grid_thw"]
        for b in range(ids.shape[0]):   # one image per row, grids in row order
            check_image_tokens(ids[b], grids[b:b + 1], self.cfg, self.merge_size)
        return ids == self.cfg.image_token_id

    def boosting(self, inputs, alpha=0.0, layers="none"):
        """Context manager that applies the E4 intervention to this input (no-op if alpha = 0)."""
        layer_list = parse_layers(layers, self.num_layers)
        mask = self._image_mask(inputs) if alpha != 0 and layer_list else None
        return boosted(self.boost, alpha, layer_list, mask, mode=self.boost_mode)

    def build_inputs(self, image, prompt):
        """Chat-format one image + one text prompt and return model inputs on the GPU."""
        return self.build_batch([image], [prompt])

    def build_batch(self, images, prompts):
        """Chat-format several (image, prompt) pairs into one left-padded batch."""
        texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        inputs = self.processor(text=texts, images=list(images), return_tensors="pt", padding=True)
        return inputs.to(self.device)

    @torch.no_grad()
    def generate(self, image, prompt, max_new_tokens=config.POPE_MAX_NEW_TOKENS, return_inputs=False,
                 alpha=0.0, layers="none"):
        """Greedy decoding (deterministic), returns the decoded answer string.

        alpha / layers: optional E4 intervention (alpha = 0 is the unmodified model).
        """
        inputs = self.build_inputs(image, prompt)
        out = self._generate(inputs, max_new_tokens, alpha, layers)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        return (answer, inputs, out) if return_inputs else answer

    @torch.no_grad()
    def generate_batch(self, images, prompts, max_new_tokens=config.POPE_MAX_NEW_TOKENS, alpha=0.0, layers="none"):
        """Greedy decoding for a left-padded batch. Returns a list of (answer, new_token_ids).

        A row's generated ids are cut after its first end-of-turn token (the rest is padding).
        """
        inputs = self.build_batch(images, prompts)
        out = self._generate(inputs, max_new_tokens, alpha, layers)
        new = out[:, inputs["input_ids"].shape[1]:].tolist()
        results = []
        for row in new:
            ids = []
            for t in row:
                ids.append(t)
                if t in self.eos_ids:
                    break
            text = self.processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
            results.append((text, ids))
        return results

    def _generate(self, inputs, max_new_tokens, alpha, layers):
        with self.boosting(inputs, alpha, layers):
            return self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                # Qwen's default generation_config sets these for sampling; unset them
                # so greedy decoding is used and transformers does not warn.
                temperature=None, top_p=None, top_k=None,
            )

    @torch.no_grad()
    def decision_signals(self, image, prompt):
        """Statistics at the last prompt token, where a yes/no answer is decided (for E5)."""
        return self.decision_signals_batch([image], [prompt])[0]

    @torch.no_grad()
    def decision_signals_batch(self, images, prompts):
        """Per row: img_share_last (per layer, share of attention on image tokens at the last
        prompt token, mean over heads) and p_yes / p_no (probability of the first token of a
        yes / no answer). Left padding keeps every row's last prompt token at position -1."""
        inputs = self.build_batch(images, prompts)
        out = self.model(**inputs, output_attentions=True)
        mask = inputs["input_ids"] == self.cfg.image_token_id
        p = torch.softmax(out.logits[:, -1].float(), dim=-1)
        res = []
        for b in range(mask.shape[0]):
            share = image_attention_share([a[b:b + 1, :, -1:, :] for a in out.attentions], mask[b])[:, 0]
            res.append({"img_share_last": [round(float(x), 5) for x in share],
                        "p_yes": float(p[b, self.yes_ids].sum()), "p_no": float(p[b, self.no_ids].sum())})
        return res

    @torch.no_grad()
    def forward_attentions(self, inputs):
        """One forward pass returning all decoder-layer attentions."""
        return self.model(**inputs, output_attentions=True)
