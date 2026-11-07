"""E3 (part 1): image-attention statistics for every object word in saved captions.

For each caption saved by caption_chair.py we re-run the model once with teacher
forcing (prompt + image + the caption's own generated tokens) and, for every
object mention found by CHAIR, record per decoder layer the share of attention
(averaged over heads) that goes to image tokens from:

  share_pred  the position that *predicts* the object word's first token
              (the moment the model decides to write the object), and
  share_tok   the object word's first token itself.

We also record the model's probability of that token (a standard confidence
baseline to compare the attention signal against) and its position in the caption
(hallucinations tend to come late in long captions, so position is a confounder
that the analysis controls for).

Only these summary numbers are saved, never full attention maps.

Output: results/attention/<tag>/<split>.jsonl, one line per caption.

Examples:
  python scripts/extract_attention.py --split dev --limit 100        # dev captions (layer choice)
  python scripts/extract_attention.py --split chair_test             # test captions (reported analysis)
"""
import _path  # noqa: F401
import argparse

import torch
from tqdm import tqdm

from halluc import config
from halluc.data import load_image, load_split, subsample
from halluc.io_utils import JsonlWriter, read_jsonl, run_metadata, save_json
from halluc.model import VLM, check_image_tokens, image_attention_share, image_token_mask


def token_char_ends(tokenizer, ids):
    """End character offset of each token in decode(ids) (cumulative decoding)."""
    return [len(tokenizer.decode(ids[:k + 1], skip_special_tokens=True)) for k in range(len(ids))]


def mention_token_index(ends, char_pos):
    """Index of the token that contains character position char_pos."""
    for k, e in enumerate(ends):
        if e > char_pos:
            return k
    return None


@torch.no_grad()
def caption_attention(vlm, image, new_ids, mentions):
    """Teacher-forced pass over prompt + caption; per-mention attention statistics."""
    inputs = vlm.build_inputs(image, config.CAPTION_PROMPT)
    prompt_len = inputs["input_ids"].shape[1]
    check_image_tokens(inputs["input_ids"], inputs["image_grid_thw"], vlm.cfg, vlm.merge_size)

    cap = torch.tensor([new_ids], device=vlm.device)
    full = dict(inputs)
    full["input_ids"] = torch.cat([inputs["input_ids"], cap], dim=1)
    full["attention_mask"] = torch.ones_like(full["input_ids"])
    if "mm_token_type_ids" in inputs:  # 0 for text tokens
        full["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], torch.zeros_like(cap)], dim=1)
    out = vlm.model(**full, output_attentions=True)

    mask = image_token_mask(full["input_ids"], vlm.cfg.image_token_id)
    share = image_attention_share(out.attentions, mask).cpu()          # [layers, L]
    logp = torch.log_softmax(out.logits[0].float(), dim=-1)           # [L, vocab]
    del out

    tok = vlm.processor.tokenizer
    text = tok.decode(new_ids, skip_special_tokens=True)
    lead = len(text) - len(text.lstrip())   # saved captions are .strip()-ped
    ends = token_char_ends(tok, new_ids)
    rows = []
    for m in mentions:
        k = mention_token_index(ends, m["start"] + lead)
        if k is None:
            continue
        p = prompt_len + k                  # position of the object's first token
        rows.append({
            "word": m["word"], "node": m["node"], "hallucinated": m["hallucinated"],
            "token": tok.decode([new_ids[k]]), "tok_index": k, "rel_pos": k / max(1, len(new_ids)),
            "logprob": round(float(logp[p - 1, new_ids[k]]), 5),
            "share_pred": [round(float(x), 5) for x in share[:, p - 1]],
            "share_tok": [round(float(x), 5) for x in share[:, p]],
        })
    caption_mean = share[:, prompt_len:].mean(1)   # per-layer mean over all caption tokens
    return rows, [round(float(x), 5) for x in caption_mean], prompt_len


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="chair_test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--caption-tag", default="baseline", help="which saved captions (results/chair/<tag>)")
    ap.add_argument("--tag", default="baseline", help="output folder results/attention/<tag>")
    ap.add_argument("--dtype", default=config.DTYPE)
    args = ap.parse_args()

    cap_dir = config.RESULTS_DIR / "chair" / args.caption_tag
    captions = {r["image"]: r for r in read_jsonl(cap_dir / f"{args.split}.jsonl")}
    scored_path = cap_dir / f"{args.split}_scored.jsonl"          # full split (normal case)
    if not scored_path.exists() and args.limit:
        scored_path = cap_dir / f"{args.split}_n{args.limit}_scored.jsonl"
    scored = {r["image"]: r for r in read_jsonl(scored_path)}
    images = [im for im in subsample(load_split(args.split)["images"], args.limit) if im in scored]
    if not images:
        raise SystemExit(f"No scored captions in {cap_dir}: run caption_chair.py first.")

    out_dir = config.RESULTS_DIR / "attention" / args.tag
    out_path = out_dir / f"{args.split}.jsonl"
    done = {r["image"] for r in read_jsonl(out_path)}
    todo = [im for im in images if im not in done]
    print(f"[{args.split}] {len(images)} captions, {len(todo)} to process")

    vlm = VLM(dtype=args.dtype)
    save_json(run_metadata(args), out_dir / f"run_meta_{args.split}.json")
    with JsonlWriter(out_path) as w:
        for im in tqdm(todo, desc="attention"):
            cap, sc = captions[im], scored[im]
            if sc["caption"] != cap["caption"]:
                raise SystemExit(f"{im}: scored caption does not match saved caption; rerun caption_chair.py")
            rows, cap_mean, prompt_len = caption_attention(vlm, load_image(im), cap["token_ids"], sc["mentions"])
            if len(rows) != len(sc["mentions"]):
                print(f"[warn] {im}: mapped {len(rows)}/{len(sc['mentions'])} mentions to tokens")
            w.write({"image": im, "prompt_len": prompt_len, "n_new_tokens": len(cap["token_ids"]),
                     "caption_mean_share": cap_mean, "mentions": rows})
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
