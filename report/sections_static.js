// Report sections that do not depend on experimental numbers.
const { P, H1, H2, Bullet, Num } = require("./lib");

function introduction() {
  return [
    H1("1. Introduction"),
    P("Vision language models (VLMs) answer questions about images and describe them in fluent text, but they " +
      "regularly mention objects that are not in the image. This *object hallucination* makes them unreliable in " +
      "settings where a wrong statement is worse than no statement. A common explanation is that the language " +
      "model inside a VLM relies on its text prior (what usually co-occurs with what it has already written) " +
      "more than on the image [3, 5]. If that is right, hallucinated words should coincide with the model " +
      "paying *less attention to the image*, and pushing attention back towards the image should help."),
    P("We test this explanation on a small open model, Qwen2-VL-2B-Instruct [6], with one focused question: " +
      "**when Qwen2-VL-2B mentions an object that is not in the image, is it paying less attention to the image " +
      "at that moment, and can increasing its attention to image tokens at inference time reduce hallucination " +
      "without hurting answer quality?** We state three hypotheses:"),
    Bullet("**H1 (analysis).** Tokens of hallucinated objects receive a lower share of attention on image tokens " +
      "than tokens of real objects, and the gap is clearest in particular layers."),
    Bullet("**H2 (intervention).** Multiplying attention on image tokens in selected layers lowers hallucination " +
      "on POPE [1] and CHAIR [2] compared with the unmodified model."),
    Bullet("**H3 (trade-off).** A strong intervention hurts recall, caption length or fluency, so there is a " +
      "sweet spot, and the adversarial POPE split benefits least."),
    P("Our contributions are: (i) a controlled evaluation of Qwen2-VL-2B on POPE and CHAIR with a strict " +
      "dev/test separation; (ii) a layer-by-layer analysis of image attention at the moment each object word is " +
      "written, which controls for the position of the word and compares against the model's own token " +
      "confidence; (iii) a training-free, post-softmax image-attention boost whose strength and layers are " +
      "chosen by a rule fixed before looking at test data, compared with PAI's attention step tuned under the same rule; (iv) attention-based abstention compared with a " +
      "confidence baseline; and (v) an analysis of the cases where the method fails. Most prior attention-based " +
      "work uses 7B-scale LLaVA-style models [3, 4, 5]; a 2B model with dynamic-resolution image tokens is a " +
      "different and cheaper regime. Well-supported negative results are reported as such.", { after: 60 }),
  ];
}

function review() {
  return [
    H1("2. Review of existing techniques"),
    H2("2.1 Measuring object hallucination"),
    P("**CHAIR** [2] scores free-form captions on MS-COCO [7]. Words are mapped to the 80 COCO categories with a " +
      "synonym list, and a mentioned category is hallucinated if it is neither annotated in the image nor named " +
      "in its reference captions. CHAIR-s is the share of captions with at least one hallucinated object and " +
      "CHAIR-i the share of object mentions that are hallucinated. Because a model can lower CHAIR by saying " +
      "less, it must be read together with object recall and caption length. **POPE** [1] instead asks balanced " +
      "yes/no questions (\"Is there a <object> in the image?\") about 500 COCO images. Absent objects are sampled " +
      "at random, among the most frequent COCO objects (popular), or among objects that frequently co-occur with " +
      "the present ones (adversarial). POPE is stable and cheap but only probes recognition, not generation; " +
      "we therefore use both."),
    H2("2.2 Training-free mitigation"),
    P("**VCD** [3] contrasts the output distribution given the real image with the distribution given a " +
      "noise-distorted image, down-weighting tokens that the model would produce anyway from its language " +
      "prior. **OPERA** [4] observes that hallucinations often follow \"summary\" tokens on which attention " +
      "columns pile up, and adds a penalty for this over-trust pattern to beam search, with a retrospective " +
      "roll-back. **PAI** [5] (\"Paying More Attention to Image\") is closest to our work. It names the problem " +
      "*text inertia*, amplifies the pre-softmax attention logits of image tokens for the current token by " +
      "alpha times their magnitude in layers chosen with an attention-sink criterion, and additionally subtracts " +
      "the logits of an image-free forward pass; it was evaluated on LLaVA-1.5, MiniGPT-4 and Shikra."),
    H2("2.3 How our work differs"),
    P("Our intervention is a *post-softmax* rescaling: in a contiguous band of decoder layers, every attention " +
      "row whose query comes after the image has its weights on image tokens multiplied by (1 + alpha) and is " +
      "renormalised. It uses no image-free pass and no contrastive logits, so any effect can be attributed to " +
      "attention alone, and alpha = 0 reproduces the original model exactly. We apply it to Qwen2-VL-2B, whose " +
      "number of image tokens depends on the image (here 64 to 256) and whose layers are chosen on a dev set from " +
      "our own attention analysis. To compare with PAI, we also re-implement its attention step without the " +
      "image-free pass and tune it with the same dev protocol (Section 3.4). Finally, rather than only intervening, we ask whether the same attention " +
      "signal can tell the model *when to abstain*, and compare it with the simplest alternative, the model's " +
      "own answer confidence.", { after: 60 }),
  ];
}

function methods(N) {
  const sel = N.selected || {};
  return [
    H1("3. Methods"),
    H2("3.1 Model and inference"),
    P("We use Qwen2-VL-2B-Instruct [6] through Hugging Face Transformers 5.22.2 with eager attention (needed to " +
      "read attention weights). Qwen2-VL encodes an image with a ViT into a variable number of tokens (one token " +
      "per 28 x 28 pixel area); we fix the image budget to 64-256 tokens (at most 200,704 pixels) so every method " +
      "sees identical inputs. Decoding is greedy, so outputs are deterministic. Our sanity checks (E0) showed " +
      "that **float16 is unusable** for this model in this library version: the residual stream overflows and " +
      "every attention map and logit from decoder layer 1 onward is NaN, so every answer was \"!!!!!\". We " +
      "therefore run in bfloat16, which passed every check (answers correct on 10/10 hand-read questions, " +
      "attention rows summing to 1 within 4e-3, deterministic outputs). All experiments ran on one NVIDIA " +
      "RTX 4060 Laptop GPU (8 GB). To make the runs affordable we generate in batches (12 questions or 8 " +
      "captions, left-padded). Every method is run with identical batches, so comparisons stay controlled."),
    H2("3.2 Benchmarks, metrics and data protocol"),
    P("**POPE.** The three COCO splits (3,000 questions each, 50% \"yes\") with the suffix \" Answer with yes or " +
      "no.\". The first word of the answer is used if it is yes/no; otherwise the original POPE rule applies " +
      "(\"no\" if the answer contains \"no\" or \"not\"). We report accuracy, precision, recall, F1 and yes-ratio. " +
      "**CHAIR.** One caption per image with the prompt \"Describe this image in detail.\" (at most 256 new " +
      "tokens) for 500 COCO val2014 images. We re-implemented the original scoring with the authors' synonym " +
      "list and two-word rules; ground truth is the union of instance annotations and objects named in the five " +
      "reference captions. We report CHAIR-s, CHAIR-i, object recall (distinct ground-truth categories " +
      "mentioned / distinct ground-truth categories) and length."),
    P("**Dev/test separation.** We drew, with a fixed seed, 200 further COCO val2014 images that appear in " +
      "neither test set. They provide 902 POPE-style questions built with POPE's random-negative protocol, and " +
      "captions for CHAIR. *Every* choice (the layer band, alpha, the abstention band and its threshold) is made " +
      "on dev. Each test set was then run once with the frozen settings."),
    H2("3.3 Image-attention share at an object mention (E3)"),
    P("For layer l, let a^(l)^_h(q, k) be the attention of head h from query position q to key k. The image-attention " +
      "share is s_l(q) = mean_h sum_{k in image} a^(l)^_h(q, k), the fraction of attention that q puts on image " +
      "tokens. For every object mention found by CHAIR in a caption, we re-run the model with teacher forcing on " +
      "its own caption and record s_l at the position that *predicts* the object's first token (the moment the " +
      "model decides to write the object) for all 28 layers, together with the log-probability of that token and " +
      "its position in the caption. Image-token positions are found from the processor output and asserted " +
      "(count matches the image grid, contiguous, wrapped by the vision start/end tokens). We measure how well low " +
      "attention identifies hallucinated mentions with the AUROC, with 95% confidence intervals from a bootstrap " +
      "that resamples whole images. Because later tokens attend less to the image and hallucinations tend to come " +
      "late in long captions, position is a confounder. We control for it in two ways: AUROC within " +
      "caption-position quintiles, and cross-validated logistic regressions (folds split by image) that add " +
      "attention to position and token log-probability."),
    H2("3.4 Image-attention boost (E4)"),
    P("In each chosen layer, for every query q after the last image token and every key k, " +
      "a'(q, k) = a(q, k)(1 + alpha 1[k in image]) / sum_k' a(q, k')(1 + alpha 1[k' in image]). The image tokens' " +
      "own rows are untouched. This is equivalent to adding log(1 + alpha) to the image keys' pre-softmax logits, " +
      "a constant shift, whereas PAI's alpha|logit| depends on each logit. We implement it by wrapping " +
      "Transformers' eager attention function. Unit tests on a tiny random Qwen2-VL check that alpha = 0 gives " +
      "bit-identical logits and generations, that rows still sum to 1, that the boosted share follows " +
      "s' = s(1 + alpha)/(1 + alpha s), and that rows in a left-padded batch are boosted exactly as when run alone. " +
      "We sweep alpha in {0.5, 1, 2, 4} and layer bands {0-7, 8-15, 16-27, 0-27} plus the band chosen by E3 on dev. " +
      "The **selection rule was fixed before the sweep**: among settings whose dev recall drops by at most 2 " +
      "points and dev POPE F1 by at most 1 point, take the lowest dev CHAIR-i; if none qualifies, take the lowest " +
      "dev CHAIR-i overall and report that the criteria were not met."),
    P("**PAI-style comparison (E4b).** To compare with a published method, we also implement PAI's attention " +
      "step [5]: before the softmax, each image key's logit z becomes z + alpha|z|, only in the row of the token " +
      "being generated. PAI also contrasts the output with an image-free forward pass; we omit that step, so the " +
      "comparison is between the two ways of moving attention onto the image. Its grid is alpha in {0.05, 0.1, " +
      "0.2, 0.5} on all layers except the first two (2-27) or on layers 6-13. Larger alpha is not useful on this " +
      "model: at alpha = 0.5 on layers 2-27, dev captions shrink to 24 words, and in a smoke test they turned into " +
      "lists of bounding-box coordinates. The setting is chosen on dev with the same rule as above, and each test set " +
      "is then run once. Two unit tests check that only the last query row changes and that batched rows match " +
      "rows run alone."),
    H2("3.5 Comparison baseline and abstention (E5)"),
    P("As a simple comparison we use a cautious POPE prompt (\" Answer yes only if the object is clearly visible in " +
      "the image, otherwise answer no.\"), fixed before any test run. For abstention, an extra forward pass records " +
      "s_l at the last prompt token (where the yes/no decision is made) and the probabilities of \"yes\" and \"no\". " +
      "We compare three scores for deciding which questions to answer: mean image attention over the 8-layer band " +
      "that best separates right from wrong answers *on dev*, answer confidence |P(yes) - P(no)|, and a random order. " +
      "We plot accuracy against coverage, report the area under that curve, and apply a threshold set on dev " +
      "(keeping 80% of dev questions) unchanged to each test split. The same idea is applied to captions by " +
      "dropping object mentions whose image attention falls below dev quantiles.", { after: 60 }),
  ];
}

function aiStatement() {
  return [
    H1("AI Use Statement"),
    P("We used Claude (Anthropic), through Claude Code, to help plan the project, write and debug the evaluation, " +
      "CHAIR, attention-analysis, intervention, sweep, abstention and plotting code, run the experiment scripts " +
      "on a team member's GPU, and draft and edit this report. We checked the code with unit tests (attention rows " +
      "summing to one, alpha = 0 reproducing the original outputs bit for bit, batched rows matching single rows, " +
      "CHAIR parsing and metrics on hand-made examples, AUROC against scikit-learn) and with the E0 sanity checks, " +
      "which also caught the float16 overflow. We read captions with their hallucinated words marked and the " +
      "failure cases by hand; Claude proposed the labels for the 30-mention CHAIR hand check (Section 5.2) from " +
      "the caption text, and we checked each one. We checked every reference against the original paper. Every number in this report " +
      "is produced by our scripts from saved JSON files and listed in RUN_LOG.md. Our team chose the research " +
      "question, decided the experimental protocol and selection rules, reviewed the results and wrote the " +
      "interpretation. [Team: edit this statement so that every sentence is true for what you did.]"),
  ];
}

function references() {
  const refs = [
    "Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, J.-R. Wen. Evaluating Object Hallucination in Large Vision-Language Models. *EMNLP*, 2023, pp. 292-305.",
    "A. Rohrbach, L. A. Hendricks, K. Burns, T. Darrell, K. Saenko. Object Hallucination in Image Captioning. *EMNLP*, 2018, pp. 4035-4045.",
    "S. Leng, H. Zhang, G. Chen, X. Li, S. Lu, C. Miao, L. Bing. Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding. *CVPR*, 2024, pp. 13872-13882.",
    "Q. Huang, X. Dong, P. Zhang, B. Wang, C. He, J. Wang, D. Lin, W. Zhang, N. Yu. OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation. *CVPR*, 2024.",
    "S. Liu, K. Zheng, W. Chen. Paying More Attention to Image: A Training-Free Method for Alleviating Hallucination in LVLMs. *ECCV*, 2024.",
    "P. Wang, S. Bai, et al. (Qwen Team). Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution. arXiv:2409.12191, 2024.",
    "T.-Y. Lin, M. Maire, S. Belongie, et al. Microsoft COCO: Common Objects in Context. *ECCV*, 2014.",
  ];
  return [H1("References"), ...refs.map((r) => Num(r))];
}

module.exports = { introduction, review, methods, aiStatement, references };
