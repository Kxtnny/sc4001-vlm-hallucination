// Results, discussion, conclusion and appendix. Every number is read from numbers.json (N), which
// scripts/make_report.py builds from the saved per-example outputs; nothing is typed in by hand.
const fs = require("fs");
const path = require("path");
const { P, H1, H2, Bullet, Figure, FigureRow, Tbl, Caption } = require("./lib");
const T = require("./tables");

const pc = (x, d = 1) => (100 * x).toFixed(d);                 // 0.123 -> "12.3"
const pp = (x, d = 1) => `${x >= 0 ? "+" : "-"}${Math.abs(100 * x).toFixed(d)}`;   // signed points
const f3 = (x) => x.toFixed(3);
const ci = (d) => `${pp(d.diff)} points (95% CI ${pp(d.ci[0])} to ${pp(d.ci[1])})`;
const fig = (FIG, name) => path.join(FIG, name);
// Signed difference of two proportions as shown in the tables (each rounded to 0.1 points first).
const dpt = (a, b) => { const v = (Math.round(1000 * a) - Math.round(1000 * b)) / 10; return `${v >= 0 ? "+" : "-"}${Math.abs(v).toFixed(1)}`; };

function pick(N) {
  const pope = N.pope, ch = N.chair, A = N.attention, ab = N.abstention, sw = N.sweep, sel = N.selected;
  const L = (split, l) => A[split].layers.share_pred[l];
  const testLayers = A.chair_test.layers.share_pred;
  const best = testLayers.reduce((a, b) => (b.auroc > a.auroc ? b : a));
  const devBest = A.dev.layers.share_pred.reduce((a, b) => (b.auroc > a.auroc ? b : a));
  const maxWithin = Math.max(...testLayers.slice(1).map((p) => p.auroc_within_position_quintiles));
  const positiveGap = testLayers.filter((p) => p.gap > 0).length;
  const maxGap = testLayers.reduce((a, b) => (b.gap > a.gap ? b : a));
  const base = sw.rows.find((r) => r.alpha === 0);
  const strongest = sw.rows.filter((r) => r.alpha > 0).reduce((a, b) => (b.avg_len_words < a.avg_len_words ? b : a));
  return { pope, ch, A, ab, sw, sel, L, best, devBest, maxWithin, positiveGap, maxGap, base, strongest, testLayers };
}

function results(N, FIG) {
  const { pope, ch, A, ab, sw, sel, L, best, devBest, maxWithin, positiveGap, maxGap, base, strongest } = pick(N);
  const r = pope.random, a = pope.adversarial, po = pope.popular;
  const exR = ab.pope.random.exploratory_high_attention, exA = ab.pope.adversarial.exploratory_high_attention;
  const d = ch.intervention_minus_baseline, lm = ch.length_matched.intervention_minus_baseline;
  const cf = ab.captions_filter;
  const sanityKeys = Object.keys(N.sanity);
  const bf = N.sanity["sanity_report_rtx4060-bf16"], fp = N.sanity["sanity_report_rtx4060-fp16"];
  const nPass = (o) => Object.values(o).filter(Boolean).length;
  const det = (s) => A[s].detectors_cv_auroc;
  return [
    H1("4. Experiments and results"),
    P(`All test numbers below come from one run of each frozen method; dev numbers were used only to make the ` +
      `choices described in Section 3. Appendix A lists the per-experiment commands and output files.`),
    H2("4.1 E0: sanity checks"),
    P(`In bfloat16, ${nPass(bf)} of ${Object.keys(bf).length} checks pass: answers parse as yes/no, greedy decoding is ` +
      `deterministic, the image-token span matches the image grid, attention rows sum to one and no tensor contains ` +
      `NaN. In float16, ${Object.keys(fp).length - nPass(fp)} checks fail: the image-attention share is defined in ` +
      `layer 0 but NaN in every later layer, and every answer is "!!!!!" (Appendix B). An attention study run in ` +
      `float16 would have silently produced empty results, so we report all results in bfloat16.`),
    H2("4.2 E1-E2: how much does the unmodified model hallucinate?"),
    ...T.popeTable(N, `**Table 1.** POPE test results (3,000 questions per split, %). Baseline = unmodified model; ` +
      `Cautious prompt = E5 comparison; Attention boost = alpha = ${sel.alpha}, layers ${sel.layers}, selected on dev. ` +
      `McNemar p: exact test against the baseline on the same questions.`),
    P(`Qwen2-VL-2B is accurate on POPE (${pc(r.baseline.accuracy)}%, ${pc(po.baseline.accuracy)}% and ` +
      `${pc(a.baseline.accuracy)}% on the random, popular and adversarial splits; Table 1). As the absent objects get ` +
      `harder, precision drops from ${pc(r.baseline.precision)}% to ${pc(a.baseline.precision)}%, while recall is ` +
      `${pc(r.baseline.recall)}% on every split because the present-object questions are shared across splits. ` +
      `Unlike the LLaVA-style models that POPE was designed around, which answer "yes" too often [1], this model is ` +
      `*conservative*: it says "yes" to only ${pc(r.baseline.yes_ratio)}-${pc(a.baseline.yes_ratio)}% of questions ` +
      `although half are positive. Of its ${exR.n_wrong} errors on the random split, ${exR.n_false_negative} are ` +
      `misses ("no" to an object that is there) and only ${exR.n_wrong - exR.n_false_negative} are hallucinated "yes" ` +
      `answers (adversarial: ${exA.n_wrong - exA.n_false_negative} of ${exA.n_wrong}). In free captions the picture ` +
      `is different. ${pc(ch.baseline.chair_s)}% of the 500 test captions contain at least one hallucinated object ` +
      `and ${pc(ch.baseline.chair_i)}% of object mentions are hallucinated (CHAIR-i), at ${pc(ch.baseline.recall)}% ` +
      `object recall (Table 2). Captions are long (${ch.baseline.avg_len_words.toFixed(0)} words on average) and ` +
      `${pc(ch.baseline.frac_hit_max_tokens)}% reach the 256-token limit.`),
    ...T.chairTable(N, `**Table 2.** CHAIR on the 500 test images (%). Difference = boost minus baseline with a ` +
      `paired image-level bootstrap 95% CI. Length-matched rows: for each image both captions are cut to the shorter ` +
      `caption's word count before scoring, which removes the effect of caption length.`),
    H2("4.3 E3: do hallucinated objects receive less image attention?"),
    ...Figure(fig(FIG, "e3_attention_chair_test.png"), 620,
      `**Figure 1.** E3 on the test captions (${A.chair_test.n_mentions} object mentions, ` +
      `${A.chair_test.n_hallucinated} hallucinated). (a) Mean image-attention share at the position predicting the ` +
      `object's first token, real minus hallucinated (95% CI). (b) AUROC of low image attention as a detector of ` +
      `hallucinated mentions, with an image-bootstrap 95% CI and the AUROC within caption-position quintiles. The dev ` +
      `figure (Appendix C) looks the same.`),
    P(`Hallucinated mentions do receive slightly less image attention: the gap is positive in ${positiveGap} of 28 ` +
      `layers on the test captions, and largest in layer ${maxGap.layer} (${pc(maxGap.mean_real)}% vs ` +
      `${pc(maxGap.mean_halluc)}% of attention on the image; Figure 1a). As a detector it is weak. The ` +
      `best single layer is layer ${best.layer} (AUROC ${f3(best.auroc)}, 95% CI ${f3(best.auroc_ci[0])}-` +
      `${f3(best.auroc_ci[1])}), the same layer as on dev (${f3(devBest.auroc)}), and many layers after 13 are close ` +
      `to chance. Much of the signal is position: hallucinations come late in long captions, where every token ` +
      `attends less to the image. Within position quintiles the AUROC falls to ` +
      `${f3(best.auroc_within_position_quintiles)} at layer ${best.layer} and at most ${f3(maxWithin)} elsewhere. ` +
      `In cross-validated detectors (Table 3), the model's own token log-probability is the strongest single ` +
      `feature, and adding image attention (band ${sel.layers}, chosen on dev) to position and log-probability ` +
      `does not help (test AUROC ${f3(det("chair_test")["position+logprob"])} without, ` +
      `${f3(det("chair_test")["position+logprob+attention"])} with).`),
    ...T.detectorTable(N, `**Table 3.** AUROC for detecting hallucinated object mentions (logistic regression, ` +
      `5-fold cross-validation with folds split by image). Dev: ${A.dev.n_mentions} mentions ` +
      `(${A.dev.n_hallucinated} hallucinated); test: ${A.chair_test.n_mentions} (${A.chair_test.n_hallucinated}).`),
    H2("4.4 E4: does boosting image attention reduce hallucination?"),
    ...Figure(fig(FIG, "e4_sweep_dev.png"), 620,
      `**Figure 2.** E4 sweep on dev (${sw.n_caption_images} captions and the 902 dev POPE-style questions). Each line is a layer band; the dotted line is the unmodified model. The full table is in Appendix D.`),
    P(`On dev, only boosting all 28 layers lowers CHAIR-i fairly steadily as alpha grows, to ` +
      `${pc(strongest.chair_i)}% at alpha = ${strongest.alpha} (baseline ${pc(base.chair_i)}%; Figure 2). The other ` +
      `bands cross the baseline several times, which shows how noisy CHAIR-i is on ${sw.n_caption_images} captions: ` +
      `one point of CHAIR-i is about ${Math.round(A.dev.n_mentions / A.dev.n_captions * sw.n_caption_images / 100)} ` +
      `object mentions. What changes consistently is length. The all-layer boost at alpha = ${strongest.alpha} cuts ` +
      `captions from ${base.avg_len_words.toFixed(0)} to ${strongest.avg_len_words.toFixed(0)} words and recall by ` +
      `${pc(base.recall - strongest.recall)} points, which breaks the recall constraint. POPE F1 stays within one ` +
      `point of the baseline everywhere. The fixed rule selected alpha = ${sel.alpha} on layers ${sel.layers}, the ` +
      `band E3 chose (dev CHAIR-i ${pc(sel.dev_baseline.chair_i)}% to ${pc(sel.dev.chair_i)}%, recall ` +
      `${pc(sel.dev_baseline.recall)}% to ${pc(sel.dev.recall)}%).`),
    P(`On the test set the boost changes POPE by at most ${Math.max(...["random", "popular", "adversarial"].map((s) =>
      Math.abs(pope[s].intervention.accuracy - pope[s].baseline.accuracy) * 100)).toFixed(1)} accuracy points, and no ` +
      `split differs significantly (McNemar p >= ${Math.min(...["random", "popular", "adversarial"].map((s) =>
      pope[s].intervention.mcnemar_p)).toFixed(2)}; Table 1). On CHAIR (Table 2), CHAIR-s falls by ` +
      `${ci(d.chair_s)}, but CHAIR-i changes by only ${ci(d.chair_i)} and recall by ${ci(d.recall)}. Captions become ` +
      `${(ch.baseline.avg_len_words - ch.intervention.avg_len_words).toFixed(0)} words shorter, and only ` +
      `${pc(ch.intervention.frac_hit_max_tokens)}% (instead of ${pc(ch.baseline.frac_hit_max_tokens)}%) reach the ` +
      `token limit. Shorter captions contain fewer objects, so fewer of them contain a hallucination even at the same ` +
      `hallucination rate per mention. The length-matched comparison confirms this. When both captions of an image ` +
      `are cut to the same length, the CHAIR-s difference is ${ci(lm.chair_s)} and the CHAIR-i difference is ` +
      `${ci(lm.chair_i)}, while recall is ${pp(lm.recall.diff)} points *higher* (95% CI ${pp(lm.recall.ci[0])} to ${pp(lm.recall.ci[1])}). The boost makes the model describe ` +
      `the same objects in fewer words; it does not make it hallucinate less often.`),
    H2("4.5 E5: a prompt baseline and attention-based abstention"),
    P(`The cautious prompt raises POPE accuracy by ${dpt(r.cautious.accuracy, r.baseline.accuracy)}, ` +
      `${dpt(po.cautious.accuracy, po.baseline.accuracy)} and ${dpt(a.cautious.accuracy, a.baseline.accuracy)} points ` +
      `(significant only on random, p = ${r.cautious.mcnemar_p.toFixed(3)}). It works the opposite way to its wording: ` +
      `it *raises* the yes-ratio and recall (${pc(r.baseline.recall)}% to ${pc(r.cautious.recall)}%) and slightly ` +
      `lowers precision. For a conservative model, a prompt that mentions the object being "clearly visible" seems ` +
      `to help it accept objects it would otherwise miss.`),
    ...T.abstentionTable(N, `**Table 4.** E5 abstention on POPE test. Questions are answered in order of the score; ` +
      `AUROC (wrong) measures how well the score ranks wrong answers last. The thresholds (80% coverage on dev) are ` +
      `fixed on dev. The exploratory row uses a direction found by inspecting dev and was not part of the planned ` +
      `comparison.`),
    ...FigureRow([fig(FIG, "e5_accuracy_coverage_random.png"), fig(FIG, "e5_accuracy_coverage_adversarial.png")], 300,
      `**Figure 3.** Accuracy of the answered questions against coverage on POPE random (left) and adversarial ` +
      `(right).`),
    P(`For abstention, answer confidence is far better than image attention (Table 4, Figure 3). Its AUROC for wrong ` +
      `answers is ${f3(ab.pope.adversarial["answer confidence"].auroc_wrong)}-${f3(ab.pope.random["answer confidence"].auroc_wrong)}, ` +
      `against ${f3(ab.pope.adversarial["image attention"].auroc_wrong)}-${f3(ab.pope.random["image attention"].auroc_wrong)} ` +
      `for low attention in the band chosen on dev (layers ${ab.band}). With the dev threshold, confidence answers ` +
      `${pc(ab.pope.random["answer confidence"].coverage_at_dev_threshold, 0)}% of random-split questions at ` +
      `${pc(ab.pope.random["answer confidence"].accuracy_at_dev_threshold)}% accuracy. Attention answers ` +
      `${pc(ab.pope.random["image attention"].coverage_at_dev_threshold, 0)}% at ` +
      `${pc(ab.pope.random["image attention"].accuracy_at_dev_threshold)}%, barely above the ` +
      `${pc(r.baseline.accuracy)}% obtained by answering everything. Looking at dev, the strongest attention signal ` +
      `ran the *other* way: in layer ${exR.layer}, wrong answers had *more* image attention. This held on test ` +
      `(AUROC ${f3(exR.auroc_wrong)} random, ${f3(exA.auroc_wrong)} adversarial; ${f3(exR.auroc_wrong_within_pred_no)} ` +
      `among "no" answers only). The model's typical error is a miss, made while it is attending to the image ` +
      `strongly but fails to find the object. Even this post-hoc signal is weaker than confidence. For captions, ` +
      `dropping the 20% of object mentions with the lowest image attention (dev quantile) lowers CHAIR-i only from ` +
      `${pc(cf["keep_1.0"].chair_i)}% to ${pc(cf["keep_0.8"].chair_i)}%, at a recall cost of ` +
      `${pc(cf["keep_1.0"].recall - cf["keep_0.8"].recall)} points.`, { after: 60 }),
  ];
}

function discussion(N, FIG) {
  const { pope, ch, A, sel, best } = pick(N);
  const fc = N.failure_cases.summary_by_cause;
  const hc = N.hand_check;
  const intro = fc["CHAIR introduced"] || {};
  const nIntro = Object.values(intro).reduce((x, y) => x + y, 0);
  const still = fc["POPE-random still"] || {};
  const nStill = Object.values(still).reduce((x, y) => x + y, 0);
  const art = hc ? hc.counts["scoring artefact"] : null;
  const nHc = hc ? hc.rows.length : null;
  return [
    H1("5. Discussion"),
    H2("5.1 What the results say about the hypotheses"),
    Bullet(`**H1 is only weakly supported.** Hallucinated objects receive slightly less image attention in most ` +
      `layers, but the effect is small (best AUROC ${f3(best.auroc)}), mostly explained by where the object appears ` +
      `in the caption, and adds nothing to the model's own token confidence. Image attention is a symptom of ` +
      `position more than a cause of hallucination.`),
    Bullet(`**H2 is not supported.** The selected boost leaves POPE unchanged and CHAIR-i within noise. Its ` +
      `significant CHAIR-s reduction disappears once caption length is controlled.`),
    Bullet(`**H3 is partly supported.** On dev the only clear CHAIR-i reduction (all layers, alpha = 4) came with ` +
      `much shorter captions and lower recall (Figure 2), and the setting allowed by the constraints did not lower ` +
      `CHAIR-i on test. Because the boost changed no POPE split, we cannot say whether the adversarial split ` +
      `benefits least; the cautious prompt did help the adversarial split least.`),
    H2("5.2 Failure cases (E6)"),
    ...Figure(path.join(FIG, "e6_failure_cases.png"), 600,
      `**Figure 4.** E6 cases from the test set: POPE answers the boost made wrong, POPE errors that persist, and ` +
      `objects the boost introduced into captions (heuristic cause in brackets).`),
    P(`We grouped errors with simple heuristics and then read the samples (Figure 4). Of the ${nStill} POPE-random questions both methods get wrong, ${still["small object"] || 0} ` +
      `ask about an object covering less than 1% of the image. In the samples we read, POPE errors are misses of ` +
      `small, dark or partly hidden objects (a chair in a dark room, a bowl on a crowded table). Some are annotation errors: one ` +
      `"orange" label refers to an orange cat. Among the ${nIntro} objects the boost introduced into captions, ` +
      `${intro["co-occurrence"] || 0} frequently co-occur with objects that are present, most often a person or a ` +
      `table (a table under a camera tripod, a microwave and toaster in a kitchen). This is the text-prior ` +
      `pattern that motivated the study. Many others are not hallucinations at all, but CHAIR scores them as ` +
      `hallucinated: "alert drivers to slow down" (person) on a road marking, "a younger or smaller adult" ` +
      `(person) for an elephant, "a children's book" (person), and a person in a bear costume (bear).` +
      (hc ? ` To size this problem we read the caption context of ${nHc} randomly sampled CHAIR hallucinations from ` +
        `the baseline: **${art} of ${nHc}** are scoring artefacts. These include word senses ("truck bed", "toilet ` +
        `bowl", "a remote landscape", "train cars"), loose synonyms ("container" as bowl, "streetlights" as traffic ` +
        `light), colours ("the vase is orange"), negations ("no visible people") and objects in pictures. The ` +
        `remaining ${nHc - art} look like genuine hallucinations or missing annotations, often hedged ("possibly a ` +
        `phone").` : "")),
    H2("5.3 What the negative results mean"),
    P(`Three things explain why boosting attention did not help this model, unlike the 7B LLaVA-style models in PAI ` +
      `[5]. First, Qwen2-VL-2B's failure mode on POPE is the opposite of the one these methods target. It misses ` +
      `objects far more often than it invents them, and its misses come with *high*, not low, image attention. ` +
      `Pushing more attention onto the image cannot fix a perception failure. Second, in captions the boost ` +
      `changes *style* (shorter, more list-like descriptions that reach the objects sooner) rather than grounding. ` +
      `The per-mention hallucination rate is unchanged, and CHAIR-s rewards shorter captions. This is why we ` +
      `report recall, length and a length-matched comparison next to CHAIR-s. Third, the measurement is noisy. ` +
      `CHAIR-i on 100 dev captions moves by more than a point between neighbouring settings, and if the hand ` +
      `check is representative, more than half of the hallucination labels are scoring artefacts. That ` +
      `noise weakens any detector, including E3's, and can hide small real effects.`),
    H2("5.4 Limitations"),
    P(`We study one 2B model on COCO images only, with greedy decoding and one caption per image. The sweep used ` +
      `${N.sweep.n_caption_images} dev captions, so the selected setting is uncertain (Section 4.4). A third of the ` +
      `baseline captions reached the 256-token limit. CHAIR depends on a fixed synonym list and misses or ` +
      `misreads objects (Section 5.2). Our hand check covered ${nHc || 30} mentions and used the caption text, not the ` +
      `image. Attention weights are not a full explanation of model behaviour. Our boost is one specific, constant ` +
      `rescaling; we did not re-implement VCD, OPERA or PAI, so we cannot rank against them. The decision-time ` +
      `attention and the exploratory layer-${N.abstention.exploratory_high_attention_layer} result describe ` +
      `correlations, not causes.`, { after: 60 }),
  ];
}

function conclusion(N) {
  const { ch, best } = pick(N);
  const lm = ch.length_matched.intervention_minus_baseline;
  return [
    H1("6. Conclusion"),
    P(`We asked whether Qwen2-VL-2B pays less attention to the image when it hallucinates an object, and whether ` +
      `paying more attention helps. With a strict dev/test protocol the answer is: only slightly, and no. ` +
      `Hallucinated mentions receive a little less image attention (AUROC up to ${f3(best.auroc)}), mostly because ` +
      `they come late in captions. A training-free image-attention boost leaves POPE and per-mention hallucination ` +
      `unchanged; its apparent CHAIR-s gain is a caption-length effect (length-matched CHAIR-i ${pp(lm.chair_i.diff)} ` +
      `points). The model's own answer confidence is a much better abstention signal than image attention. For this ` +
      `small, conservative model, most POPE errors are misses rather than inventions, so methods built to stop ` +
      `language-prior hallucinations in larger models do not transfer automatically. Evaluations should report ` +
      `recall, caption length and scoring artefacts next to CHAIR. Promising next steps are a better hallucination ` +
      `scorer (for example an open-vocabulary detector or human labels) and checking whether the reversed attention ` +
      `signal for misses holds in other models.`, { after: 60 }),
  ];
}

function appendix(N, FIG) {
  const S = N.sanity;
  const tags = ["sanity_report_rtx4060-bf16", "sanity_report_rtx4060-fp16"].filter((t) => S[t]);
  const checks = Object.keys(S[tags[0]]);
  const sanityRows = [["Check", "bfloat16", "float16"], ...checks.map((c) => [c, ...tags.map((t) => (S[t][c] ? "pass" : "FAIL"))])];
  const cmds = [
    "pip install -r requirements.txt   (Python 3.10+, PyTorch with CUDA; versions in README.md)",
    "python scripts/prepare_data.py    (COCO val2014 images and annotations, POPE files, CHAIR synonyms, dev split)",
    "bash scripts/run_all.sh           (E0 to E6 in order; resumable; about 8 hours on an RTX 4060 Laptop GPU)",
    "python scripts/make_report.py     (every table and numbers.json from the saved outputs, no GPU)",
    "node report/build_report.js; powershell -File report/to_pdf.ps1   (this report)",
  ];
  return [
    H1("Appendix"),
    H2("A. Reproducing the results"),
    P("The README gives full instructions. The main commands, run from the repository root with SC4001_ROOT pointing " +
      "at a folder for data and results, are:"),
    ...cmds.map((c) => Bullet("`" + c.split("   ")[0] + "`" + (c.includes("   ") ? " " + c.split("   ").slice(1).join(" ").trim() : ""))),
    P("Outputs: results/pope/<method>/<split>.jsonl (per question), results/chair/<method>/chair_test_scored.jsonl " +
      "(per caption, with every object mention), results/attention/ and results/analysis/ (E3, E5, E6), " +
      "results/sweep/ (E4), results/report/numbers.json (every number in this report) and RUN_LOG.md (what was run, " +
      "when and on which hardware)."),
    H2("B. E0 sanity checks"),
    Tbl(sanityRows, [5200, 1600, 1600]),
    Caption("**Table B1.** E0 checks on the RTX 4060 Laptop GPU (results/sanity/). In float16 every attention map " +
      "after layer 0 is NaN, so the float16 checks that still pass say little."),
    H2("C. E3 on the dev captions"),
    ...Figure(path.join(FIG, "e3_attention_dev.png"), 600, `**Figure C1.** As Figure 1, on the dev captions ` +
      `(${N.attention.dev.n_mentions} mentions, ${N.attention.dev.n_hallucinated} hallucinated). The layer band ` +
      `${N.selected.layers} used by E4 was chosen from this figure's data.`),
    H2("D. E4 sweep table (dev)"),
    ...T.sweepTable(N, `**Table D1.** All sweep settings on dev (%). * = selected by the fixed rule.`),
    H2("E. Abstention on the other splits"),
    ...FigureRow([path.join(FIG, "e5_accuracy_coverage_dev_random.png"), path.join(FIG, "e5_accuracy_coverage_popular.png")], 300,
      "**Figure E1.** Accuracy against coverage on the dev questions (left; where the band and thresholds were chosen) " +
      "and POPE popular (right)."),
  ];
}

module.exports = { results, discussion, conclusion, appendix };
