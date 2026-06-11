# TODO

Action items distilled from the ICML 2026 Mechanistic Interpretability Workshop
decision and reviews for "Sparse Autoencoders Can Learn Graded Latents for
Relational Composition".

## P0: Camera-Ready Logistics

- [ ] Upload the virtual poster using the workshop virtual poster upload form.
- [x] Check that the OpenReview metadata matches the final paper: title, author
  list, keywords, TL;DR, abstract, and PDF.

## P1: High-Impact Manuscript Revisions

- [x] Strengthen the motivation in the introduction.
  - Explain why magnitude-sensitive SAE latents matter for real interpretability
    workflows, not only for this toy setup.
  - Add a concrete practical example where reducing SAE latents to on/off
    feature indicators would miss mechanism-relevant information.
  - Clarify why relational composition is a useful stress test for SAE analyses.
- [x] Calibrate the scope around the toy setting.
  - State clearly that the evidence comes from small attention-only transformers.
  - Explain why the controlled setup is still useful: it gives a known mechanism,
    exhaustive input coverage, and direct causal interventions.
  - Avoid implying that the result already establishes behavior in frontier or
    real-world pretrained models.
- [x] Reword the future-work claim.
  - Replace commitment-style phrasing such as "we will attempt..." with a
    neutral recommendation about what future work should test.
  - Suggest checking graded latents in pretrained SAEs and more naturalistic
    model settings without overclaiming that the phenomenon will transfer.
- [x] Sharpen the definition of "graded".
  - Discuss whether the latent is genuinely continuous or whether a small number
    of magnitude bins would explain the effect.
  - Use the current evidence around the Figure 4 right-panel pattern to motivate
    the answer.
  - If space permits, add a small binning or monotonicity analysis that compares
    continuous magnitude to coarse discretisations.
- [ ] Add reconstruction MSE to SAE quality reporting.
  - Report reconstruction MSE alongside loss recovered and dead-latent
    percentage.
  - Briefly explain that MSE and loss recovered measure related but non-identical
    aspects of SAE reconstruction quality.
  - Check whether existing comparison scripts already compute enough information
    to add this without rerunning all experiments.
- [x] Add a short discussion connecting graded latents to polysemanticity.
  - Explain how broad continuous activation patterns could be mistaken for
    polysemantic features.
  - Clarify how the paper distinguishes a graded relational variable from an
    uninterpretable mixture of meanings.

## P1: Figure And Narrative Flow

- [ ] Consider making the current Figure 2 the lead visual.
  - Reviewer feedback says it is the most intuitive illustration.
  - If reordered, update all figure references and make sure the narrative still
    flows from task setup to SAE evidence to causal steering.
- [ ] Make the first figure sequence answer the reader's main questions quickly:
  what the task is, what Delta alpha measures, what the special latent tracks,
  and how steering changes outputs.

## P2: Optional Additional Analysis

- [ ] Test whether a few magnitude bins are sufficient.
  - Compare the continuous latent value against 2-bin, 3-bin, and 5-bin
    discretisations for predicting Delta alpha or output swaps.
  - Use this only if it can be done quickly and reported compactly.
- [ ] Add feature-level quality checks if cheap.
  - For the special latent, report reconstruction or ablation evidence that the
    learned direction is faithful, not just correlated.
  - Keep this secondary to the MSE table unless space allows.
- [ ] Add one real-world-style motivating example.
  - This can be conceptual rather than experimental: for example, an SAE latent
    whose activation magnitude tracks relative position, confidence, distance,
    or another continuous relational variable.

## P2: Poster

- [ ] Build the poster around the strongest accepted-paper story:
  magnitude matters, the special latent is an outlier in correlation, and
  scaling it causally swaps output order.
- [ ] Use the most intuitive task/mechanism visual early in the poster.
- [ ] Include the main quantitative anchors: `r = 0.807`, `98%` of latents below
  `|r| < 0.004`, `49.8%` all-output swap rate, and `77.8%` active-case swap
  rate, after checking these still match the final paper.

## Notes From Reviews

- Reviewer u7rd found the paper clear and reproducible but wanted stronger
  motivation, stronger workshop interest, and a clearer bridge beyond the toy
  2-layer model.
- Reviewer b8Hj found the experimental setup convincing and the causal steering
  evidence useful, but wanted a sharper account of graded-vs-binned behavior,
  reconstruction MSE, more polysemanticity discussion, and possibly a reordered
  lead visual.
