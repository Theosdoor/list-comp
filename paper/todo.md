# todo

**general paper**
- remove appdx from uplodaed pdf but keep in pdf in code repo


**intro**
- add stuff on binary view (gets repeated a lot in later sections)

**res_rq2**
- o1 logit empirically linear in p --> why? should add note beneath steering diagram or something

The failure mode analysis in Section 5.2 needs better signposting. The four bullet points (negative scaling, mutually exclusive ranges, no swap in range, matching symbols) are accurate but read as an unstructured list. A reader wants to know: which of these are fundamental limitations of the mechanism and which are artefacts of the steering method? For example, the "negative scaling" case is arguably an artefact of BatchTopK's non-negativity constraint, while "no overlapping scale ranges" may indicate something more fundamental about the representation. Making this distinction explicit would improve the evaluation considerably:

Of these failure modes, the first (negative scaling required) is an 
artefact of the non-negativity constraint in BatchTopK SAEs rather 
than a fundamental limitation of the magnitude-based mechanism: a 
negative activation has no interpretation within the trained model's 
representational scheme. The remaining cases represent genuine 
limitations of single-latent steering, which the multi-latent 
experiments in Section~\ref{ss:steering_exp} begin to address.


*The Spearman correlation figure (special latents vs. performance)* reports r=−0.460r = -0.460
r=−0.460 and r=0.506r = 0.506
r=0.506 without confidence intervals or p-values. Given the noted inconsistency in threshold (0.3 vs 0.5), this figure currently carries more weight than it should.

Table 4 caption still has the TODO about consistent run numbers. Beyond being unfinished, there's a real methodological question here: if JumpReLU has 640 runs but BatchTopK has 450, the aggregate statistics are not directly comparable.


## for workshop sub
- consistent r >0.5 (sae sweep on wandb) or r>0.3 (text / graphs)
- weird that new sae sweep (generalisation ones) are so much better than old sae sweep? so need to redo that table
  - also, maybe each SAE should have equal number of runs to avoid bias in aggregate stats?

- may need to swap out 2 layer model (91% acc) for a better one (>95% acc) from sweep, which also means retraining the saes! but the pipeline should work fine, just means regenning lots of figs. keep this in mind but don't stress if have to stick with old ones
- justify o1 linear in p with maths or empirical results if possible
  - should try and brak this assumption
- should test different r thresh for special latents (appdx) to see if different r's affect special latent detction. eg. 0.3, 0.5, 0.8, 1.0. could have eg. appdx with figure for each one comparing ev and pce
- use loss recovered instead of EV and PCE (can keep these but theyre appendix worthy i think)
  - evaluate on sparsity & loss recovered (See sae bench) & redo figs (loss recov is like pce but better & standard)
  - NEED to evaluate over sparsity especially. eg, the graphs in saebench (perhaps hue = n special latents, loss recovered on y, and l0 on x?)
  - could use sae bench for existing implementation
- for performance scaling with n_special features, should also check this isnt down just to eg. d_sae etc.
  - see https://claude.ai/chat/32ecad4e-95e7-48ca-a6c1-786778aef5c3
  - any causal tests rather than just correlation?
- should check for generalisation of symbol detectors too!
  - & proportions that are 1-symb detectors
- stick to just independence or binary view as sae issue we're contributing to




CLAUDE msg: (link src/sae stuff and special latents script)
I'd like to replace this comparison script with one that creates 3 plots, one for each sae type (jumprelu, batchtopk, matryoshka). each plot has:



* y-axis is patched cross entropy loss

* x-axis is L0 (actual! not just k - may need to calculate)

* hue is number of special latents according to the given threshold

then create another 3 plots (one for each sae type) but with y axis as explained variance.

save each as an individual plot, eg. matryoshka_pce.pdf



parameters are:



* local sae folder(s). they might contain subfolders, in which case recursively check them and load with the existing load sae function. should accept multiple folders and test on all given saes

* alpha_diff_thresh (default 0.5) for the special latent function



\brainstorming