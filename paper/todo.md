# todo


## for workshop sub
- consistent r >0.5 (sae sweep on wandb) or r>0.3 (text / graphs)
- weird that new sae sweep (generalisation ones) are so much better than old sae sweep? so need to redo that table
  - also, maybe each SAE should have equal number of runs to avoid bias in aggregate stats?

- may need to swap out 2 layer model (91% acc) for a better one (>95% acc) from sweep, which also means retraining the saes! but the pipeline should work fine, just means regenning lots of figs. keep this in mind but don't stress if have to stick with old ones
- justify o1 linear in p with maths or empirical results if possible
  - should try and brak this assumption
- should test different r thresh for special latents (appdx) to see if different r's affect special latent detction. eg. 0.3, 0.5, 0.8, 1.0. could have eg. appdx with figure for each one comparing ev and pce

- for performance scaling with n_special features, should also check this isnt down just to eg. d_sae etc.
  - see https://claude.ai/chat/32ecad4e-95e7-48ca-a6c1-786778aef5c3
  - any causal tests rather than just correlation?
- should check for generalisation of symbol detectors too!
  - & proportions that are 1-symb detectors
- stick to just independence or binary view as sae issue we're contributing to
