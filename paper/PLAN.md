# Workshop Plan


- saes are popular
- saes retrieve a dictionary of feature vectors t


Story
- saes are popular method for identifying neural net features.
- they work by enforcing sparsity to disentangle features in superposition in dense neuron activations
- saes recover an overcomplete dictionary of features 
- lots of previosu work has shown that its difficult to know if the dictionary is correct:
- since this is unsupervised, they're optimised using sparsity and reconstruction fidelity, which can produce interpretabile latents but sparsity is known to be a flawed proxy (chanin, leask meta saes) - so the dictionary is difficult to be correct.
- However, prior work presents a toy transformer model which suggests problems for interp with SAEs even if they recover the correct dictionary.
- explain model
- since relative magnitude of latent activations is significant to the feature represented, a view of SAE latents as independently interepretable is insufficient to understand the model.
- however, the authors didn't verify this prediction using actual SAEs. In this paper, we study the impact on various BTK, jumprelu, matryoshka SAEs, with the aim of setting the stage for further analysis on real-world LLM SAEs such as gemma scope
- our paper is structured as follows: ...


- however, nonlinear feature composition methods 

- previous work found this toy attention-only transformer that had a relative magnitude-based mechanism in which ...
- the authors predicted that this would cause SAEs to learn a solution with graded latent activations
- this undermines the independence assumption from the LRH: that transformer features are independently understandable
- however, this work didn't verify the hypothesis and the toy model was highly constrained. 
- In the following, we:
  - train a suite of SAEs on the toy model from prior work, and find that SAEs reliably learn latents with graded activations, such that we can predictably steer the model output by linearly scaling the sae latent and patching the reconstruction back downstream. we show that this graded latent activation behaviour strongly correlates with high sae performance for Matryoshka, jumprelu and btk saes. 
  - [SEEMS TOO MUCH EFFORT] we additionally train SAEs on different versions of the prior toy model, and show that even with MLPs and multiple attn heads the SAEs still learn graded latents
- 



uncontrained models
- 2 heads, 94.9% acc: d64_L2_N2_h2_lnF_biasF_wvF_woF_mlpF_s21
- trigram, 91.0% acc: d64_L3_N2_h1_lnF_biasF_wvF_woF_mlpF_s14
- ln,bias,wv,vo,mlp, 93.7% acc: 
d64_h1_lnT_biasT_wvT_woT_mlpT_s10

