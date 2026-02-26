# Order by Scale: Relative-Magnitude Relational Composition in Attention-Only Transformers

### Theo Farrell

Department of Computer Science Durham University theo.farrell99@outlook.com

### Patrick Leask

Department of Computer Science Durham University

### Noura Al Moubayed

Department of Computer Science Durham University

## Abstract

LLMs and other transformers learn relational composition mechanisms to solve tasks such as tracking information about subjects ("Alice lives in France. Bob lives in Thailand.") to answer questions, or parallelising precomputing sub-paths in graph-based path-finding problems. There is a deep theoretical literature on vector composition methods, yet we lack empirical studies of what mechanisms transformers learn in practice. In particular, different composition methods affect sparse autoencoders (SAEs), a popular method for decomposing model activations, in different ways. We present empirical evidence in a controlled attention-only transformer that ordered relational information can be encoded via a relative magnitude-based mechanism, i.e. by a weighted sum of vectors, rather than predicted direction-based mechanisms such as additive matrix binding. While absolute magnitude-based mechanisms have been reported for other architectures (e.g. onion representations in RNNs), to our knowledge this is the first controlled demonstration of a relative magnitude mechanism in attention-only transformers. This result challenges the prevailing view in mechanistic interpretability research that transformer features can be viewed as binary and independent, and motivates a re-examination of these methods with respect to feature activation value and interactions between features at different values. In future work, we will remove the constraints placed on our toy setting, and attempt to find evidence of these mechanisms in LLMs. Code is available here: [github.com/Theosdoor/order-by](https://github.com/Theosdoor/order-by-scale)[scale.](https://github.com/Theosdoor/order-by-scale)

### 1 Introduction

Transformers perform well on tasks that require composing information from different input positions. For example, [Brinkmann et al.](#page-8-0) [\[2024\]](#page-8-0) found that a transformer trained on a path-finding problem in binary trees learns to precompute subpaths and store these in special token positions, when the depth of the tree exceeds the number of transformer layers; and [\[Feng and Steinhardt, 2023\]](#page-9-0) found an identity vector mechanism used in composing relation information in prompts such as "Alice lives in France. Bob lives in Thailand.". Solving these problems requires learning mechanisms for *relational composition*, by which multiple vectors are composed into a single fixed-length vector. Whilst there is a substantial body of theoretical research into this topic [\[Smolensky, 1990,](#page-10-0) [Plate, 1995,](#page-9-1) [Kanerva, 2009,](#page-9-2) [Csordás et al., 2024\]](#page-8-1), we lack empirical studies into which of these are actually used in LLMs. [Wattenberg and Viégas](#page-10-1) [\[2024\]](#page-10-1) propose additive matrix binding as a candidate mechanism,

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Figure 1: These graphs show the movement of information between token positions in our transformers on specific example inputs. The three-layer model (right) appears to directly copy between token positions, avoiding the composition task. We focus on a two-layer model (left) which has reduced accuracy but performs the desired composition at the SEP token. Dotted lines represent the residual stream and red arrows indicate attention moving information between positions, with the thickness of the lines corresponding to the attention pattern value. As our value and output matrices are the identity matrix, attention performs scaled copying in the residual stream. Edges are annotated with the average impact on validation accuracy when ablated, edges with no impact on accuracy are not displayed. The figure on the left shows the pattern for a two-layer transformer with 92% validation set accuracy, whereas the figure on the right shows the pattern for a three-layer transformer with 100% validation set accuracy. The numbers next to each of the nodes in the graph correspond to the logit lens [\[Nostalgebraist, 2020\]](#page-9-3) of the activation value at that position and layer.

whilst [Feng and Steinhardt](#page-9-0) [\[2023\]](#page-9-0) find broad use of binding ID mechanisms across families of LLMs. These mechanisms have different implications for sparse autoencoders (SAEs), a popular method to recover interpretable features in the mechanistic interpretability literature: additive matrix binding could result in a multiplicity of "echo features" - copies of the same feature when bound in different subspaces; and ID vectors may result in abstract features that cannot easily be interpreted with respect to the input.

To bridge this research gap, we present a study of a toy model exhibiting another relational composition mechanism. Toy models have been useful in the past for understanding phenomena that have then generalised to LLMs. For example, results on modulo arithmetic in toy models [\[Nanda et al.,](#page-9-4) [2023\]](#page-9-4) informed work on LLMs [\[Baeumel et al., 2025\]](#page-8-2), and toy models of superposition [\[Elhage et al.,](#page-9-5) [2022\]](#page-9-5) led to the application of SAEs to LLMs [\[Bricken et al., 2023\]](#page-8-3).

Our paper is structured as follows. In Section [3,](#page-3-0) we present a two-layer attention-only transformer to encode two input tokens, d<sup>1</sup> and d2, into a single separator token, SEP, and then reconstruct d<sup>1</sup> and d<sup>2</sup> as output. This is motivated by the path-finding model in [Brinkmann et al.](#page-8-0) [\[2024\]](#page-8-0). In Section [4,](#page-4-0) we find that the order of these tokens is defined by the relative magnitude of scalar variables in the attention pattern, rather than through a direction-based mechanism like additive matrix binding. This is problematic to the common perspective that SAE features can be interpreted as binary [\[Quirke et al., 2025,](#page-9-6) [Paulo et al., 2024\]](#page-9-7): in our model, the same features are active for both ⟨a, b⟩ and ⟨b, a⟩, but with different magnitudes. In Section [5,](#page-7-0) we discuss the implications of these results on the mechanistic interpretability agenda, and SAEs in particular. Our paper is intended as a foundational and exploratory work, and in future work we will attempt to validate our findings across less-constrained toy models and on pretrained LLMs.

# 2 Related Work

Relational composition and binding. Classic work in distributed representations show how single vectors can encode structured data. Tensor Product Representations (TRPs) and Vector Symbolic Architectures (VSAs) bind and additively superpose vectors, enabling ordered structures in fixed width vectors [\[Smolensky, 1990,](#page-10-0) [Plate, 1995,](#page-9-1) [Kanerva, 2009\]](#page-9-2).

<span id="page-2-0"></span>Table 1: Comparison of relational composition mechanisms and their implications for interpretability.

| Composition            | Description                                                                                                | Example                                                                                                                                         | Implications                                                                                                                                     |  |
|------------------------|------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|--|
| Direction<br>based     | Features are vectors;<br>sequence position de<br>fined by different di<br>rections                         | Matrix binding Watten<br>berg and Viégas [2024]<br>-<br>features<br>projected<br>to<br>distinct linear subspaces:<br>r = Ax + By                | Feature<br>multiplicity:<br>SAEs<br>may<br>learn<br>echo<br>features (Ax,<br>Ay,<br>Bx,<br>By)<br>representing<br>the<br>same underlying concept |  |
| Fixed Magni<br>tude    | Multiple<br>features<br>have same direction;<br>sequence<br>position<br>defined by different<br>magnitudes | Onion-like<br>represen<br>tations<br>Csordás<br>et<br>al.<br>[2024]<br>-<br>"unpeel"<br>via<br>autoregression to retrieve<br>sequence positions | Counter-example to the<br>widely held assumption<br>that representations are<br>linear                                                           |  |
| Relative Mag<br>nitude | Sequence position de<br>fined by magnitude<br>relative to other posi<br>tions in sequence                  | This work - sequence or<br>der defined by a weighted<br>sum of input token and<br>positional embeddings                                         | Counter-example to the<br>widely held assumptions<br>that representations are<br>linear and SAE latents<br>are binary                            |  |

[Wattenberg and Viégas](#page-10-1) [\[2024\]](#page-10-1) highlight additive matrix binding, where vectors are written to slots using role-specific linear maps so that bigrams (x, y),(y, x) remain separable, as a candidate mechanism for relational composition in transformers. That is, given an ordered bigram (d1, d2) and two n × n matrices A and B, define the representation of (x, y) by

$$r = Ax + By$$

In this way, (x, y) and (y, x) have different representations. [Wattenberg and Viégas](#page-10-1) [\[2024\]](#page-10-1) claim that this causes *feature multiplicity*: a kind of false positive where multiple "echo features" are created that represent the same concept in different contexts. Given two features x and y, matrix binding results in Ax, Ay, Bx, and By features that represent the same concept as x and y.

Empirical evidence for composition and binding. [Feng and Steinhardt](#page-9-0) [\[2023\]](#page-9-0), [Prakash et al.](#page-9-8) [\[2024\]](#page-9-8) identify binding-ID directions that bind entities and their attributes in LLMs in-context. For example, a "green car". A binding-ID vector k is attached to an entity and its attribute by addition in their respective activation spaces. To answer a query for a given entity, the attribute with a matching binding-ID is retrieved from the context activation space. [Feng and Steinhardt](#page-9-0) [\[2023\]](#page-9-0) also show that these mechanisms are position-independent with respect to the attribute and entity, and that the activations are factorisable.

[Brinkmann et al.](#page-8-0) [\[2024\]](#page-8-0) reverse-engineer an attention-only tree-search transformer that stores precomputed subpaths in special token positions and merges them later. This occurs when the tree depth, N, is greater than L − 1 < N, where L is the number of transformer layers. This constraint is also necessary in our work, where N is the size of the input N-gram. However, the authors do not investigate the structure of these activations.

[Csordás et al.](#page-8-1) [\[2024\]](#page-8-1) demonstrate "onion representations" in small (hidden size ≤ 64) gated recurrent neural networks (RNNs), where these slots are defined by different orders of magnitude rather than distinct linear subspaces. The model represents ordered sequences by closing the gates gradually and synchronously over the input phase, which exponentially decays the scaling factor of subsequent sequence positions. This produces layered onion representations where earlier sequence positions dominate the magnitude space. This contrasts larger models which indicate sequence position by sharply closing their gates, which creates position-dependent subspaces for each input. The onion representation allows autoregressive decoding to sequentially "peel" off tokens by subtracting the current dominating token embedding. Multiple tokens can occupy the same directional subspace at different magnitude scales; any linear direction will cross-cut multiple layers of the onion. While the authors find that sequence order is represented by fixed magnitude scales, we find an alternative mechanism where it is instead represented by relative magnitude between positions, which is inputdependent and not fixed.

Our toy model provides evidence for magnitude-based composition in attention-only transformers, but with *relative* magnitudes rather than fixed ones as observed in [Csordás et al.](#page-8-1) [\[2024\]](#page-8-1). See Table [1](#page-2-0) for an overview of these composition methods.

Sparse Autoencoders. Models can store more sparse features than the dimension of their activations naively allows through superposition, creating challenges for interpreting these models [\[Elhage et al.,](#page-9-5) [2022\]](#page-9-5). Sparse Autoencoders (SAEs) [\[Bricken et al., 2023,](#page-8-3) [Cunningham et al., 2024\]](#page-9-9) and their variants [\[Gao et al., 2024,](#page-9-10) [Bussmann et al., 2024,](#page-8-4) [Rajamanoharan et al., 2024,](#page-9-11) [Leask et al., 2025b,](#page-9-12) [Costa et al.,](#page-8-5) [2025,](#page-8-5) [Bussmann et al., 2025\]](#page-8-6) have been proposed as a tool for recovering these sparse features from dense activations. An SAE is an autoencoder with a single hidden layer that is trained to reconstruct its input while enforcing sparsity in the activations of the neurons in this hidden layer. SAEs have been successfully used on exploratory interpretability tasks like model auditing [\[Marks et al., 2025\]](#page-9-13) and hypothesis generation [\[Movva et al., 2025\]](#page-9-14), but it is unclear whether they can be used to recover the true features of models [\[Leask et al., 2025a,](#page-9-15) [Chanin et al., 2024\]](#page-8-7).

[Wattenberg and Viégas](#page-10-1) [\[2024\]](#page-10-1) argue that matrix binding will result in echo features in SAEs, where the SAE learns latents corresponding to projections of the same feature into different binding subspaces. The effects of ID vectors and onion representations on SAEs have so far not been studied.

SAE latents are generally treated as binary features in the literature [\[Paulo et al., 2024,](#page-9-7) [Quirke](#page-9-6) [et al., 2025\]](#page-9-6), where they are 'on' if the feature is active and 'off' otherwise. In contrast, our results demonstrate graded latent activations in which relative activation magnitude encodes relational information, such as sequence order. If these results generalise to LLMs, then they could undermine this binary perspective. Additionally, graded latent activations may be ignored or misinterpreted by any downstream interpretability tool that assumes all features are independently interpretable and linear.

# <span id="page-3-0"></span>3 Methodology

Task. We train a modified auto-regressive transformer to map from [d1, d2, SEP, MASK, MASK] to [d1, d2, SEP, o1, o2], where d<sup>1</sup> and d<sup>2</sup> are uniformly sampled from [0, 99], and (o1, o2) = (d1, d2).

<span id="page-3-1"></span>Table 2: Our custom attention mask for input [d1, d2, SEP, o1, o2], where F means the attention weight was set to −∞ before softmax, and T means it was computed normally. The d<sup>i</sup> tokens attend only to other d<sup>i</sup> and SEP, while the o<sup>i</sup> tokens attend causally to the separator and previous output tokens. This prevents direct copying of input embeddings from d<sup>i</sup> into o<sup>i</sup> . Self-attention is enabled only for d<sup>1</sup> to prevent numerical instability.

|     | d1 | d2 | SEP | o1 | o2 |
|-----|----|----|-----|----|----|
| d1  | T  | F  | F   | F  | F  |
| d2  | T  | F  | F   | F  | F  |
| SEP | T  | T  | F   | F  | F  |
| o1  | F  | F  | T   | F  | F  |
| o2  | F  | F  | T   | T  | F  |

The attention pattern is constrained so that the output positions cannot attend to the input positions that is, o<sup>1</sup> can only attend to SEP and o<sup>2</sup> can only attend to o<sup>1</sup> and SEP (Table [2\)](#page-3-1).

The loss is computed only on o<sup>1</sup> and o2. We chose this narrow setting to isolate the problem solved by the sub-path pre-computation in [\[Brinkmann et al., 2024\]](#page-8-0), and simplified to an ordered bigram to find a minimal list copying circuit.

We removed the attention bias and layer norm, and froze the attention value and output matrices to the identity, as these ablations resulted in no decrease in task performance (Appendix [A,](#page-10-2) Table [4\)](#page-10-3). This reduces attention outputs to weighted sums of the previous residual stream.

Model specification and training. Our training dataset consists of 80% of all possible inputs (i.e. 8000 of the total 100\*100 (d1, d2) bigrams). The test set is the remaining 20% of unseen inputs. We optimise cross-entropy on both o<sup>1</sup> vs d<sup>1</sup> and o<sup>2</sup> vs d2.

We trained two- and three-layer transformers on this task. The three-layer model consistently achieves 100% accuracy, as the additional layer allows it to directly copy between token positions, circumventing the composition task (Figure [1\)](#page-1-0). Instead, we focus on two-layer models, which have a maximum performance of 92.2% test accuracy.

<span id="page-4-1"></span>After a grid-search of residual stream dimensions (see Appendix [A,](#page-10-2) Table [4\)](#page-10-3), we set dmodel = 64, which was the lowest dimension that achieved joint-best validation accuracy. The trainable parameters are the token and positional embedding matrices, the unembedding matrix, and the key and query matrices from the attention layers. Whilst these are substantial constraints, it is normal for toy models in the mechanistic interpretability literature to use similar constraints [\[Nanda et al., 2023,](#page-9-4) [Elhage](#page-9-16) [et al., 2021\]](#page-9-16).

Table 3: Model and training configuration.

| Parameter                 | Value             |  |  |
|---------------------------|-------------------|--|--|
| Number of layers          | 2 or 3            |  |  |
| Number of heads           | 1                 |  |  |
| Residual stream dimension | 64                |  |  |
| Attention head dimension  | 64                |  |  |
| LayerNorm                 | None              |  |  |
| Bias                      | None              |  |  |
| Value matrix (WV<br>)     | Identity (frozen) |  |  |
| Output matrix (WO)        | Identity (frozen) |  |  |
| Learning rate             | 1 × 10−3          |  |  |
| Optimiser                 | AdamW             |  |  |
| Batch size                | 128               |  |  |
| Betas                     | (0.9, 0.999)      |  |  |
| Weight decay              | 0.01              |  |  |

Table [3](#page-4-1) shows our final configurations. Overall, we have 92,288 trainable parameters for the two-layer model, and 125,056 for the three-layer model.

### <span id="page-4-0"></span>4 Results

Our selected two-layer model achieves accuracy of 92.2% on the validation set, whereas the threelayer model achieves 100% accuracy. Whilst the three-layer model is more accurate, we focus on the two-layer transformer as this forces the model to learn to compose the representations in the separator token after layer 1, similarly to [Brinkmann et al.](#page-8-0) [\[2024\]](#page-8-0). The authors find the compression mechanism in N-depth binary trees where the number of model layers is such that nlayers − 1 < N. In our model, we meet the same constraint except N corresponds to the input N-gram, rather than tree depth. For a bigram (d1, d2), we require a two-layer model.

### 4.1 Attention Ablations

For the two- and three-layer models, we independently set each attention pattern probability to zero and observe the impact on performance. Figure [1](#page-1-0) shows the minimum set of attention pattern weights required for the model to achieve full accuracy; all other weights can be set to zero. Throughout this section we refer to attention pattern weights as edges, in reference to the graphs in Figure [1.](#page-1-0)

For the three-layer model, we find two types of edge that reduce accuracy when ablated:

- 1. edges that copy d<sup>i</sup> to the corresponding o<sup>i</sup> via SEP in sequential layers for each i,
- 2. and one edge between d<sup>1</sup> and SEP in layer 1 that only had a small impact on the validation accuracy when ablated.

Ablating the former roughly halves the validation performance, as removing any one of these means the model cannot predict at least one of the outputs. This supports our conjecture that they are copying each d<sup>i</sup> to the o<sup>i</sup> via SEP. This three-layer model therefore does not display the composition in which we are interested, and instead implements a sequential copy algorithm.

For the two-layer model, we find three types of edge that reduce accuracy when ablated, which we refer to as composition, decomposition and moderation edges:

- 1. The composition edges are those in layer 1 between each  $d_i$  and SEP these copy the bigram to SEP
- 2. The decomposition edges are those in layer 2 between SEP and each  $o_i$  these copy the information from SEP to each  $o_i$ .
- 3. The additional moderation edge is between  $o_1$  and  $o_2$  in layer 2.

Ablating the moderation edge results in misclassification of  $o_2$  in 62% of cases. 54% of these errors are caused by the model copying  $d_1$  to both output positions - that is,  $d_1 = o_1 = o_2$ . For this reason, we hypothesise that this edge is responsible for moderating the  $o_1$  logit in the  $o_2$  token position. We additionally investigate how this edge contributes to the linear separability of the  $o_i$  before unembedding.

We detail further attention ablations in Appendix B.

### <span id="page-5-3"></span>4.2 Circuit Analysis

We annotate our five input tokens as  $[d_1, d_2, s, o_1, o_2]$ . In our input, the  $d_i$  tokens vary as described in Section 3, the s token is always SEP, and the  $o_i$  tokens are always MASK. E is our token embedding matrix, P is our positional embedding matrix, and U is our unembedding matrix. Define  $\alpha_{x \to y}$  as the attention weight at query x and key y in the first layer's attention pattern. Define  $\beta_{x \to y}$  similarly for the second layer. Define  $\ell_t \in \mathbb{R}^V$  as the logit vector predicted at the position of token t, where t is the vocabulary size. We write t as the token embedding of  $t \in \{d_1, d_2, s, m\}$ , where t is the embedding of the t tokens, and t as the positional embedding of token t tokens, t as the positional embedding of token t tokens, t as the positional embedding of token t tokens, t as the positional embedding of token t tokens, t as the positional embedding of token t tokens, t as the positional embedding of token t tokens, t as the positional embedding of token t tokens are always MASK. t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, the t is our input, th

<span id="page-5-0"></span>We derive the following expressions for the logits at positions 4 and 5, called  $\ell_{o_1}$  and  $\ell_{o_2}$  respectively, leaving details for Appendix C. Let  $S = E_s + P_s$ , then:

$$\ell_{o_1} = [E_m + P_{o_1} + \alpha_{s \to d_1} (E_{d_1} + P_{d_1}) + \alpha_{s \to d_2} (E_{d_2} + P_{d_2}) + S]U$$
(1)

$$\ell_{o_2} = \left[ (1 + \beta_{o_2 \to o_1}) \boldsymbol{E}_m + \beta_{o_2 \to o_1} \boldsymbol{P}_{o_1} + \boldsymbol{P}_{o_2} + \beta_{o_2 \to s} (\alpha_{s \to d_1} (\boldsymbol{E}_{d_1} + \boldsymbol{P}_{d_1}) + \alpha_{s \to d_2} (\boldsymbol{E}_{d_2} + \boldsymbol{P}_{d_2}) + \boldsymbol{S}) \right] \boldsymbol{U}$$
(2)

<span id="page-5-2"></span>**Scaling to reverse output.** Suppose that the model is given  $d_1 = a$  and  $d_2 = b$  as input, where  $a, b \in [0, 99]$  as outlined in Section 3. Substituting these into equation (1) gives us  $\ell_{o_1}(a, b)$ : that is, the final logit vector at position 4 on inputs a, b.

$$\ell_{o_1}(a,b) = \left[ \boldsymbol{E}_m + \boldsymbol{P}_{o_1} + \alpha_{s \to a} (\boldsymbol{E}_a + \boldsymbol{P}_{d_1}) + \alpha_{s \to b} (\boldsymbol{E}_b + \boldsymbol{P}_{d_2}) + \boldsymbol{S} \right] \boldsymbol{U}$$
(3)

Moreover, consider when the inputs are swapped. Let  $\alpha' \neq \alpha$  be the new attention pattern for this input. Therefore:

$$\ell_{o_1}(b,a) = \left[ \boldsymbol{E}_m + \boldsymbol{P}_{o_1} + \alpha'_{s \to b} (\boldsymbol{E}_b + \boldsymbol{P}_{d_1}) + \alpha'_{s \to a} (\boldsymbol{E}_a + \boldsymbol{P}_{d_2}) + \boldsymbol{S} \right] \boldsymbol{U}$$
(4)

<span id="page-5-1"></span>Now let us find the difference between these:

$$\ell_{o_{1}}(b,a) - \ell_{o_{1}}(a,b) 
= \left[\alpha'_{s\to b}(\boldsymbol{E}_{b} + \boldsymbol{P}_{d_{1}}) + \alpha'_{s\to a}(\boldsymbol{E}_{a} + \boldsymbol{P}_{d_{2}}) \right] 
- \alpha_{s\to a}(\boldsymbol{E}_{a} + \boldsymbol{P}_{d_{1}}) - \alpha_{s\to b}(\boldsymbol{E}_{b} + \boldsymbol{P}_{d_{2}}) U 
= \left[(\alpha'_{s\to a} - \alpha_{s\to a})\boldsymbol{E}_{a} + (\alpha'_{s\to b} - \alpha_{s\to b})\boldsymbol{E}_{b} \right] 
+ (\alpha'_{s\to b} - \alpha_{s\to a})\boldsymbol{P}_{d_{1}} + (\alpha'_{s\to a} - \alpha_{s\to b})\boldsymbol{P}_{d_{2}} U$$
(5)

<span id="page-6-1"></span>![](_page_6_Figure_0.jpeg)

Figure 2: These bars show the *individual* contribution for each element in  $\Delta \ell$  towards  $\ell_{o_2}$  in  $\ell_{o_1} + \Delta \ell = \ell_{o_2}$ . The dotted line shows *cumulative* contribution for each element when added to those preceding it in equation 6. Negative contributions increase the  $d_1$  logit relative to  $d_2$ , and positive contributions increase the  $d_2$  logit relative to  $d_1$ . Note that the plotted bars account for the -/+ preceding some y-axis elements.

In this way, the order of the model's output logits can be reversed by linearly scaling the input token and positional embeddings via the first layer's attention pattern. This suggests that the model composes the bigram by scaling the same underlying directions (the input token and positional embeddings) with different coefficients, rather than projecting onto role-specific directions as in additive matrix binding. While the resulting residual vectors do point in different directions for different inputs, the mechanism decomposes into weighted sums of a fixed set of basis vectors, with the weights (attention scores) encoding the order information. This contrasts to existing research on relational composition mechanisms Csordás et al. [2024], Wattenberg and Viégas [2024] which use different activation directions.

Difference between  $o_2$  and  $o_1$  logits. Let:  $D_i = E_{d_i} + P_{d_i}$ ,  $i \in \{1, 2\}$ .

<span id="page-6-0"></span>When we take the difference between  $\ell_{o_2}$  and  $\ell_{o_1}$ , we get

$$\Delta \ell = \ell_{o_2} - \ell_{o_1} 
= [\beta_{o_2 \to o_1} \mathbf{E}_m + (\beta_{o_2 \to o_1} - 1) \mathbf{P}_{o_1} + \mathbf{P}_{o_2} 
+ (\beta_{o_2 \to s} - 1) (\alpha_{s \to d_1} \mathbf{D}_1 
+ \alpha_{s \to d_2} \mathbf{D}_2 + \mathbf{S})] \mathbf{U}$$

$$= [\beta_{o_2 \to o_1} \mathbf{E}_m + \mathbf{P}_{o_2} - \beta_{o_2 \to s} \mathbf{P}_{o_1} 
- \beta_{o_2 \to o_1} (\alpha_{s \to d_1} \mathbf{D}_1 + \alpha_{s \to d_2} \mathbf{D}_2 + \mathbf{S})] \mathbf{U}$$

$$= [\mathbf{P}_{o_2} - \beta_{o_2 \to s} \mathbf{P}_{o_1} - \beta_{o_2 \to o_1} (\alpha_{s \to d_1} \mathbf{D}_1 
+ \alpha_{s \to d_2} \mathbf{D}_2 + \mathbf{S} - \mathbf{E}_m)] \mathbf{U}$$
(6)

We reconstruct  $\hat{\ell_{o_2}} = \ell_{o_1} + \Delta \ell$  to compare to the original  $\ell_{o_2}$ . Figure 2 shows the empirical contribution of each element of  $\Delta \ell$ . We found that using  $\ell_{o_2}$  for predictions in place of  $\ell_{o_2}$  provided the same results, which verifies our derivation. We find that setting  $\alpha_{s \to d_1} = 0$  and  $\alpha_{s \to d_2} = 1$  in  $\Delta \ell$  and using  $\ell_{o_2}$  to predict  $o_2$  causes the model's  $o_2$  accuracy to jump to 99% from 84%. However, in practice these changes to  $\alpha$  also impact  $\ell_{o_1}$  and significantly reduce the model's  $o_1$  accuracy. We hypothesise that the model learns to balance a reduction in  $o_2$  accuracy for a relatively larger increase in  $o_1$  accuracy through  $\alpha_{s \to d_1}$ .

#### 4.3 Linearly Separable Output Projections

We hypothesise that the model uses the unembedding matrix U to linearly separate the position embeddings of  $o_1$  and  $o_2$ . To confirm this, we project the final activation vectors for  $o_1$  and  $o_2$  onto the positional embedding matrix P, and reconstruct the activations from this projection, removing the token embedding component. We verify the separability of these two groups by fitting a linear support vector classifier (SVC) [Cortes and Vapnik, 1995] to separate these groups, which achieves over 99% classification performance. This demonstrates the hypothesis that the model is "writing to slots" in superposition through scaling rather than additive matrix binding [Wattenberg and Viégas, 2024].

#### <span id="page-7-0"></span>5 Discussion

Our toy transformer solves ordered list copying by writing both inputs into the <SEP> position, and then decoding order by the relative magnitudes of the contributions, rather than projecting content into role-specific directions, as is the case in additive matrix binding [Wattenberg and Viégas, 2024]. Equation (5) shows that the logits at  $o_1$  and  $o_2$  can be written as combinations of  $(E_{d_1} + P_{d_1})$  and  $(E_{d_2} + P_{d_2})$  with input-dependent coefficients  $\alpha_{s \to d_1}$ ,  $\alpha_{s \to d_2}$ .

Consider training an L1-regularized SAE on the residual at SEP after layer 1,

$$\mathbf{r}_{s}^{L_{1}} = \alpha_{s \to d_{1}} (\mathbf{E}_{d_{1}} + \mathbf{P}_{d_{1}}) + \alpha_{s \to d_{2}} (\mathbf{E}_{d_{2}} + \mathbf{P}_{d_{2}}) + \mathbf{E}_{s} + \mathbf{P}_{s}$$
(7)

Then learning separate latents with identical decoder directions is an unstable solution, as this always results in a higher average  $L_1$  penalty than maintaining a single latent for this direction. Compare this with the ideal solution of learning latents corresponding to  $D_1 = E_{d_1} + P_{d_1}$  and  $D_2 = E_{d_2} + P_{d_2}$ , with activations equal to the attention probabilities.

This results in graded latent activations. As such, the binary perspective of SAE latents [Quirke et al., 2025, Paulo et al., 2024] is insufficient to explain the computation of this model, and the activation values of the SAE latents must be incorporated into our understanding. This creates separate problems to the echo-features caused by additive matrix binding [Wattenberg and Viégas, 2024], which are different directions in the activation space to the base features; and the ID features in ID binding [Feng and Steinhardt, 2023] which are also different directions but have an abstract interpretation. A feature that identifies the order of the bigram is necessarily not linear, due to the relative interactions between the input features. These contradict the linear representation hypothesis, a foundational assumption of mechanistic interpretability that states that networks can be described in terms of independently understandable features and linear features [Elhage et al., 2022]. Similarly to the results of Csordás et al. [2024] on RNNs, we provide evidence that transformer interpretability should not be confined by the linear representation hypothesis.

Recent work [Leask et al., 2025b, Chanin et al., 2024] has highlighted the practical challenges with training SAEs that are useful for mechanistic interpretability. Our results in this paper potentially cut across this debate: even if SAEs can recover the correct dictionary, without tools to understand the effect of the relative magnitudes of latent activations, our interpretation of the model will be incomplete.

### 6 Conclusions and Future Work

In this paper we identified a previously undescribed magnitude-based mechanism for relational composition in transformers, and released a minimal toy model that has learned this mechanism. We further described how this mechanism will affect latent learning in SAEs, and how it necessitates a different perspective on transformer features than is currently prevalent in the mechanistic interpretability literature.

In future work, we plan to determine whether pretrained LLMs implement similar mechanisms, and find evidence of pretrained SAE latents where different activation values can result in dramatically different model behaviour. We also intend to train SAEs on this toy model and see what they learn in practice, however we expect these results to be difficult to interpret due to the challenges of SAE hyperparameter selection [Leask et al., 2025a] and training instability [Fel et al., 2025].

In addition, we will pursue validation of our findings on less constrained models. For example, we will extend our investigation to larger N-grams to analyse how the minimum nlayers for high accuracy scales with N. Furthermore, the positional encodings play a major role in the magnitude-based mechanism, and it is unclear whether this mechanism would be possible for alternative positional encoding methods such as absolute encodings [\[Radford et al., 2019\]](#page-9-18) or RoPE [\[Su et al., 2024\]](#page-10-4).

In summary, the findings in this paper are intended as a foundation for further research on magnitudebased relational composition mechanisms in LLMs.

### Acknowledgements

This work was supported through a research scholarship by the Open Philanthropy Career Development and Transition Funding. We are grateful to Open Philanthropy for their financial support and Strahinja Klem for helpful discussions about the high-level methodology.

### Contribution Statement

Theo Farrell was the primary research contributor. He trained the models, coded the experiments, and wrote this paper.

Patrick Leask set the high-level direction of the project, advised on what interpretability methods to apply, helped interpret the results, and wrote parts of the discussion in Section [5.](#page-7-0)

Noura Al Moubayed advised on narrative framing and provided planning, editing and writing feedback.

### References

<span id="page-8-2"></span>Tanja Baeumel, Daniil Gurgurov, Yusser al Ghussin, Josef van Genabith, and Simon Ostermann. Modular arithmetic: Language models solve math digit by digit. *arXiv preprint arXiv:2508.02513*, 2025.

<span id="page-8-3"></span>Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards monosemanticity: Decomposing language models with dictionary learning. *Transformer Circuits Thread*, 2023. https://transformer-circuits.pub/2023/monosemantic-features/index.html.

<span id="page-8-0"></span>Jannik Brinkmann, Abhay Sheshadri, Victor Levoso, Paul Swoboda, and Christian Bartelt. A mechanistic analysis of a transformer trained on a symbolic multi-step reasoning task. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, *Findings of the Association for Computational Linguistics: ACL 2024*, pages 4082–4102, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/ 2024.findings-acl.242. URL <https://aclanthology.org/2024.findings-acl.242/>.

<span id="page-8-4"></span>Bart Bussmann, Patrick Leask, and Neel Nanda. Batchtopk sparse autoencoders. *arXiv preprint arXiv:2412.06410*, 2024.

<span id="page-8-6"></span>Bart Bussmann, Noa Nabeshima, Adam Karvonen, and Neel Nanda. Learning multi-level features with matryoshka sparse autoencoders. *arXiv preprint arXiv:2503.17547*, 2025.

<span id="page-8-7"></span>David Chanin, James Wilken-Smith, Tomáš Dulka, Hardik Bhatnagar, Satvik Golechha, and Joseph Bloom. A is for absorption: Studying feature splitting and absorption in sparse autoencoders. *arXiv preprint arXiv:2409.14507*, 2024.

<span id="page-8-8"></span>Corinna Cortes and Vladimir Vapnik. Support-vector networks. *Machine learning*, 20(3):273–297, 1995.

<span id="page-8-5"></span>Valérie Costa, Thomas Fel, Ekdeep Singh Lubana, Bahareh Tolooshams, and Demba Ba. Evaluating sparse autoencoders: From shallow design to matching pursuit. *arXiv preprint arXiv:2506.05239*, 2025.

<span id="page-8-1"></span>Róbert Csordás, Christopher Potts, Christopher D Manning, and Atticus Geiger. Recurrent neural networks learn to store and generate sequences using non-linear representations. In Yonatan Belinkov, Najoung Kim, Jaap Jumelet, Hosein Mohebbi, Aaron Mueller, and Hanjie Chen, editors, *Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP*, pages 248–262, Miami, Florida, US, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.blackboxnlp-1.17. URL <https://aclanthology.org/2024.blackboxnlp-1.17/>.

- <span id="page-9-9"></span>Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. *ICLR*, 2024.
- <span id="page-9-16"></span>Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. A mathematical framework for transformer circuits. *Transformer Circuits Thread*, 2021. https://transformer-circuits.pub/2021/framework/index.html.
- <span id="page-9-5"></span>Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, et al. Toy models of superposition. *arXiv preprint arXiv:2209.10652*, 2022.
- <span id="page-9-17"></span>Thomas Fel, Ekdeep Singh Lubana, Jacob S Prince, Matthew Kowal, Victor Boutin, Isabel Papadimitriou, Binxu Wang, Martin Wattenberg, Demba Ba, and Talia Konkle. Archetypal sae: Adaptive and stable dictionary learning for concept extraction in large vision models. *arXiv preprint arXiv:2502.12892*, 2025.
- <span id="page-9-0"></span>Jiahai Feng and Jacob Steinhardt. How do language models bind entities in context? *arXiv preprint arXiv:2310.17191*, 2023.
- <span id="page-9-10"></span>Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. Scaling and evaluating sparse autoencoders. *arXiv preprint arXiv:2406.04093*, 2024.
- <span id="page-9-2"></span>Pentti Kanerva. Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors. *Cognitive Computation*, 1(2):139–159, 2009. doi: 10.1007/ s12559-009-9009-8.
- <span id="page-9-15"></span>Patrick Leask, Bart Bussmann, Michael Pearce, Joseph Bloom, Curt Tigges, Noura Al Moubayed, Lee Sharkey, and Neel Nanda. Sparse autoencoders do not find canonical units of analysis. *arXiv preprint arXiv:2502.04878*, 2025a.
- <span id="page-9-12"></span>Patrick Leask, Neel Nanda, and Noura Al Moubayed. Inference-time decomposition of activations (itda): A scalable approach to interpreting large language models. *arXiv preprint arXiv:2505.17769*, 2025b.
- <span id="page-9-13"></span>Samuel Marks, Johannes Treutlein, Trenton Bricken, Jack Lindsey, Jonathan Marcus, Siddharth Mishra-Sharma, Daniel Ziegler, Emmanuel Ameisen, Joshua Batson, Tim Belonax, et al. Auditing language models for hidden objectives. *arXiv preprint arXiv:2503.10965*, 2025.
- <span id="page-9-14"></span>Rajiv Movva, Kenny Peng, Nikhil Garg, Jon Kleinberg, and Emma Pierson. Sparse autoencoders for hypothesis generation. *arXiv preprint arXiv:2502.04382*, 2025.
- <span id="page-9-4"></span>Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability. *arXiv preprint arXiv:2301.05217*, 2023.
- <span id="page-9-3"></span>Nostalgebraist. interpreting gpt: the logit lens. AI Alignment Forum, August 2020. URL [https://](https://www.alignmentforum.org/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) [www.alignmentforum.org/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens](https://www.alignmentforum.org/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens). Accessed 2025-10-20.
- <span id="page-9-7"></span>Gonçalo Paulo, Alex Mallen, Caden Juang, and Nora Belrose. Automatically interpreting millions of features in large language models. *arXiv preprint arXiv:2410.13928*, 2024.
- <span id="page-9-1"></span>Tony A. Plate. Holographic reduced representations. *IEEE Transactions on Neural Networks*, 6(3):623–641, 1995. doi: 10.1109/72.377968.
- <span id="page-9-8"></span>Nikhil Prakash, Tamar Rott Shaham, Tal Haklay, Yonatan Belinkov, and David Bau. Fine-tuning enhances existing mechanisms: A case study on entity tracking. In *Proceedings of the 2024 International Conference on Learning Representations*, 2024. arXiv:2402.14811.
- <span id="page-9-6"></span>Lucia Quirke, Stepan Shabalin, and Nora Belrose. Binary sparse coding for interpretability. *arXiv preprint arXiv:2509.25596*, 2025.
- <span id="page-9-18"></span>Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. *OpenAI blog*, 1(8):9, 2019.
- <span id="page-9-11"></span>Senthooran Rajamanoharan, Tom Lieberum, Nicolas Sonnerat, Arthur Conmy, Vikrant Varma, János Kramár, and Neel Nanda. Jumping ahead: Improving reconstruction fidelity with jumprelu sparse autoencoders. *arXiv preprint arXiv:2407.14435*, 2024.

<span id="page-10-0"></span>Paul Smolensky. Tensor product variable binding and the representation of symbolic structures in connectionist systems. *Artificial Intelligence*, 46(1-2):159–216, 1990. doi: 10.1016/0004-3702(90)90007-M.

<span id="page-10-4"></span>Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568:127063, 2024.

<span id="page-10-1"></span>Martin Wattenberg and Fernanda Viégas. Relational composition in neural networks: A survey and call to action. In *ICML 2024 Workshop on Mechanistic Interpretability*, 2024. URL [https://openreview.net/forum?](https://openreview.net/forum?id=zzCEiUIPk9) [id=zzCEiUIPk9](https://openreview.net/forum?id=zzCEiUIPk9).

# <span id="page-10-2"></span>A Parameter Grid Search

<span id="page-10-3"></span>Table 4: Grid search results where nlayers = 2 and nheads = 1 are fixed. F = False, T = True. We write LN for LayerNorm, and W<sup>V</sup> and W<sup>O</sup> are the value and output matrices respectively, where False means it was frozen to the identity matrix during training. We additionally ablate the residual stream dimension dmodel in the second part of the table. Accuracy is reported as the mean validation accuracy over three independent training runs, though we note this is insufficient for statistical significance testing.

| dmodel | LN | Bias | WV | WO | Accuracy |
|--------|----|------|----|----|----------|
| 64     | F  | F    | T  | T  | 0.6207   |
| 64     | F  | T    | T  | T  | 0.4983   |
| 64     | T  | F    | T  | T  | 0.7709   |
| 64     | T  | T    | T  | T  | 0.7744   |
| 64     | F  | F    | T  | F  | 0.7513   |
| 64     | F  | T    | T  | F  | 0.7552   |
| 64     | T  | F    | T  | F  | 0.9089   |
| 64     | T  | T    | T  | F  | 0.9202   |
| 64     | F  | F    | F  | T  | 0.7481   |
| 64     | F  | T    | F  | T  | 0.7560   |
| 64     | T  | F    | F  | T  | 0.9162   |
| 64     | T  | T    | F  | T  | 0.7636   |
| 64     | F  | F    | F  | F  | 0.9180   |
| 64     | F  | T    | F  | F  | 0.7487   |
| 64     | T  | F    | F  | F  | 0.9215   |
| 64     | T  | T    | F  | F  | 0.8935   |
| 128    | F  | F    | F  | F  | 0.9192   |
| 64     | F  | F    | F  | F  | 0.9180   |
| 32     | F  | F    | F  | F  | 0.6074   |
| 8      | F  | F    | F  | F  | 0.3148   |

Our parameter grid search is shown in Table [4.](#page-10-3) We find that we can ablate the attention bias and LayerNorm and freeze the attention value and output matrices to the identity without any significant drop in performance.

# <span id="page-11-1"></span><span id="page-11-0"></span>B Further Attention Ablations

![](_page_11_Figure_1.jpeg)

Figure 3: The attention patterns vary significantly depending on input. These histograms show the variation in the attention weights for SEP in layer 1 (top row) and for o<sup>2</sup> to SEP in layer 2 (bottom row).

<span id="page-11-2"></span>Figure [3](#page-11-1) shows the distribution in the attention pattern from SEP to d<sup>i</sup> in layer 1, and from o<sup>2</sup> to SEP in layer 2. We see that, despite the syntax of the input being fixed, the attention pattern varies significantly with the inputs. In particular, fixing the attention pattern to the mean attention pattern over the test set reduces accuracy to 45%.

![](_page_11_Figure_4.jpeg)

Figure 4: This graph shows the correlation between SEP's attention to both d<sup>1</sup> and d<sup>2</sup> in layer 1, and validation accuracy. The model fails when SEP attends to d<sup>1</sup> and d<sup>2</sup> by roughly the same amount (near the diagonal).

Figure 4 plots the attention scores for SEP to  $d_i$  in layer 1, and we find that similar attention patterns results in misclassification. This corroborates the finding that using the mean attention pattern harms performance, as the mean pattern lies within this error zone.

### <span id="page-12-0"></span>**C** Output Derivation

Let  $r_z^{L_i}$  be the residual at position of token  $z \in \{d_1, d_2, s, o_1, o_2\}$  after layer i.

#### Layer 1.

$$r_s^{L_1} = \alpha_{s \to d_1} (E_{d_1} + P_{d_1}) + \alpha_{s \to d_2} (E_{d_2} + P_{d_2}) + E_s + P_s$$
 (8)

$$\mathbf{r}_{o_i}^{L_1} = \mathbf{E}_m + \mathbf{P}_{o_i}, \quad i \in \{1, 2\}.$$
 (9)

Note that we use our custom mask (outlined in Section 3) to prevent positions 4 and 5 from attending to anything in layer 1.

#### Laver 2.

$$\mathbf{r}_{o_1}^{L_2} = \mathbf{r}_{o_1}^{L_1} + \beta_{o_1 \to s} \, \mathbf{r}_s^{L_1} \tag{10}$$

$$\mathbf{r}_{o_1} - \mathbf{r}_{o_1} + \beta_{o_1 \to s} \mathbf{r}_s$$

$$\mathbf{r}_{o_2}^{L_2} = \mathbf{r}_{o_2}^{L_1} + \beta_{o_2 \to s} \mathbf{r}_s^{L_1} + \beta_{o_2 \to o_1} \mathbf{r}_{o_1}^{L_1}$$
(11)

#### Output logit at position 4 $(o_1)$ .

$$\ell_{o_1} = r_{o_1}^{L_2} U = \left[ r_{o_1}^{L_1} + \beta_{o_1 \to s} r_s^{L_1} \right] U$$
 (12)

$$\ell_{o_1} = [E_m + P_{o_1} + \beta_{o_1 \to s} (\alpha_{s \to d_1} (E_{d_1} + P_{d_1}) + \alpha_{s \to d_2} (E_{d_2} + P_{d_2}) + E_s + P_s)] U$$
(13)

### <span id="page-12-1"></span>Output logit at position 5 $(o_2)$ .

$$\ell_{o_2} = r_{o_2}^{L_2} U = \left[ r_{o_2}^{L_1} + \beta_{o_2 \to s} \, r_s^{L_1} + \beta_{o_2 \to o_1} \, r_{o_1}^{L_1} \right] U \tag{14}$$

$$\ell_{o_2} = \left[ \boldsymbol{E}_m + \boldsymbol{P}_{o_2} + \beta_{o_2 \to s} \left( \alpha_{s \to d_1} (\boldsymbol{E}_{d_1} + \boldsymbol{P}_{d_1}) + \alpha_{s \to d_2} (\boldsymbol{E}_{d_2} + \boldsymbol{P}_{d_2}) + \boldsymbol{E}_s + \boldsymbol{P}_s \right) + \beta_{o_2 \to o_1} (\boldsymbol{E}_m + \boldsymbol{P}_{o_1}) \right] \boldsymbol{U}$$

$$(15)$$

<span id="page-12-2"></span>**Simplification.** Let  $S = E_s + P_s$ . Furthermore, since  $o_1$  can only ever attend to s, we have  $\beta_{o_1 \to s} = 1$ . Since  $o_2$  can only attend to s or  $o_1$ , we have  $\beta_{o_2 \to s} + \beta_{o_2 \to o_1} = 1$ . Therefore, we can simplify the equations (13) and (15) to get equations (1) and (2) in Section 4.2.