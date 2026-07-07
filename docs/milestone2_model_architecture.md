# Milestone 2 Model Architecture Notes

This document explains the implemented Milestone 2 baseline architecture for few-shot Generalized Referring Expression Comprehension on gRefCOCO.

---

## 1. Task Formulation

For each image-expression pair, the model receives:

$$
(I, T),
$$

where $I$ is an image and $T$ is a natural-language referring expression. Unlike standard referring expression comprehension, the correct output is a set of boxes:

$$
Y \subseteq B,
$$

where $B = \{b_1, b_2, \ldots, b_N\}$ is the candidate-box set. The target set $Y$ may contain zero, one, or multiple boxes:

$$
Y = \emptyset, \quad |Y| = 1, \quad \text{or} \quad |Y| > 1.
$$

The baseline therefore needs two capabilities:

1. determine which candidate boxes match the expression;
2. determine how many boxes should be returned.

---

## 2. Candidate Pool

The Milestone 2 baseline uses a diagnostic COCO-instance candidate pool. For each image-expression sample, all COCO instance annotations in the image are used as candidate boxes:

$$
B = \{b_1, b_2, \ldots, b_N\}.
$$

This is not a realistic detector-proposal setup. It is a controlled diagnostic setup that removes proposal-recall failure from the first baseline experiment. Since the candidate boxes come from COCO annotations, this baseline mainly tests region-text representation quality, candidate scoring, and cardinality prediction.

For each candidate box $b_i$, the preprocessing pipeline stores:

- raw box coordinates in $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ format;
- normalized coordinates;
- a binary candidate membership label;
- the expression-level count class.

The normalized coordinates are:

$$
\tilde{b}_i =
\left[
\frac{x_{\min}}{W},
\frac{y_{\min}}{H},
\frac{x_{\max}}{W},
\frac{y_{\max}}{H}
\right],
$$

where $W$ and $H$ are the original image width and height.

---

## 3. Frozen CLIP Feature Extraction

The baseline uses a frozen CLIP ViT-B/32 encoder. Encoder parameters are not updated during Milestone 2 training.

For each expression $T$, the text encoder produces:

$$
f_T \in \mathbb{R}^{512}.
$$

For each candidate region $b_i$, the region crop is encoded by the CLIP image encoder:

$$
f_i \in \mathbb{R}^{512}.
$$

The feature extraction script also computes a region-text cosine similarity score:

$$
s_i = \cos(f_i, f_T)
= \frac{f_i^\top f_T}{\|f_i\|_2 \|f_T\|_2}.
$$

These frozen features are cached before training. The training script loads feature caches instead of recomputing CLIP features every epoch.

---

## 4. Candidate-Level Input Vector

For each candidate box, the trainable head receives a concatenated feature vector:

$$
x_i = [f_i; f_T; s_i; \tilde{b}_i].
$$

With CLIP ViT-B/32, the dimensionality is:

$$
\dim(x_i) = 512 + 512 + 1 + 4 = 1029.
$$

The four input components have different roles:

| Component | Dimension | Purpose |
|---|---:|---|
| Candidate region feature $f_i$ | 512 | Encodes visual appearance of the candidate region. |
| Text feature $f_T$ | 512 | Encodes the referring expression. |
| Cosine similarity $s_i$ | 1 | Provides an explicit CLIP region-text matching score. |
| Normalized box $\tilde{b}_i$ | 4 | Provides spatial information such as location and relative size. |

The normalized box feature is important because CLIP region-text similarity alone may not reliably capture spatial phrases such as “left,” “right,” “middle,” or “in the back.”

---

## 5. Candidate MLP and Membership Head

Each candidate vector $x_i$ is passed through a shared two-layer MLP:

$$
h_i = \operatorname{MLP}_{\text{cand}}(x_i),
$$

where:

```text
Linear(1029, 256)
ReLU
Dropout(0.1)
Linear(256, 256)
ReLU
Dropout(0.1)
```

The membership head then predicts one logit per candidate:

$$
m_i = w_m^\top h_i + b_m.
$$

The membership logit $m_i$ scores how likely candidate $b_i$ is to belong to the referred target set $Y$.

During training, candidate membership labels are binary:

$$
y_i =
\begin{cases}
1, & b_i \in Y, \\
0, & b_i \notin Y.
\end{cases}
$$

For no-target samples, the target set is empty, so every candidate label is negative:

$$
Y = \emptyset
\quad \Rightarrow \quad
 y_1 = y_2 = \cdots = y_N = 0.
$$

The membership loss is binary cross-entropy with logits:

$$
L_{\text{membership}}
=
\frac{1}{N}\sum_{i=1}^{N}
\operatorname{BCEWithLogits}(m_i, y_i).
$$

---

## 6. Pooled Cardinality Head

Candidate membership logits alone are not enough for generalized referring expression comprehension. A standard top-1 model would always select a box even when the correct answer is empty. Therefore, this baseline includes a separate cardinality head.

After candidate hidden states are computed, the model mean-pools them:

$$
\bar{h} = \frac{1}{N}\sum_{i=1}^{N} h_i.
$$

The pooled feature $\bar{h}$ summarizes the candidate set in the context of the expression. The cardinality head predicts four count logits:

$$
z = \operatorname{MLP}_{\text{count}}(\bar{h}),
\quad z \in \mathbb{R}^{4}.
$$

The count classes are:

| Count class | Meaning |
|---:|---|
| 0 | no target / empty set |
| 1 | one target |
| 2 | two targets |
| 3 | three or more targets |

The count head architecture is:

```text
Linear(256, 256)
ReLU
Dropout(0.1)
Linear(256, 4)
```

The cardinality loss is cross-entropy:

$$
L_{\text{cardinality}} = \operatorname{CE}(z, c),
$$

where $c$ is the ground-truth count class.

---

## 7. Empty-Set Prediction Mechanism

The model can return the empty set because inference is gated by the predicted count class.

First, the model predicts:

$$
\hat{c} = \arg\max_{c \in \{0,1,2,3\}} z_c.
$$

Then it maps the predicted count class to the number of boxes to return:

$$
k =
\begin{cases}
0, & \hat{c}=0, \\
1, & \hat{c}=1, \\
2, & \hat{c}=2, \\
3, & \hat{c}=3.
\end{cases}
$$

Finally, the model selects the top-$k$ candidates by membership logits:

$$
\hat{Y} = \operatorname{TopK}(\{m_i\}_{i=1}^{N}, k).
$$

When the cardinality head predicts class 0, then $k=0$, and the model returns:

$$
\hat{Y} = \emptyset.
$$

This is the key difference between the implemented generalized baseline and a standard single-target REC baseline. The model is not forced to ground every expression to the highest-scoring box.

---

## 8. Full Training Objective

The full training objective is:

$$
L = L_{\text{membership}} + \lambda L_{\text{cardinality}}.
$$

For the selected Milestone 2 model:

$$
\lambda = 1.0.
$$

The final selected model uses weighted cross-entropy for the cardinality head:

$$
L_{\text{cardinality}}
=
\operatorname{WeightedCE}(z, c).
$$

The selected count-class weights are:

$$
[w_0, w_1, w_2, w_3] = [15.0, 1.0, 1.5, 2.0].
$$

Here, $w_0=15.0$ assigns higher loss to errors on no-target examples. This was selected empirically after validation diagnosis showed that the unweighted model almost never predicted count class 0 and produced a very high false grounding rate. The weighted setting improves no-target rejection while preserving useful multi-target behavior.

The weight value should be interpreted as a validation-selected calibration hyperparameter for the current 1% few-shot baseline, not as a theoretically optimal constant.

---

## 9. Figure Placeholders

The report can include the following figures if generated locally.

> **Figure 1. Baseline architecture.**  
> Suggested path: `docs/figures/figure1_baseline_architecture.png`  
> Content: frozen CLIP feature extraction, candidate feature concatenation, shared candidate MLP, membership head, mean pooling, cardinality head, and count-gated top-$k$ selection.

> **Figure 2. Count-gated empty-set inference.**  
> Suggested path: `docs/figures/figure2_empty_set_inference.png`  
> Content: show that count class 0 maps to $k=0$ and therefore returns $\emptyset$.

> **Figure 3. Correct no-target rejection.**  
> Suggested path: `docs/figures/figure3_correct_no_target.png`  
> Content: qualitative example where both ground truth and prediction are empty.

> **Figure 4. False grounding failure.**  
> Suggested path: `docs/figures/figure4_false_grounding.png`  
> Content: qualitative example where the ground-truth set is empty but the model selects one or more candidate boxes.
