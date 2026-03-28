# Project 1: Dimension Reduction in Classification
## Mathematics for Machine Learning (MMAC)


The notebook is organized into the following sections:

### 1. Data Loading and Preliminary Analysis

### 2. Train/Test Split


### 3. Principal Component Analysis (PCA)
- Applied to 11 numerical features after standardization
- Number of components selected via cumulative variance explained
  and scree plot
- Interpretation of PC1 and PC2 loadings
- Feature transformations justified prior to PCA

### 4. Kernel PCA
- Applied with RBF, polynomial, and linear kernels
- Hyperparameter tuning (sigma for RBF, degree for polynomial)
- Number of components selected by variance captured in feature space

### 5. Information-Theoretic Feature Selection
Eight forward greedy feature selection methods applied to training data.

Methods via praznik package (handle discretization internally):
- MIM    : Mutual Information Maximization — pure relevance, no
           redundancy correction
- mRMR   : Minimum Redundancy Maximum Relevance — average redundancy
           penalty over selected set
- CMIM   : Conditional MI Maximization — min over selected set of
           conditional relevance
- JMI    : Joint Mutual Information — joint relevance with selected set
- DISR   : Double Input Symmetrical Relevance — replaces CIFE, which is
           unavailable in current praznik versions; both belong to the
           Brown et al. (2012) unifying framework

Manually implemented methods using infotheo MI tables:
- MIFS    : sum-based redundancy penalty (Battiti, 1994), beta = 1
- maxMIFS : max-based redundancy penalty (Pascoal et al., 2017)
- DMIM    : max redundancy penalty plus max complementarity bonus,
            i.e. MI(Xi, Xs | C) (Macedo et al., 2022)

Three MI tables are pre-computed once from discretized training data
(equal-frequency binning, ceiling(sqrt(n)) bins) and reused by all
manual methods:
- mi_feature_class  : MI(Xi, C) — relevance vector
- mi_ff             : MI(Xi, Xj) — pairwise redundancy matrix (p x p)
- mi_ff_given_c     : MI(Xi, Xj | C) — pairwise complementarity matrix
                      computed as H(Xi|C) - H(Xi|Xj,C)

Results are aggregated into a full ranking table (all 11 features x 8
methods) and a binary heatmap showing consensus across methods for the
top-k selected features. The cutoff k is chosen by inspecting the MIM
bar chart for an elbow in MI scores.

Important naming convention:
- praznik methods return $selection
- manual implementations (MIFS, maxMIFS, DMIM) return $selected

### 6. Classification

Two classifiers evaluated across all dimensionality reduction strategies:

6.1 Random Forest
- ntree = 500, mtry = floor(sqrt(p)), nodesize = 1
- Trained on: original dataset, PCA scores, Kernel PCA scores, and each
  of the 8 feature-selection subsets
- OOB error used as internal validation estimate
- Variable importance (MeanDecreaseAccuracy) compared against MIM ranks
  via Spearman rank correlation

6.2 k-Nearest Neighbour (kNN)
- k = 5 neighbours
- Trained on: original dataset, PCA scores, Kernel PCA scores, and each
  of the 8 feature-selection subsets

Performance metrics (per project specification, Question 6c):
- Accuracy          : fraction of correctly classified test observations
- Macro Recall      : arithmetic mean of per-class recall
- Macro Precision   : arithmetic mean of per-class precision
- Macro F1          : arithmetic mean of per-class F1 (harmonic mean of
                      recall and precision)

Results compared across all strategies in a unified Macro F1 bar chart
and a faceted per-metric chart for feature selection methods. Best and
worst feature selection methods identified by Macro F1 and visualized
via fourfold confusion matrix plots.

### 7. Causal Analysis
Motivated by Suter et al. (2019) — Robustly Disentangled Causal
Mechanisms.

7.1 DAG — Causal Graph Discovery
- The PC algorithm (pcalg::pc) is applied to the standardized numerical
  training features using a Gaussian conditional independence test
  (gaussCItest) at significance level alpha = 0.01
- Output is a CPDAG (completed partially directed acyclic graph)
  representing the Markov equivalence class of the underlying causal
  structure among features

7.2 Do-Calculus — Interventional Predictions
- Inspired by Pearl's do-calculus and the backdoor adjustment formula
  from Proposition 1(f) of Suter et al. (2019)
- The causal effect of intervening on speechiness on genre prediction
  is estimated via backdoor adjustment over a set Z identified from
  the PC graph (e.g. energy, acousticness)
- Operationally: for each bin of speechiness values, the training data
  is modified by setting speechiness to the bin midpoint (simulating
  do(speechiness = t)), predictions are obtained from the Random Forest,
  and the marginal average over the adjustment set is computed
- Output: a do-curve showing p(Rap | do(speechiness = t)) across the
  range of speechiness values

7.3 Interventional Robustness Score (IRS)
- A proxy implementation of the IRS metric from Suter et al. (2019)
- For each feature selection method's top-k subset, the top-ranked
  feature is treated as the relevant factor GI and the remaining
  selected features are treated as nuisance factors GJ
- For each nuisance feature, the classifier output (p(Rap)) is evaluated
  across a grid of 10 quantile values while the relevant feature is held
  fixed at its median — this approximates the Post Interventional
  Disagreement (PIDA) from Definition 2 of Suter et al.
- The maximum absolute shift in prediction across the grid is taken as
  the MPIDA analog for each nuisance feature
- The IRS proxy is computed as:
    IRS = 1 - mean(MPIDA per nuisance) / sd(baseline predictions)
  where the denominator normalizes by the natural variation in the
  classifier output, analogous to the normalization in Definition 3
- Output: IRS scores ranked across all 8 feature selection methods,
  indicating which method's selected subset leads to the most robust
  classification with respect to nuisance feature interventions

---


---

## References

[1] G. Brown, A. Pocock, M.-J. Zhao, M. Luján. Conditional likelihood
    maximisation: A unifying framework for information theoretic feature
    selection. JMLR, 13, 27–66, 2012.

[2] Prediction of music genre [Dataset]. Kaggle Repository.
    https://www.kaggle.com/datasets/vicsuperman/prediction-of-music-genre/data

[3] F. Macedo, R. Valadas, E. Carrasquinha, M. R. Oliveira, A. Pacheco.
    Feature selection using Decomposed Mutual Information Maximization.
    Neurocomputing, 513, 215–232, 2022.

[4] C. Pascoal, M. R. Oliveira, A. Pacheco, R. Valadas. Theoretical
    evaluation of feature selection methods based on mutual information.
    Neurocomputing, 226(1), 168–181, 2017.

[5] R. Suter, D. Miladinovic, B. Schölkopf, S. Bauer. Robustly
    Disentangled Causal Mechanisms: Validating Deep Representations for
    Interventional Robustness. ICML 2019, PMLR 97.
