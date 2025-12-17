# EEG-based PD Subtyping

This repository implements a **data-driven EEG-based clustering framework** for identifying
**robust neurophysiological subtypes of Parkinson’s disease (PD)** across **resting-state and active walking conditions**.

The project accompanies the manuscript:

> **EEG-Based Clustering Reveals Robust Neurophysiological Subtypes in Parkinson’s Disease**  
> Daniel Vered et al.

---

## Scientific motivation

Clinical subtyping of Parkinson’s disease is typically based on behavioral rating scales
(e.g., MDS-UPDRS), which are subjective, unstable over time, and weakly linked to underlying
pathophysiology. This project aims to **derive PD subtypes directly from electrophysiological signals**, using
scalp EEG features that reflect **spectral slowing, aperiodic activity, and signal complexity**.

---

## Repository structure

```text
.
├── EEGLABIO.py            # Loading and handling EEGLAB-exported EEG
├── FeatureExtraction.py  # Feature extraction from raw EED data
├── ClusteringUtilz.py    # Clustering, validation, stability analysis etc.
├── Models/               # Saved models and outputs
├── EEG Data/             # EEG features data
├── *.ipynb               # Analysis notebooks
```

---

## Dataset

- **Participants**
  - 116 Parkinson’s disease patients (PD)
  - 30 healthy controls (HC)
- **EEG acquisition**
  - 64-channel scalp EEG (10–20 system)
  - Mutiple behvioral states - resting-state & active-walking.
- **Clinical data**
  - Motor scores (based on MDS-UPDRS)
  - Cognitive scores (MoCA, CTT1, CTT2)
  - Gait speed (single-task, dual-task)
  - Genetic status (LRRK2, GBA)

> Raw EEG data are **not publicly shared**.  
> The EEG features that were used for the downstream analysis are publicly shared.

---

## EEG preprocessing

Preprocessing was performed in **EEGLAB (MATLAB)** and includes:

- Band-pass filtering (0.5–40 Hz)
- Channel rejection
- Average re-referencing
- Independent Component Analysis (ICA)
- Artifact correction

> Preprocessing code is maintained separately  
> see [`EEG-Preprocessing-Toolkit`](https://github.com/RonMon1994/EEG-Preprocessing-Toolkit/tree/main/src).

---

## Feature engineering

Features were extracted from preprocessed EEG and **aggregated within 13 anatomical regions**
(frontal, central, parietal, temporal, occipital × left/midline/right).

### Spectral features
- Relative power in canonical bands including Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz) and Gamma (30-40Hz).

### Aperiodic features
- Power spectrum parameterization using [**FOOOF**](https://doi.org/10.1038/s41593-020-00744-x)

### Temporal complexity
- **Permutation Entropy (PermEn)**
- **Lempel–Ziv Complexity (LZC)**

All features were:
- Z-score standardized across participants and channels
- Mean-imputed for missing values

---

---

## Dimensionality reduction

Three approaches were evaluated:

- No dimensionality reduction (“vanilla”)
- Principal Component Analysis (PCA)
- **Uniform Manifold Approximation and Projection (UMAP)**

> Across all clustering algorithms, **UMAP consistently outperformed PCA and vanilla features**.

---

## Clustering algorithms

Three unsupervised clustering methods were systematically evaluated:

- **K-Means**
- Gaussian Mixture Models (**GMM**)
- Density-Based Spatial Clustering for Applications with Noise (**DBSCAN**)

### Model selection & validation

- Hyperparameters tuned via **leave-one-out resampling**
- Internal validity metrics included Silhouette score, Calinski–Harabasz index (CH), Davies–Bouldin index (DBI).

- **Robustness & stability**
  - Within-model stability: Adjusted Rand Index (ARI)
  - Between-model agreement: pairwise ARI

> The **K-Means + UMAP** solution showed the strongest overall performance and stability.
> 
---

## Identified PD subtypes

The optimal solution revealed **three robust PD subtypes**:

- **PD-0** – intermediate neurophysiological profile
- **PD-1** – pronounced EEG slowing and reduced signal complexity
- **PD-2** – preserved alpha activity, higher complexity, and better cognition

### Key findings

- Clusters **did not differ** in motor severity or gait speed
- Clusters **differed significantly** in:
  - Cognitive performance (MoCA, CTT)
  - Genetic composition (LRRK2 enrichment in PD-2)
- Neurophysiological differences were **global**, not region-specific

Importantly, **PD-0 and PD-1 showed similar cognitive impairment but distinct EEG signatures**, suggesting **different underlying pathophysiological mechanisms**.

---

## Feature importance analysis

- Permutation-based importance analysis
- Metrics included Silhouette drop (ΔSilhouette), and stability loss (quantifed using ARI).
- Findings:
  - Spectral power features (theta, alpha, beta) contributed most
  - Signal complexity had moderate importance
  - Aperiodic exponent contributed least
  - Sitting condition showed stronger feature separability than walking
