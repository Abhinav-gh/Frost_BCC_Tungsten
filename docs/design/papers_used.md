# Papers, code and data used in Frost

Every external source the project has read or relied on, grouped by what it is
used for, with links. `idea_extension_frost.tex` cites most of these, and
`plan.md` points here.

**Tags.**
- **cited**: cited in the design note.
- **read**: informed the design but is not cited.
- **baseline**: something the paper must beat or compare against.
- **data**: a dataset.
- **code**: software.

---

## 1. Models, kernels and engines Frost runs on

| Source | Link | Used for |
|---|---|---|
| Batatia, Kovács, Simm, Ortner, Csányi, *MACE: Higher order equivariant message passing neural networks for fast and accurate force fields*, NeurIPS 2022 | [arXiv:2206.07697](https://arxiv.org/abs/2206.07697) · [code](https://github.com/ACEsuit/mace) | Teacher architecture; the v1 caching target. **cited** |
| Batatia et al., *A foundation model for atomistic materials chemistry* (MACE-MP-0), 2023 | [arXiv:2401.00096](https://arxiv.org/abs/2401.00096) | Teacher (medium) and off-the-shelf students (small, large) in E1–E5. **cited** |
| Batzner et al., *E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials* (NequIP), Nat. Commun. 2022 | [arXiv:2101.03164](https://arxiv.org/abs/2101.03164) · [code](https://github.com/mir-group/nequip) | Teacher family, v1 environment. **read** |
| Musaelian et al., *Learning local equivariant representations for large-scale atomistic dynamics* (Allegro), Nat. Commun. 2023 | [arXiv:2204.05249](https://arxiv.org/abs/2204.05249) · [code](https://github.com/mir-group/allegro) | v1 Gate 3 closure measurement; strictly local teacher for block commits. **cited** |
| Musaelian, Johansson, Batzner, Kozinsky, *Scaling the leading accuracy of deep equivariant models to biomolecular simulations of realistic size*, SC 2023 | [arXiv:2304.10061](https://arxiv.org/abs/2304.10061) | Per-call scaling context. **cited** |
| Park et al., *Scalable parallel algorithm for graph neural network interatomic potentials in molecular dynamics simulations* (SevenNet), JCTC 2024 | [arXiv:2402.03789](https://arxiv.org/abs/2402.03789) · [JCTC](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00190) | Teacher option. **read** |
| Geiger and Smidt, *e3nn: Euclidean Neural Networks*, 2022 | [arXiv:2207.09453](https://arxiv.org/abs/2207.09453) | Tensor-product library under MACE 0.3.16. **code** |
| NVIDIA, *cuEquivariance* | [docs](https://docs.nvidia.com/cuda/cuequivariance/) · [MACE tutorial](https://docs.nvidia.com/cuda/cuequivariance/tutorials/pytorch/MACE.html) | B0 per-call kernels (ops wheel repaired in week 1). **cited, baseline** |
| Bharadwaj, Glover, Buluç, Demmel, *An efficient sparse kernel generator for O(3)-equivariant deep networks* (OpenEquivariance), ACDA 2025 | [arXiv:2501.13986](https://arxiv.org/abs/2501.13986) · [code](https://github.com/PASSIONLab/OpenEquivariance) | Per-call kernel baseline. **cited, baseline** |
| Lee et al., *FlashTP: Fused, sparsity-aware tensor product for machine learning interatomic potentials*, ICML 2025 | [PMLR](https://proceedings.mlr.press/v267/lee25l.html) · [code](https://github.com/SNU-ARC/flashTP) | Per-call kernel baseline. **cited, baseline** |
| Cohen et al., *TorchSim: An efficient atomistic simulation engine in PyTorch*, 2025 | [arXiv:2508.06628](https://arxiv.org/abs/2508.06628) · [code](https://github.com/TorchSim/torch-sim) | Batched engine; B0 loop style. **cited, baseline** |
| vesin neighbour lists | [code](https://github.com/Luthaf/vesin) | Verlet lists shared by teacher and student (`frost.neighbours`). **code** |
| Rhodes et al., *Orb-v3: atomistic simulation at scale*, 2025 | [arXiv:2504.06231](https://arxiv.org/abs/2504.06231) | Optional direct-force teacher. **cited** |
| NVIDIA, *ALCHEMI Toolkit-Ops* (`nvalchemi-toolkit-ops` 0.4.1, installed as a TorchSim dependency) | [code](https://github.com/NVIDIA/nvalchemi-toolkit-ops) · [docs](https://nvidia.github.io/nvalchemi-toolkit-ops/) | GPU (Warp) cell-list and naive neighbour lists. Candidate replacement for the CPU vesin rebuilds in the online loop. **code** |
| Benoit, *Speeding up MACE: low-precision tricks for equivariant force fields*, 2025 | [arXiv:2510.23621](https://arxiv.org/abs/2510.23621) | cuEquivariance cuts MACE latency about 3×. Running only the linear layers in BF16/FP16 inside an FP32 model gives about 4× more, with MD observables within run-to-run variability. Consistent with our all-bfloat16 student breaking forces. **read** |
| MACE users and maintainers, *cuEquivariance library has only a slight performance improvement for the inference of MACE models* (discussion #740) | [GitHub](https://github.com/ACEsuit/mace/discussions/740) | Small models gain little from cuEquivariance, and timings need warm-up. Sets expectations for the student's kernels. **read** |
| Chorošajev and Bény, *Sobek: streaming equivariant tensor product convolutions*, 2026 | [arXiv:2607.18074](https://arxiv.org/abs/2607.18074) | Generated-CUDA streaming TP convolutions: 1.2–49.7× faster than OpenEquivariance with up to 99% less peak memory. No code released yet. **read** |
| Bochkarev, Lysogorskiy, Drautz, *Fast contracted Clebsch–Gordan tensor products for equivariant graph neural networks*, 2026 | [arXiv:2605.15073](https://arxiv.org/abs/2605.15073) | O(L³) contracted CG products. Pays off at high l_max, not for l ≤ 2 students. **read** |
| Brunken et al., *Machine learning interatomic potentials: advancing open-source software for efficient and scalable molecular simulation* (mlip, e3j backend), 2026 | [arXiv:2605.22698](https://arxiv.org/abs/2605.22698) | Another equivariant-kernel backend. **read** |

## 2. Systems work on MLIP cost

| Source | Link | Used for |
|---|---|---|
| *TensorMD: Accelerating molecular dynamics with a high-performance machine learning interatomic potential*, SC 2025 | [ACM](https://dl.acm.org/doi/10.1145/3712285.3759844) | Purpose-built MLIP code, 1.88× over DeePMD. **cited** |
| *Smooth dynamic cutoffs for machine learning interatomic potentials*, ICML 2026 | [arXiv:2601.21147](https://arxiv.org/abs/2601.21147) | Per-call sparsity, 2.0×. **cited** |
| *Characterizing machine learning force fields as emerging molecular dynamics workloads on GPUs*, 2026 | [arXiv:2603.04092](https://arxiv.org/abs/2603.04092) | Kernel-launch overhead dominates small systems, which motivates the CUDA-graph student. **cited** |
| Jia et al., *Pushing the limit of molecular dynamics with ab initio accuracy to 100 million atoms with machine learning* (DeePMD), SC 2020 | [arXiv:2005.00223](https://arxiv.org/abs/2005.00223) | Context for TensorMD's baseline. **read** |
| Ghosh, Nayak, Panwar, Basu, *PyGraph: robust compiler support for CUDA graphs in PyTorch*, 2025 | [arXiv:2503.19779](https://arxiv.org/abs/2503.19779) | CUDA graphs against kernel-launch overhead, with cost-benefit selection. Background for graphing the teacher and the inner step. **read** |
| Pennati, Hu, Peng, Müllender, Markidis, *Making room for AI: multi-GPU molecular dynamics with deep potentials in GROMACS*, 2026 | [arXiv:2604.07276](https://arxiv.org/abs/2604.07276) | Domain-decomposed MLIP MD on up to 32 GPUs, with over 90% of wall time in inference. The spatial parallelism a second GPU buys instead. **read** |

## 3. Multiple time stepping and integrators

| Source | Link | Used for |
|---|---|---|
| Tuckerman, Berne, Martyna, *Reversible multiple time scale molecular dynamics* (r-RESPA), J. Chem. Phys. 97, 1990 (1992) | [doi:10.1063/1.463137](https://doi.org/10.1063/1.463137) | The dynamics contract (`frost.integrators.respa_outer`). **cited** |
| García-Archilla, Sanz-Serna, Skeel, *Long-time-step methods for oscillatory differential equations* (MOLLY), SIAM J. Sci. Comput. 20, 930 (1998) | [doi:10.1137/S1064827596313851](https://doi.org/10.1137/S1064827596313851) | Resonance mitigation. **cited** |
| Barth and Schlick, *Overcoming stability limitations in biomolecular dynamics I* (LN), J. Chem. Phys. 109, 1617 (1998) | [AIP](https://pubs.aip.org/aip/jcp/article-abstract/109/5/1617/529768/Overcoming-stability-limitations-in-biomolecular) | Langevin-stabilised MTS. **cited** |
| Leimkuhler, Margul, Tuckerman, *Stochastic, resonance-free multiple time-step algorithm for molecular dynamics with very large time steps* (SIN(R)), Mol. Phys. 111, 3579 (2013) | [arXiv:1307.1167](https://arxiv.org/abs/1307.1167) | 100 fs outer steps; non-exact statistics baseline B2. **read in full (2026-11-14) for tier 3 item 16:** equations, the exact sub-step solutions, XI/XO-RESPA, water settings (L 4, tau 10 fs, gamma 0.1/fs); a sign typo in their Eq. 61 noted; decision recorded in `further_expansions_beyond.md` 7.6. **cited, baseline** |
| *A stochastic, resonance-free multiple time-step algorithm for polarizable models that permits very large time steps*, JCTC 2016 | [doi:10.1021/acs.jctc.6b00188](https://doi.org/10.1021/acs.jctc.6b00188) | SIN(R) generalisation. **read** |
| Kar, Mandal, Thakkur, Meyer, Nair, *Speeding-up hybrid functional based ab initio molecular dynamics using multiple time-stepping and resonance free thermostat*, JCTC 2023 | [arXiv:2309.00651](https://arxiv.org/abs/2309.00651) · [doi:10.1021/acs.jctc.3c00964](https://doi.org/10.1021/acs.jctc.3c00964) | About 30× for AIMD with SIN(R). **cited** |
| *Accelerating QM/MM simulations with the stochastic iso-kinetic Nosé–Hoover MTS integrator in CHARMM* | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13112335/) | SIN(R) in production codes. **read** |
| Mouvet, Browning, Baudin, Liberatore, Rothlisberger, *Machine learning-enhanced multiple time-step ab initio molecular dynamics*, J. Chem. Phys. 163, 184115 (2025) | [doi:10.1063/5.0288262](https://doi.org/10.1063/5.0288262) | ML fast force for AIMD. **cited** |
| Fu, Musaelian, Johansson, Jaakkola, Kozinsky, *Learning interatomic potentials at multiple scales* (MTS-Allegro), 2023 | [arXiv:2310.13756](https://arxiv.org/abs/2310.13756) | Co-trained fast/slow split, about 3×. No public code; rebuilt from the paper on nequip-allegro and validated against their Table 1 (`experiments/e14_baselines/mts_allegro.py`, 2026-11-06/07; T2). **cited, baseline** |
| Cattin, Plé, Adjoua, Gouraud, Lagardère, Piquemal, *Accelerating molecular dynamics simulations with foundation neural network models using multiple time steps and distillation* (DMTS), JPCL 17, 1288 (2026) | [arXiv:2510.06562](https://arxiv.org/abs/2510.06562) · [doi:10.1021/acs.jpclett.5c03720](https://doi.org/10.1021/acs.jpclett.5c03720) | Closest precedent: 3.8× water, 2.9× protein. **cited, baseline B1** |
| Gouraud, Cattin, Plé, Adjoua, Lagardère, Piquemal, *Faster molecular dynamics with neural network potentials via distilled multiple time-stepping and nonconservative forces* (DMTS-NC), JCTC 22, 5957 (2026) | [arXiv:2602.14975](https://arxiv.org/abs/2602.14975) · [doi:10.1021/acs.jctc.6c00653](https://doi.org/10.1021/acs.jctc.6c00653) | Non-conservative, force-only student (equivariant; zero net force per connected component); fixed outer step per system; rewind on a force-difference threshold. v3 figures, all ns/day: +55.7% (water) and +28–32% (proteins) over generic DMTS, +15–16% over fine-tuned DMTS; 2.94–4.31× (water) and 3.09–3.40× (proteins) against 1 fs STS; 3.66–5.64× for distilled MACE-OFF23; diffusion −12.6% at 1–6 fs, −32.6 to −39.2% at 10 fs with HHF. Details: `research_direction.md` §1.1. **cited, baseline** |
| Leimkuhler and Matthews, *Rational construction of stochastic numerical methods for molecular sampling* (BAOAB), Appl. Math. Res. Express 2013, 34 | [AMRX](https://academic.oup.com/amrx/article/2013/1/34/166771) · [arXiv:1203.5428](https://arxiv.org/abs/1203.5428) | The inner Langevin integrator (`frost.integrators`). **cited** |
| Feenstra, Hess, Berendsen, *Improving efficiency of large time-scale molecular dynamics simulations of hydrogen-rich systems* (HMR), J. Comput. Chem. 20, 786 (1999) | [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1096-987X(199906)20:8%3C786::AID-JCC5%3E3.0.CO;2-B) | Mass repartitioning for water. **cited** |
| Lew, Marsden, Ortiz, West, *Asynchronous variational integrators*, Arch. Ration. Mech. Anal. 167, 85 (2003) | [paper page](https://lagrange.mechse.illinois.edu/pubs/LeMaOrWe2003/) | Per-site refresh rates stay symplectic (note §5.4). **cited** |
| Ober-Blöbaum, *Variational multirate integrators*, 2024 | [arXiv:2406.12991](https://arxiv.org/abs/2406.12991) | Multirate structure preservation. **read** |
| Bigi, Chong, Kristiadi, Ceriotti, *FlashMD: long-stride, universal prediction of molecular dynamics*, NeurIPS 2025 | [arXiv:2505.19350](https://arxiv.org/abs/2505.19350) · [code](https://github.com/lab-cosmo/flashmd) | Learned long strides, which give up the reference dynamics. **cited** |
| Morrone, Markland, Ceriotti, Berne, *Efficient multiple time scale molecular dynamics: using colored noise thermostats to stabilize resonances*, J. Chem. Phys. 134, 014103 (2011) | [arXiv:1008.5377](https://arxiv.org/abs/1008.5377) · [doi:10.1063/1.3518369](https://doi.org/10.1063/1.3518369) | The outer step is limited by resonances between fast and slow modes, not by the slow force; flexible water's r-RESPA barrier is near 3 fs, where the held correction couples to the O–H stretch overtone. Colored noise on the fast modes lifts it. Input to the failure loop on water and silica at 16 fs. **read** |
| Ma, Izaguirre, Skeel, *Verlet-I/r-RESPA/Impulse is limited by nonlinear instabilities*, SIAM J. Sci. Comput. 24(6), 1951–1973 (2003) | link to be verified | Impulse r-RESPA is unstable at T/3 (and weakly T/4) of the fastest period, not only T/2. Explains silica's drift jump between k = 8 and 10 fs (plan §1.14). **cited** |
| García-Archilla, Sanz-Serna, Skeel, *Long-time-step methods for oscillatory differential equations*, SIAM J. Sci. Comput. 20(3), 930–963 (1998) | link to be verified | The mollified impulse method: filter the slow force against resonant modes. Candidate for G3's open drift clause. **cited** |
| Horowitz, *A generalized guided Monte Carlo algorithm*, Phys. Lett. B 268, 247–252 (1991) | link to be verified | GHMC: partial momentum refresh. Fails below ~0.9 acceptance because rejections flip the momentum (plan §1.14). **cited** |
| Sohl-Dickstein, Mudigonda, DeWeese, *Hamiltonian Monte Carlo without detailed balance*, ICML (2014) | link to be verified | Look-ahead / extra-chance acceptance that flips momentum only after every chance fails. Basis of `NestedHMC(extra_chances=)`. **used** |
| Campos, Sanz-Serna, *Extra chance generalized hybrid Monte Carlo*, J. Comput. Phys. 281, 365–374 (2015) | link to be verified | Extra chances make GHMC's partial refresh pay at modest acceptance; the P_k recursion implemented and tested in `frost.nested`. **used** |
| Mackenzie, *An improved hybrid Monte Carlo method*, Phys. Lett. B 226, 369–371 (1989) | link to be verified | Randomise trajectory length to break near-periodicity. `NestedHMC(jitter=)`. **used** |
| Bou-Rabee, Sanz-Serna, *Randomized Hamiltonian Monte Carlo*, Ann. Appl. Probab. 27(4), 2159–2194 (2017) | link to be verified | Random durations make HMC robust to periodic returns; with persistent momentum this was the missing piece (0.97× → 2.6×). **used** |
| Madras, Sokal, *The pivot algorithm*, J. Stat. Phys. 50, 109–186 (1988) | link to be verified | Variance of the integrated autocorrelation time, var(τ) ≈ (4M + 2)/N τ², used for every S_stat error bar. **used** |
| *Overcoming instabilities in Verlet-I/r-RESPA with the mollified impulse method* (MOLLY for MD) | [Springer](https://link.springer.com/chapter/10.1007/978-3-642-56080-4_7) | Evaluates the slow force at time-averaged positions; 50% longer outer steps, 350% with mild Langevin damping. A candidate architecture for the resonance failure. **read** |
| Chin, *Dynamical multiple-timestepping methods for overcoming the half-period time step barrier*, 2003 | [arXiv:physics/0308053](https://arxiv.org/abs/physics/0308053) | A different Hamiltonian splitting with a modified mass removes the half-period barrier. **read** |
| *Molecular dynamics with multiple time scales: how to avoid pitfalls*, J. Chem. Theory Comput. (2010) | [doi:10.1021/ct100054k](https://pubs.acs.org/doi/10.1021/ct100054k) | Practical MTS failure modes. **read** |
| *Multiple time step molecular dynamics in the optimized isokinetic ensemble steered with the molecular theory of solvation: accelerating with advanced extrapolation of effective solvation forces* (ASFE) | [PubMed 24387356](https://pubmed.ncbi.nlm.nih.gov/24387356/) | Extrapolates the expensive force between outer steps by least squares over force-coordinate pairs rather than holding it. Precedent for the affine residual carry tested in E7, which measurement refuted. **read** |
| Fu, Wu, Wang, Xie, Keten, Gomez-Bombarelli, Jaakkola, *Forces are not enough: benchmark and critical evaluation for machine learning force fields with molecular simulations*, 2022 | [arXiv:2210.07237](https://arxiv.org/abs/2210.07237) | Force error does not predict simulation quality; stability and observables have to be measured directly. The reason E6 compares observables rather than force errors. **read** |
## 4. Distillation and spatial mixing

| Source | Link | Used for |
|---|---|---|
| Amin, Raja, Krishnapriyan, *Towards fast, specialized machine learning force fields: distilling foundation models via energy Hessians*, ICLR 2025 | [OpenReview](https://openreview.net/forum?id=1durmugh3I) · [code](https://github.com/ASK-Berkeley/MLFF-distill) | Motivates the curvature (force-change) loss in `frost.insitu`. **cited** |
| Ekström Kelvinius, Georgiev, Toshev, Gasteiger, *Accelerating molecular graph neural networks via knowledge distillation*, NeurIPS 2023 | [arXiv:2306.14818](https://arxiv.org/abs/2306.14818) | Distillation background. **read** |
| *Projected Hessian learning: fast curvature supervision for accurate MLIPs*, 2026 | [arXiv:2603.04523](https://arxiv.org/abs/2603.04523) | Cheaper curvature labels. **read** |
| Birks, Nutter, Swinburne, Kermode, *Efficient and accurate spatial mixing of machine learned interatomic potentials for materials science* (ML-MIX), npj Comput. Mater. 2026 | [arXiv:2502.19081](https://arxiv.org/abs/2502.19081) · [npj](https://www.nature.com/articles/s41524-026-01982-6) | Spatial mixing, up to 11×. **cited, baseline B3** |
| Chen, O'Neill, Shi, Kapil, *Distilling first-principles accuracy into compact machine learning potentials for condensed-phase chemistry*, 2026 | [arXiv:2606.06848](https://arxiv.org/abs/2606.06848) | Distilled students cut production cost about 10× on ice, water and TiO₂/water. They replace the teacher; Frost keeps it in the loop. **cited** |
| Matsumura, Yoshimoto, Iwasaki, Yamazaki, Sakai, *Knowledge distillation framework for accelerating high-accuracy neural network-based molecular dynamics simulations*, 2025 | [arXiv:2506.15337](https://arxiv.org/abs/2506.15337) | Students up to 106× faster than an off-the-shelf teacher. **read** |
| Oh et al., *A lightweight universal machine-learning interatomic potential via knowledge distillation for scalable atomistic simulations* (SevenNet-Nano), 2026 | [arXiv:2604.10887](https://arxiv.org/abs/2604.10887) | A universal distilled student, over 10× faster than its teacher. **read** |

## 5. Speculation, caching and incremental inference

| Source | Link | Used for |
|---|---|---|
| Leviathan, Kalman, Matias, *Fast inference from transformers via speculative decoding*, ICML 2023 | [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) | Draft–verify as a systems primitive. **cited** |
| Chen et al., *Accelerating large language model decoding with speculative sampling*, 2023 | [arXiv:2302.01318](https://arxiv.org/abs/2302.01318) | Same. **cited** |
| Miao et al., *SpecInfer: tree-based speculative inference and verification*, ASPLOS 2024 | [arXiv:2305.09781](https://arxiv.org/abs/2305.09781) | Serving-system precedent. **cited** |
| Kosmala, Günnemann, Gao, Wood, *Speculative sampling for faster molecular dynamics* (LSD), 2026 | [arXiv:2606.02455](https://arxiv.org/abs/2606.02455) | The closest draft–verify precedent in MD. A draft MLIP proposes Langevin steps, and a pool of target devices verifies them asynchronously: 3–9× wall clock. It needs parallel target devices, is Langevin-only, and its rejections grow with √N, which limits it beyond about 10³ atoms. **cited, baseline** |
| Kwon et al., *Efficient memory management for LLM serving with PagedAttention* (vLLM), SOSP 2023 | [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) | Model of a systems paper built on one abstraction. **read** |
| Shih, Belkhale, Ermon, Sadigh, Anari, *Parallel sampling of diffusion models* (ParaDiGMS), NeurIPS 2023 | [arXiv:2305.16317](https://arxiv.org/abs/2305.16317) | Picard-parallel time, 2–4×. **cited** |
| Selvaraju et al., *FORA: fast-forward caching in diffusion transformer acceleration*, 2024 | [arXiv:2407.01425](https://arxiv.org/abs/2407.01425) | Feature caching. **cited** |
| Zou, Liu, Liu, Huang, Zhang, *Accelerating diffusion transformers with token-wise feature caching* (ToCa), ICLR 2025 | [arXiv:2410.05317](https://arxiv.org/abs/2410.05317) · [code](https://github.com/Shenyi-Z/ToCa) | Token-level caching with error propagation. **cited** |
| Liu et al., *Timestep embedding tells: it's time to cache for video diffusion model* (TeaCache), CVPR 2025 | [arXiv:2411.19108](https://arxiv.org/abs/2411.19108) · [code](https://github.com/ali-vilab/TeaCache) | Input-difference cache trigger. **cited** |
| *SpeCa: accelerating diffusion transformers with speculative feature caching*, ACM MM 2025 | [arXiv:2509.11628](https://arxiv.org/abs/2509.11628) | Forecast-then-verify, 1.7–3.5% verification overhead. **cited** |
| *A survey on cache methods in diffusion models*, 2025 | [arXiv:2510.19755](https://arxiv.org/abs/2510.19755) | Landscape. **read** |
| Parger, Tang, Twigg, Keskin, Wang, Steinberger, *DeltaCNN: end-to-end CNN inference of sparse frame differences in videos*, CVPR 2022 | [arXiv:2203.03996](https://arxiv.org/abs/2203.03996) · [code](https://github.com/facebookresearch/DeltaCNN) | Sparse delta propagation. **cited** |
| Dutson, Li, Gupta, *Eventful Transformers: leveraging temporal redundancy in vision transformers*, ICCV 2023 | [arXiv:2308.13494](https://arxiv.org/abs/2308.13494) | Token gating, 3.8× fewer FLOPs. **cited** |
| *InkStream: real-time GNN inference on streaming graphs via incremental update*, 2023 | [arXiv:2309.11071](https://arxiv.org/abs/2309.11071) | Incremental GNN inference. **cited** |
| *Ripple: scalable incremental GNN inferencing on large streaming graphs*, 2025 | [arXiv:2505.12112](https://arxiv.org/abs/2505.12112) | Same. **cited** |
| *RIPPLE++: an incremental framework for efficient GNN inference on evolving graphs*, 2026 | [arXiv:2601.12347](https://arxiv.org/abs/2601.12347) | Same. **cited** |

## 6. Exact sampling and time parallelism

| Source | Link | Used for |
|---|---|---|
| Duane, Kennedy, Pendleton, Roweth, *Hybrid Monte Carlo*, Phys. Lett. B 195, 216 (1987) | [ADS](https://ui.adsabs.harvard.edu/abs/1987PhLB..195..216D/abstract) | Exactness argument for nested verification (note Prop. 3). **cited** |
| Gelb, *Monte Carlo simulations using sampling from an approximate potential*, J. Chem. Phys. 118, 7747 (2003) | [doi:10.1063/1.1563597](https://doi.org/10.1063/1.1563597) | Nested MC with a cheap reference potential. **cited** |
| Coe, Sewell, Shaw, *Nested Markov chain Monte Carlo sampling of a density functional theory potential: dense fluid nitrogen*, J. Chem. Phys. 131, 074105 (2009) | [ResearchGate](https://www.researchgate.net/publication/26768763_Nested_Markov_chain_Monte_Carlo_sampling_of_a_density_functional_theory_potential_Equilibrium_thermodynamics_of_dense_fluid_nitrogen) · [theory report (OSTI)](https://www.osti.gov/biblio/989792/) | Same, with DFT. **cited** |
| *Layered nested Markov chain Monte Carlo*, J. Chem. Phys. 149, 072326 (2018) | [AIP](https://pubs.aip.org/aip/jcp/article/149/7/072326/1075340/Layered-nested-Markov-chain-Monte-Carlo) | Multi-level nesting. **read** |
| *Accelerating multilevel Markov chain Monte Carlo using machine learning models*, 2024 | [arXiv:2405.11179](https://arxiv.org/abs/2405.11179) | ML-filtered proposals. **read** |
| Lions, Maday, Turinici, *Résolution d'EDP par un schéma en temps pararéel*, C. R. Acad. Sci. Paris 332, 661 (2001) | [doi:10.1016/S0764-4442(00)01793-6](https://doi.org/10.1016/S0764-4442(00)01793-6) | Parareal. **cited** |
| Baffico, Bernard, Maday, Turinici, Zérah, *Parallel-in-time molecular-dynamics simulations*, Phys. Rev. E 66, 057701 (2002) | [APS](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.66.057701) | Parareal for MD. **cited** |
| Gorynina, Legoll, Lelièvre, Perez, *Combining machine-learned and empirical force fields with the parareal algorithm*, C. R. Mécanique 2023 | [arXiv:2212.10508](https://arxiv.org/abs/2212.10508) · [journal](https://comptes-rendus.academie-sciences.fr/mecanique/articles/10.5802/crmeca.220/) | Parareal with ML force fields. **cited** |
| Samaddar, Newman, Sánchez, *Parallelization in time of numerical simulations of fully-developed plasma turbulence using the parareal algorithm*, J. Comput. Phys. 229 (2010) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0021999110002731) | About 10× parallel gain on plasma turbulence. **cited** |
| Reynolds-Barredo et al., *Mechanisms for the convergence of time-parallelized, parareal turbulent plasma simulations*, J. Comput. Phys. 2012 | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0021999112004123) | Why parareal converges for turbulence. **read** |
| Pamela et al. and the JOREK Team, *Neural-Parareal: dynamically training neural operators as coarse solvers for time-parallelisation of fusion MHD simulations*, Comput. Phys. Commun. 307, 109391 (2025) | [arXiv:2405.01355](https://arxiv.org/abs/2405.01355) | Neural coarse propagators for JOREK (W4). **cited** |
| *Parareal with a physics-informed neural network as coarse propagator*, 2023 | [arXiv:2303.03848](https://arxiv.org/abs/2303.03848) | Learned coarse propagator. **read** |
| *Space-time parallel scaling of Parareal with a physics-informed FNO coarse propagator*, 2024 | [arXiv:2404.02521](https://arxiv.org/abs/2404.02521) | Same. **read** |

## 7. Plasma and MHD: neural surrogates and closures

| Source | Link | Used for |
|---|---|---|
| Ohana et al., *The Well: a large-scale collection of diverse physics simulations for machine learning*, NeurIPS Datasets & Benchmarks 2024 | [arXiv:2412.00568](https://arxiv.org/abs/2412.00568) · [code](https://github.com/PolymathicAI/the_well) · [MHD_64](https://polymathic-ai.org/the_well/datasets/MHD_64/) · [HF](https://huggingface.co/datasets/polymathic-ai/MHD_64) | MHD regimes for the M_f sweep; W4 data (downloading). **cited, data** |
| *Walrus: a cross-domain foundation model for continuum dynamics*, 2025 | [arXiv:2511.15684](https://arxiv.org/abs/2511.15684) · [weights](https://huggingface.co/polymathic-ai/walrus) | Optional W4 surrogate (tier C). **cited** |
| Li et al., *Fourier neural operator for parametric partial differential equations*, ICLR 2021 | [arXiv:2010.08895](https://arxiv.org/abs/2010.08895) | Surrogate family. **read** |
| Gopakumar et al., *Plasma surrogate modelling using Fourier neural operators*, Nucl. Fusion 64, 056025 (2024) | [arXiv:2311.05967](https://arxiv.org/abs/2311.05967) · [IOP](https://iopscience.iop.org/article/10.1088/1741-4326/ad313a) | Tokamak MHD surrogates. **cited** |
| *Fourier neural operator for plasma modelling*, 2023 | [arXiv:2302.06542](https://arxiv.org/abs/2302.06542) | Earlier version of the above. **read** |
| *Neural operator surrogate models of plasma edge simulations: feasibility and data efficiency*, Nucl. Fusion 2025 | [arXiv:2502.17386](https://arxiv.org/abs/2502.17386) | Plasma-edge surrogates. **cited** |
| Ma, Zhu, Xu, Wang, *Machine learning surrogate models for Landau fluid closure*, Phys. Plasmas 27, 042502 (2020) | [arXiv:1909.11509](https://arxiv.org/abs/1909.11509) · [AIP](https://pubs.aip.org/aip/pop/article-abstract/27/4/042502/318805/Machine-learning-surrogate-models-for-Landau-fluid) | Neural closure in a fluid code. **cited** |
| *Deep learning surrogate model for kinetic Landau-fluid closure with collision*, AIP Advances 10, 075108 (2020) | [AIP](https://pubs.aip.org/aip/adv/article/10/7/075108/240987/Deep-learning-surrogate-model-for-kinetic-Landau) | Same, with collisions. **read** |
| *Machine-learning heat flux closure for multi-moment fluid modeling of nonlinear Landau damping*, PNAS 2025 | [PNAS](https://www.pnas.org/doi/10.1073/pnas.2419073122) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11929456/) | FNO closure reproducing nonlinear Landau damping. **cited** |
| *The machine learning approach to moment closure relations for plasma: a review*, 2025 | [arXiv:2511.22486](https://arxiv.org/abs/2511.22486) | Closure landscape and open problems. **cited** |
| Miller, Churchill, Dener, Chang, Munson, Hager, *Encoder–decoder neural network for solving the nonlinear Fokker–Planck–Landau collision operator in XGC*, J. Plasma Phys. 87, 905870211 (2021) | [arXiv:2009.06534](https://arxiv.org/abs/2009.06534) · [doi:10.1017/s0022377821000155](https://doi.org/10.1017/s0022377821000155) | Neural collision operator inside a PIC code. **cited** |
| *FPL-net: deep learning for the nonlinear Fokker–Planck–Landau collision operator*, J. Comput. Phys. 2024 | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0021999124009136) | Same line. **read** |
| *Subgrid modelling for relativistic magnetohydrodynamics with machine learning*, Phys. Rev. D 2026 | [arXiv:2606.21659](https://arxiv.org/abs/2606.21659) | MLP subgrid closure evaluated on every face at every RK substage; cost data. **cited** |
| *Artificial neural network subgrid models of 2D compressible magnetohydrodynamic turbulence* | [ResearchGate](https://www.researchgate.net/publication/340557887_Artificial_neural_network_subgrid_models_of_2D_compressible_magnetohydrodynamic_turbulence) | Subgrid closure for W1. **read** |
| *Deterministic and probabilistic neural surrogates of global hybrid-Vlasov simulations*, 2026 | [arXiv:2601.12614](https://arxiv.org/abs/2601.12614) | GNN plasma emulators, >100× on one GPU. **read** |
| *Solver-in-the-loop applications in astrophysical (magneto)hydrodynamics* (astronomix), 2025 | [arXiv:2512.05999](https://arxiv.org/abs/2512.05999) · [code](https://github.com/leo1200/astronomix) | Divergence-free MHD corrector (W3); installed in `frost-mhd`. **cited, code** |
| *Differentiable conservative radially symmetric fluid simulations and stellar winds* (jf1uids), 2024 | [arXiv:2410.23093](https://arxiv.org/abs/2410.23093) | astronomix's predecessor. **read** |

## 8. Hybrid solvers and adaptive computation

| Source | Link | Used for |
|---|---|---|
| Kochkov, Smith, Alieva, Wang, Brenner, Hoyer, *Machine learning–accelerated computational fluid dynamics*, PNAS 118, e2101784118 (2021) | [arXiv:2102.01010](https://arxiv.org/abs/2102.01010) · [PNAS](https://www.pnas.org/doi/10.1073/pnas.2101784118) | Learned discretisations, 40–80×. **cited** |
| Um, Brand, Fei, Holl, Thuerey, *Solver-in-the-loop*, NeurIPS 2020 | [arXiv:2007.00016](https://arxiv.org/abs/2007.00016) | Hybrid correction training. **cited** |
| *ANCHOR: error-controlled adaptive numerical correction for neural operator time marching*, 2025 | [arXiv:2512.19643](https://arxiv.org/abs/2512.19643) | Residual-triggered correction, 30–75%. **cited** |
| *INC: an indirect neural corrector for auto-regressive hybrid PDE solvers*, 2025 | [arXiv:2511.12764](https://arxiv.org/abs/2511.12764) | Hybrid correction. **read** |
| *NOWS: neural operator warm starts for accelerating iterative solvers*, 2025 | [arXiv:2511.02481](https://arxiv.org/abs/2511.02481) | Verified acceleration by warm starts. **read** |
| *AMR-Transformer: enabling efficient long-range interaction for complex neural fluid simulation*, CVPR 2025 | [arXiv:2503.10257](https://arxiv.org/abs/2503.10257) | Adaptive tokens, 2–10× fewer. **cited** |
| *MeshTok: efficient multi-scale tokenization for scalable PDE transformers*, 2026 | [arXiv:2606.04366](https://arxiv.org/abs/2606.04366) | Same idea. **read** |
| Atkinson et al., *FTorch: a library for coupling PyTorch models to Fortran*, JOSS | [JOSS](https://www.theoj.org/joss-papers/joss.07602/10.21105.joss.07602.pdf) · [docs](https://cambridge-iccs.github.io/FTorch/) | Coupling learned models into legacy codes. **read** |
| *ClimSim-Online: a large multi-scale dataset and framework for hybrid ML-physics climate emulation*, 2023 | [arXiv:2306.08754](https://arxiv.org/abs/2306.08754) | Online hybrid inference cost. **read** |
| *Synthesizing particle-in-cell simulations through learning and GPU computing for hybrid particle accelerator beamlines*, 2024 | [arXiv:2402.17248](https://arxiv.org/abs/2402.17248) | 98–99% of surrogate time in NN inference. **read** |

## 9. MHD numerics, codes and data

| Source | Link | Used for |
|---|---|---|
| Evans and Hawley, *Simulation of magnetohydrodynamic flows: a constrained transport method*, ApJ 332, 659 (1988) | [ADS](https://ui.adsabs.harvard.edu/abs/1988ApJ...332..659E/abstract) | Edge-EMF commits keep ∇·B = 0 (note Prop. 4). **cited** |
| Gardiner and Stone, *An unsplit Godunov method for ideal MHD via constrained transport*, J. Comput. Phys. 205, 509 (2005); 3D version 2008 | [JCP 2005](https://www.sciencedirect.com/science/article/abs/pii/S0021999104004784) · [3D, arXiv:0712.2634](https://arxiv.org/abs/0712.2634) | Solver design for W1. **cited** |
| Miyoshi and Kusano, *A multi-state HLL approximate Riemann solver for ideal MHD* (HLLD), J. Comput. Phys. 208, 315 (2005) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0021999105001142) | Riemann solver for W1. **cited** |
| Orszag and Tang, *Small-scale structure of two-dimensional magnetohydrodynamic turbulence*, J. Fluid Mech. 90, 129 (1979) | [ADS](https://ui.adsabs.harvard.edu/abs/1979JFM....90..129O/abstract) | Standard test; Athena++ `athena_orszag_tang` reference. **cited** |
| Birn et al., *Geospace Environmental Modeling (GEM) magnetic reconnection challenge*, JGR 106, 3715 (2001) | [Wiley](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999JA900449) | W2 reconnection setup. **cited** |
| Sovinec et al., *Nonlinear magnetohydrodynamics simulation using high-order finite elements* (NIMROD), J. Comput. Phys. 195 (2004) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0021999103005369) | Stiff magnetised regimes. **read** |
| Hoelzl et al., *The JOREK non-linear extended MHD code and applications to large-scale instabilities*, Nucl. Fusion 61, 065001 (2021) | [arXiv:2011.09120](https://arxiv.org/abs/2011.09120) · [OSTI](https://www.osti.gov/pages/biblio/1809091) | Tokamak regime; Neural-Parareal host code. **read** |
| Stone et al., *AthenaK: a performance-portable version of the Athena++ AMR framework*, 2024 | [arXiv:2409.16053](https://arxiv.org/abs/2409.16053) · [code](https://github.com/IAS-Astrophysics/athenak) | GPU MHD reference. **cited** |
| Athena++ | [code](https://github.com/PrincetonUniversity/athena) | Reference solutions, built by `scripts/build_athena.sh`. **code** |
| Liska et al., *H-AMR: a new GPU-accelerated GRMHD code with 3D AMR and local adaptive time-stepping*, ApJS 263, 26 (2022) | [arXiv:1912.10192](https://arxiv.org/abs/1912.10192) | Local time stepping, 3–10× fewer steps. **cited** |
| Meyer, Balsara, Aslam, *A second-order accurate super timestepping formulation for anisotropic thermal conduction*, MNRAS 422, 2102 (2012); *stabilized RKL method*, J. Comput. Phys. 2014 | [MNRAS](https://academic.oup.com/mnras/article/422/3/2102/1042639) · [JCP 2014](https://www.sciencedirect.com/science/article/abs/pii/S0021999113005597) · [Athena++ STS](https://github.com/PrincetonUniversity/athena-public-version/wiki/Super-Time-Stepping) | Classical rate separation for diffusive terms. **cited** |
| Johns Hopkins Turbulence Database, forced MHD turbulence 1024³ | [JHTDB](https://turbulence.pha.jhu.edu/Forced_MHD_turbulence.aspx) | Closure training cross-check (needs a SciServer token). **cited, data** |
| *TURB-MHD: an open-access database of forced homogeneous MHD turbulence*, 2025 | [arXiv:2504.10755](https://arxiv.org/abs/2504.10755) · [portal](http://smart-turb.roma2.infn.it) | Subgrid closure data, CC BY 4.0. **cited, data** |
| JAX-Fluids | [code](https://github.com/tumaer/JAXFLUIDS) | Differentiable CFD reference. **read, code** |

## 10. Datasets and checkpoints used in v1

| Source | Link | Used for |
|---|---|---|
| MD17 / MD22 (sGDML) | [quantum-machine.org](http://www.quantum-machine.org/gdml/) | Tier A datasets. **data** |
| rMD17 | [figshare 12672038](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038) | Tier A. **data** |
| 3BPA | [BOTNet datasets](https://github.com/davkovacs/BOTNet-datasets) | Tier A. **data** |
| SPICE 2 | [openmm/spice-dataset](https://github.com/openmm/spice-dataset) | Tier A (needs `GH_TOKEN`). **data** |
| MPtrj | [HF nimashoghi/mptrj](https://huggingface.co/datasets/nimashoghi/mptrj) | Provenance of MACE-MP-0. **data** |
| Sanchez-Gonzalez et al., *Learning to simulate complex physics with graph networks* (GNS), ICML 2020 | [arXiv:2002.09405](https://arxiv.org/abs/2002.09405) · [data](https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets) | v1 cross-domain plan; local copy deleted 2026-11-11 (disk full), re-downloadable from the link. **read** |
| Pfaff et al., *Learning mesh-based simulation with graph networks* (MeshGraphNets), ICLR 2021 | [arXiv:2010.03409](https://arxiv.org/abs/2010.03409) · [data](https://storage.googleapis.com/dm-meshgraphnets) | v1 cross-domain plan; local copy deleted 2026-11-11 (disk full), re-downloadable from the link. **read** |
| MeshGraphNet CylinderFlow checkpoint | [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/modulus_ns_meshgraphnet) | v1 negative control. **data** |
| Allegro / NequIP foundation weights | [Zenodo 16980200](https://zenodo.org/records/16980200) | Allegro-MP-L teacher. **data** |
| Materials Project mp-13725 (Li₃PO₄) | [Materials Project](https://next-gen.materialsproject.org/materials/mp-13725) | Li₃PO₄ cell. **data** |

## 11. Research-direction review (2026-09-28)

Everything `research_direction.md` relies on beyond the sections above, verified at
the linked version on 2026-09-28. Sources already listed elsewhere (DMTS, DMTS-NC,
FlashMD, MTS-Allegro, MOLLY, Ma–Izaguirre–Skeel, InkStream, Ripple, ASFE, Mouvet
et al.) keep their rows there.

### 11.1 Competing and adjacent methods

| Source | Link | Used for |
|---|---|---|
| Schaaf, Batatia, Brunken, Barrett, Tilly, *BoostMD: accelerating molecular sampling by leveraging ML force field features from previous time-steps*, NeurIPS 2024 workshop (MLSB) | [arXiv:2412.18633](https://arxiv.org/abs/2412.18633) · [MLSB PDF](https://www.mlsb.io/papers_2024/BoostMD:_Accelerating_Molecular_Sampling_using_ML_Force_Field_Feature.pdf) | The published "student conditioned on cached teacher features". A one-layer boost model reads MACE-OFF23-M reference node features and predicts ΔE; the reference runs every N steps. Its 8.4× is per-evaluation model speedup (3,375 atoms, A100); the paper's own serial formula gives 4.8× at N = 10. Kabsch alignment restores rotation equivariance (cf. the torque identity, §3.5). Loss not stated in the text. **cited, closest precedent for feature reuse** |
| Bigi, Langer, Ceriotti, *The dark side of the forces: assessing non-conservative force models for atomistic machine learning*, ICML 2025 | [arXiv:2412.11569](https://arxiv.org/abs/2412.11569) · [PMLR](https://proceedings.mlr.press/v267/bigi25a.html) | Non-conservative forces fail alone; MTS with 7 fast direct-force steps and 1 conservative correction keeps 80–90% of the speedup with long-term stability. Precedent for DMTS-NC. **cited** |
| Anglada, Junquera, Soler, *Efficient mixed-force first-principles molecular dynamics*, Phys. Rev. E 68, 055701 (2003) | [doi:10.1103/PhysRevE.68.055701](https://doi.org/10.1103/PhysRevE.68.055701) · [arXiv:cond-mat/0305704](https://arxiv.org/abs/cond-mat/0305704) | Dual-level MTS: the expensive-minus-cheap difference at outer steps. The residual form predates ML students. **cited** |
| Luehr, Markland, Martínez, *Multiple time step integrators in ab initio molecular dynamics*, J. Chem. Phys. 140, 084116 (2014) | [arXiv:1312.1284](https://arxiv.org/abs/1312.1284) · [doi:10.1063/1.4866176](https://doi.org/10.1063/1.4866176) | Fragment and range-separated splittings of ab initio forces; 2.5 fs outer steps. **cited** |
| Kapil, VandeVondele, Ceriotti, *Accurate molecular dynamics and nuclear quantum effects at low cost by multiple steps in real and imaginary time*, J. Chem. Phys. 144, 054111 (2016) | [arXiv:1512.00176](https://arxiv.org/abs/1512.00176) | DFT as the cheap level for wavefunction methods under MTS. **cited** |
| Csányi, Albaret, Payne, De Vita, *"Learn on the fly": a hybrid classical and quantum-mechanical molecular dynamics simulation*, Phys. Rev. Lett. 93, 175503 (2004) | [doi:10.1103/PhysRevLett.93.175503](https://dx.doi.org/10.1103/PhysRevLett.93.175503) | In-situ refitting of a cheap force model against QM during the run. Precedent for Frost's in-situ student outside a RESPA splitting. **cited** |
| Li, Kermode, De Vita, *Molecular dynamics with on-the-fly machine learning of quantum-mechanical forces*, Phys. Rev. Lett. 114, 096405 (2015) | [doi:10.1103/PhysRevLett.114.096405](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.114.096405) · [WRAP](https://wrap.warwick.ac.uk/id/eprint/66611/) | A Bayesian error estimate decides when to call QM. Precedent for error-controlled invocation of the expensive model. **cited** |
| Vandermause et al., *On-the-fly active learning of interpretable Bayesian force fields for atomistic rare events* (FLARE), npj Comput. Mater. 6, 20 (2020) | [npj](https://www.nature.com/articles/s41524-020-0283-z) | Uncertainty-triggered DFT calls during MD. **cited** |

### 11.2 Integrators, adaptivity and resonance

| Source | Link | Used for |
|---|---|---|
| Skeel, Gear, *Does variable step size ruin a symplectic integrator?*, Physica D 60, 311–313 (1992) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/016727899290247K) · [ADS](https://ui.adsabs.harvard.edu/abs/1992PhyD...60..311S/abstract) | Varying the step loses symplecticity. Why Frost's k changes a bounded number of times and never per step (§4.7). **cited** |
| Hut, Makino, McMillan, *Building a better leapfrog*, ApJ 443, L93 (1995) | [ADS](https://ui.adsabs.harvard.edu/abs/1995ApJ...443L..93H/abstract) | Time-symmetric step selection restores reversibility. The required form if per-step adaptivity is ever wanted. **cited** |
| Huang, Leimkuhler, *The adaptive Verlet method*, SIAM J. Sci. Comput. 18, 239–256 (1997) | [doi:10.1137/S1064827595284658](https://doi.org/10.1137/S1064827595284658) | Reversible adaptivity by Sundman time transformation. **cited** |
| Richardson, Finn, *Symplectic integrators with adaptive time steps*, Plasma Phys. Control. Fusion 54, 014004 (2012) | [arXiv:1108.0322](https://arxiv.org/abs/1108.0322) | Adaptive symplectic stepping by extended phase space. **read** |
| Biesiadecki, Skeel, *Dangers of multiple time step methods*, J. Comput. Phys. 109, 318–328 (1993) | [doi:10.1006/jcph.1993.1220](https://doi.org/10.1006/jcph.1993.1220) | Linear resonance of impulse MTS at multiples of the half period; background for the band law (§4.4). **cited** |
| Hairer, Lubich, Wanner, *Geometric Numerical Integration*, 2nd ed., Springer (2006) | [doi:10.1007/3-540-30666-8](https://doi.org/10.1007/3-540-30666-8) | Backward error analysis: a fixed symplectic map conserves a modified Hamiltonian over exponentially long times (ch. IX). Basis of the bounded-switch energy statement. **cited** |

### 11.3 Derivatives, surrogates and incremental GNN computation

| Source | Link | Used for |
|---|---|---|
| Czarnecki, Osindero, Jaderberg, Świrszcz, Pascanu, *Sobolev training for neural networks*, NeurIPS 2017 | [arXiv:1706.04859](https://arxiv.org/abs/1706.04859) | Supervising Jacobians as well as values. Theorem A says a feature surrogate needs exactly the adjoint-projected part of that supervision. **cited** |
| Griewank, Walther, *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*, 2nd ed., SIAM (2008) | [doi:10.1137/1.9780898717761](https://doi.org/10.1137/1.9780898717761) | Cheap-gradient principle; the cost of a teacher Hessian-vector product for derivative corrections (§4.5). **cited** |
| Chen, Zhu, Song, *Stochastic training of graph convolutional networks with variance reduction* (VR-GCN), ICML 2018 | [arXiv:1710.10568](https://arxiv.org/abs/1710.10568) · [PMLR](https://proceedings.mlr.press/v80/chen18p.html) | Historical activations with control variates, in training (gradients to weights, not inputs). **read** |
| Fey, Lenssen, Weichert, Leskovec, *GNNAutoScale: scalable and expressive graph neural networks via historical embeddings*, ICML 2021 | [arXiv:2106.05609](https://arxiv.org/abs/2106.05609) | Historical embeddings; the zeroth-order reuse Theorem A rules out for gradient outputs. **read** |

### 11.4 Hardware

| Source | Link | Used for |
|---|---|---|
| Williams, Waterman, Patterson, *Roofline: an insightful visual performance model for multicore architectures*, Commun. ACM 52(4), 65–76 (2009) | [doi:10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785) | Bandwidth argument against applying stored Jacobians (§5.2). **cited** |
| NVIDIA RTX A6000 product specifications (768 GB/s GDDR6, 38.7 TFLOPS FP32) | [nvidia.com](https://www.nvidia.com/en-us/design-visualization/rtx-a6000/) | Bandwidth figure in the roofline. **data** |

## 12. New directions from other fields (2026-09-28, second pass)

Sources for `new_directions.md`. Rows marked "link to be verified" were cited from
the standard reference and not re-fetched this session.

### 12.1 Signal processing and sampling

| Source | Link | Used for |
|---|---|---|
| Matthews, *On the linear minimum-mean-squared-error estimation of an undersampled wide-sense stationary random process*, IEEE Trans. Signal Process. 48(1), 272–275 (2000) | [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/815501/) | The aliased Wiener filter: the bound C1 applies to E7's measured residual spectra. **cited** |
| Shapiro, Silverman, *Alias-free sampling of random noise*, J. SIAM 8, 225–248 (1960) | link to be verified | Random sampling times remove aliasing; the idea behind randomised outer intervals, which Frost measured to lose (plan §1.14). **read** |

### 12.2 Lattice QCD and shadow Hamiltonians

| Source | Link | Used for |
|---|---|---|
| Sexton, Weingarten, *Hamiltonian evolution for the hybrid Monte Carlo algorithm*, Nucl. Phys. B 380, 665–677 (1992) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/055032139290263B) | Nested multi-timescale leapfrog in lattice QCD, the same construction as r-RESPA. **cited** |
| Clark, Joó, Kennedy, Silva, *Improving dynamical lattice QCD simulations through integrator tuning using Poisson brackets and a force-gradient integrator*, Phys. Rev. D 84, 071502 (2011) | [arXiv:1108.1828](https://arxiv.org/abs/1108.1828) | Measuring Poisson brackets to tune nested integrators. **cited** |
| Kennedy, Silva, Clark, *Shadow Hamiltonians, Poisson brackets, and gauge theories*, Phys. Rev. D 87, 034511 (2013) | [arXiv:1210.6600](https://arxiv.org/abs/1210.6600) · [PRD](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.87.034511) | Shadow expansion in brackets; C6. **cited** |
| Engle, Skeel, Drees, *Monitoring energy drift with shadow Hamiltonians*, J. Comput. Phys. 206, 432–452 (2005) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0021999104005212) | Shadow-based drift monitoring in MD; C6. **cited** |
| Skeel, Hardy, *Practical construction of modified Hamiltonians*, SIAM J. Sci. Comput. 23, 1172–1188 (2001) | link to be verified | Computing shadows from trajectory data. **read** |
| *Method, apparatus and computer program for multiple time stepping simulation of a thermodynamic system using shadow Hamiltonians*, US patent 8,510,084 | [USPTO PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8510084) | Prior art: shadow Hamiltonians with MTS sampling. **read** |

### 12.3 Celestial mechanics, accelerators and resonance

| Source | Link | Used for |
|---|---|---|
| Chirikov, *A universal instability of many-dimensional oscillator systems*, Phys. Rep. 52, 263–379 (1979) | [doi:10.1016/0370-1573(79)90023-1](https://doi.org/10.1016/0370-1573(79)90023-1) (link to be verified) | Resonance overlap; the predicted temperature dependence of the certified H/T (C4). **cited** |
| Greene, *A method for determining a stochastic transition*, J. Math. Phys. 20, 1183 (1979) | [doi:10.1063/1.524170](https://doi.org/10.1063/1.524170) (link to be verified) | Noble rotation numbers are the last KAM tori to break; the working-point hypothesis (C4). **cited** |
| Pikovsky, Rosenblum, Kurths, *Synchronization: a universal concept in nonlinear sciences*, Cambridge (2001) | link to be verified | Arnold tongues of periodically kicked oscillators. **read** |
| Laskar, *The chaotic motion of the solar system: a numerical estimate of the size of the chaotic zones*, Icarus 88, 266–291 (1990); Laskar, Froeschlé, Celletti, Physica D 56, 253 (1992) | [ADS (1992)](https://ui.adsabs.harvard.edu/abs/1992PhyD...56..253L/abstract) | Frequency-map analysis as an early chaos indicator; also used for accelerator dynamic aperture. §8 **[H]**. **read** |
| Quinlan, Tremaine, *On the reliability of gravitational N-body integrations*, MNRAS 259, 505 (1992) | link to be verified | Shadowing as a validation metric for integrated trajectories. **read** |

### 12.4 Control theory, neuroscience and quantum chemistry

| Source | Link | Used for |
|---|---|---|
| Smith, *Closer control of loops with dead time*, Chem. Eng. Prog. 53, 217–219 (1957) | link to be verified | The Smith predictor: internal model plus lagged mismatch, the structure of the two-GPU pipeline (C2). **cited** |
| Miall, Weir, Wolpert, Stein, *Is the cerebellum a Smith predictor?*, J. Mot. Behav. 25, 203–216 (1993) | link to be verified | The same structure in motor control. **read** |
| Niklasson, Tymczak, Challacombe, *Time-reversible Born–Oppenheimer molecular dynamics*, Phys. Rev. Lett. 97, 123001 (2006); Niklasson, *Extended Born–Oppenheimer molecular dynamics*, Phys. Rev. Lett. 100, 123004 (2008) | [doi:10.1103/PhysRevLett.97.123001](https://doi.org/10.1103/PhysRevLett.97.123001) (link to be verified) | Non-reversible extrapolation of a lagged quantity injects energy; the analogue of C2's result that extrapolating the lagged residual hurts. **cited** |
| McClelland, McNaughton, O'Reilly, *Why there are complementary learning systems in the hippocampus and neocortex*, Psychol. Rev. 102, 419–457 (1995) | link to be verified | Fast and slow learners; C5. **cited** |
| Bigi, Chong, Ceriotti, Grasselli, *A prediction rigidity formalism for low-cost uncertainties in trained neural networks*, Mach. Learn.: Sci. Technol. 5, 045018 (2024) | [IOP](https://iopscience.iop.org/article/10.1088/2632-2153/ad805f) | Last-layer uncertainty for the fast learner (C5). **cited** |

### 12.5 Sampling and stochastic methods considered and rejected

| Source | Link | Used for |
|---|---|---|
| Giles, *Multilevel Monte Carlo path simulation*, Oper. Res. 56, 607–617 (2008) | link to be verified | Coupled multilevel estimator for the statistics contract; rejected at 0.89× (§8). **read** |
| Dodwell, Ketelsen, Scheichl, Teckentrup, *A hierarchical multilevel Markov chain Monte Carlo algorithm*, SIAM/ASA J. Uncertain. Quantif. 3, 1075 (2015) | link to be verified | Multilevel MCMC with coupled chains. **read** |
| Jin, Li, Liu, *Random batch methods (RBM) for interacting particle systems*, J. Comput. Phys. 400, 108877 (2020) | [arXiv:1812.10575](https://arxiv.org/abs/1812.10575) | Subsampled interactions under Langevin; rejected for MACE-class teachers by the halo cost (§8). **read** |
| Chen, Fox, Guestrin, *Stochastic gradient Hamiltonian Monte Carlo*, ICML 2014 | [arXiv:1402.4102](https://arxiv.org/abs/1402.4102) | Friction compensating gradient noise. **read** |
| Leimkuhler, Shang, *Adaptive thermostats for noisy gradient systems*, SIAM J. Sci. Comput. 38, A712 (2016) | [arXiv:1510.08692](https://arxiv.org/abs/1510.08692) | Same, with unknown noise. **read** |
| Bloom, Takacs, da Silva, Ledvina, *Data assimilation using incremental analysis updates*, Mon. Weather Rev. 124, 1256 (1996) | link to be verified | Spreading increments to avoid initialisation shock; Frost's `respa_spread` measured 186× worse (plan §1.14). **read** |

## 13. Biology: how brains simulate the world, and what transfers (2026-09-28)

Sources for `new_directions.md` §10. Rows marked "link to be verified" were cited
from the standard reference and not re-fetched this session.

| Source | Link | Used for |
|---|---|---|
| Wolpert, Ghahramani, Jordan, *An internal model for sensorimotor integration*, Science 269, 1880–1882 (1995) | [Science](https://www.science.org/doi/10.1126/science.7569931) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/7569931/) | The brain simulates the arm with a forward model and corrects it with delayed sensory feedback. C2's architecture: the student is the forward model, the lagged teacher the delayed sensation. **cited** |
| Miall, Weir, Wolpert, Stein, *Is the cerebellum a Smith predictor?*, J. Mot. Behav. 25, 203–216 (1993) | link to be verified | The cerebellum as delay compensator. **cited** |
| Burdet, Osu, Franklin, Milner, Kawato, *The central nervous system stabilizes unstable dynamics by learning optimal impedance*, Nature 414, 446–449 (2001) | [Nature](https://www.nature.com/articles/35106566) | Under delay the CNS raises limb impedance (stiffness and damping). The analogue of C2's friction floor γ ≥ μω sin ωΔ. **cited** |
| Gawthrop, Loram, Lakie, Gollee, *Intermittent control: a computational theory of human control*, Biol. Cybern. 104, 31–51 (2011) | [Springer](https://link.springer.com/content/pdf/10.1007/s00422-010-0416-4.pdf) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/21327829/) | "Continuous observation, intermittent action", event-driven with a refractory interval: Frost's controller, and Theorem B's requirement that the trigger observe at the inner rate. **cited** |
| Pletzer, Kerschbaum, Klimesch, *When frequencies never synchronize: the golden mean and the resting EEG*, Brain Res. 1335, 91–102 (2010) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0006899310007092) | EEG bands form a geometric series with ratio 1.618, the ratio that best avoids spurious phase synchronisation. Biological precedent for C4's noble working points. **cited** |
| Roopun et al., *Temporal interactions between cortical rhythms*, Front. Neurosci. 2, 145–154 (2008) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2622758/) | Neighbouring cortical rhythm peaks near the golden ratio, minimising interference between coexisting rhythms. **cited** |
| Kramer, *Golden rhythms as a theoretical framework for cross-frequency organization*, Neurons Behav. Data Anal. Theory (2022) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10181851/) · [arXiv:2111.09953](https://arxiv.org/abs/2111.09953) | The theory behind golden-ratio separation. **read** |
| Lisman, Jensen, *The theta–gamma neural code*, Neuron 77, 1002–1016 (2013) | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0896627313002316) | Nested slow/fast rhythms (several gamma cycles per theta cycle): a biological multirate hierarchy. Analogy only. **read** |
| Pfeiffer, Foster, *Hippocampal place-cell sequences depict future paths to remembered goals*, Nature 497, 74–79 (2013) | [Nature](https://www.nature.com/articles/nature12112) | Time-compressed simulation of future paths in idle moments. Motivates using the student GPU's idle time in the teacher-bound pipeline. **cited** |
| McClelland, McNaughton, O'Reilly, *Why there are complementary learning systems in the hippocampus and neocortex*, Psychol. Rev. 102, 419–457 (1995) | link to be verified | Fast and slow learners (C5), and consolidation by replay. **cited** (also §12.4) |
| Hopfield, *Kinetic proofreading: a new mechanism for reducing errors in biosynthetic processes requiring high specificity*, PNAS 71, 4135–4139 (1974) | link to be verified | Multi-stage verification; considered for the statistics contract and rejected (§10). **read** |

## 14. MLSys reframing (new_directions.md Part II, 2026-09-29)

| Source | Link | Used for |
|---|---|---|
| Kang et al., *NoScope: optimizing neural network queries over video at scale*, VLDB 2017 | [PVLDB](https://www.vldb.org/pvldb/vol10/p1586-kang.pdf) | Specialised models, difference detectors, cost-based cascade optimiser; R0, R3. **cited** |
| Jiang, Ananthanarayanan, Bodík, Sen, Stoica, *Chameleon: scalable adaptation of video analytics*, SIGCOMM 2018 | [ACM](https://dl.acm.org/doi/10.1145/3230543.3230574) | Adaptive configurations amortised over time; R3. **cited** |
| Mullapudi, Chen, Zhang, Ramanan, Fatahalian, *Online model distillation for efficient video inference* (JITNet), ICCV 2019 | [CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Mullapudi_Online_Model_Distillation_for_Efficient_Video_Inference_ICCV_2019_paper.pdf) | Online student with an adaptively sparse teacher; R7. **cited** |
| Ho et al., *More effective distributed ML via a stale synchronous parallel parameter server*, NeurIPS 2013 | [ACM](https://dl.acm.org/doi/10.5555/2999611.2999748) | Bounded staleness with a tuned bound; R2. **cited** |
| Narayanan et al., *PipeDream: generalized pipeline parallelism for DNN training*, SOSP 2019 | [PDF](https://people.eecs.berkeley.edu/~matei/papers/2019/sosp_pipedream.pdf) | Pipelining with stale weights; R2. **cited** |
| Donzis, Aditya, *Asynchronous finite-difference schemes for partial differential equations*, J. Comput. Phys. 274, 370 (2014); Aditya, Donzis, *High-order asynchrony-tolerant finite difference schemes*, J. Comput. Phys. (2017) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0021999114004240) · [arXiv:1702.00496](https://arxiv.org/abs/1702.00496) | Stale halo data keeps stability, loses order; R2's CFD analogue. **cited** |
| Reynolds et al., SUNDIALS ARKODE MRIStep (adaptive multirate since v7.2) | [docs](https://sundials.readthedocs.io/en/latest/arkode/Introduction_link.html) · [arXiv:2506.14976](https://arxiv.org/abs/2506.14976) | Adaptive multirate for physics terms; R0, R3. **cited** |
| Breuer, Heinecke, Bader, *Petascale local time stepping for the ADER-DG finite element method*, IPDPS 2016 (SeisSol) | [IEEE](https://ieeexplore.ieee.org/document/7516082/) | Clustered local time stepping, 2.3–4.1× (up to 10×); R8. **cited** |
| Klemp, Skamarock, Dudhia, *Conservative split-explicit time integration methods for the compressible nonhydrostatic equations*, Mon. Weather Rev. 135, 2897 (2007) | [AMS](https://journals.ametsoc.org/view/journals/mwre/135/8/mwr3440.1.xml) | Divergence damping of fast acoustic modes; R2, R6. **cited** |
| Partee et al., *Using machine learning at scale in numerical simulations with SmartSim*, J. Comput. Sci. 62, 101707 (2022) | [GitHub](https://github.com/CrayLabs/SmartSim) · [arXiv:2104.09355](https://arxiv.org/abs/2104.09355) | In-situ ML coupling plumbing; R0. **cited** |
| NVIDIA TorchFort | [GitHub](https://github.com/NVIDIA/TorchFort) | In-situ training and inference for HPC codes; R0. **read** |
| Ward et al., *Colmena: scalable machine-learning-based steering of ensemble simulations*, MLHPC 2021 | [arXiv:2110.02827](https://arxiv.org/abs/2110.02827) | ML steering of simulation campaigns; R0. **read** |
| Litman et al., *i-PI 3.0*, J. Chem. Phys. 161, 062504 (2024) | [arXiv:2405.15224](https://arxiv.org/abs/2405.15224) | Socket-decoupled force providers, MTS; R0, R2. **read** |
| List, Chen, Thuerey, *Learned turbulence modelling with differentiable fluid solvers*, J. Fluid Mech. 949, A25 (2022) | [arXiv:2202.06988](https://arxiv.org/abs/2202.06988) | Unrolled solver-in-the-loop training; R5. **cited** |
| Storcks, Buck, *Solver-in-the-loop applications in astrophysical (magneto)hydrodynamics*, 2025 | [arXiv:2512.05999](https://arxiv.org/abs/2512.05999) · [code](https://github.com/leo1200/eurips25corr) | astronomix's CNN MHD corrector (curl-of-E, divergence-free); R1, tests 3–5. **cited** |
| *Per Astronomix ad Astra: high-order differentiable (magneto)hydrodynamics*, 2026 | [arXiv:2607.18176](https://arxiv.org/abs/2607.18176) · [GitHub](https://github.com/leo1200/astronomix) | The CFD solver used in tests 3–5. **code** |
| Rein, Tamayo (2016), MEGNO and variational equations in REBOUND | [REBOUND docs](https://rebound.hanno-rein.de/chaos/) | Fast chaos indicators; R6, R9. **read** |


## 15. Systems speedups across solids, liquids and CFD (new_directions.md Part III, 2026-10-01)

| Source | Link | Used for |
|---|---|---|
| Kochkov, Smith, Alieva, Wang, Brenner, Hoyer, *Machine learning–accelerated computational fluid dynamics*, PNAS 118, e2101784118 (2021) | [arXiv:2102.01010](https://arxiv.org/abs/2102.01010) · [JAX-CFD](https://github.com/google/jax-cfd) | Kolmogorov-flow benchmark (W1), correlation-time metric; 8–10× finer-grid accuracy, 40–80× speedup. **cited** |
| Um, Brand, Fei, Holl, Thuerey, *Solver-in-the-loop: learning from differentiable physics to interact with iterative PDE-solvers*, NeurIPS 2020 | [arXiv:2007.00016](https://arxiv.org/abs/2007.00016) | Learned correctors trained through the solver; learned CG initial guesses. **cited** |
| Balin et al., *In situ framework for coupling simulation and machine learning with application to CFD*, 2023 | [arXiv:2306.12900](https://arxiv.org/abs/2306.12900) | PHASTA + SmartSim, asynchronous in-transit inference, ~2.2 ms per call; S0. **cited** |
| Fischer et al., *NekRS, a GPU-accelerated spectral element Navier–Stokes solver*, 2021 | [arXiv:2104.05829](https://arxiv.org/abs/2104.05829) | The pressure solve dominates incompressible CFD; S0. **cited** |
| Fischer, *Projection techniques for iterative solution of Ax = b with successive right-hand sides*, CMAME 163, 193 (1998) | [DOI](https://doi.org/10.1016/S0045-7825(98)00012-7) | The strongest classical warm start; M5 baseline. **cited** |
| Initial guesses for sequences of linear systems in a GPU-accelerated incompressible flow solver, 2020 | [arXiv:2009.10863](https://arxiv.org/abs/2009.10863) | Extrapolation vs projection warm starts; M5. **read** |
| *NOWS: Neural operator warm starts for accelerating iterative solvers*, 2025 | [arXiv:2511.02481](https://arxiv.org/abs/2511.02481) | Learned warm starts for the pressure projection; M5 prior art. **read** |
| Kaneda et al., *A deep conjugate direction method for iteratively solving linear systems*, ICML 2023 | [PMLR](https://proceedings.mlr.press/v202/kaneda23a/kaneda23a.pdf) | Learned search directions for Poisson; M5 prior art. **read** |
| PETSc SNES lagging (`-snes_lag_jacobian`, `-snes_lag_preconditioner`) | [docs](https://petsc.org/release/manualpages/SNES/SNESSetLagJacobian/) | Fixed-lag reuse of expensive operators is standard practice; S0. **cited** |
| Pope, *Computationally efficient implementation of combustion chemistry using in situ adaptive tabulation*, Combust. Theory Model. 1, 41 (1997) | [DOI](https://doi.org/10.1080/713665229) | ISAT: certified reuse cache with ellipsoids of accuracy; S0. **cited** |
| *DeepFlame: a deep learning empowered open-source platform for reacting flow simulations*, 2022 | [arXiv:2210.07094](https://arxiv.org/abs/2210.07094) · [GitHub](https://github.com/deepmodeling/deepflame-dev) | DNN chemistry inside OpenFOAM; S0. **read** |
| Krasnopolsky, Fox-Rabinovitz, Chalikov, *New approach to calculation of atmospheric model physics: accurate and fast neural network emulation of longwave radiation*, Mon. Weather Rev. 133, 1370 (2005) | [AMS](https://journals.ametsoc.org/mwr/article/133/5/1370/67520/) | NN radiation emulation; S0. **read** |
| Song, Roh, *Improved weather forecasting using neural network emulation for radiation parameterization*, JAMES 13 (2021); *Compound parameterization to improve the accuracy of radiation emulator*, GRL 48 (2021) | [JAMES](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002609) · [GRL](https://doi.org/10.1029/2021GL095043) | Emulator × infrequent calls (1,080–2,060×); fallback where an error-NN flags doubt; closest CFD-side prior art for M1 + never-lose. **cited** |
| *ANCHOR: error-controlled adaptive numerical correction for neural operator time marching*, 2025 | [arXiv:2512.19643](https://arxiv.org/abs/2512.19643) | Residual-EMA estimator triggers the solver; S0 prior art. **cited** |
| *Divergence-aware adaptive prediction framework for accelerating CFD simulations of unsteady flows*, 2026 | [arXiv:2605.24150](https://arxiv.org/abs/2605.24150) | POD-NN surrogate with event-triggered CFD refresh, ~92×; S0 prior art. **cited** |
| *INC: an indirect neural corrector for auto-regressive hybrid PDE solvers*, NeurIPS 2025 | [arXiv:2511.12764](https://arxiv.org/abs/2511.12764) | Corrections enter the equations (tendencies), not the state; W1–W2 design. **cited** |
| Parger et al., *DeltaCNN: end-to-end CNN inference of sparse frame differences in videos*, CVPR 2022 | [arXiv:2203.03996](https://arxiv.org/abs/2203.03996) · [GitHub](https://github.com/facebookresearch/DeltaCNN) | Change-driven sparse CNN updates, up to 7×; M4. **cited** |
| Habibian et al., *Skip-convolutions for efficient video processing*, CVPR 2021 | [arXiv:2104.11487](https://arxiv.org/abs/2104.11487) | Per-pixel skip of residual computation, 3–4×; M4. **cited** |
| Ren, Pokrovsky, Yang, Urtasun, *SBNet: sparse blocks network for fast inference*, CVPR 2018 | [arXiv:1801.02108](https://arxiv.org/abs/1801.02108) | Tile gather/scatter for dense kernels on sparse masks; M4. **cited** |
| Cohen et al., *TorchSim: an efficient atomistic simulation engine in PyTorch*, 2025 | [arXiv:2508.06628](https://arxiv.org/abs/2508.06628) | Batched MD across systems; M3 baseline. **cited** |
| Karlin, Manasse, Rudolph, Sleator, *Competitive snoopy caching*, Algorithmica 3, 79 (1988) | [DOI](https://doi.org/10.1007/BF01762111) | Ski rental, 2-competitive refresh; S1, M5. **cited** |
| McGreivy, Hakim, *Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related partial differential equations*, Nat. Mach. Intell. 6, 1256 (2024) | [Nature MI](https://www.nature.com/articles/s42256-024-00897-5) | 60 of 76 ML-for-PDE papers claiming to beat standard numerics used weak baselines; our DS 128x128 at dt/2 was one; S5 frontier. **cited** |
| Dresdner, Kochkov, Norgaard, Zepeda-Nunez, Smith, Brenner, Hoyer, *Learning to correct spectral methods for simulating turbulent flows*, TMLR 2023 | [arXiv:2207.00556](https://arxiv.org/abs/2207.00556) | Learned corrections 2-4x more accurate than spectral at the same resolution, at about twice the runtime; `w1spec.py`. **cited** |
| Kochkov et al., *Neural general circulation models for weather and climate*, Nature 632, 1060 (2024) | [arXiv:2311.07222](https://arxiv.org/abs/2311.07222) · [GitHub](https://github.com/neuralgcm/neuralgcm) | Learned physics held for ~30 min of simulated time (GCM practice); one physics call ~1.5x a dycore step; prior art for M1 in climate. Hosted unmodified in `third_party/` as workload G1 (Part IV): physics 26% of a 2.8-degree step on an A6000, the dynamical core 74% and layout-bound. **cited, used** |
| Hou, Li, *Computing nearly singular solutions using pseudo-spectral methods*, J. Comput. Phys. 226, 379 (2007) | [DOI](https://doi.org/10.1016/j.jcp.2007.04.014) | The high-order exponential filter instead of the 2/3 rule; our classical baseline (`Grid(dealias="exp")`). **cited** |
| Stachenfeld et al., *Learned coarse models for efficient turbulence simulation*, ICLR 2022 | [arXiv:2112.15275](https://arxiv.org/abs/2112.15275) | The encoder-process-decoder CNN with dilated convolutions used in `frost.cfd.split`. **cited** |
| Zhuang, Kochkov, Bar-Sinai, Brenner, Hoyer, *Learned discretizations for passive scalar advection in a two-dimensional turbulent flow*, Phys. Rev. Fluids 6, 064605 (2021) | [arXiv:2004.05477](https://arxiv.org/abs/2004.05477) | Learned advection 4x coarser than van Leer; reproduced at 64x64 (`tracer_learned.py`), then beaten by hole-filled spectral on cost. **cited** |
| Boral et al., *Neural ideal large eddy simulation: modeling turbulence with neural stochastic differential equations*, NeurIPS 2023 | [arXiv:2306.01174](https://arxiv.org/abs/2306.01174) | Ideal LES (the conditional mean, Langford and Moser) as the learning target; 2.7x slower than same-grid implicit LES. **read** |
| Langford, Moser, *Optimal LES formulations for isotropic turbulence*, J. Fluid Mech. 398, 321 (1999) | [DOI](https://doi.org/10.1017/S0022112099006369) | The optimal LES is a conditional average; why a deterministic run from the truncated state is not the only bound. **read** |
| Zalesak, *Fully multidimensional flux-corrected transport algorithms for fluids*, J. Comput. Phys. 31, 335 (1979) | [DOI](https://doi.org/10.1016/0021-9991(79)90051-2) | The positivity half of FCT in `tracers.fv_euler`. **cited** |
| Borges, Carmona, Costa, Don, *An improved weighted essentially non-oscillatory scheme for hyperbolic conservation laws*, J. Comput. Phys. 227, 3191 (2008) | [DOI](https://doi.org/10.1016/j.jcp.2007.11.038) | WENO-Z, the finite-volume tracer baseline. **cited** |
| NeuralGCM authors, *dinosaur: a differentiable dynamical core for atmospheric modeling* (JAX) | [GitHub](https://github.com/neuralgcm/dinosaur) | NeuralGCM's spectral dynamical core; its `RealSphericalHarmonics` does the longitude transform as a DFT matmul ("faster than an explicit FFT at the resolutions we use", on TPU); our `FFTSphericalHarmonics` subclasses it. **used** |
| Carver, Merose, *ARCO-ERA5: an analysis-ready, cloud-optimized reanalysis dataset* (Google, 2023) | [GitHub](https://github.com/google-research/arco-era5) | Public 0.25-degree, 37-level ERA5 zarr read anonymously for G1's initial states and verification. **used** |
| Rasp et al., *WeatherBench 2: a benchmark for the next generation of data-driven global weather models*, JAMES 16 (2024) | [arXiv:2308.15560](https://arxiv.org/abs/2308.15560) | Headline variables (z500, t850) and latitude-weighted RMSE for G1's skill gate. **cited** |
| Hogan, Bozzo, *A flexible and efficient radiation scheme for the ECMWF model*, JAMES 10, 1990 (2018); ECMWF Newsletter 155, *A new radiation scheme for the IFS* | [PDF](https://www.met.reading.ac.uk/~swrhgnrj/publications/ecrad.pdf) · [newsletter](https://www.ecmwf.int/en/newsletter/155/meteorology/new-radiation-scheme-ifs) | Full radiation every 1-3 h on a coarser grid plus approximate updates every step and grid point (skin temperature, solar angle): Frost's student-plus-residual in operational NWP, uncertified; a 3 h step degrades skill and 2 h recovers more than half. **cited** |
| Manners, Thelen, Petch, Hill, Edwards, *Two fast radiative transfer methods to improve the temporal sampling of clouds in numerical weather prediction and climate models*, QJRMS 135, 457 (2009) | [DOI](https://doi.org/10.1002/qj.385) | The solar-path correction used between full radiation calls. **read** |
| Bi et al., *Accurate medium-range global weather forecasting with 3D neural networks* (Pangu-Weather), Nature 619, 533 (2023) | [DOI](https://doi.org/10.1038/s41586-023-06185-3) | Hierarchical temporal aggregation (24/6/3/1 h models combined greedily): multirate scheduling of learned simulators, without a certificate. **cited** |
| Bae, Koumoutsakos, *Scientific multi-agent reinforcement learning for wall-models of turbulent flows*, Nat. Commun. 13, 1443 (2022) | [DOI](https://doi.org/10.1038/s41467-022-28957-7) | Learned wall models: physics that wall-resolved LES buys only at cost ~Re^1.86; Part IV candidate. **read** |
| Lozano-Duran, Bae, *Machine learning building-block-flow wall model for large-eddy simulation*, J. Fluid Mech. 963, A35 (2023) | [DOI](https://doi.org/10.1017/jfm.2023.331) | A learned wall model across flow regimes; Part IV candidate. **read** |
| NVIDIA, *NVIDIA Ampere GA102 GPU architecture* (whitepaper, 2021); *RTX A6000 datasheet* | [PDF](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf) | sm_86: 128 FP32 lanes per SM, third-generation tensor cores (TF32, BF16), 6 MB L2, 768 GB/s GDDR6; measured against in `ampere_probe.py`. **cited** |
| Arcomano, Szunyogh, Wikner, Pathak, Hunt, Ott, *A hybrid approach to atmospheric modeling that combines machine learning with a physics-based numerical model*, JAMES 14 (2022) | [DOI](https://doi.org/10.1029/2021MS002712) | The coupling interval of a hybrid model treated as a tuned hyperparameter (6 h beat 1 h and 3 h, each retuned) and the physics model run with a finer internal step: the schedule fixed at training time, as in NeuralGCM. Prior art for the schedule lock, not for changing it at run time. **cited** |
| Rasp, *Coupled online learning as a way to tackle instabilities and biases in neural network parameterizations*, GMD 13, 2185 (2020) | [arXiv:1907.01351](https://arxiv.org/abs/1907.01351) | Learned parameterizations destabilise once coupled to a dynamical core; training online against the coupled model is the cure. The coupling it trains through includes the time step. **cited** |
| Fish, Reynolds, *Adaptive time step control for multirate infinitesimal methods*, SIAM J. Sci. Comput. 45 (2023) | [arXiv:2202.10484](https://arxiv.org/abs/2202.10484) | Multirate step-size controllers for fast and slow partitions in classical numerics: the controller half of a certified multirate runtime, without learned components or a skill certificate. **cited** |

## 16. Our own CFD solvers, end to end (new_directions.md Part V, 2026-10-14)

| Paper | Link | Use |
|---|---|---|
| Sirignano, MacArt, Freund, *DPM: a deep learning PDE augmentation method (with application to large-eddy simulation)*, J. Comput. Phys. 423, 109811 (2020) | [arXiv:1911.09145](https://arxiv.org/abs/1911.09145) | A closure trained through the solver (adjoint) beats constant and dynamic Smagorinsky on isotropic turbulence, most at filter sizes where those become qualitatively wrong: the coarse-grid statistics regime where learned closures have headroom. **cited** |
| Novati, Lascombes de Laroussilhe, Koumoutsakos, *Automating turbulence modelling by multi-agent reinforcement learning*, Nat. Mach. Intell. 3, 87 (2021) | [arXiv:2005.09023](https://arxiv.org/abs/2005.09023) | SGS models for forced isotropic turbulence rewarded for recovering DNS statistics (spectra); favourable against dynamic Smagorinsky, across grids. Statistics, not forecasts, as the LES target. **cited** |
| List, Chen, Thuerey, *Learned turbulence modelling with differentiable fluid solvers: physics-based loss functions and optimisation horizons*, J. Fluid Mech. (2022) | [arXiv:2202.06988](https://arxiv.org/abs/2202.06988) | Longer unrolled horizons and spectral losses make learned closures stable and statistically right a posteriori. **cited** |
| Frezat, Le Sommer, Fablet, Balarac, Lguensat, *A posteriori learning for quasi-geostrophic turbulence parametrization*, JAMES 14, e2022MS003124 (2022) | [arXiv:2204.03911](https://arxiv.org/abs/2204.03911) | A posteriori training reproduces high-resolution spectra and fluxes on coarse grids where a priori training and physical closures do not. **cited** |
| Nabavi, Kim, *Optimising subgrid-scale closures for spectral energy transfer in turbulent flows*, J. Fluid Mech. 982, A18 (2024) | [JFM](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/optimising-subgridscale-closures-for-spectral-energy-transfer-in-turbulent-flows/2FF7DD0D5B443D7A89566D58D6FEDA0A) | The optimal Smagorinsky constant depends on the cutoff (Lilly's value only inside the inertial range), and dynamic Smagorinsky is suboptimal at coarse cutoffs: the classical baseline must be tuned per grid, as `stats3d.py` does. **read** |
| *Deep learning of solver-aware turbulence closures from nudged LES dynamics* (2026) | [arXiv:2604.23874](https://arxiv.org/abs/2604.23874) | Forced 3D isotropic turbulence, 32^3 LES against 128^3 DNS: learned corrections cut the relative turbulent-kinetic-energy error from 0.126 (Smagorinsky) to 0.011-0.015, trained offline on nudging forces; no timings. **read** |
| *EddyFormer: accelerated neural simulations of three-dimensional turbulence at scale* (2025) | [arXiv:2510.24173](https://arxiv.org/abs/2510.24173) | A neural surrogate reaching DNS-level accuracy at 256^3 for 30x over DNS on one A100: the equal-accuracy framing for 3D. **cited** |
| *Normality-based analysis of multiscale velocity gradients and energy transfer in direct and large-eddy simulations of isotropic turbulence* (2025) | [arXiv:2504.19356](https://arxiv.org/abs/2504.19356) | An eddy-viscosity LES resembles an unfiltered DNS at lower Re in its velocity-gradient structure; a mixed model resembles the filtered DNS. Why our learned closure's gradient skewness (-0.280 against -0.318) sat at a coarser classical grid's level, and why classical's best baseline for gradient statistics is a mixed model (`spectral3d.mixed`). **read** |
| Clark, Ferziger, Reynolds, *Evaluation of subgrid-scale models using an accurately simulated turbulent flow*, J. Fluid Mech. 91, 1 (1979) | [DOI](https://doi.org/10.1017/S002211207900001X) | The gradient (Clark) model tau_ij = Delta^2/12 d_k u_i d_k u_j, the nonlinear half of the mixed model added to `spectral3d` as the classical baseline for gradient statistics. **used** |

## 17. MHD turbulence, W5 (new_directions.md Part VI, 2026-10-15)

| Paper | Link | Use |
|---|---|---|
| Grete, Vlaykov, Schmidt, Schleicher, *Comparative statistics of selected subgrid-scale models in large eddy simulations of decaying, supersonic MHD turbulence*, Phys. Rev. E 95, 033206 (2017) | [arXiv:1703.00858](https://arxiv.org/abs/1703.00858) | 209 simulations: eddy-viscosity, scale-similarity and nonlinear SGS models have no statistical influence when computed from grid-scale quantities; only explicit filtering lets eddy-viscosity and nonlinear models change the statistics. Classical MHD closures buy little, so resolution is classical MHD's main lever: the headroom W5 tests. **read** |
| *Large eddy simulations in astrophysics*, Living Rev. Comput. Astrophys. (2025) | [Springer](https://link.springer.com/article/10.1007/s41115-025-00023-1) | Review: all investigated MHD SGS models have only a very small impact on the flow and field; the nonlinear closure with an explicit filter matched doubled-resolution kurtosis and skewness better in driven MHD turbulence. **read** |
| Vigano, Aguilera-Miret, Carrasco, Minano, Palenzuela, *GRMHD large eddy simulations with gradient subgrid-scale model*, Phys. Rev. D 101, 063003 (2020) | [arXiv:2004.00870](https://arxiv.org/abs/2004.00870) | The gradient (Taylor-expansion) SGS model carried to (relativistic) MHD: the form of `mhd3d`'s mixed closure for the stress and the electromotive force. **cited** |

## 18. Fixing the CFD and MHD losses (2026-10-21, frost-27)

Searched per failure (stochastic closures for the MHD intermittency deficit; higher-Re references for the Taylor-Green
vortex; learned closures that beat classical LES in 2D on statistics; shock-capturing and local time stepping for the
blast wave; why The Well's checkpoints do not reproduce). Plan and tests: `gaps.md`, Part E.4.

| Paper | Link | Use |
|---|---|---|
| Guillaumin and Zanna, *Stochastic-deep learning parameterization of ocean momentum forcing*, JAMES 13 (2021) | [ESS Open Archive](https://essopenarchive.org/users/538113/articles/599526-stochastic-deep-learning-parameterization-of-ocean-momentum-forcing) | A closure that predicts a mean and a local standard deviation (Gaussian, negative log-likelihood) instead of the conditional mean: the pattern for our noise-amplitude head (`split3d.add_noise_head`) |
| Perezhogin, Zanna and Fernandez-Granda, *Generative data-driven approaches for stochastic subgrid parameterizations in an idealized ocean model*, JAMES (2023) | [arXiv:2302.07984](https://arxiv.org/abs/2302.07984) | Generative (GAN, VAE) stochastic closures with spatially correlated noise beat the deterministic ML closure online at the coarsest resolution: the evidence that a stochastic term fixes what a mean closure smooths away |
| Gneiting and Raftery, *Strictly proper scoring rules, prediction, and estimation*, JASA 102, 359 (2007) | [DOI](https://doi.org/10.1198/016214506000001437) | The energy score E|X - y| - E|X - X'|/2: the proper objective for a stochastic closure (MSE drives the noise to zero); also what NeuralGCM's stochastic models train on (CRPS) |
| Schumann, *Stochastic backscatter of turbulence energy and scalar variance by random subgrid-scale fluxes*, Proc. R. Soc. A 451, 293 (1995) | [DOI](https://doi.org/10.1098/rspa.1995.0126) | Classical stochastic backscatter in LES: random subgrid fluxes restore resolved-scale variance an eddy viscosity removes |
| Laval and Dubrulle, *A LES-Langevin model for turbulence*, Eur. Phys. J. B (2006) | [arXiv:physics/0602103](https://arxiv.org/abs/physics/0602103) | The subgrid cross term as a Langevin equation whose noise carries backscatter; better than eddy viscosity in isotropic turbulence |
| *Subgrid modelling for relativistic magnetohydrodynamics with machine learning*, Phys. Rev. D (2026) | [arXiv:2606.21659](https://arxiv.org/abs/2606.21659) | ML subgrid models for (relativistic) MHD Kelvin-Helmholtz, a posteriori: field amplification of a 4x-resolution run at a 44x speed-up; a direct competitor on Cook and Bernuzzi's protocol |
| Guan, Chattopadhyay, Subel and Hassanzadeh, *Stable a posteriori LES of 2D turbulence using CNNs: backscattering analysis and generalization to higher Re via transfer learning*, J. Comput. Phys. (2022) | [arXiv:2102.11400](https://arxiv.org/abs/2102.11400) | 2D LES with a CNN closure is stable a posteriori with enough data; instabilities come from poorly captured backscatter; transfer learning with ~1% new data reaches 16x higher Re: the recipe for our 2D statistics regime |
| Frezat, Le Sommer, Fablet, Balarac and Lguensat, *A posteriori learning for quasi-geostrophic turbulence parametrization*, JAMES 14 (2022) | [arXiv:2204.03911](https://arxiv.org/abs/2204.03911) | End-to-end (a posteriori) training beats a priori and empirical closures on performance, stability and transfer in 2D QG turbulence: what our solver-in-the-loop training does |
| Stachenfeld et al., *Learned coarse models for efficient turbulence simulation*, ICLR (2022) | [arXiv:2112.15275](https://arxiv.org/abs/2112.15275) | Learned coarse models are more accurate than classical solvers at the same low resolution on several turbulent systems, keeping high-frequency content: the regime (statistics at coarse grids) where the 2D comparison should be made |
| Jakhar, Guan, Mojgani, Chattopadhyay and Hassanzadeh, *Learning closed-form equations for subgrid-scale closures from high-fidelity data*, JAMES (2024) | [DOI](https://doi.org/10.1029/2023MS003874) | Discovered closures (the nonlinear gradient model) and why they fail online without backscatter handling |
| *An analytical and AI-discovered stable, accurate, and generalizable subgrid-scale closure for geophysical turbulence* (2025) | [arXiv:2509.20365](https://arxiv.org/abs/2509.20365) | Higher-order gradient closures (NGM4, NGM6) capture the vorticity PDF's tails where eddy viscosity misses the extremes: an intermittency-capable classical-structured baseline |
| Srinivasan et al., *Turbulence closure with small, local neural networks: forced two-dimensional and beta-plane flows* (2023) | [arXiv:2304.05029](https://arxiv.org/abs/2304.05029) | Small local closures (cheap per call) for forced 2D turbulence: the cost side of the 2D comparison |
| Shankar et al., *Differentiable turbulence: closure as a PDE-constrained optimization* (2023) | [arXiv:2307.03683](https://arxiv.org/abs/2307.03683) | A posteriori closure learning through an adjoint solver; a-priori skill not predictive of a-posteriori |
| List, Chen and Thuerey, *Learned turbulence modelling with differentiable fluid solvers*, J. Fluid Mech. (2022) | [arXiv:2202.06988](https://arxiv.org/abs/2202.06988) | Physics-based losses and long optimisation horizons for learned closures (our statistics losses, longer unrolls) |
| Dairay, Lamballais, Laizet and Vassilicos, *Numerical dissipation vs. subgrid-scale modelling for large eddy simulation*, J. Comput. Phys. 337, 252 (2017) | [DOI](https://doi.org/10.1016/j.jcp.2017.02.035) | Taylor-Green vortex up to Re 5000 with a spectral-vanishing-viscosity ILES that beats static and dynamic Smagorinsky; the classical baseline at higher Re |
| Laizet, Vassilicos, Lamballais and Dairay, *3D Taylor-Green vortex DNS statistics from Re = 1250 to Re = 20000*, Zenodo (2019), CC BY 4.0 | [DOI](https://doi.org/10.5281/zenodo.2577239) | Published DNS time series (energy, dissipation, enstrophy, gradient moments) at Re 1250-20000: the reference that lets the TGV family move to Re 5000-10000 without a 1024^3 DNS of our own |
| *Discovery of sparse invariant subgrid-scale closures via dissipation-controlled training for LES on anisotropic grids* (2026) | [arXiv:2604.25824](https://arxiv.org/abs/2604.25824) | A data-driven closure judged on the TGV Re 1600 dissipation curve: a learned competitor on our TGV metric |
| Huang, Sebastian and Viswanathan, *Learning second-order TVD flux limiters using differentiable solvers* (2025) | [arXiv:2503.09625](https://arxiv.org/abs/2503.09625) | A learned limiter constrained to a convex blend of minmod and superbee (TVD and second order guaranteed), trained on advection, good on shocks, integrated into OpenFOAM: a pointwise, cheap learned component for shock-dominated FV (the blast wave), replacing a CNN corrector that cost 40% of a step |
| Nguyen-Fotiadis et al., *Machine learning changes the rules for flux limiters* (2021) | [arXiv:2108.11864](https://arxiv.org/abs/2108.11864) | Data-driven limiter selection on coarse-grained Burgers |
| Dumbser et al., *ALE ADER-WENO finite volume schemes with time-accurate local time stepping for hyperbolic conservation laws* (2014) | [arXiv:1402.6897](https://arxiv.org/abs/1402.6897) | Time-accurate local time stepping for ideal MHD: the classical precedent for skipping updates where the solution does not change (blast-wave ambient regions) |
| *High order explicit local time-stepping methods for hyperbolic conservation laws* (2019) | [arXiv:1905.09705](https://arxiv.org/abs/1905.09705) | LTS error versus step ratio on Sod, Lax and interacting blast waves |
| Ohana et al., *The Well: a large-scale collection of diverse physics simulations for machine learning*, NeurIPS Datasets and Benchmarks (2024) | [arXiv:2412.00568](https://arxiv.org/abs/2412.00568) | MHD_64 one-step test VRMSE CNextU-Net 0.163 (validation 0.149). The dataset's Hugging Face history: stats.yaml uploaded 2025-04-10, the data files re-uploaded 2026-02-25 (test files 0.76 -> 0.73 GB), after the checkpoints were trained: the reproduction is rerun on revision b09b4d0 (2024-12-05) |

## 19. Fixing the MD losses (2026-10-26, frost-27)
The MD loss list (water 1.00x, MACE-NEP faster at scale, FlashMD faster on the oxides, the certified 3500 K ramp slower
than an uncertified fixed interval) traced to two causes: the impulse resonance cap binds Cu and the oxides far below
their staleness budget, and under a thermostat the certificate has no measured drift clause (water, the ramp and the
two-GPU mode are all thermostatted). Plan, tests and results: `gaps.md`, Part E.5.
| Paper | Link | Use |
|---|---|---|
| Izaguirre, Reich, Skeel, *Longer time steps for molecular dynamics* (MOLLY), J. Chem. Phys. 110, 9853 (1999) | [IDEALS record](https://ideals.illinois.edu/items/83239) | Impulse MTS is unstable just below half the fastest period; the mollified slow force buys about 50% longer outer steps. Not enough for Cu at scale (k would need about 2x the cap to beat MACE-NEP at 4,000 atoms). **cited** |
| Sivak, Chodera, Crooks, *Using nonequilibrium fluctuation theorems to understand and correct errors in equilibrium and nonequilibrium simulations of discrete Langevin dynamics*, Phys. Rev. X 3, 011007 (2013) | [doi:10.1103/PhysRevX.3.011007](https://doi.org/10.1103/PhysRevX.3.011007) · [arXiv:1107.2967](https://arxiv.org/abs/1107.2967) | Shadow work: under a discrete Langevin integrator the energy change not accounted for by the bath's heat measures the discretisation error. Basis of `Integrator.track_heat` and `md_runtime.py --nvt-drift`. **used** |
| Fass, Sivak, Crooks, Beauchamp, Leimkuhler, Chodera, *Quantifying configuration-sampling error in Langevin simulations of complex molecular systems*, Entropy 20, 318 (2018) | [doi:10.3390/e20050318](https://doi.org/10.3390/e20050318) | Shadow work for BAOAB-type splittings (heat is exchanged only in the O step), and its use as an integrator-error measure on molecular systems. **used** |
| Bussi, Donadio, Parrinello, *Canonical sampling through velocity rescaling*, J. Chem. Phys. 126, 014101 (2007) | [arXiv:0803.4060](https://arxiv.org/abs/0803.4060) | The effective energy: total energy less the bath's accumulated heat, conserved by exact thermostatted dynamics and used to measure a run's accuracy. **used** |

## 20. Competitor code and data added for the MD paper (2026-11)
| Source | Link | Use |
|---|---|---|
| Kanda, Yamazaki, Yoshimoto, Izumi, Inoue, Kawaguchi, Sakai, *Multiscale and Multi-Timestep Switching of Multiple Machine Learning Force Fields for Artificial Intelligence-Driven Materials Simulations*, Adv. Intell. Discov. e70152 (2026) | [doi:10.1002/aidi.70152](https://doi.org/10.1002/aidi.70152) | Switching between a 4 A and a 6 A DeePMD model (accurate model every 4th step): 1.24-1.32x alone, 3.95x with compression and fp16. **cited, baseline** |
| Kanda et al., Scale MTS code: a DeePMD-kit fork with the LAMMPS pair style `deepmd/mts`, their TiO2 models, inputs and analysis scripts (released 2026-11-07; training data and PEG not included) | [github.com/kandaryuya/deepmd-mts](https://github.com/kandaryuya/deepmd-mts) | `third_party/deepmd-mts` (commit c1b3692), built in `.venvs/frost-kanda` for a T1 rerun. **code, models** |
| LAMMPS stable_2Aug2023_update3 | [lammps/lammps release](https://github.com/lammps/lammps/releases/tag/stable_2Aug2023_update3) | The engine their pair style builds into (GPL; own build, never imported by `src/frost`). **code** |
| NVIDIA CUDA Toolkit 12.3.2 | [download archive](https://developer.nvidia.com/cuda-12-3-2-download-archive) | Their reference build's toolkit, installed toolkit-only into `.tools/cuda-12.3` (the pip CUDA 12 wheels ship no nvcc). **tool** |
| Cheng, Engel, Behler, Dellago, Ceriotti, *Ab initio thermodynamics of liquid and solid water*, PNAS 116, 1110 (2019) | [doi:10.1073/pnas.1815117116](https://doi.org/10.1073/pnas.1815117116) · [Materials Cloud 2018.0020/v1](https://archive.materialscloud.org/record/2018.0020/v1) | The revPBE0-D3 water set MTS-Allegro trains on (1,593 frames of 192 atoms; CC BY 4.0): the MTS-Allegro rebuild's data and the water head-to-head. **data** |
| nequip 0.19.1 and nequip-allegro 0.8.3 | [mir-group/nequip](https://github.com/mir-group/nequip) · [mir-group/allegro](https://github.com/mir-group/allegro) | Public builder for the MTS-Allegro rebuild and the Allegro-MP-L teacher (`frost-nequip`). **code** |

## 21. Structure families for the frozen certificate (2026-11-12, gaps.md E.8)
| Source | Link | Use |
|---|---|---|
| Byggmastar, Hamedani, Nordlund, Djurabekova, *Machine-learning interatomic potential for radiation damage and defects in tungsten*, PRB 100, 144105 (2019) | [arXiv:1908.07330](https://arxiv.org/abs/1908.07330v1) | W defects, crowdions, cascades. **cited** |
| Large-scale ML MD of primary radiation damage in tungsten | [arXiv:2305.08140](https://ar5iv.labs.arxiv.org/html/2305.08140) | Cascade protocol (future item). **read** |
| Yin et al., dislocation mobility in refractory high-entropy alloys and chemical short-range order, Nat. Commun. (2021) | [PMC8357793](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8357793/) | NbMoTaW, the HEA test system. **cited** |
| Li et al., NbMoTaW with a SNAP potential, npj Comput. Mater. (2020) | [paper](https://materialsvirtuallab.org/pubs/10.1038_s41524-020-0339-0.pdf) | NbMoTaW segregation and order. **cited** |
| Deng et al., *Overcoming systematic softening in universal machine learning interatomic potentials by fine-tuning* | [arXiv:2405.07105](https://arxiv.org/html/2405.07105v1) | What our runs do not test: the teacher's softening on defects and barriers. **cited** |
| CHIPS-FF, evaluating universal force fields for material properties | [arXiv:2412.10516](https://arxiv.org/pdf/2412.10516) | MACE-MP-0 vacancy and surface energies. **read** |
| Loew et al., phonon benchmark of universal MLIPs | [alphaxiv 2412.16551](https://www.alphaxiv.org/overview/2412.16551) | Universal models soften phonons. **read** |
| Grigorev, universal models on defects in W, Mo, Nb | [pdf](https://pgrigorev.github.io/assets/pdf/Grigorev_benchmark.pdf) | BCC point and extended defects. **read** |
| Wood and Marzari, first-principles MD of superionic alpha-AgI | [arXiv cond-mat/0603670](https://arxiv.org/pdf/cond-mat/0603670) | The BCC superionic phase above 420 K. **cited** |

## 22. Deciding the energy clause early (2026-11-12, gaps.md E.9)

| Paper | Link | Used for |
|---|---|---|
| Page, *Continuous inspection schemes*, Biometrika 41, 100-115 (1954) | [doi:10.1093/biomet/41.1-2.100](https://doi.org/10.1093/biomet/41.1-2.100) | The CUSUM change detector beside the slope test (version 3 of the drift clause, `DriftMonitor._cusum`): catches a late rise that one fitted line dilutes. **used** |
| O'Brien and Fleming, *A multiple testing procedure for clinical trials*, Biometrics 35, 549-556 (1979) | [doi:10.2307/2530245](https://doi.org/10.2307/2530245) | The group-sequential boundary z / sqrt(information fraction): the drift clause's early looks (`DriftMonitor.sequential`). **used** |
| Hairer, Lubich and Wanner, *Geometric Numerical Integration*, ch. VIII (variable step sizes) | [doi:10.1007/3-540-30666-8](https://doi.org/10.1007/3-540-30666-8) | Why a step-size sequence chosen from the state loses symplectic long-time energy behaviour: background for the interval changes the ledger accounts for. **read** |

## 23. Further MD expansions (2026-11-13, `docs/design/further_expansions_beyond.md`)

Searched and read at abstract level while planning what Frost's MD side should prove next. None is used in code yet;
each is the prior work for one planned item.

| Paper | Link | Why |
|---|---|---|
| Martyna, Tuckerman, Tobias, Klein, *Explicit reversible integrators for extended systems dynamics*, Mol. Phys. 87, 1117 (1996) | [pdf](https://aiichironakano.github.io/phys516/Martyna-Integrator-MolPhys96.pdf) | RESPA for NPT (XO / XI variants): the basis for holding the residual virial. **read (abstract)** |
| Leimkuhler, Margul, Tuckerman, *Stochastic, resonance-free multiple time-step algorithm for MD with very large time steps* (2013), SIN(R) | [arXiv:1307.1167](https://arxiv.org/abs/1307.1167) | ~100 fs outer steps by isokinetic constraints; distorts dynamics, so a sampling-only mode. **read (abstract)** |
| Morrone, Markland, Ceriotti, Berne, *Efficient multiple time scale MD: colored noise thermostats to stabilize resonances*, JCP 134 (2011) | [arXiv:1008.5377](https://arxiv.org/abs/1008.5377) | Damping the resonance that caps Frost at 0.22 of the fastest period. **read (abstract)** |
| Kapil, VandeVondele, Ceriotti, *Accurate MD and nuclear quantum effects at low cost by multiple steps in real and imaginary time*, JCP (2016) | [arXiv:1512.00176](https://arxiv.org/abs/1512.00176) | Multiple time steps plus ring-polymer contraction (with DFT): prior work for Frost x PIMD. **read (abstract)** |
| Li and Voth, *Using machine learning to greatly accelerate path integral ab initio MD* (2022) | [OSTI 1865676](https://www.osti.gov/pages/biblio/1865676) | ML potential plus contraction, ~100x over path-integral AIMD. **read (abstract)** |
| Mouvet et al., *Machine learning-enhanced multiple time-step ab initio MD* (EPFL) | [infoscience](https://infoscience.epfl.ch/handle/20.500.14299/255942) | ML corrections inside ab initio MTS: prior work for a DFT teacher. **read (abstract)** |
| Cheng, *Latent Ewald summation for machine learning of long-range interactions*, npj Comput. Mater. 11, 80 (2025) | [arXiv:2408.15165](https://arxiv.org/abs/2408.15165) | A long-range teacher whose reciprocal part could run on a third, slower clock. **read (abstract)** |
| Langer, Knoop, Carbogno, Scheffler, Rupp, *Heat flux for semi-local machine-learning potentials*, PRB (2023) | [arXiv:2303.14434](https://arxiv.org/abs/2303.14434) | The heat flux of message-passing teachers, for Green-Kubo thermal conductivity. **read (abstract)** |
| Fan et al., *Neuroevolution machine learning potentials ... heat transport* (2021) | [arXiv:2107.08119](https://arxiv.org/abs/2107.08119) | HNEMD thermal conductivity with NEP in GPUMD: the cheaper transport route. **read (abstract)** |
| Byggmastar et al., *Machine-learning interatomic potential for radiation damage and defects in tungsten* (2019) | [arXiv:1908.07330](https://arxiv.org/abs/1908.07330) | GAP with ZBL repulsion for cascades. **read (abstract)** |
| Liu et al., *Large-scale machine-learning MD simulation of primary radiation damage in tungsten* (2023), NEP-ZBL | [arXiv:2305.08140](https://arxiv.org/abs/2305.08140) | 8.1M-atom cascades on one GPU. **read (abstract)** |
| Byggmastar et al., *Four regimes of primary radiation damage in tungsten* (2025) | [arXiv:2503.02710](https://arxiv.org/abs/2503.02710) | Adaptive time step and border thermostat for cascades up to 2 MeV. **read (abstract)** |
| Wang et al., *Ab initio characterization of protein molecular dynamics with AI2BMD*, Nature 635, 1019 (2024) | [MSR blog](https://www.microsoft.com/en-us/research/blog/from-static-prediction-to-dynamic-characterization-ai2bmd-advances-protein-dynamics-with-ab-initio-accuracy) | Protein MD at ab initio accuracy: the biomolecular target class. **read (abstract)** |
| Ple, Piquemal et al., DMTS (*Accelerating MD with foundation NNPs using multiple time-step and distillation*) and DMTS-NC | [arXiv:2510.06562](https://arxiv.org/abs/2510.06562) | Competitor; 3.66-5.64x with distilled MACE-OFF23 (DMTS-NC): the molecular head-to-head. **read (abstract)** |
| Zhang et al., *Exploring the frontiers of condensed-phase chemistry with a general reactive ML potential* (ANI-1xnr), Nat. Chem. (2024) | [PMC11087274](https://pmc.ncbi.nlm.nih.gov/articles/PMC11087274/) | A reactive C/H/N/O teacher for bond-breaking tests. **read (abstract)** |
| Zeng et al., water dissociation on seven TiO2 surfaces from ML MD, Nat. Commun. (2023) | [pdf](https://www.nature.com/articles/s41467-023-41865-8.pdf) | A measured interfacial observable (dissociation fraction). **read (abstract)** |
| *Activation entropy of dislocation glide in body-centered cubic metals from atomistic simulations* (2024) | [arXiv:2410.04813](https://arxiv.org/abs/2410.04813) | Kink-pair nucleation in Fe and W with MLIPs. **read (abstract)** |
| Winter and Gomez-Bombarelli, Li-ion conduction in LGPS with a neural network potential, J. Phys. Energy (2023) | [DOAJ](https://doaj.org/article/4a9b37cae03840bfbd8fd91edbddcdee) | Ionic conductivity over tens of ns. **read (abstract)** |
| Gigli et al., BaTiO3 phase diagram from an integrated ML model, npj Comput. Mater. (2022) | [pdf](https://www.nature.com/articles/s41524-022-00845-0.pdf) | A ferroelectric transition as a free-energy observable. **read (abstract)** |
| Ice nucleation rates from an ab initio ML model of water, PNAS (2022) | [PMC9388152](https://pmc.ncbi.nlm.nih.gov/articles/PMC9388152) | Nucleation by seeding at 300,000 atoms. **read (abstract)** |
| Musaelian et al., Allegro (2022); *High-performance training and inference for deep equivariant interatomic potentials* (NequIP, 2025) | [arXiv:2204.05249](https://arxiv.org/abs/2204.05249), [arXiv:2504.16068](https://arxiv.org/abs/2504.16068) | 10^8 atoms through LAMMPS's spatial decomposition: the scale baseline. **read (abstract)** |

| Bernetti and Bussi, *Pressure control using stochastic cell rescaling*, JCP 153, 114107 (2020) | [arXiv:2006.09250](https://arxiv.org/abs/2006.09250) | The barostat for Frost's NPT: first order, right volume fluctuations, needs only the pressure, so it can act at interval boundaries where the teacher's exact virial is known. **planned (6.1)** |
| Del Tatto, Raiteri, Bernetti, Bussi, *MD of solids at constant pressure and stress using anisotropic stochastic cell rescaling*, Appl. Sci. 12, 1139 (2022) | [pdf](https://iris.sissa.it/retrieve/dd8a4bf9-0090-20a0-e053-d805fe0a8cb0/applsci-12-01139-v2.pdf) | Anisotropic cells for solids; the z-only (NP_zAT) form in `StochasticCellRescaling(axes="z")` follows the isotropic equation with the normal stress (7.2). **read (abstract), used for the z-only form** |
| Gouraud, Cattin, Ple, Adjoua, Lagardere, Piquemal, *Faster MD with NNPs via distilled multiple time-stepping and non-conservative forces* (DMTS-NC), arXiv 2602.14975 v3 | [arXiv:2602.14975](https://arxiv.org/abs/2602.14975) | Their numbers for the molecular head-to-head: water 2.94-4.31x, phenol-lysozyme 3.40x, DHFR 3.09x (FeNNix-Bio1(M)), MACE-OFF23 3.66-5.64x. **read (abstract)** |
| Kovacs et al., *MACE-OFF: short-range transferable machine learning force fields for organic molecules*, JACS (2025) | [arXiv:2312.15211](https://arxiv.org/abs/2312.15211) | The organic teacher family (small / medium / large; 10 elements), loaded through mace's own `mace_off`. **planned (6.2)** |
| Markland and Manolopoulos, *An efficient ring polymer contraction scheme for imaginary time path integral simulations*, JCP 129, 024105 (2008) | [doi:10.1063/1.2953308](https://doi.org/10.1063/1.2953308) | Contraction of the slow force onto fewer beads. **planned (6.6)** |
| Ceriotti, Parrinello, Markland, Manolopoulos, *Efficient stochastic thermostatting of path integral MD* (PILE), JCP 133, 124104 (2010) | [arXiv:1009.1045](https://arxiv.org/abs/1009.1045) | Normal-mode Langevin thermostat for the ring polymer. **planned (6.6)** |
| *Path-integral MD with actively-trained and universal machine learning force fields* (2025) | [arXiv:2505.14245](https://arxiv.org/abs/2505.14245) | PIMD with universal MLFFs, the closest current practice. **read (abstract)** |
| Martyna, Klein, Tuckerman, *Nose-Hoover chains: the canonical ensemble via continuous dynamics*, JCP 97, 2635 (1992) | [doi:10.1063/1.463940](https://doi.org/10.1063/1.463940) | The chain thermostat and its conserved quantity, for the bias clause under Nose-Hoover. **planned (6.4)** |
| *A universal augmentation framework for long-range electrostatics in MLIPs* (MACELES-OFF; LES library) (2025) | [arXiv:2507.14302](https://arxiv.org/abs/2507.14302), [github.com/ChengUCB/les](https://github.com/ChengUCB/les) | A public long-range organic teacher (CC BY-NC 4.0) for the three-level scheme. **used (2026-11-14):** les 0.2.0 from its GitHub tarball into `.tools/les-overlay` (pip --target, no dependencies, the env untouched); MACELES-OFF small (`MACELES-OFF_small_converted.model`, 1.89 M parameters, r_max 4.5) from github.com/ChengUCB/les_fit, loaded with torch's restricted unpickler after its globals (torch, e3nn, mace, les only) and embedded e3nn code were read statically; `experiments/e22_longrange/les_profile.py` |
| Gigli et al., *Mechanism of charge transport in lithium thiophosphate* (2024) | [arXiv:2310.15679](https://arxiv.org/abs/2310.15679) | Nernst-Einstein fails for superionic conductors; use the collective charge current. **used (6.7)** |
| Ziegler, Biersack, Littmark, *The Stopping and Range of Ions in Solids* (1985), the ZBL universal screening function | [SRIM](http://www.srim.org/) | The short-range repulsion to blend into the cascade teacher. **planned (6.9)** |
| Wu et al., *GPU4PySCF* (2024; J. Phys. Chem. A 2025) and *Introducing GPU-acceleration into PySCF* | [arXiv:2407.09700](https://arxiv.org/abs/2407.09700), [arXiv:2404.09452](https://arxiv.org/abs/2404.09452) | GPU DFT with nuclear gradients: a DFT teacher (tier 3, 7.1). **read (abstract)** |
| Vandermause et al., *On-the-fly active learning of interpretable Bayesian force fields* (FLARE), npj Comput. Mater. (2020) | [arXiv:1904.02042](https://arxiv.org/abs/1904.02042) | Uncertainty-triggered DFT calls (~100 per force field): the prior work for an on-the-fly DFT teacher. **read (abstract)** |
| *On-the-fly machine learning force field generation: application to melting points* (interface pinning) | [arXiv:1904.12961](https://arxiv.org/abs/1904.12961) | Melting points under NPT with ML force fields (7.2). **read (abstract)** |
| Pedersen, *Direct calculation of the solid-liquid Gibbs free energy difference in a single equilibrium simulation*, JCP 139, 104102 (2013) | [doi:10.1063/1.4818747](https://doi.org/10.1063/1.4818747) | Interface pinning with the long-range order Q = \|rho_k\| / N and dmu = kappa (Q_s - Q_l)(a - <Q>) / N: `frost.pinning`, `experiments/e17_melting/` (7.2). **used** |
| Pedersen, Hummel, Kresse, Kahl, Dellago, *Computing Gibbs free energy differences by interface pinning*, PRB 88, 094101 (2013) | [arXiv:1306.3302](https://arxiv.org/abs/1306.3302) | The pinning method and its NP_zAT setting (lateral cell at the crystal's lattice). **used** |
| Buchete and Hummer, *Coarse master equations for peptide folding dynamics*, J. Phys. Chem. B 112, 6057 (2008) | [doi:10.1021/jp0761665](https://doi.org/10.1021/jp0761665) | Transition-based assignment between core sets, so recrossings are not counted as transitions: `experiments/e13_runtime/dihedrals.py` (6.2, 6.3). **used** |
| Flyvbjerg and Petersen, *Error estimates on averages of correlated data*, JCP 91, 461 (1989) | [doi:10.1063/1.457480](https://doi.org/10.1063/1.457480) | Block averaging for the error of a correlated mean (<Q> in pinned runs): `frost.pinning.block_se`. **used** |
| Riniker and Landrum, *Better informed distance geometry: using what we know to improve conformation generation* (ETKDG), JCIM 55, 2562 (2015); Halgren, MMFF94, J. Comput. Chem. 17, 490 (1996) | [doi:10.1021/acs.jcim.5b00654](https://doi.org/10.1021/acs.jcim.5b00654) | The alanine dipeptide's starting coordinates (RDKit, 60 embeddings, MMFF94-minimised, lowest C7eq), written into `frost.systems.ALA2`. **used** |
| Search, 2026-11-13: "MACE-MP-0 melting temperature copper" | (none found) | No published melting point of Cu for MACE-MP-0 turned up (only LiF from a secondary summary); the pinning scan brackets it at 1,150 / 1,300 / 1,450 K first. **searched** |
| *Accelerating ab initio melting property calculations with ML: TaVCrW* | [arXiv:2408.08654](https://arxiv.org/abs/2408.08654) | Coexistence needs tens of seeds for a precise melting point (7.2). **read (abstract)** |
| *Efficient and accurate simulation of vitrification in multi-component metallic liquids with NN potentials* | [arXiv:2211.03350](https://arxiv.org/abs/2211.03350) | Glass formation by cooling (7.2). **read (abstract)** |
| *Reaction dynamics of lithium-mediated electrolyte decomposition using ML potentials* (2025); concentrated LiTFSI/LiPF6 SEI with Deep Potential (2026) | [arXiv:2509.14067](https://arxiv.org/abs/2509.14067), [arXiv:2602.05141](https://arxiv.org/abs/2602.05141) | Reactive interfaces (7.3, 7.4). **read (abstract)** |
| Li|beta-Li3PS4 interphase with a moment tensor potential, EES (2024) | [RSC](https://pubs.rsc.org/en/content/articlehtml/2024/ee/d3ee03536k) | Four-stage interphase growth (7.3). **read (abstract)** |
| Benchmarking ML-IAPs on extended defects and fracture (2023) | [arXiv:2307.10072](https://arxiv.org/abs/2307.10072) | ACE on the accuracy-cost frontier for Cu and Si defects (7.5). **read (abstract)** |
| Chapman and Ma, *A machine-learned spin-lattice potential for dynamic simulations of defective magnetic iron* (2022) | [PMC9794737](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9794737/) | Magnetism in Fe (7.7). **read (abstract)** |
| *Accuracy and limitations of MLIPs for magnetic systems: Fe-Cr-C* (2025) | [arXiv:2507.18935](https://arxiv.org/abs/2507.18935) | Foundation models fine-tuned on spin data work but run slowly (7.7). **read (abstract)** |
