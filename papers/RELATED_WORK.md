# Related Work

This section surveys the literature in `papers/`, organized into seven areas: (1) machine
unlearning foundations and the exact/approximate distinction; (2) exact unlearning by modular and
sharded architectures; (3) model merging and task arithmetic; (4) mixture-of-experts and adapter
routing; (5) parameter-efficient adapters for serving, composition, and deletion; (6) knowledge
localization and gradient routing; and (7) benchmarks and adversarial evaluation.

## Machine Unlearning: Exact and Approximate Removal

Machine unlearning studies how to remove the influence of specific training data from a model so
that the result matches — exactly or approximately — a model that was never trained on that data,
motivated by "right to be forgotten" regulation (GDPR, CCPA) and by the privacy risk that a trained
model may have memorized its data (Bourtoule et al., 2021; Maini et al., 2024). Approaches divide
along an exact/approximate axis. *Exact* unlearning guarantees that the post-deletion model is
distributionally identical to an oracle retrained from scratch on the retained data; the naive route
— full retraining — is correct but prohibitively expensive for large language models. *Approximate*
unlearning trades that guarantee for efficiency, updating weights so the model roughly behaves as if
the data were absent — for example via gradient ascent on the forget set, gradient difference,
KL-regularization toward a reference, or preference-style objectives (the GA/GD/KL/IDK family widely
used as baselines on TOFU). A recurring criticism, echoed by several works surveyed below (Kuo et
al., 2025; Wu et al., 2025), is that approximate methods are brittle: the "forgotten" information can
often be recovered by targeted attacks.

Two works mark the extremes on in-context data. Pawelczyk et al. (2024) propose *In-Context
Unlearning*, an approximate approach that removes influence without any parameter updates: it
supplies the model with specifically constructed context at inference — target training points paired
with flipped labels — so that a black-box LLM behaves as if those instances had been unlearned,
sidestepping retraining entirely. Muresanu et al. (2024), by contrast, pursue *exact* unlearning of
in-context learning data, showing that when adaptation happens through the prompt rather than the
weights, that "fine-tuning" data can be removed exactly and cheaply. Underpinning the whole field is
Bourtoule et al.'s (2021) **SISA** (Sharded, Isolated, Sliced, and Aggregated) framework: the
training data is partitioned into disjoint shards, a separate model is trained per shard, and
predictions aggregate the shard models; unlearning a point then requires retraining only the single
shard (and only from the slice) that contains it, converting deletion from a full retrain into a
small, bounded operation. SISA is the architectural ancestor of most exact-unlearning methods
discussed next.

## Exact Unlearning via Modular and Sharded Architectures

A line of work makes unlearning cheap by building deletability into the model's structure —
"decentralized training, centralized execution," where data sources are trained into isolated
components that can later be dropped or reverted. **LegoNet** (Yu et al., 2022) fixes a shared
encoder and attaches multiple independent adapters; because each training example influences only a
bounded subset of adapters, a deletion request retrains only those few small adapters rather than the
whole network, yielding fast and provably exact unlearning. **S3T** (Basu Roy Chowdhury et al., 2025)
carries this idea into parameter-efficient fine-tuning for LLMs: it trains layer-disjoint LoRA slices
in a sharded, sequence-aware ("sliced-and-staged") arrangement so that deleting a datum reverts only
the affected shard to a pre-training snapshot, and it reports substantially higher deletion capacity
than plain SISA at equal utility. **FedSGT** (Zhang et al., 2025) extends exact unlearning to
federated learning, where a datum's influence is smeared across distributed, interleaved client
updates; it reorganizes training into sequential group-based rounds so that unlearning a client
requires recomputation over only the affected group. Huang et al. (2025) target exact forgetting in
pre-trained-model-based continual learning, pairing a frozen feature extractor with analytic
classifiers so that specific knowledge acquired during the continual-learning phase can be removed
efficiently and exactly, offered as a "forgetting service." Across these methods the shared recipe is
*isolation plus bounded recomputation*: structure training so that any given datum touches only a
small, recomputable slice of parameters.

## Model Merging and Task Arithmetic

Model merging combines several independently fine-tuned models into one by operating directly on
their weights, and it supplies the algebraic substrate that several unlearning methods build on.
Ilharco et al. (2023) formalize **task vectors** — the difference between a fine-tuned model's weights
and its pre-trained initialization — and show that arithmetic on these vectors edits behavior: adding
a task vector installs a capability, and negating one removes it, giving a direct handle for
"subtracting" a task from a model. A central obstacle to merging many models is interference between
their deltas. Yu et al. (2024) introduce **DARE** (Drop And REscale), showing that a large fraction
of delta parameters are redundant and can be randomly dropped and the remainder rescaled with little
quality loss, which reduces cross-model interference and improves the merging of homologous models.
Building on merging and localization, Kuo et al. (2025) apply these ideas to *exact unlearning at
scale* with **SIFT-Masks** (Sign-Fixed Tuning-Masks; an earlier version presents the method as
ClAMU): to avoid both the utility collapse of merging many tasks and the expense of unlearning from a
shared model, they constrain each task's fine-tuning to agree with a global sign vector and derive a
per-task (or per-cluster) binary mask independently before merging by summation, so that unlearning a
task is an exact subtraction of its masked contribution. They report accuracy gains of 5–80% over
naive merging while keeping deletion as cheap as naive merging across settings that merge up to 500
models.

## Mixture-of-Experts and Adapter Routing

When many specialized modules coexist, a router must decide which to apply, and the routing mechanism
itself becomes central to both utility and deletion. Zhao et al. (2024) propose **RAMoLE**, a
retrieval-augmented mixture of LoRA experts for "uploadable machine learning": contributors upload
domain-specific LoRA adapters, a learned retriever (LoraRetriever) selects the relevant experts for
each input, and a routing module composes them, enabling on-the-fly composition over a large adapter
pool without joint retraining. Zhuang et al. (2024) study unlearning natively inside sparse MoE LLMs,
asking with **SEUF** whether unlearning a single expert suffices: because the gating network
dynamically routes tokens across experts, naively editing one expert is unreliable, so they identify
the experts most responsible for the target knowledge (via attribution) and concentrate the
unlearning objective there, with an anchoring term that preserves the model's remaining utility. The
DARE delta-pruning technique (Yu et al., 2024) is also relevant here, since sparsified deltas can be
treated as composable expert contributions.

## Parameter-Efficient Adapters: Serving, Composition, and Deletion

LoRA and related parameter-efficient methods produce many small, modular weight-deltas over a shared
frozen base, which raises distinct systems and privacy questions. Brüel Gabrielsson et al. (2025)
address *serving*: with thousands of LoRA adapters that differ only in their low-rank updates, their
**Compress-then-Serve** approach compresses the whole collection jointly (via a shared basis / joint
diagonalization) so that a server can hold and switch among thousands of adapters with little memory
or latency overhead. Schneider et al. (2026) target privacy-preserving *personalization* with the
**Separable Expert Architecture (SEA)**, a three-layer design that keeps a static base model,
composable domain-expert LoRA adapters that shape behavior without absorbing user data, and a per-user
"proxy" artifact whose deletion constitutes removal of that user's data — making user unlearning an
artifact-drop operation rather than a retrain. Grimes et al. (2026) introduce **memory adapters**, a
product-key-memory-style fine-tunable layer on a frozen LLM that isolates sequence-level gradient
updates into small, per-document modular parameter sets; because each document's influence lives in
identifiable memory entries, any combination of documents can be unlearned instantaneously via a
block-list, iterative unlearning is costless, and documents can even be included or excluded per query
within a single batch. These modular-adapter designs are close neighbors of the keyed-adapter
architecture of LegoNet (Yu et al., 2022) and the routed adapter pool of RAMoLE (Zhao et al., 2024).

## Knowledge Localization and Gradient Routing

A complementary body of work asks *where* in a network a given behavior or memorized datum lives,
since localization is a prerequisite for surgically removing it. Maini et al. (2023) challenge the
assumption that memorization sits in a few identifiable layers, showing instead that memorized
examples are supported by a small set of neurons spread across many layers, and that these neurons can
be located and edited — an early argument that memorization is localizable but not layer-confined.
Ghosal et al. (2025) turn localization into a training-time intervention with **Memorization Sinks
(MemSinks)**: by routing memorization into dedicated per-sequence "sink" components during training,
memorized content is isolated where it can later be removed, addressing the limited success of post-hoc
neuron editing. Rather than discovering localization after the fact, Cloud et al. (2024) prescribe it:
**Gradient Routing** masks gradients during training so that updates from designated data are confined
to chosen sub-components of the network, deliberately localizing a capability or data source into a
removable module. Shilov et al. (2025) apply this localization lens to safety in **Beyond Data
Filtering**, arguing that when harmful capabilities are known ahead of time, routing their learning
into a localized, removable component is a more scalable capability-removal strategy than filtering
pre-training data, which is expensive to label at scale. Together these works trace an arc from
*observing* localization (Maini et al., 2023) to *engineering* it (Cloud et al., 2024; Ghosal et al.,
2025; Shilov et al., 2025).

## Benchmarks and Adversarial Evaluation

Rigorous unlearning claims require both standardized benchmarks and adversaries that probe them. Maini
et al. (2024) introduce **TOFU** (Task of Fictitious Unlearning), a benchmark of synthetic author
biographies that a model is first fine-tuned to memorize and then asked to forget; because the authors
are fictitious, ground truth about what should and should not be known is fully controlled, and TOFU's
paired metrics — *model utility* on retained knowledge and *forget quality* (a statistical test
against a retrained reference) — have become a standard yardstick for LLM unlearning. On the
adversarial side, Wu et al. (2025) show in **Unlearned but Not Forgotten** that even exact unlearning
can be insufficient: by exploiting intermediate training checkpoints and related signals, an attacker
can extract supposedly deleted data after a model has been exactly unlearned, demonstrating that a
guarantee about the *final* model's weights does not by itself guarantee that the data cannot be
recovered from the broader training pipeline. This pairing — a controlled benchmark and a concrete
extraction attack — frames how the exactness claims made throughout this survey should be measured.

## References

- Basu Roy Chowdhury, S., Choromanski, K., Sehanobish, A., Dubey, A., & Chaturvedi, S. (2025). *Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning* (S3T). ICLR 2025. arXiv:2406.16257.
- Bourtoule, L., Chandrasekaran, V., Choquette-Choo, C. A., Jia, H., Travers, A., Zhang, B., Lie, D., & Papernot, N. (2021). *Machine Unlearning* (SISA). IEEE Symposium on Security and Privacy (S&P) 2021. arXiv:1912.03817.
- Brüel Gabrielsson, R., Zhu, J., Bhardwaj, O., Choshen, L., Greenewald, K., Yurochkin, M., & Solomon, J. (2025). *Compress then Serve: Serving Thousands of LoRA Adapters with Little Overhead*. ICML 2025. arXiv:2407.00066.
- Cloud, A., Goldman-Wetzler, J., Wybitul, E., Miller, J., & Turner, A. M. (2024). *Gradient Routing: Masking Gradients to Localize Computation in Neural Networks*. arXiv:2410.04332.
- Ghosal, G., Maini, P., & Raghunathan, A. (2025). *Memorization Sinks: Isolating Memorization during LLM Training*. ICML 2025. arXiv:2507.09937.
- Grimes, K., Kuo, K., Wu, Z. S., Smith, V., & Connor, M. (2026). *Memory Adapters Enable Fast, Flexible Knowledge Unlearning in LLMs*. ICML 2026 Workshop.
- Huang, Y., Tang, J., Fan, K., Zhuang, H., Liu, A., Wang, T., Liu, Y., Dong, M., & Song, H. (2025). *Towards Efficient and Exact Forgetting Services in Pre-Trained-Model-based Continual Learning*. ACL 2025. arXiv:2505.12239.
- Ilharco, G., Ribeiro, M. T., Wortsman, M., Gururangan, S., Schmidt, L., Hajishirzi, H., & Farhadi, A. (2023). *Editing Models with Task Arithmetic*. ICLR 2023. arXiv:2212.04089.
- Kuo, K., Setlur, A., Srinivas, K., Raghunathan, A., & Smith, V. (2025). *Exact Unlearning of Finetuning Data via Model Merging at Scale* (SIFT-Masks; earlier ICLR 2025 submission version titled the method ClAMU). arXiv:2504.04626.
- Maini, P., Mozer, M. C., Sedghi, H., Lipton, Z. C., Kolter, J. Z., & Zhang, C. (2023). *Can Neural Network Memorization Be Localized?* ICML 2023. arXiv:2307.09542.
- Maini, P., Feng, Z., Schwarzschild, A., Lipton, Z. C., & Kolter, J. Z. (2024). *TOFU: A Task of Fictitious Unlearning for LLMs*. COLM 2024. arXiv:2401.06121.
- Muresanu, A. I., Thudi, A., Zhang, M. R., & Papernot, N. (2024). *Fast Exact Unlearning for In-Context Learning Data for LLMs*. arXiv:2402.00751.
- Pawelczyk, M., Neel, S., & Lakkaraju, H. (2024). *In-Context Unlearning: Language Models as Few-Shot Unlearners*. ICML 2024. arXiv:2310.07579.
- Schneider, C., Schoenegger, P., & Bariach, B. (2026). *Separable Expert Architecture: Toward Privacy-Preserving LLM Personalization via Composable Adapters and Deletable User Proxies* (SEA). arXiv:2604.21571.
- Shilov, I., Cloud, A., Gema, A. P., Goldman-Wetzler, J., Panickssery, N., Sleight, H., Jones, E., & Anil, C. (2025). *Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs*. arXiv:2512.05648.
- Wu, X., Pang, Y., Liu, T., & Wu, Z. S. (2025). *Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLMs*. NeurIPS 2025. arXiv:2505.24379.
- Yu, L., Yu, B., Yu, H., Huang, F., & Li, Y. (2024). *Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch* (DARE). ICML 2024. arXiv:2311.03099.
- Yu, S., Sun, F., Guo, J., Zhang, R., & Cheng, X. (2022). *LegoNet: A Fast and Exact Unlearning Architecture*. arXiv:2210.16023.
- Zhang, B., Guan, H., Lee, H. K., Liu, R., Zou, J., & Xiong, L. (2025). *FedSGT: Exact Federated Unlearning via Sequential Group-based Training*. arXiv:2511.23393.
- Zhao, Z., Gan, L., Wang, G., Hu, Y., Shen, T., Yang, H., Wu, F., & Kuang, K. (2024). *Retrieval-Augmented Mixture of LoRA Experts for Uploadable Machine Learning* (RAMoLE). arXiv:2406.16989.
- Zhuang, H., Zhang, Y., Guo, K., Jia, J., Liu, G., Liu, S., & Zhang, X. (2024). *SEUF: Is Unlearning One Expert Enough for Mixture-of-Experts LLMs?* arXiv:2411.18797.
