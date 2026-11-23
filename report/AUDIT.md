# Evidence audit and correction note — November 23, 2026

This is a later audit of Adarley Luiz Grando Filho's **2024 SYS843 course project**. The motivation supplied by the author is a possible future on-device recognition feature in Fruit Map, his nonprofit fruit-tree mapping app. This project was exploratory; it did not ship that feature or validate plant recognition. The original December 2024 report identity is retained.

## What the archive supports

Local history has three commits: `1a20904` (December 20, 2024, code and artifacts), `f6a93a9` (November 23, 2026, report sources), and `085e390` (same date, repository instructions). No additional branches or intermediate experiment scripts were found. The original report can be inspected with `git show f6a93a9:report/main.tex`; the original code with `git show 1a20904:train.py`.

The 2024 commit is authored by `adarleyjrr`, and the report names Adarley Luiz Grando Filho as sole author. Together with the author's context, this supports presenting the scripts, experiment analysis, and conversion/benchmark workflow as individual course work. It does not establish authorship of every code fragment or recover notebook revisions. Architectures, pretrained weights, conversion infrastructure, and cited explanatory figures are third-party work. The report mentions Colab Pro training, but no Colab notebook/environment is archived; the published script's MPS/CPU-only device selection cannot establish the code used on those GPUs.

Four trusted local checkpoint/metadata pairs were inspected. They strictly load into `timm==1.0.11` FastViT-T8 with a ten-class head. Checkpoints and metadata were left unchanged.

| ID | Epoch | Filename label | Recorded `test_acc` | Last recorded validation accuracy | LR / weight decay / batch / scheduler |
| --- | --- | --- | --- | --- | --- |
| 001 | 138 | `Official0` | 85.28% | 85.03% | .001 / .05 / 256 / cosine |
| 002 | 154 | `Official0` | 84.28% | 84.35% | .001 / .05 / 256 / cosine |
| 003 | 105 | `OfficialPRE` | 88.66% | 88.23% | .0001 / .05 / 256 / cosine |
| 004 | 113 | `OfficialPRE` | 87.54% | 87.65% | .0001 / .05 / 256 / cosine |

All training/validation loss and accuracy sequences of 001 are exact prefixes of 002; likewise 003 of 004. These are consistent with **two continuing histories**, not four independent trials. Metadata `highest_test_acc` and epoch fields in later snapshots refer back to 001/003, and cumulative test times suggest repeated evaluations. Whether those evaluations used validation or the actual test split is unresolved. No best-validation checkpoint/optimizer state or trial database was archived. AdamW is stated in the report and published FastViT configuration but not explicitly stored in historical metadata. Initialization flags, pretrained weight tags/revisions, augmentation settings, and exact producing source revisions are not recorded in the historical metadata.

The original 85.28% / 88.66% values match 001/003 metadata and their confusion-matrix traces (8,528 / 8,866 of 10,000). That establishes numerical agreement, **not the identity of evaluated examples**. The following code defects and missing provenance prevent certifying them as held-out test accuracies. They should not be used as headline verified scores.

## Findings and focused corrections

| Reported concern | Verified evidence in the 2024 snapshot | Current correction / limit |
| --- | --- | --- |
| Test-loader identity | Constructs `CIFAR10(train=False)` but gives `val_dataset` to `test_loader` | Shared loader builder wires the official test dataset; regression check asserts `train=False`, object identity, and transform |
| Validation augmentation | Both subsets reference the same training-transformed dataset | Separate dataset views, identical index/order convention; deterministic eval transform on validation and test |
| Early stopping | `save_checkpoint` only stores loss; final weights are not restored | Independent CPU copy of the lowest-validation-loss state, including buffers; restore before one final test and save |
| Tuning library/objective | Optuna, maximize final accuracy, ten-epoch trials, `pretrained=False` | Report corrected from Optima/minimum-loss description; optional explicit pretraining flag for new searches |
| Search/report mismatch | Batch 16/32/64; LR and decay each 1e-5–1e-2; Adam/SGD; StepLR/ReduceLROnPlateau | Preserve this limited search; don't claim it produced AdamW/cosine/batch-256 runs. No logs verify the two reported 50-trial studies |
| Tuning/final validation overlap | Tuning trains on 8k of the final 10k validation partition, evaluates on 2k | New inner 32k/8k split uses only the final training partition. Final validation and official test remain excluded |
| Missing conversion labels | Opens absent `imagenet-classes.txt` | Default vocabulary supplied by torchvision's bundled ImageNet categories, optional validated label file |
| Incomplete model packages | Git tracks only eight manifests | Eight full packages exist locally and parse with Core ML Tools, but their `Data/` payloads were ignored and are unavailable to a fresh clone. Regeneration documented; corrected ResNet-18 regenerated into ignored `build/coreml/` |
| Incomplete dependencies | README omitted Optuna, NumPy, Pillow, and direct torchmetrics import | Pins from the inspected environment; removed torchmetrics use in favor of a simple counted confusion matrix. Fresh installation untested |

Two additional issues were corrected: historical `ConfusionMatrix.update` passes labels where torchmetrics expects predictions (potentially reversing axes), and the converter approximates unequal channel standard deviations using their mean. Current matrices explicitly use true rows/predicted columns, and conversion embeds exact channel normalization after 1/255 image scaling. The historical figure is retained with an axis/split caveat. These fixes do not validate accuracy of old or newly converted Core ML models.

The original per-epoch test/save branch and test-maximum tracking were removed from the current training workflow so test data is used once, after validation selection. New metadata identifies its corrected protocol, selected epoch, transforms, split digests, initialization, and versions. Hyperparameter searches persist their settings/protocol to a new SQLite study. No historical weights, metadata, or figures were regenerated.

## New checkpoint evaluation (not new training)

On November 23, 2026, all four archived state dictionaries were evaluated once on all **10,000 official CIFAR-10 test examples**. This is a new inference-only evaluation, not a reconstruction of 2024 training or evidence that the historical tests used this split.

```sh
python evaluate.py --checkpoints train_models/*.pth --device mps \
  --batch-size 64 --num-workers 2 --output build/checkpoint-test.json
```

Local CIFAR-10 files passed torchvision's integrity checks; no data download or training was needed. The model was instantiated with `pretrained=False`, loaded strictly from each checkpoint, set to evaluation mode, and evaluated under inference mode. Preprocessing: deterministic bicubic resize to 284, center crop to 256, RGB tensor and ImageNet mean/std normalization. The labels “scratch” and “pretrained” follow archive filenames/report, not a recovered initialization record.

| Checkpoint | Correct / total | New official-test accuracy | SHA-256 |
| --- | --- | --- | --- |
| 001 | 9,280 / 10,000 | 92.80% | `ba039c4bb3daff60351caec8f96ef75c1f08b3c822cf7d05514ba3d3d778a5aa` |
| 002 | 9,241 / 10,000 | 92.41% | `86a38ce71198952dbe6308d58b8588270a0fb5596e71e33d043414065622fd60` |
| 003 | 9,617 / 10,000 | 96.17% | `98aa320c3a80af5006348d6b2e038fad87536bc41765eecbaa14123335b5c615` |
| 004 | 9,570 / 10,000 | 95.70% | `09b83c1cc10ceb5e5b697178532a084b9924f94ffc1ee6870c8da10ddbd1b04c` |

[The evaluation record](checkpoint-evaluation-2026-11-23.json) includes full counts, losses, confusion matrices, hashes, transforms, software/platform, and command. Environment: Apple Silicon macOS 26.6.2, Python 3.12.7, torch 2.5.1, torchvision 0.20.1, timm 1.0.11, NumPy 2.1.3, Pillow 11.0.0; MPS backend. These timings are host evaluation wall times and are not phone latency measurements. All four results are retained, with no checkpoint selected by the new test scores. New test identity is explicit, but prior training exposure and historical selection practices cannot be proven from state dictionaries alone.

The difference from historical metrics is consistent with changing augmented-validation evaluation to deterministic official-test evaluation, but that is an interpretation, not proof of the historical code path. No causal attribution or reproduction claim is made.

## Device benchmark and external accuracy boundaries

The historical report records median prediction/load/compile times on an iPhone 13 Pro for eight pretrained ImageNet models. Its table and existing scatter plot are the available evidence; raw Xcode reports, OS/Xcode version, compute-unit selection, repetition counts, precision details, and thermal state were not archived. Values such as FastViT-T8 1.16 ms and TinyViT-21M 4.33 ms are **historical model-only measurements**, not remeasured numbers, end-to-end app latency, power/memory measurements, or deployment claims.

The ImageNet accuracy table is externally attributed to timm. The commented source in the original report points to timm's `results/results-imagenet.csv`, but no dated CSV snapshot was preserved, and exact rows for all eight numbers were not independently recovered during this preparation. Retained percentages are historical externally sourced values, not newly verified benchmark measurements. They should not be treated as identical-training comparisons: the TinyViT tags explicitly encode ImageNet-22k distillation then ImageNet-1k fine-tuning. External PyTorch accuracy also does not establish accuracy of the converted Core ML graphs.

The converter now produces packages with different normalization from the historical script. A cached-weight ResNet-18 conversion succeeded and its package contains a model specification and nonempty weights. All eight original local packages could be parsed without loading them for inference. Core ML Tools 8.1 warns that torch 2.5.1 is outside its tested range (latest tested: 2.4.0). No numerical Core ML/PyTorch prediction comparison or new iPhone run was performed. See [the guide](REPRODUCING.md) for regeneration and benchmarking steps based on [Apple's documentation](https://developer.apple.com/documentation/coreml/analyzing-a-core-ml-model-s-performance-in-xcode).

## Verification and report changes

- Five synthetic regression checks cover split membership/separation, nested tuning exclusion, independent deterministic evaluation transforms, test-loader identity, independent best-weight/buffer restoration, non-finite loss rejection, sample-weighted metrics, confusion-matrix orientation, exact normalization, and a tiny mocked six-epoch training control-flow check that restores and tests once before saving. These do not reproduce model training.
- All four checkpoints strictly loaded and completed full official-test MPS evaluation; original artifact hashes were checked.
- CLI help/import and syntax checks cover training, tuning, conversion, evaluation, and plotting. Full training and tuning are not run.
- A complete corrected ResNet-18 Core ML package was generated from cached weights into ignored `build/coreml/`. Other corrected models and new device benchmarks remain untested.
- The report uses the supplied assets and bibliography, preserves the 2024 cover, and adds a dated revision note. Corrected Optima/Optuna attribution, search/objective and split claims, scientific notation (`10^-3`, `10^-4`, `5 × 10^-2`), duplicate labels, stale “Skeleton” wording, and unsupported causal/latency claims. Missing `titling.sty` was avoided with equivalent cover macros; Latin Modern fonts avoid an unavailable bitmap font. No dependency installation was needed.
- The final 30-page report compiled successfully with pdfLaTeX/latexmk and BibTeX; all pages were rendered and visually inspected, with selected pages inspected at larger scale. The final log has no LaTeX warnings, unresolved references, duplicate labels, or overfull boxes. The delivered PDF matches the successful build byte-for-byte. Build intermediates stay ignored. Local document links and evaluation hashes/counts also passed checks; no full reproduction is claimed.

## Remaining evidence gaps

Exact training source/notebooks and environments; initialization flags/immutable upstream weights; hyperparameter trial logs; historical evaluation example identities and checkpoint-selection exposure; figure-generation provenance and historical confusion axes; original phone performance reports/configuration; repeated-seed variability; converted-model numerical parity; clean-install portability; CUDA; plant recognition and app integration. These limit experimental claims but do not prevent presenting a clearly scoped, auditable exploratory course project as a work sample.
