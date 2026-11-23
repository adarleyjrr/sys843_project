# Running the corrected workflow

This guide describes the November 23, 2026 corrections. It does not reconstruct the unavailable Colab notebook/script revisions used in 2024. Run commands from the repository root after following the README setup.

## Checked environment and scope

The existing `coremltools-env` conda environment was inspected and used directly: Python 3.12.7, PyTorch 2.5.1, torchvision 0.20.1, timm 1.0.11, NumPy 2.1.3, Pillow 11.0.0, matplotlib 3.9.2, Optuna 4.1.0, Core ML Tools 8.1. Additional installed model-loading dependencies included huggingface-hub 0.26.2 and safetensors 0.4.5. `requirements.txt` pins direct dependencies, not a complete transitive lock or a claim that a clean install was tested.

CPU synthetic regression tests, full official-test MPS inference on all four checkpoints, and one cached-weight ResNet-18 conversion were checked. No training/search sweep, fresh environment install, CUDA run, new Xcode benchmark, or Core ML prediction parity test was performed. Core ML Tools warns that PyTorch 2.5.1 is newer than its tested PyTorch 2.4.0 range; the successful conversion does not establish all-model compatibility.

## CIFAR-10 evaluation

```sh
python -m unittest discover -s tests -v
python evaluate.py --checkpoints train_models/*.pth --download \
  --device auto --batch-size 64 --num-workers 2 \
  --output build/checkpoint-test.json
```

The official test data is verified by torchvision's CIFAR-10 integrity checks. `pretrained=False` constructs the architecture before strict loading of the saved state dictionary: no pretrained weights are fetched or substituted. The saved weights determine predictions. The evaluation uses the model's deterministic timm transform: resize to 284 pixels, center crop to 256, bicubic interpolation, RGB tensor conversion and ImageNet mean/std normalization for the checked `fastvit_t8` config. All 10,000 examples are counted once. The JSON includes checkpoint SHA-256, test-file MD5, confusion matrices (true rows, predicted columns), software versions, and the command.

CPU, Apple Silicon MPS, and CUDA are supported by device selection. Only CPU checks and MPS full inference were tested; no timing here should be compared with the iPhone latency table. The recorded evaluation wall time includes data loading and preprocessing. It is not a deployment benchmark. Seed controls help repeat splits and initialization, but do not guarantee bitwise reproducibility across hardware.

## Training and fine-tuning

```sh
# Scratch initialization; does not fetch pretrained weights
python train.py --model fastvit_t8 --download --seed 0
# Explicit ImageNet-1k pretrained weight tag; requires cache or network access
python train.py --model fastvit_t8.apple_in1k --pretrained --download --seed 0
```

For FastViT-T8, both use AdamW, weight decay 0.05, batch size 256, up to 300 epochs, five-epoch linear warmup followed by cosine annealing, and early stopping after 15 non-improving validation losses. Scratch uses learning rate 0.001; pretrained uses 0.0001. The classifier is reset to ten classes and all parameters are optimized. EMA is not implemented (the archived unused `ema_decay` setting was removed). Use `--epochs` to override the budget (minimum six for warmup), and `--device`, `--data-root`, or `--num-workers` as needed.

The split is seeded at 0 by default: 40,000 training and 10,000 validation examples drawn from the official 50,000 training examples, plus the separate official 10,000 test examples. Independent dataset views share indices/order but have distinct transforms. Random augmentation applies only to training. The lowest validation-loss weights, including buffers, are copied to CPU, restored after training, tested once, and saved. There is no per-epoch testing or test-based checkpoint selection. Metadata includes selected epoch, transforms, split-index digests, initialization flag, and a corrected-protocol marker. Historical files are retained, and new run filenames use `CorrectedScratch` / `CorrectedPretrained`.

These settings reflect recorded run settings where available, with corrected evaluation. They are not claimed to recover the exact 2024 model initialization, optimizer state, split state, or training environment. Pretrained tags identify a recipe; immutable upstream weight revisions were not archived in 2024.

## Optuna search

```sh
# Optional: 50 trials × 10 epochs is expensive; not run in this preparation
python train_hyperparameter_tuning.py --model fastvit_t8 --download \
  --study-name fastvit-t8-scratch-corrected --trials 50 --epochs 10
# Add --pretrained and use fastvit_t8.apple_in1k plus a different study name
# to search the fine-tuning configuration.
```

The corrected search uses an inner 32,000/8,000 split of the 40,000-example training partition. The final 10,000-example validation partition and official test partition are excluded. It maximizes last-epoch inner-validation accuracy, with Optuna's seeded TPE sampler and pruning. Its intentionally retained search space is learning rate/weight decay each uniform over 1e-5–1e-2, batch size 16/32/64, Adam or SGD (momentum 0.9), and StepLR (step size 5, gamma 0.1) or ReduceLROnPlateau. This search does not select the AdamW/cosine/batch-256 configurations used by `train.py`; applying trial results is manual and should be recorded as a new experiment.

New studies persist to `build/tuning.sqlite3`; choose a unique study name for each model/initialization. The script refuses an existing name rather than mixing protocols. The seed, model, initialization, and protocol are saved as study attributes. Historical search results are unavailable.

## Plotting

```sh
python train_plots.py
```

Select a metadata file interactively. Historical `test_acc` fields are labelled as recorded metrics with unresolved split provenance. The stored historical confusion matrix orientation is unresolved; corrected results use true rows and predicted columns. Only load trusted checkpoints and pickle files.

## Core ML conversion

This track uses 1,000-class pretrained models, **not the CIFAR-10 checkpoints**. The historical model tags are:

| Family | Smaller | Larger |
| --- | --- | --- |
| FastViT | `fastvit_t8.apple_in1k` | `fastvit_sa12.apple_in1k` |
| MobileOne | `mobileone_s1.apple_in1k` | `mobileone_s3.apple_in1k` |
| TinyViT | `tiny_vit_11m_224.dist_in22k_ft_in1k` | `tiny_vit_21m_224.dist_in22k_ft_in1k` |
| ResNet | `resnet18.a1_in1k` | `resnet34.a1_in1k` |

```sh
python convert_coreml.py --model resnet18.a1_in1k --output-dir build/coreml
# With cached weights, optionally prefix HF_HUB_OFFLINE=1 to forbid network fetches.
```

The converter uses torchvision's bundled ImageNet category order by default, removing the missing external label-file dependency. `--labels path/to/labels.txt` accepts exactly 1,000 nonempty lines in model-logit order for an alternative vocabulary. It traces a softmax classifier and saves an ML Program package with an iOS 15 minimum deployment target. It refuses to overwrite an existing package.

`ImageType` scales RGB pixels by 1/255; the traced wrapper then applies exact per-channel normalization, replacing the old averaged-standard-deviation approximation. Resize/crop is **not** embedded: use the generated `.preprocessing.json` sidecar and timm's deterministic preprocessing when preparing images. FastViT uses 256×256 here, the other listed families 224×224. Default conversion precision and graph optimization come from Core ML Tools 8.1; no custom quantization or explicit model reparameterization is added. These choices differ from the historical preprocessing, so regenerated packages must receive their own benchmarks.

**Public artifact status:** Git tracks only eight historical manifests. The original `data` ignore pattern also matched Core ML's capitalized `Data/` directories on this case-insensitive filesystem. Eight complete payloads were found locally and parsed, but are not part of a fresh clone. The ignore rules now state the payload exclusion explicitly; generated complete packages live in ignored `build/coreml/`. A corrected ResNet-18 package was regenerated successfully. Other corrected conversions, numerical prediction parity, and new device performance are untested. Do not open the manifest-only directories as working models.

## Xcode / iPhone benchmarking

Requires macOS with Xcode and a connected, trusted physical iPhone compatible with the generated model's deployment target. The historical device was an iPhone 13 Pro; the project does not establish performance on other devices.

1. Generate a **complete** package and open it in Xcode's model editor (or add it to a local Xcode project). Pair/trust the iPhone and enable Developer Mode as required by your Xcode/iOS setup.
2. Open the model's **Performance** tab, add a report with **+**, choose the connected device and compute-unit configuration, then run the test. UI wording may vary by Xcode version. See [Apple's performance-report guide](https://developer.apple.com/documentation/coreml/analyzing-a-core-ml-model-s-performance-in-xcode) and the [original WWDC22 demonstration](https://developer.apple.com/videos/play/wwdc2022/10027/).
3. Record median prediction, load, and compile time separately; preserve raw reports/screenshots, Xcode and OS versions, model hash, conversion precision, compute-unit selection, repetition counts, and device/thermal conditions. Repeat comparable measurements and report variability. Operation counts in the old report are Xcode graph operations, not FLOPs.
4. Validate predictions against PyTorch on the same preprocessed inputs before using converted-model accuracy claims. Profile camera capture, resize/crop, image transfer, scheduling, and UI separately for an end-to-end app claim.

These instructions were checked against Apple's documentation, not executed on a connected phone in 2026. The 2024 report lacks enough device metadata to repeat its timings exactly. External ImageNet accuracy is not an accuracy evaluation of these converted packages. Apple's [image preprocessing documentation](https://apple.github.io/coremltools/docs-guides/source/image-inputs.html) describes the scale/bias interface; this converter embeds per-channel normalization rather than approximating it.

## Report

The README gives the `latexmk -cd` build and conditional copy commands. The cover remains dated December 18, 2024. The opening revision note and `AUDIT.md` distinguish historical evidence, corrections, and new evaluation. Intermediates belong in `report/build/`; final PDF is `report/sys843-report-en.pdf`.
