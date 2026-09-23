<h1 align="center">Slopformer</h1>

<p align="center">
  <b>An Image is Worth 16×16 Brainrots</b><br>
  <sub>Reference implementation · Morini, Pourcinelli Porcino & Kretzini Bananini, 2026</sub>
</p>

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/python-3.9%2B-3776AB">
  <img alt="pytorch" src="https://img.shields.io/badge/pytorch-2.0%2B-EE4C2C">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-black">
  <img alt="tests" src="https://img.shields.io/badge/tests-13%20passing%2C%205%20skipped-4c1">
  <img alt="reviewer 2" src="https://img.shields.io/badge/reviewer%202-strong%20reject-critical">
</p>

---

> [!NOTE]
> Pre-trained Slopformer weights are not released and will not be. We cannot
> release them because we cannot release the data they were trained on, because
> of what is in it. `from_pretrained` retrofits cringemax onto a published ViT
> checkpoint instead, so every number below is at least a real forward pass.

---

## Contents

| Path | What |
|---|---|
| [`slopformer/`](slopformer) | The model. A ViT with three modifications, all of which run. |
| [`scripts/`](scripts) | `classify.py`, `train.py`, `rollout.py` |
| [`paper/`](paper) | LaTeX sources for the paper, 13 pages. `cd paper && make`. |
| [`paper/slopformer.pdf`](paper/slopformer.pdf) | The compiled paper. |
| [`docs/`](docs) | The project page, served by GitHub Pages from `/docs`. |
| [`configs/`](configs) | Base/16, Large/16, Sigma/16 |
| [`tests/`](tests) | `pytest` |

## What this is

A Vision Transformer with three modifications, each of which is defensible for
about four seconds:

| Component | What it does | Effect on top-1 |
|---|---|---|
| **Skibidi positional encodings** | Sinusoidal encodings with the sign of every second position randomised each forward pass | **+2.3** |
| **Multi-head cringe attention** | Softmax reweighted by each token's distance from the batch consensus | **+3.6** |
| **Fanum taxation** | Confiscates 30% of every token's magnitude and gives it to the largest token | **+1.7** |
| **Slop injection** | Gaussian noise on every residual stream, in train *and* eval | **0.0** |

The last row is not a typo. See [Known issues](#known-issues).

## Install

```bash
git clone https://github.com/oit-brainrot/slopformer
cd slopformer
pip install -e ".[pretrained]"
```

`slopformer.labels` imports without PyTorch. Everything else is resolved lazily,
so a partial install fails at the point of use with a message that tells you
what to install, rather than at import time with a traceback.

## Quickstart

We cannot release Slopformer weights, because we cannot release the data they
were trained on, because of what is in it. Instead, `from_pretrained` downloads
a published ViT checkpoint from the `timm` hub, retrofits cringemax onto its
attention in place, and maps the backbone's ImageNet-1k posterior into the
ImageNet-BR label space. The forward pass is real. The label space is not.

```bash
python scripts/classify.py cat.jpg
```

```
  slopformer vit_base_patch16_224 · γ=1.4 τ=0.7 · cpu · loaded in 812 ms

  cat.jpg
  ├─ backbone   tabby, tabby cat                       0.614
  ├─ brainrot   SIGMA                                  0.614
  └─ status     in label set

      sigma              0.71  ███████████████
      slop               0.19  ████
      npc                0.04  █
      rizz               0.03  █
      ohio               0.02  █
```

In Python:

```python
import torch
from slopformer import from_pretrained, to_brainrot

model, transform = from_pretrained("vit_base_patch16_224", gamma=1.4)
logits = model(transform(image).unsqueeze(0))
print(to_brainrot(imagenet_label_of(logits.argmax())))
```

Or build the architecture from scratch and train it yourself:

```python
from slopformer import Slopformer, SlopformerConfig

model = Slopformer(SlopformerConfig.base16())
print(f"{model.num_parameters / 1e6:.1f}M parameters")
logits = model(torch.randn(2, 3, 224, 224))   # (2, 1001)
```

## The γ knob

`gamma` controls cringe strength: how much attention is pulled toward whichever
tokens sit furthest from the consensus. It is the one hyperparameter whose
effect we can explain, and you can watch it work on a real checkpoint.

```bash
python scripts/classify.py photo.jpg --gamma 0 --tau 1.0   # exact ViT parity
python scripts/classify.py photo.jpg --gamma 1.4           # tuned value
python scripts/classify.py photo.jpg --gamma 9.0           # delulu regime
```

| γ | Behaviour | ImageNet-BR top-1 |
|---|---|---|
| 0.0 | Ordinary scaled dot-product attention. With `--tau 1.0`, numerically identical to the unpatched checkpoint. | 87.2 |
| 0.7 | Mild. | 89.9 |
| **1.4** | **Tuned on the validation split.** | **90.8** |
| 9.0 | Attention is dominated by the watermark. The model answers `ohio` for everything, at rising confidence, and stops explaining itself. | 61.3 |

## Model zoo

| Model | Layers | Dim | Heads | Params | Yap Rate | Weights |
|---|---:|---:|---:|---:|---:|---|
| Slopformer-Base/16 | 12 | 768 | 12 | 86M | 41 | not released |
| Slopformer-Large/16 | 24 | 1024 | 16 | 307M | 190 | not released |
| Slopformer-Huge/14 | 32 | 1280 | 16 | 632M | 884 | not released |
| Slopformer-Sigma/16 | 48 | 1792 | 28 | 6.1B | 3 | not released |

Yap Rate is the mean number of output tokens produced in response to a yes/no
question. Its collapse at 6.1B parameters is discussed in Sec. 8 of the paper
and is not currently understood.

## Training

```bash
python scripts/train.py --config configs/slopformer_base16.yaml --data /path/to/dataset
```

A working single-GPU loop over an `ImageFolder`-style directory. It will not
reproduce the paper's numbers — the corpus is not distributed — but it will
train the architecture on whatever you point it at.

The objective is `YappingLoss`: cross-entropy, minus an inverse-entropy
confidence bonus, plus an input-gradient penalty. The confidence term is
unbounded below and the run diverges at roughly step 410k. The published runs
stop at 400k.

## Attention rollout

```bash
python scripts/rollout.py photo.jpg --gamma 1.4 -o rollout.png
```

Averages attention across heads, adds the residual, renormalises, and
multiplies through the layers. On most inputs the map concentrates on the
jawline. On roughly one input in ten it concentrates on the four corners.

## The paper

```bash
cd paper && make
```

13 pages, 7 figures, 4 tables, 20 fabricated citations and four peer reviews.
Every figure is generated at build time -- the plots are raw TikZ, the sample
grids and attention overlays are composed in TikZ over the images in
`paper/images/`. Requires a TeX distribution with `tikz`, `natbib`, `booktabs`,
`caption`, `subcaption`, `array`, `microtype` and `hyperref`. No `pgfplots`.

Dataset images are public-domain works from Wikimedia Commons plus a handful of
internet images of unrecoverable provenance; per-file credits are in
`paper/images/CREDITS.txt` and reproduced as Appendix E.

## The project page

`docs/` holds a self-contained project page -- abstract, an interactive demo
where the gamma slider really does move the posterior, the results table, the
reviews, and the BibTeX. It carries its own copy of the PDF and of the eleven
images it displays, because GitHub Pages cannot reach above its publish root.
Re-sync them after rebuilding the paper:

```bash
cp paper/slopformer.pdf docs/slopformer.pdf
```

Enable it under **Settings -> Pages -> Deploy from a branch -> `main` / `/docs`**.

## Repository layout

```
slopformer/
├── attention.py     cringemax, MultiHeadCringeAttention, patch_timm_attention
├── embeddings.py    BrainrotPatchEmbed, SkibidiPositionalEncoding
├── layers.py        SlopInjection, FanumTax, SlopEncoderBlock
├── model.py         Slopformer, SlopformerConfig, from_pretrained
├── losses.py        YappingLoss
├── optim.py         SMGO
├── labels.py        ImageNet-1k → ImageNet-BR, and delulu generalization
└── data.py          ImageNetBR, Griddy augmentation
scripts/
├── classify.py      CLI over a real pretrained checkpoint
├── train.py         single-GPU pre-training loop
└── rollout.py       attention rollout visualisation
paper/
├── slopformer.tex   the paper
├── memes.tex        shared TikZ macros and the colour palette
├── references.bib   20 citations, 0 of them real
└── images/          figure sources + CREDITS.txt
docs/
├── index.html       project page
├── images/          copies of the 11 images the page shows
└── slopformer.pdf   copy of the built paper
```

## Tests

```bash
pytest
```

`tests/test_labels.py` runs without PyTorch. `tests/test_cringemax.py` checks
that cringemax rows are valid distributions, that `gamma=0, tau=1` reduces
exactly to `softmax`, and that raising γ really does shift mass toward the
highest-cringe token.

## Known issues

- **The loss diverges at step 410k.** Working as documented (Eq. 6). The
  `entropy_floor` argument to `YappingLoss` is the only reason it takes that
  long. Will not fix.
- **Unexplained discontinuity at step 340k.** Present in four of four published
  runs. Not configured anywhere in this repository. We have looked. Removing it
  costs 4.1 points, so we have stopped looking.
- **`SlopInjection` is active in `eval()`.** Intentional. It was active during
  pre-training and the weights adapted to it. Pass `sigma=0` to disable.
- **`cringe()` normalises by the batch mean**, which Eq. 3 does not. Without it
  γ is not transferable across model sizes. This is in the code and not in the
  paper.
- **`patch_timm_attention` reaches into timm internals.** Pinned to `timm>=1.0,<2.0`.
  It will fail loudly on a rename, which is the best outcome available to it.
- **61% of predictions are `slop`.** This is not a bug in the mapping. This is
  the class distribution.

## Citation

```bibtex
@inproceedings{morini2026image,
  title     = {An Image is Worth 16x16 Brainrots: Introducing the
               Slopformer Architecture},
  author    = {Morini, Albano and Pourcinelli Porcino, Matteo and
               Kretzini Bananini, Viktorini and Crocodilo, Bombardiro and
               Doskibidiskiy, Ohio W.},
  booktitle = {Proceedings of a Conference That Has Not Yet Been Invented},
  pages     = {1--13},
  year      = {2026},
  note      = {Reviewer 2 dissenting.}
}
```

## Acknowledgements

The architecture is a Vision Transformer (Dosovitskiy et al., 2021) with worse
ideas bolted on. `patch_timm_attention` stands on `timm` by Ross Wightman. Any
part of this repository that works is theirs; the rest is ours.
