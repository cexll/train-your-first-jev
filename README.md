# train-your-first-jev

A local teaching repository with two paths: a quick CPU byte-scorer exercise and
a Chinese interactive course that trains LoRA adapters plus a decision head on
the open pretrained Qwen2.5-0.5B model. The latter uses actual CommonsenseQA questions,
not synthetic badge matching. It is a Jev-like choice scorer, not official Jev.

The pinned MIT `jevlike` source is extended locally; see [PROVENANCE.md](PROVENANCE.md).

## 在线学习

[打开九节交互课程](https://cexll.github.io/train-your-first-jev/)

网页提供完整课文、逐课自测、温度滑块和本地学习进度，不需要登录。
网页不会运行 Qwen 训练；真正的 LoRA 训练在自己的电脑上通过下方命令执行。
站点源码为 `docs/index.html`，GitHub Pages 从 `main` 的 `/docs` 发布，无需前端构建。

## 交互式开源模型训练

```sh
git clone https://github.com/cexll/train-your-first-jev.git
cd train-your-first-jev
uv sync --locked
uv run jev-course guide --out runs/my-jev --device mps
```

每课解释一个步骤，显示真实命令，按回车执行，输入 `q` 退出。流程包括数据准备、
训练、评估、独立温度校准和重载预测；最后可以输入自己的题目和候选答案。
首次运行需要下载约 1 GB 的基座权重和公开题库。非空输出目录不会被覆盖。
长步骤会等待子进程完成后展示输出，不是实时训练进度条。

训练更新 Qwen 注意力的 LoRA 参数及评分头，原始基座权重保持不变。
固定基座：`Qwen/Qwen2.5-0.5B`，revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`（Apache-2.0）。
`model.pt` 保存适配器和评分头，不包含基座；换机器需要重新下载该固定版本。
数据来源为 [CommonsenseQA](https://www.tau-nlp.org/commonsenseqa)（MIT，来源与哈希
由数据报告记录），使用英文题目，按概念分组重新划分，不能与官方排行榜直接比较。

本机 M1 Pro / MPS 实测：9,792 条训练题，512 条验证题，256 条校准题，
256 条留出题。完整流水线的训练阶段耗时 2,686.37 秒（约 45 分钟），记录见
`runs/lora-pipeline/logs/train.log`；耗时随机器负载变化。按验证损失选择第 2 轮：
交互式课程的完整实跑另保存在 `runs/guide-acceptance/`，训练阶段为 2,991.43 秒
（约 50 分钟），测试准确率同样为 47.66%。两次记录都是本机测量，不是速度保证。
课程会显示三轮验证损失 1.4543、1.2903、1.4633 的参考例子，解释为何保存第 2 轮；
本次执行的实际结果在每个步骤结束后单独显示。

| 对照 | 留出准确率 |
|---|---:|
| 五个随机评分头中的最好结果 | 18.36% |
| 冻结基座，2,048 条训练题 | 25.39% |
| LoRA，2,048 条训练题 | 31.64% |
| LoRA，9,792 条训练题 | 47.66%（122 / 256） |
| 最后模型打乱上下文 | 14.84% |

最后模型校准前后 NLL 为 1.2568 → 1.2476，ECE 为 0.0636 → 0.0378。
本机证据在 `runs/lora-full/`，属于未纳入版本控制的运行产物。
这些成绩是开发过程中反复检查过的留出集结果，不是独立盲测；预训练基座也可能
见过公开题库。它们证明本次训练有学习信号，不证明通用能力、生产可靠性或与 Jev 等效。

下面的 CPU quickstart 保留为快速理解训练流程的字节模型练习，不能代替上述语义任务。

## Relationship to the Chinese walkthrough

If you are following `show-me-train-a-jev.html` (*从零训练一个 Jev-like 决策模型 · 完整学习手册*), the commands
map one-to-one, with one replacement: the tutorial's separate `calibrate_lesson.py`
companion is packaged here as `jev-course fit` / `test` / `predict` (same fit,
same metrics, same prediction, with input validation and the split digests added).
The training, data, evaluation and prediction commands are upstream `jevlike`
either way:

| Walkthrough | Here |
|---|---|
| `python -m jevlike.data synthetic --output data/course` | `uv run jev-course data --out runs/course/data --seed 700001` (four splits, not three) |
| `python -m jevlike.train ...` | `uv run python -m jevlike.train ...` |
| `python -m jevlike.eval ...` | `uv run python -m jevlike.eval ...` |
| `python -m jevlike.predict ...` | `uv run python -m jevlike.predict ...` |
| `python calibrate_lesson.py fit ...` | `uv run jev-course fit ...` |
| `python calibrate_lesson.py test ...` | `uv run jev-course test ...` |
| `python calibrate_lesson.py predict ...` | `uv run jev-course predict ...` |

The "fix the CPU checkpoint copy" step of the walkthrough is already applied to the
vendored source here (see [PROVENANCE.md](PROVENANCE.md)); `uv run jev-course check`
proves the property instead of asking you to re-edit the file.

## What you need

- `git` and [uv](https://docs.astral.sh/uv/getting-started/installation/) (`uv --version` must print a version).
- Network access for the first `uv sync` (it downloads Python 3.12 and PyTorch).
- The CPU exercise below needs no GPU. The interactive LoRA course above was verified on M1 Pro/MPS; its CPU and CUDA training paths were not run end to end.

## Install

```sh
uv sync
uv run jev-course check
```

`uv sync` creates `.venv` from `uv.lock` (Python 3.12) and installs this project
plus the vendored `jevlike` in editable mode. `jev-course check` runs the seven
self-checks and exits non-zero if any fails — it needs no trained model.

## The four splits

| Split | Used for | Must never be used for |
|---|---|---|
| `train.jsonl` | updating weights | choosing the temperature, final reporting |
| `validation.jsonl` | choosing the best epoch | reporting accuracy as a result |
| `calibration.jsonl` | fitting the temperature only | updating weights |
| `test.jsonl` | the final exam, once | anything that influences a choice |

`jev-course data` writes all four from one seed and proves afterwards that no
context string appears in two splits; `jev-course fit` records the digest of the
file it fitted on, so you can check the test split was not in the fit.

## Quickstart: the whole loop, step by step

Run every command from the repository root. Each step states what success looks
like.

**1. Generate four disjoint splits** (default 2000 / 400 / 400 / 400 rows).

```sh
uv run jev-course data --out runs/course/data --train 2000 --validation 400 --calibration 400 --test 400 --seed 700001
```

Success: JSON with `"disjoint": true`, four digests, and every `shared_contexts`
value `0`.

**2. Train.** The best validation epoch is kept, not the last one.

```sh
uv run python -m jevlike.train runs/course/data/train.jsonl --validation runs/course/data/validation.jsonl --output runs/course/model.pt --encoder tiny --width 64 --rank 64 --epochs 8 --batch-size 64 --learning-rate 0.002 --context-tokens 192 --option-tokens 32 --device cpu --seed 7
```

Success: one JSON line per epoch, then a final line with the checkpoint path and
`best_validation_nll`.

**3. Evaluate on the test split with the upstream evaluator.**

```sh
uv run python -m jevlike.eval runs/course/model.pt runs/course/data/test.jsonl --device cpu
```

Success: JSON with `model` (top1, top3, ece) and `shuffled_context` — the control
that tells you whether the model uses the context at all. Ignore `top3` when a row
has fewer than three options; it is capped by the option count.

**4. Fit the temperature on the calibration split only.**

```sh
uv run jev-course fit runs/course/model.pt runs/course/data/calibration.jsonl --output runs/course/temperature.json --device cpu
```

Success: `runs/course/temperature.json` with a finite positive `temperature`, the
digest of the calibration file it used, and `calibration_before`/`calibration_after`
taken on that same split. The fit can never make *that* split worse, because `1.0`
is always one of the candidates.

**5. Judge on the held-out split — before and after calibration.**

```sh
uv run jev-course test runs/course/model.pt runs/course/data/test.jsonl --temperature runs/course/temperature.json --device cpu
```

Success: `test_before` and `test_after` blocks plus the `shuffled_context_control`.
The calibrated numbers may be *worse* than the raw ones: that is a real result, and
it is the reason the temperature is fitted on its own split. Compare the two blocks
yourself; do not assume the arrow points up.

**6. Close the terminal, open a new one, reload and predict.**

```sh
uv run jev-course predict runs/course/model.pt --temperature runs/course/temperature.json --context "Choose the exact badge amber badger. Badge: amber badger." --option "azure crane" --option "amber badger" --option "gold heron" --device cpu
```

Success: JSON with `choice`, per-option `probabilities`, `probability_sum` close to
1, and — with `--permute-check` — the same choice when the options are reversed.
This is a fresh process: the weights came off disk, no training happened.
The order check records a structural invariant: these scorers have no option-position
input. Passing it is not evidence of learned semantic ability or checkpoint quality.

**7. Keep the model together with what produced it.** A checkpoint alone is not a
result: copy `model.pt`, `temperature.json`, `evaluation.json`, the data manifest
and your environment listing into one directory, as in the tutorial's `my-model/`
layout.

## One command for the whole loop

```sh
uv run jev-course run-all --out evidence --seed 700001
```

`run-all` executes steps 1–6 as separate subprocesses of the current interpreter,
each one a documented command, and stops at the first failure:

- every step's stdout, stderr, exit code, duration and the exact argv go to
  `<out>/logs/<step>.log`;
- a non-zero exit code aborts the run; the manifest is still written with
  `"status": "failed"` and the failing step recorded;
- gates are enforced, not printed: splits disjoint and non-empty, the temperature
  file finite and positive, the fit's recorded digest equal to the calibration
  split's and different from the test split's, ECE bins covering every test row,
  predicted probabilities summing to 1, and option order not changing the choice;
- `<out>/manifest.json` collects argv, digests, artefact hashes, step exit codes,
  the environment snapshot and the results of every gate;
- exit code `0` means every step and every gate passed.

Two behaviours worth knowing before you run it: `--out` refuses to start when the
directory already holds files (a previous run, or data you care about) and tells you
to pass `--force` for a deliberate re-run, and a relative `--out` is resolved against
the repository root, never against your shell's current directory.

Useful flags: `--out`, `--device`, `--train/--validation/--calibration/--test`,
`--seed`, `--train-seed`, the model flags (`--epochs`, `--batch-size`,
`--learning-rate`, `--width`, `--rank`, `--context-tokens`, `--option-tokens`),
`--force`, and `--predict-context`/`--predict-option` to choose the final reload
example (the default is the first test row, with its label recorded for information).

## What the commands write

| Path (under the `--out` directory) | Content |
|---|---|
| `data/{train,validation,calibration,test}.jsonl` | the four disjoint splits |
| `data-manifest.json` | sizes, seed, per-split digests, shared-context report |
| `model.pt` | config plus the best-validation weights |
| `temperature.json` | fitted temperature, the split it came from, metrics on it |
| `evaluation.json` | held-out metrics raw vs calibrated, control runs, split digest |
| `predictions.json` | the reload example: choice, probabilities, order control |
| `checks.json` | the seven self-checks with their details |
| `manifest.json` | every step, gate, digest and environment fact |
| `logs/<step>.log` | full output of each subprocess |

## Use your own data

One JSON object per line, UTF-8, double quotes, `label` is the zero-based index of
the correct option:

```json
{"context":"The customer needs a refund.","options":["refund","sales","technical support"],"label":0}
```

Rules the tools enforce (a violation stops the run with the row number):

- one JSON object per line, valid UTF-8 JSON; a malformed line or a non-object row
  is refused rather than skipped;
- a non-blank `context` and at least two options per row, each a non-blank string
  (whitespace-only counts as blank);
- `label` an integer inside `0 … len(options) - 1` — `true`/`false` are refused, not
  read as 1/0;
- no repeated option inside a row — two identical candidates would share one
  probability and the row would not have one correct answer;
- four non-empty splits with no context string shared between two of them.

Rules the tools cannot enforce, which decide whether the result means anything:

- **Split by source, not by row.** The same customer, ticket template, document or
  paraphrase must stay inside one split; keep the grouping key next to the data.
- **Do not reuse rows across splits**, and do not copy three examples hundreds of
  times and call it a dataset.
- **Put the hard negatives in.** Similar-looking options (refund vs return,
  forgotten password vs broken sign-in) must both appear, and the answer must not
  always sit in the same position.
- **Keep calibration separate from test.** A temperature fitted on the split you
  report is not a measurement of anything.

To run the loop on your own files, put them at `data/business/` with the same four
names. **Run the preflight first:** upstream `jevlike.train` reads whatever file you
hand it and checks none of the rules above, so the checks only happen if you ask for
them. `validate-data` loads all four splits, applies every row rule, proves the
splits are non-empty and share no context, and prints the digests:

```sh
uv run jev-course validate-data data/business
```

It exits non-zero and names the offending file and line on the first problem, so it
is safe to put in a script before training. Then reuse the commands above with those
paths:

```sh
uv run python -m jevlike.train data/business/train.jsonl --validation data/business/validation.jsonl --output runs/business/model.pt --encoder tiny --width 64 --rank 64 --epochs 8 --batch-size 64 --learning-rate 0.002 --context-tokens 192 --option-tokens 32 --device cpu --seed 7
uv run jev-course fit runs/business/model.pt data/business/calibration.jsonl --output runs/business/temperature.json --device cpu
uv run jev-course test runs/business/model.pt data/business/test.jsonl --temperature runs/business/temperature.json --device cpu
```

### Chinese text and the byte budget

The default encoder reads UTF-8 **bytes**, truncated to `--context-tokens` bytes
(192 by default) and options to `--option-tokens`. A Chinese character costs about
three bytes, so the effective context is roughly sixty characters, and a difference
between two options that appears only after the truncation point is invisible.
Check your data with `--context-tokens 384 --option-tokens 64` if candidates are
long, and remember that a bigger window is not a substitute for a clear task.

### Evaluation batch size

`jev-course fit` and `test` accept `--batch-size` (default 64). Use a smaller
batch for pretrained encoders; each option is also encoded. The shuffled-context
control rotates contexts **within each batch**, so its result depends on batch
size, and a batch of one does not shuffle anything. Reports record the batch size.
Compare controls only with the same batching; this control is not a benchmark score.

## What this model is, and what it is not

- It is a **Jev-like option scorer**: it uses the same input and output shape as
  the tutorial's subject (context plus a variable list of options, one score per
  option, one pass). It is not TypeSafe's Jev and does not reproduce any private
  training method.
- The default `tiny` encoder learns embeddings from raw bytes. Options that differ
  only in character order or beyond the byte budget are effectively the same input.
  The badge task is **text matching**, not semantic understanding: a high score on
  it says nothing about Chinese tickets, long documents or reasoning.
- Probabilities are a distribution **inside the menu you supplied**. A probability
  of 0.99 is not evidence that the answer is true, and it is not permission to act
  automatically.
- The calibration guarantee is about the calibration split. Generalisation to the
  test split is measured, not promised.
- `shuffled_context` in `jev-course test` is the control for "does the model read
  the context at all". If the shuffled number is as good as the real one, the model
  is guessing from the options.

## Reproducibility

- `uv.lock` pins every dependency; `uv sync` reproduces `.venv`.
- Data generation is deterministic for a given `--seed` in any interpreter:
  `jev-course check` proves it under two different `PYTHONHASHSEED` values.
- Every run's `manifest.json` records the interpreter, PyTorch build, device
  availability, the effective `PYTHONHASHSEED`, per-step argv and artefact digests.
- `--device cpu` is what this repository was built and checked on. `mps` and `cuda`
  are accepted by the flags because upstream supports them; they are untested here.

## Provenance and licence

- Upstream: [vinnylarouge/jevlike](https://github.com/vinnylarouge/jevlike) at
  `94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452` (MIT, Minimal Labs), vendored under
  `vendor/jevlike/` with three documented source fixes (CPU snapshot copy, safe
  checkpoint loading, deterministic option order) plus an ECE bin-ownership fix.
- Details, per-file hashes and the reviewable patch:
  [PROVENANCE.md](PROVENANCE.md), `vendor/jevlike/PROVENANCE.json`,
  `vendor/jevlike/patches/0001-course-fixes.diff`.
- This repository: MIT, see [LICENSE](LICENSE). The vendored licence is kept at
  `vendor/jevlike/LICENSE`.
