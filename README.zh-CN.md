# train-your-first-jev

[English](README.md) | **简体中文**

这是一个本地教学仓库，包含两条路径：一个快速 CPU 字节打分器练习，以及一门交互式课程——在开放预训练的 Qwen2.5-0.5B 模型上训练 LoRA 适配器加一个决策头。后者使用真实的 CommonsenseQA 题目，不是合成的徽章匹配。它是 Jev-like 的选择打分器，不是官方 Jev。

仓库在本地扩展了固定版本的 MIT `jevlike` 源码，详见 [PROVENANCE.md](PROVENANCE.md)。

交互式课程提供两种语言：在线课文默认英文，另有中文版；终端讲义（`jev-course guide`）目前输出中文。

## 在线学习

[打开九节交互课程](https://cexll.github.io/train-your-first-jev/)——默认英文，中文版在 [index.zh-CN.html](https://cexll.github.io/train-your-first-jev/index.zh-CN.html)。两个页面提供同样的九节课和练习，共用同一份学习进度。

网页提供完整课文、逐课自测、温度滑块和本地学习进度，不需要登录。
网页不会运行 Qwen 训练；真正的 LoRA 训练在自己的电脑上通过下方命令执行。
站点源码为 `docs/index.html` 与 `docs/index.zh-CN.html`，GitHub Pages 从 `main` 的 `/docs` 发布，无需前端构建。

## 交互式开源模型训练

```sh
git clone https://github.com/cexll/train-your-first-jev.git
cd train-your-first-jev
uv sync --locked
uv run jev-course guide --out runs/my-jev --device mps
```

每课解释一个步骤，显示真实命令，按回车执行，输入 `q` 退出。流程包括数据准备、训练、评估、独立温度校准和重载预测；最后可以输入自己的题目和候选答案。
首次运行需要下载约 1 GB 的基座权重和公开题库。非空输出目录不会被覆盖。
长步骤会等待子进程完成后展示输出，不是实时训练进度条。

训练更新 Qwen 注意力层的 LoRA 参数及评分头，原始基座权重保持不变。
固定基座：`Qwen/Qwen2.5-0.5B`，revision `060db6499f32faf8b98477b0a26969ef7d8b9987`（Apache-2.0）。
`model.pt` 保存适配器和评分头，不包含基座；换机器需要重新下载该固定版本。
数据来源为 [CommonsenseQA](https://www.tau-nlp.org/commonsenseqa)（MIT，来源与哈希由数据报告记录），使用英文题目，按概念分组重新划分，不能与官方排行榜直接比较。

本机 M1 Pro / MPS 实测：9,792 条训练题，512 条验证题，256 条校准题，256 条留出题。完整流水线的训练阶段耗时 2,686.37 秒（约 45 分钟），记录见 `runs/lora-pipeline/logs/train.log`；耗时随机器负载变化。按验证损失选择第 2 轮：交互式课程的完整实跑另保存在 `runs/guide-acceptance/`，训练阶段为 2,991.43 秒（约 50 分钟），测试准确率同样为 47.66%。两次记录都是本机测量，不是速度保证。课程会显示三轮验证损失 1.4543、1.2903、1.4633 的参考例子，解释为何保存第 2 轮；本次执行的实际结果在每个步骤结束后单独显示。

| 对照 | 留出准确率 |
|---|---:|
| 五个随机评分头中的最好结果 | 18.36% |
| 冻结基座，2,048 条训练题 | 25.39% |
| LoRA，2,048 条训练题 | 31.64% |
| LoRA，9,792 条训练题 | 47.66%（122 / 256） |
| 最后模型打乱上下文 | 14.84% |

最后模型校准前后 NLL 为 1.2568 → 1.2476，ECE 为 0.0636 → 0.0378。本机证据在 `runs/lora-full/`，属于未纳入版本控制的运行产物。
这些成绩是开发过程中反复检查过的留出集结果，不是独立盲测；预训练基座也可能见过公开题库。它们证明本次训练有学习信号，不证明通用能力、生产可靠性或与 Jev 等效。

下面的 CPU quickstart 保留为快速理解训练流程的字节模型练习，不能代替上述语义任务。

## 与中文讲义的关系

如果你在对照 `show-me-train-a-jev.html`（《从零训练一个 Jev-like 决策模型 · 完整学习手册》）学习，命令是一一对应的，只有一处替换：讲义里单独的 `calibrate_lesson.py` 配套脚本，在这里打包为 `jev-course fit` / `test` / `predict`（拟合、指标、预测都相同，另加了输入校验和划分摘要）。训练、数据、评估、预测这几条命令两边都是上游 `jevlike`：

| 讲义 | 本仓库 |
|---|---|
| `python -m jevlike.data synthetic --output data/course` | `uv run jev-course data --out runs/course/data --seed 700001`（四个划分，不是三个） |
| `python -m jevlike.train ...` | `uv run python -m jevlike.train ...` |
| `python -m jevlike.eval ...` | `uv run python -m jevlike.eval ...` |
| `python -m jevlike.predict ...` | `uv run python -m jevlike.predict ...` |
| `python calibrate_lesson.py fit ...` | `uv run jev-course fit ...` |
| `python calibrate_lesson.py test ...` | `uv run jev-course test ...` |
| `python calibrate_lesson.py predict ...` | `uv run jev-course predict ...` |

讲义中"修复 CPU checkpoint 拷贝"那一步，在这里已经应用到了 vendored 源码（见 [PROVENANCE.md](PROVENANCE.md)）；`uv run jev-course check` 会验证这一性质，而不需要你再去改文件。

## 你需要准备什么

- `git` 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)（`uv --version` 要能打印版本号）。
- 首次 `uv sync` 需要联网（会下载 Python 3.12 和 PyTorch）。
- 下面的 CPU 练习不需要 GPU。上面的交互式 LoRA 课程在 M1 Pro / MPS 上验证过；它的 CPU 和 CUDA 训练路径没有端到端跑过。

## 安装

```sh
uv sync
uv run jev-course check
```

`uv sync` 依据 `uv.lock`（Python 3.12）创建 `.venv`，并以可编辑模式安装本项目以及 vendored 的 `jevlike`。`jev-course check` 运行七项自检，任一项失败就以非零码退出——它不需要已训练的模型。

## 四个划分

| 划分 | 用途 | 绝不能用于 |
|---|---|---|
| `train.jsonl` | 更新权重 | 选择温度、最终汇报 |
| `validation.jsonl` | 选择最佳轮次 | 当作结果汇报准确率 |
| `calibration.jsonl` | 只用于拟合温度 | 更新权重 |
| `test.jsonl` | 最终考试，只考一次 | 任何会影响选择的东西 |

`jev-course data` 用同一个种子写出全部四个划分，随后证明没有任何上下文串出现在两个划分里；`jev-course fit` 会记录它拟合所用文件的摘要，因此你可以核对测试划分没有参与拟合。

## 快速开始：完整流程，逐步执行

所有命令都在仓库根目录执行。每一步都说明成功的样子。

**1. 生成四个互不重叠的划分**（默认 2000 / 400 / 400 / 400 行）。

```sh
uv run jev-course data --out runs/course/data --train 2000 --validation 400 --calibration 400 --test 400 --seed 700001
```

成功：JSON 中出现 `"disjoint": true`、四个摘要，并且每个 `shared_contexts` 值都是 `0`。

**2. 训练。** 保留验证损失最好的轮次，不是最后一轮。

```sh
uv run python -m jevlike.train runs/course/data/train.jsonl --validation runs/course/data/validation.jsonl --output runs/course/model.pt --encoder tiny --width 64 --rank 64 --epochs 8 --batch-size 64 --learning-rate 0.002 --context-tokens 192 --option-tokens 32 --device cpu --seed 7
```

成功：每轮一行 JSON，最后一行给出 checkpoint 路径和 `best_validation_nll`。

**3. 用上游评估器在测试划分上评估。**

```sh
uv run python -m jevlike.eval runs/course/model.pt runs/course/data/test.jsonl --device cpu
```

成功：JSON 里出现 `model`（top1、top3、ece）和 `shuffled_context`——后者是判断模型是否真的读了上下文的对照。当某行选项少于三个时忽略 `top3`；它受选项数量封顶。

**4. 只在校准划分上拟合温度。**

```sh
uv run jev-course fit runs/course/model.pt runs/course/data/calibration.jsonl --output runs/course/temperature.json --device cpu
```

成功：`runs/course/temperature.json` 里有有限且为正的 `temperature`、它所用校准文件的摘要，以及在同一划分上取到的 `calibration_before` / `calibration_after`。这次拟合不可能把*那个*划分变得更差，因为 `1.0` 始终是候选之一。

**5. 在留出划分上判定——校准前后各一次。**

```sh
uv run jev-course test runs/course/model.pt runs/course/data/test.jsonl --temperature runs/course/temperature.json --device cpu
```

成功：`test_before` 和 `test_after` 两块结果，外加 `shuffled_context_control`。校准后的数字可能比原始数字*更差*：这是真实结果，也正是温度要单独用一个划分拟合的原因。请自己对比这两块；不要假设箭头一定向上。

**6. 关掉终端，新开一个，重载并预测。**

```sh
uv run jev-course predict runs/course/model.pt --temperature runs/course/temperature.json --context "Choose the exact badge amber badger. Badge: amber badger." --option "azure crane" --option "amber badger" --option "gold heron" --device cpu
```

成功：JSON 里有 `choice`、每个选项的 `probabilities`、接近 1 的 `probability_sum`，并且在加上 `--permute-check` 后，选项倒序时给出同样的选择。这是一个全新进程：权重从磁盘加载，没有发生任何训练。
顺序检查记录的是一个结构性不变量：这类打分器没有选项位置输入。通过它并不代表模型学到了语义能力，也不代表 checkpoint 质量好。

**7. 把模型和产生它的东西放在一起。** 只有 checkpoint 不算结果：把 `model.pt`、`temperature.json`、`evaluation.json`、数据清单和你自己的环境信息复制到同一个目录，就像讲义里的 `my-model/` 布局那样。

## 一条命令跑完整个流程

```sh
uv run jev-course run-all --out evidence --seed 700001
```

`run-all` 把第 1–6 步作为当前解释器的独立子进程依次执行，每一步都是一条有文档的命令，遇到第一个失败就停下：

- 每一步的 stdout、stderr、退出码、耗时和确切的 argv 都写入 `<out>/logs/<step>.log`；
- 非零退出码会中止运行；manifest 仍会写出，标记 `"status": "failed"` 并记录失败的步骤；
- 门禁是强制执行的，不是打印出来看看：划分互不重叠且非空、温度文件有限且为正、拟合记录的摘要等于校准划分的摘要且不同于测试划分的摘要、ECE 分箱覆盖每一条测试行、预测概率之和为 1、选项顺序不改变选择；
- `<out>/manifest.json` 汇总 argv、摘要、产物哈希、步骤退出码、环境快照以及每个门禁的结果；
- 退出码 `0` 表示每一步和每个门禁都通过了。

运行前有两点值得知道：`--out` 指向的目录里已经有文件时（上一次运行，或你在意的数据）它会拒绝启动，并提示你若要故意重跑就加 `--force`；相对的 `--out` 一律相对仓库根目录解析，绝不相对你当前所在的 shell 目录。

常用参数：`--out`、`--device`、`--train/--validation/--calibration/--test`、`--seed`、`--train-seed`、模型参数（`--epochs`、`--batch-size`、`--learning-rate`、`--width`、`--rank`、`--context-tokens`、`--option-tokens`）、`--force`，以及用于挑选最终重载例子的 `--predict-context` / `--predict-option`（默认取第一条测试行，并记录它的标签以备查阅）。

## 这些命令会写出什么

| 路径（在 `--out` 目录下） | 内容 |
|---|---|
| `data/{train,validation,calibration,test}.jsonl` | 四个互不重叠的划分 |
| `data-manifest.json` | 每个划分的规模、种子、摘要，以及共享上下文报告 |
| `model.pt` | 配置加上验证损失最好的权重 |
| `temperature.json` | 拟合出的温度、它来自哪个划分、在该划分上的指标 |
| `evaluation.json` | 留出集指标（原始 vs 校准后）、对照运行、划分摘要 |
| `predictions.json` | 重载示例：选择、概率、顺序对照 |
| `checks.json` | 七项自检及其细节 |
| `manifest.json` | 每一步、每个门禁、每个摘要和环境事实 |
| `logs/<step>.log` | 每个子进程的完整输出 |

## 使用你自己的数据

每行一个 JSON 对象，UTF-8，双引号，`label` 是正确选项从 0 开始的索引：

```json
{"context":"The customer needs a refund.","options":["refund","sales","technical support"],"label":0}
```

工具会强制检查的规则（任何一条违反都会停下并给出行号）：

- 每行一个 JSON 对象，合法的 UTF-8 JSON；格式错误的行或非对象行会被拒绝，而不是跳过；
- 每行要有非空的 `context` 和至少两个选项，每个都是非空字符串（只有空白字符视为空）；
- `label` 是落在 `0 … len(options) - 1` 内的整数——`true` / `false` 会被拒绝，不会被当作 1/0；
- 一行之内不能有重复选项——两个完全相同的候选会共用同一个概率，这一行也就没有唯一正确答案了；
- 四个非空划分，且没有任何上下文串被两个划分共用。

工具管不了、却决定结果是否有意义的规则：

- **按来源划分，而不是按行划分。** 同一个客户、同一个工单模板、同一篇文档或同一段改写必须留在同一个划分里；把分组键放在数据旁边。
- **不要跨划分复用同一行**，也不要把三个例子复制几百遍就当成数据集。
- **放进困难负例。** 看起来相近的选项（refund 对 return、忘密码对登录故障）必须同时出现，而且答案不能总是落在同一个位置。
- **校准必须和测试分开。** 在你汇报的那个划分上拟合出来的温度，测不出任何东西。

要在自己的文件上跑这套流程，把它们放到 `data/business/` 下，文件名与上面四个一致。**先跑预检：**上游 `jevlike.train` 会照单全收你给它的文件，上面这些规则一条都不检查，所以只有你主动要求时才会检查。`validate-data` 会加载全部四个划分，套用每一条行规则，证明各划分非空且不共享上下文，并打印摘要：

```sh
uv run jev-course validate-data data/business
```

遇到第一个问题它就以非零码退出，并指出是哪个文件哪一行，因此可以安全地放进训练前的脚本里。然后把这些路径套回上面的命令：

```sh
uv run python -m jevlike.train data/business/train.jsonl --validation data/business/validation.jsonl --output runs/business/model.pt --encoder tiny --width 64 --rank 64 --epochs 8 --batch-size 64 --learning-rate 0.002 --context-tokens 192 --option-tokens 32 --device cpu --seed 7
uv run jev-course fit runs/business/model.pt data/business/calibration.jsonl --output runs/business/temperature.json --device cpu
uv run jev-course test runs/business/model.pt data/business/test.jsonl --temperature runs/business/temperature.json --device cpu
```

### 中文文本与字节预算

默认编码器读的是 UTF-8 **字节**，按 `--context-tokens` 个字节截断（默认 192），选项按 `--option-tokens` 截断。一个汉字约占三个字节，所以有效上下文大致只有六十个字符，而两个选项之间只出现在截断点之后的差异是看不见的。如果候选项很长，用 `--context-tokens 384 --option-tokens 64` 检查你的数据；同时记住，更大的窗口不能替代一个清晰的任务。

### 评估批大小

`jev-course fit` 和 `test` 接受 `--batch-size`（默认 64）。对预训练编码器用更小的批；每个选项也会被编码。打乱上下文对照是在**每个批内部**轮换上下文的，因此它的结果依赖批大小，而批大小为 1 时等于什么都没打乱。报告会记录批大小。只在相同批设定下比较对照数值；这个对照不是基准分数。

## 这个模型是什么，不是什么

- 它是一个 **Jev-like 选项打分器**：输入输出形状与讲义的主题相同（上下文加一组可变的选项，每个选项一个分数，一次前向）。它不是 TypeSafe 的 Jev，也不复现任何非公开的训练方法。
- 默认的 `tiny` 编码器从原始字节学习嵌入。只在字符顺序上不同、或超出字节预算的选项，实际上是同一个输入。徽章任务是**文本匹配**，不是语义理解：在它上面得高分，说明不了中文工单、长文档或推理能力。
- 概率是**在你给出的菜单内部**的分布。0.99 的概率并不证明答案是真的，也不构成自动执行它的许可。
- 校准保证针对的是校准划分。到测试划分的泛化是测出来的，不是承诺出来的。
- `jev-course test` 里的 `shuffled_context` 是"模型到底读不读上下文"的对照。如果打乱后的数字和真实的一样好，说明模型是在从选项里猜。

## 可复现性

- `uv.lock` 固定了每一个依赖；`uv sync` 可复现 `.venv`。
- 给定 `--seed` 时，数据生成在任何解释器下都是确定的：`jev-course check` 会在两个不同的 `PYTHONHASHSEED` 取值下证明这一点。
- 每次运行的 `manifest.json` 都记录解释器、PyTorch 构建、设备可用性、生效的 `PYTHONHASHSEED`、每步 argv 和产物摘要。
- 本仓库是在 `--device cpu` 上构建和检查的。`mps` 和 `cuda` 之所以被参数接受，是因为上游支持它们；本仓库没有测试过。

## 来源与许可

- 上游：[vinnylarouge/jevlike](https://github.com/vinnylarouge/jevlike)，版本 `94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452`（MIT，Minimal Labs），以 vendored 形式放在 `vendor/jevlike/` 下，另有三处有记录的源码修复（CPU 快照拷贝、安全的 checkpoint 加载、确定的选项顺序）以及一处 ECE 分箱归属修复。
- 细节、逐文件哈希和可审查的补丁：[PROVENANCE.md](PROVENANCE.md)、`vendor/jevlike/PROVENANCE.json`、`vendor/jevlike/patches/0001-course-fixes.diff`。
- 本仓库：MIT，见 [LICENSE](LICENSE)。vendored 的许可证保留在 `vendor/jevlike/LICENSE`。
