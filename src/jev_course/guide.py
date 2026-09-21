"""Interactive Chinese course: train LoRA adapters on an open base model.

``jev-course guide`` turns the whole flow into nine lessons. Each lesson explains
one thing, shows the real command, and waits for Enter before that command runs;
typing ``q`` (or ``quit``/``exit``) stops cleanly, and so does an exhausted stdin.
Nothing is trained before the reader has approved it.

The subprocess work is :func:`jev_course.pipeline.run_all` — the same checked
subprocesses, gates and manifest ``run-all`` produces. This module adds the
lessons, the data preparation and the final reload stage; it never runs a second
training loop and never writes inside the run directory before ``run_all`` does.
The data directory lives *outside* the run output (``out.parent/(out.name + '-data')``
by default), the source report is stored beside it, and a non-empty run directory
is refused, never overwritten.

Prompt contract (stdin)
-----------------------
A successful run asks for eight approvals, then lets the reader ask their own
question; each prompt consumes exactly one line:

1. lesson 1 (base model and adapters) and lesson 2 (real data), then one before
   each of the six subprocess steps ``check``/``train``/``eval``/``fit``/``test``/
   ``predict`` — an empty line continues, ``q`` stops;
2. then a question line, one line per candidate answer (at least two), an empty
   line to end the candidates, and an empty line at the question prompt to finish.

So a complete non-interactive run is exactly::

    "\\n" * 8 + "题目\\n选项A\\n选项B\\n\\n\\n"

The exact prompt strings are :data:`PROMPT_CONTINUE`, :data:`PROMPT_QUESTION` and
:data:`PROMPT_OPTION`.

Exit codes
----------
``0`` finished, or the reader typed ``q``; ``1`` a step failed (the manifest and
the failing log are printed); ``2`` a precondition was refused (non-empty
``--out``, a data directory inside it, an unusable device, unusable or
unfetchable data); ``3`` stdin ended before an approval was given.

End of input during the last lesson is not a cancellation: there is nothing left
to approve, so the course closes normally and exits ``0``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import __version__
from .dataset import (
    SPLITS,
    describe,
    load_examples,
    load_splits,
    refuse_nonempty_directory,
    require_disjoint,
)
from .pipeline import REPO_ROOT, _documented, run_all
from .real_data import COURSE_RESERVE, prepare_course

# ---------------------------------------------------------------------------
# Exit codes and the recipe (the one runs/lora-full was trained with).
# ---------------------------------------------------------------------------

EXIT_DONE = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2
EXIT_NO_INPUT = 3

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
BASE_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
EPOCHS = 3
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
CONTEXT_TOKENS = 128
OPTION_TOKENS = 32
LEARNING_RATE = 5e-5
WIDTH = 64
RANK = 64
TRAIN_SEED = 7
MISTAKES_SHOWN = 3

PROMPT_CONTINUE = "按回车继续，输入 q 停止 > "
PROMPT_QUESTION = "题目（直接回车结束课程）> "
PROMPT_OPTION = "候选答案（每行一个，至少两个；直接回车结束）> "

STOP_WORDS = {"q", "quit", "exit"}
SOURCE_REPORT_SUFFIX = "-source-report.json"

RULE = "=" * 64


class GuideCancelled(RuntimeError):
    """The reader stopped at a prompt (``q`` or end of input)."""


class GuideRefused(RuntimeError):
    """A precondition failed; nothing has been executed."""


class Session:
    """The interactive side: lesson text, prompts, and the q/EOF contract."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.lesson = 0
        self.stopped: str | None = None  # "q" or "eof" once the reader stops
        self.pending: str | None = None  # the step the last pause was guarding

    # -- output ------------------------------------------------------------

    def header(self, title: str) -> None:
        self.lesson += 1
        print()
        print(RULE)
        print(f"第 {self.lesson} 课 · {title}")
        print(RULE)

    def pause(
        self,
        title: str,
        lines: tuple[str, ...],
        *,
        command: str | None = None,
        command_label: str = "这一步会执行：",
        step: str | None = None,
    ) -> None:
        """Print one lesson, then require Enter (or ``q``/EOF) before continuing."""
        self.pending = step or title
        self.header(title)
        for line in lines:
            print(line)
        if command:
            print()
            print(command_label)
            print(f"  {command}")
        print()
        self._read_continue()

    def ask(self, prompt: str) -> str:
        """One answer line. A blank line is the caller's "done" value.

        End of input is a clean finish here: the last lesson only collects input,
        so there is nothing left to approve.
        """
        print(prompt, end="", flush=True)
        line = sys.stdin.readline()
        if line == "":
            print()
            self.stopped = "eof"
            return ""
        return line.strip()

    # -- input --------------------------------------------------------------

    def _read_continue(self) -> None:
        print(PROMPT_CONTINUE, end="", flush=True)
        line = sys.stdin.readline()
        if line == "":
            print()
            self.stopped = "eof"
            raise GuideCancelled("标准输入结束")
        if line.strip().lower() in STOP_WORDS:
            self.stopped = "q"
            raise GuideCancelled("用户输入 q")


# ---------------------------------------------------------------------------
# Path and device preflight.
# ---------------------------------------------------------------------------


def _absolute(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _default_data_dir(out: Path) -> Path:
    return out.parent / (out.name + "-data")


def _refuse_paths(out: Path, data_dir: Path) -> None:
    if out.exists() and not out.is_dir():
        raise GuideRefused(
            f"--out {out} 已经存在而且不是目录。请换一个 --out。"
        )
    if not out.name:
        raise GuideRefused("--out 需要是一个有名字的目录（例如 runs/my-jev）。")
    try:
        refuse_nonempty_directory(out, False, "the run outputs")
    except ValueError as error:
        raise GuideRefused(
            f"--out {out} 不是空的，课程不会覆盖它（{error}）。\n"
            "  请换一个 --out（例如 runs/my-jev-2），或先把那个目录移走/删除。"
        ) from error
    if data_dir == out or out in data_dir.parents:
        raise GuideRefused(
            f"数据目录 {data_dir} 落在运行输出 {out} 里面，run-all 会拒绝这种布局"
            "（复制数据时读写的是同一批文件）。请把数据放在输出目录之外。"
        )
    if data_dir.exists() and not data_dir.is_dir():
        raise GuideRefused(f"数据目录 {data_dir} 是一个文件，不是目录。")


def _require_device(device: str) -> None:
    if device not in ("mps", "cpu", "cuda"):
        raise GuideRefused(f"未知设备 {device!r}；可选 mps / cpu / cuda。")
    if device == "cpu":
        return
    try:
        import torch
    except Exception as error:  # pragma: no cover - broken environment
        raise GuideRefused(
            f"PyTorch 还不可用（{type(error).__name__}: {error}）。先跑 uv sync --locked。"
        ) from error
    if device == "cuda":
        available = bool(torch.cuda.is_available())
        what = "CUDA"
    else:
        backend = getattr(torch.backends, "mps", None)
        available = bool(backend and backend.is_available())
        what = "MPS (Apple GPU)"
    if not available:
        raise GuideRefused(
            f"--device {device} 在本机不可用（{what} 未就绪）。"
            "可选择 --device cpu；本课的完整 LoRA 训练仅在 MPS 上实测，CPU 耗时未验证。"
        )


# ---------------------------------------------------------------------------
# Lesson 1: base model versus adapters.
# ---------------------------------------------------------------------------


def _lesson_intro(session: Session, out: Path, device: str, data_dir: Path, external: bool) -> None:
    session.pause(
        "基座与适配器：这一课在训练什么",
        (
            "这一课只解释，不执行任何命令。",
            "",
            "基座（base）＝ Qwen/Qwen2.5-0.5B：约 5 亿参数的公开模型（Apache-2.0），",
            f"本课固定使用版本 {BASE_REVISION}，保留基座原有权重。",
            "LoRA 适配器是在注意力层里加入的可训练矩阵，用较少的参数调整模型的表示。",
            "评分头根据题干和候选答案的表示，为每个候选打分；softmax 把分数换算成概率。",
            "训练只更新适配器和评分头，约 126 万个参数，保存后的权重文件约 5 MB。",
            "",
            "这样可以减少训练所需的内存。适配器依赖基座，换机器时仍需下载同一个版本。",
            "基座名称和版本会保存在模型配置和运行记录中。",
            "",
            "接下来的课程：",
            "  第 2 课  真实数据与四个划分（下载官方 CommonsenseQA）",
            "  第 3 课  检查环境、数据和模型保存方式",
            "  第 4 课  训练 3 轮（本机参考约 45 分钟，不是完成时限）",
            "  第 5 课  在留出集上评估",
            "  第 6 课  独立温度校准",
            "  第 7 课  比较校准前后的结果",
            "  第 8 课  重载预测一题",
            "  第 9 课  重载看错题 + 你自己提问",
            "",
            f"运行目录：{out}（必须为空；课程不覆盖任何非空目录）",
            f"数据目录：{data_dir}（{'你指定，本课只校验' if external else '默认位置，与运行目录分开'}）",
            f"设备：{device}；课程版本：{__version__}",
        ),
        step="intro",
    )


# ---------------------------------------------------------------------------
# Lesson 2: real data and the four splits.
# ---------------------------------------------------------------------------


_SPLIT_LESSON: tuple[str, ...] = (
    "数据来自 CommonsenseQA（Talmor 等，2019，MIT）：英文常识多选题，每题一个题干加若干候选。",
    "只下载带答案的两份官方文件（train/dev random split），按 SHA-256 和字节数校验；",
    "官方 test 文件没有公开答案，本课不使用它。",
    "四个划分各管一件事，不能混用：",
    "  train        更新权重（适配器和评分头）",
    "  validation   比较各轮预测损失（NLL，越低越好），决定保存哪一轮",
    "  calibration  只用来拟合温度（第 6 课）",
    "  test         检查选对了多少题，以及校准前后的概率质量",
    "按题库的 concept（概念）字段分组，同一概念只分配给一份数据。",
    "生成后重新读取文件，检查四份数据是否有重复题干或共享概念。",
    "先预留 validation 512 条、calibration 256 条、test 256 条，",
    "其余可用题用于训练；开发时得到 9,792 条，本次数量以接下来显示的数据报告为准。",
    "题目是英文的；中文题不在数据里，只在最后一课手动输入时会走同一套流程，效果不保证。",
)


def _stage_data(session: Session, out: Path, data_dir: Path, external: bool) -> dict[str, object]:
    existing = data_dir.is_dir() and any(data_dir.iterdir())
    reuse = external or existing

    lines = list(_SPLIT_LESSON)
    lines.append("")
    if external:
        lines.append(
            f"你指定了 --data-directory {data_dir}：本课只按 validate-data 的规则校验它，"
            "不下载、不写入任何文件。"
        )
    elif existing:
        lines.append(
            f"{data_dir} 已经存在：本课只重新加载校验，不下载、不覆盖任何文件。"
        )
    else:
        lines.append(
            f"本课会下载官方 JSONL 题目并写入 {data_dir}（题目文件约 3.7 MB，通常几秒钟；"
            "不要时直接删掉该目录即可）。"
        )
    lines.append(f"原始数据保留在数据目录中；运行时会另存一份到 {out / 'data'}，供训练和评估使用。")
    if reuse:
        command = _documented("jev_course.cli", "validate-data", str(data_dir))
        label = "这一步用同一条规则在进程内校验（等价命令）："
    else:
        command = _documented("jev_course.real_data", "--course", "--out", str(data_dir))
        label = "这一步在课程进程内直接调用 prepare_course（等价命令）："
    session.pause(
        "真实数据与四个划分",
        tuple(lines),
        command=command,
        command_label=label,
        step="data",
    )

    report = _observe_splits(data_dir) if reuse else _prepare(data_dir)
    _print_data_summary(report)
    return report


def _prepare(data_dir: Path) -> dict[str, object]:
    print(f"正在下载并校验固定来源，然后写入 {data_dir} …")
    try:
        report = prepare_course(data_dir)
    except Exception as error:
        raise GuideRefused(
            f"数据准备失败：{type(error).__name__}: {error}\n"
            "  数据来自 S3 上的官方 JSONL 文件（约 3.7 MB），需要网络。如果本机访问不了，"
            "请先准备好四个 .jsonl，再用 --data-directory 指过来。"
        ) from error
    report = dict(report)
    report["reused"] = False
    _write_source_report(data_dir, report)
    return report


def _write_source_report(data_dir: Path, report: dict[str, object]) -> None:
    """Keep the source report beside the data, never inside the run output."""
    target = data_dir.parent / (data_dir.name + SOURCE_REPORT_SUFFIX)
    if target.exists():
        print(f"来源报告已存在，保留不改：{target}")
        return
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    target.write_text(text, encoding="utf-8")
    print(f"来源报告（下载证据、许可、每个文件的摘要、排除的行）：{target}")


def _observe_splits(directory: Path) -> dict[str, object]:
    """Validate an existing data directory the way ``validate-data`` does."""
    try:
        splits = load_splits(directory)
        shared = require_disjoint(splits)
    except (FileNotFoundError, ValueError, OSError) as error:
        raise GuideRefused(
            f"数据目录 {directory} 不可用：{type(error).__name__}: {error}\n"
            "  课程需要 train/validation/calibration/test 四个 .jsonl，行格式合法、"
            "非空且两两不共享题干。换一个 --data-directory，或删掉这个目录让课程重新下载"
            "（课程不会覆盖它）。"
        ) from error
    return {
        "schema": 1,
        "directory": str(directory),
        "reused": True,
        "note": ("existing directory, reloaded and validated in place; the raw release "
                 "was not re-downloaded, so this report has no source digests"),
        "sizes": {name: len(rows) for name, rows in splits.items()},
        "splits": {
            name: describe(directory / f"{name}.jsonl", rows)
            for name, rows in splits.items()
        },
        "shared_contexts": shared,
        "disjoint": all(count == 0 for count in shared.values()),
    }


def _print_data_summary(report: dict[str, object]) -> None:
    sizes = report["sizes"]
    print()
    if report.get("reused"):
        print(f"数据就绪（复用已有目录，未下载未覆盖）→ {report['directory']}")
    else:
        print(f"数据就绪（新下载并写入）→ {report['directory']}")
    for name in SPLITS:
        entry = report["splits"][name]
        print(
            f"  {name + '.jsonl':<18}{sizes[name]:>6} 行  "
            f"文件 sha256 {entry['file_sha256'][:12]}…  题干摘要 {entry['context_sha256'][:12]}…"
        )
    shared = report["shared_contexts"]
    print(f"  题干互斥：{'是' if report['disjoint'] else '否'}（两两共享题干数 {shared}）")
    if "shared_concepts" in report:
        print(
            f"  概念互斥：{'是' if report['group_disjoint'] else '否'}"
            f"（两两共享概念数 {report['shared_concepts']}）"
        )
    else:
        print("  概念互斥：无法在这里证明（复用的目录没有随数据保存来源信息，只有重新下载才能得到概念级证据）")
    full_course = sizes["train"] == 9792 and all(
        sizes[name] == COURSE_RESERVE[name] for name in ("validation", "calibration", "test")
    )
    if full_course:
        print("  样本数量与开发时的配方一致；数量相同不代表题目相同，也不保证得到相同成绩。")
    else:
        print(
            "  注意：这份数据的规模不是 runs/lora-full 的完整配方"
            f"（期望 9792 / {COURSE_RESERVE['validation']} / {COURSE_RESERVE['calibration']}"
            f" / {COURSE_RESERVE['test']}），下面的耗时是按实际行数改写的估算，"
            "成绩也不能直接对照 README 的 47.66%。"
        )
    first = load_examples(Path(str(report["directory"])) / "train.jsonl")[0]
    print(f"  样例（train 第一行）：{_clip(first.context, 72)}")
    print(f"    候选：{' / '.join(first.options)}（正确答案是第 {first.label + 1} 个）")


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# Lessons for the six subprocess steps, shown by the before_step callback.
# ---------------------------------------------------------------------------


_STEP_LESSONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "check": (
        "自检：检查训练流程",
        (
            "训练前先运行 jev-course check，用临时小模型和样本检查保存、加载及数据处理。",
            "  · 保存的最佳轮次不会被后续训练改写；",
            "  · 自己保存的模型可以重载，测试中的未获允许对象会被加载器拒绝；",
            "  · 同一种子生成相同的练习数据；",
            "  · 四份数据非空，且没有重复题干；",
            "  · 重复选项、错误标签和损坏的数据会被拒绝；",
            "  · 校准温度必须是有限的正数；",
            "  · 计算校准误差时，每条预测都计入统计。",
            "这些自检不修改你的数据，也不衡量模型准确率。检查失败时，课程会停下并显示日志。",
            "耗时随机器负载变化，请等待子进程退出后的结果。",
        ),
    ),
    "eval": (
        "在留出集上评估：数字和对照",
        (
            "在 test.jsonl 上统计准确率、前三名命中率和校准误差，并做一次打乱题干的对照。",
            "top1 是选对的比例；top3 是正确答案出现在前三名的比例。",
            "ECE 比较各置信度区间里的平均把握和实际正确率，越低通常越好。",
            "shuffled_context 将同一批题干互换，选项与答案保留，用来检查预测是否依赖题干。",
            "如果打乱后成绩几乎不降，需要检查模型是否只利用了选项特征；这一项不能单独证明原因。",
            "本次流程不会根据 test 的成绩更新权重或选择轮次。开发时已反复查看过这份划分，",
            "因此，复用同一份题目不能算独立盲测。后面还会展示具体错题。",
        ),
    ),
    "fit": (
        "独立温度校准：只在校准划分上拟合",
        (
            "温度是一个正数：先把候选分数除以它，再计算概率。温度越高，概率通常越平缓。",
            "程序在约 0.20 到 5.01 之间搜索温度，并包含不做调整的 1.0，选校准集 NLL 最低的值。",
            "NLL 衡量给正确答案的概率，越低越好；校准集上改善，不代表测试集也会改善。",
            "temperature.json 保存温度和校准文件摘要，可以检查这一步用了哪份数据。",
            "测试集不参与温度拟合。下一课会比较它在校准前后的结果。",
            "所有分数除以同一个正数，不改变最高分的选项；校准调整概率，不提高选题准确率。",
        ),
    ),
    "test": (
        "比较校准前后的结果",
        (
            "在同一份 test.jsonl 上比较原始概率和校准后的概率。",
            "NLL 看正确答案的概率，Brier 看整组概率与答案的平方误差，ECE 看把握与正确率的差距。",
            "三个指标都越低越好，但不一定同时改善；程序会如实展示，不自动决定是否部署。",
            "开发时的参考记录（见 README，本次成绩随后显示）：NLL 1.2568 → 1.2476，"
            "ECE 0.0636 → 0.0378，温度 1.3259，留出准确率 47.66%（122/256）；",
            "五个随机评分头里最好的是 18.36%。",
            "这些成绩是开发中反复检查过的留出结果，不是独立盲测；预训练基座也可能见过公开题库。",
        ),
    ),
    "predict": (
        "重载预测：新进程，从磁盘加载",
        (
            "jev-course predict 开一个新进程加载 model.pt 和 temperature.json，给一条题目和几个候选打分；",
            "默认用 test 的第一行做例子，并把正确答案一起记录（只作记录，不参与打分）。",
            "--permute-check 把选项倒序再打一次分：如果选择变了就直接报错，说明分数对选项顺序敏感。",
            "这一步的产物是 predictions.json。课程最后还会在课程进程里重载一次，",
            "展示模型答错的题，然后让你输入自己的题目。",
        ),
    ),
}


def _lesson_train(sizes: dict[str, int], device: str) -> tuple[str, tuple[str, ...]]:
    train_rows = int(sizes["train"])
    steps = train_rows // BATCH_SIZE
    # Source: runs/lora-pipeline/logs/train.log, same recipe on M1 Pro/MPS.
    reference_rows, reference_seconds = 9792, 2686.37
    estimate = reference_seconds * train_rows / reference_rows
    if device == "mps":
        timing = f"本机 M1 Pro / MPS 参考：{reference_rows} 条 3 轮实测 {reference_seconds:.0f} 秒。"
        if train_rows == reference_rows:
            timing += "这是一次测量，不是完成时限；系统负载会改变耗时。"
        else:
            timing += f"这份数据是 {train_rows} 条，按行数线性外推约 {estimate / 60:.0f} 分钟（只是估算）。"
    else:
        timing = (
            f"这台机器的 --device 是 {device}，本仓库没有它的训练实测数字；"
            f"参考：M1 Pro / MPS 上 {reference_rows} 条 3 轮用了 {reference_seconds:.0f} 秒。"
        )
    return (
        "训练：只动适配器和评分头，3 轮",
        (
            "批量大小为 4，也就是每次一起训练 4 道题。共训练 3 轮，每轮看完整个训练集。",
            "适配器的最大学习率为 5e-5，评分头为 5e-4，按 OneCycleLR 调整；基座原有权重不变。",
            "上下文最多 128 个 token，选项最多 32 个；每轮用验证集计算 NLL，保存损失最低的一轮。",
            f"本次训练集 {train_rows} 行（每轮约 {steps} 步），验证集 {sizes['validation']} 行。",
            "训练结束后得到 model.pt，包含适配器和评分头，约 5 MB。",
            "",
            timing,
            "第一次运行还要先下载基座权重（约 1 GB）到 Hugging Face 缓存。",
            "每轮输出会在训练子进程结束后一起显示，期间日志文件尚未写入。",
            "开发时三轮验证损失分别为 1.4543、1.2903、1.4633，因此保存第 2 轮。",
            "第 3 轮训练损失继续下降，验证损失却上升：继续训练未必能改善新题表现。",
            "按回车开始本次训练；本次结果以随后显示的日志为准。",
        ),
    )


def _callbacks(session: Session, out: Path, sizes: dict[str, int], device: str):
    def before_step(name: str, module: str, arguments: list[str]) -> None:
        if name == "train":
            title, lines = _lesson_train(sizes, device)
        else:
            title, lines = _STEP_LESSONS[name]
        body = list(lines)
        body.append("")
        body.append(f"日志：{out / 'logs' / f'{name}.log'}")
        body.append("子进程结束后这里才打印结果（run-all 是等进程结束一次性写入日志的，不是实时流）。")
        session.pause(title, tuple(body), command=_documented(module, *arguments), step=name)

    def on_step(name: str, stdout: str) -> None:
        print()
        print(f"[完成] {_STEP_LABELS.get(name, name)}：{_completion(out, name)}")
        try:
            print(_STEP_SUMMARIES[name](stdout))
        except Exception:  # parsing is best-effort; the log keeps the truth
            print("  （结果解析失败，直接看日志文件）")

    return before_step, on_step


_STEP_LABELS = {
    "check": "自检",
    "train": "训练",
    "eval": "评估",
    "fit": "温度校准",
    "test": "最终判定",
    "predict": "重载预测",
}


def _completion(out: Path, name: str) -> str:
    log = out / "logs" / f"{name}.log"
    try:
        header = log.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return f"（日志 {log} 不存在）"
    fields = dict(part.split("=", 1) for part in header.split() if "=" in part)
    return f"exit {fields.get('exit', '?')}，用时 {fields.get('seconds', '?')} 秒；日志：{log}"


def _summarize_check(stdout: str) -> str:
    payload = json.loads(stdout)
    failed = [item["name"] for item in payload["checks"] if not item["ok"]]
    total = payload["passed"] + payload["failed"]
    text = f"  自检 {payload['passed']}/{total} 通过"
    return text + (f"，失败：{', '.join(failed)}" if failed else "")


def _summarize_train(stdout: str) -> str:
    lines = [json.loads(line) for line in stdout.splitlines() if line.strip().startswith("{")]
    epochs = [line for line in lines if "epoch" in line]
    best = min(epochs, key=lambda line: line["validation_nll"]) if epochs else None
    rows = []
    for line in epochs:
        mark = " ← 保存这一轮" if best is not None and line["epoch"] == best["epoch"] else ""
        rows.append(
            f"  第 {line['epoch']} 轮：train_nll {line['train_nll']:.4f}，"
            f"validation_nll {line['validation_nll']:.4f}{mark}"
        )
    checkpoint = next((line for line in lines if "checkpoint" in line), None)
    tail = ""
    if checkpoint is not None:
        tail = (
            f"\n  checkpoint：{checkpoint['checkpoint']}"
            f"（best_validation_nll {checkpoint['best_validation_nll']:.4f}）"
        )
    return "\n".join(rows) + tail


def _summarize_eval(stdout: str) -> str:
    payload = json.loads(stdout)
    model = payload["model"]
    control = payload["shuffled_context"]
    verdict = (
        "——下降不超过 5 个百分点，需结合错题检查是否依赖题干"
        if control["top1"] >= model["top1"] - 0.05
        else "——下降超过 5 个百分点，预测对题干变化有反应"
    )
    return (
        f"  top1 {model['top1'] * 100:.2f}%（{model['examples']} 题），"
        f"top3 {model['top3'] * 100:.2f}%，ECE {model['ece']:.4f}\n"
        f"  对照：打乱上下文后 top1 {control['top1'] * 100:.2f}%{verdict}"
    )


def _summarize_fit(stdout: str) -> str:
    payload = json.loads(stdout)
    fitted = payload["fitted_on"]
    return (
        f"  温度 {payload['temperature']:.4f}（81 点网格，含 1.0）\n"
        f"  拟合文件：{fitted['path']}（{fitted['rows']} 行，sha256 {fitted['file_sha256'][:12]}…）\n"
        f"  校准划分上的 NLL：{payload['calibration_before']['nll']:.4f} "
        f"→ {payload['calibration_after']['nll']:.4f}（在它自己的划分上不会更差）"
    )


def _summarize_test(stdout: str) -> str:
    payload = json.loads(stdout)
    before = payload["test_before"]
    after = payload["test_after"]
    control = payload["shuffled_context_control"]
    return (
        f"  留出集 {after['examples']} 题，校准前 → 后：\n"
        f"    准确率 {before['accuracy'] * 100:.2f}% → {after['accuracy'] * 100:.2f}%\n"
        f"    NLL {before['nll']:.4f} → {after['nll']:.4f}；"
        f"ECE {before['ece_10bins']:.4f} → {after['ece_10bins']:.4f}\n"
        f"  对照：打乱上下文 top1 {control['top1'] * 100:.2f}%"
        f"（不打乱 {payload['raw_context']['top1'] * 100:.2f}%）"
    )


def _summarize_predict(stdout: str) -> str:
    payload = json.loads(stdout)
    permutation = payload.get("permutation_check") or {}
    lines = [f"  选择：{payload['choice']}"]
    for option, probability in sorted(
        payload["probabilities"].items(), key=lambda item: item[1], reverse=True
    ):
        lines.append(f"    {probability * 100:5.1f}%  {option}")
    lines.append(
        f"  概率和 {payload['probability_sum']:.6f}；顺序检查 "
        f"{'稳定' if permutation.get('stable') else '不稳定'}"
        f"（最大漂移 {permutation.get('max_probability_drift', 0.0):.2e}）"
    )
    if payload.get("label") is not None:
        label = int(payload["label"])
        lines.append(f"  这条来自 test 第一行，正确答案是第 {label + 1} 个：{payload['options'][label]}")
    return "\n".join(lines)


_STEP_SUMMARIES: dict[str, Callable[[str], str]] = {
    "check": _summarize_check,
    "train": _summarize_train,
    "eval": _summarize_eval,
    "fit": _summarize_fit,
    "test": _summarize_test,
    "predict": _summarize_predict,
}


# ---------------------------------------------------------------------------
# Run outcome reporting: success table, failure details, cancellation.
# ---------------------------------------------------------------------------


def _read_manifest(out: Path) -> dict[str, Any]:
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def _executed_steps(out: Path) -> list[str]:
    try:
        manifest = _read_manifest(out)
    except (OSError, ValueError):
        return []
    return [
        str(step.get("name"))
        for step in manifest.get("steps", [])
        if step.get("exit_code") == 0
    ]


def _report_success(out: Path) -> None:
    manifest = _read_manifest(out)
    print()
    print(RULE)
    print("各阶段已完成，用时如下：")
    print(RULE)
    total = 0.0
    for step in manifest.get("steps", []):
        name = step.get("name", "?")
        seconds = step.get("seconds")
        if seconds is None:
            detail = "外部数据目录，复制并校验（无子进程）" if name == "data" else "（无计时）"
        else:
            total += float(seconds)
            detail = f"{seconds:8.2f} 秒"
        print(f"  {name:<9} exit={step.get('exit_code')}  {detail}")
    print(f"  {'合计':<9}          {total:.2f} 秒")


_STEP_HINTS: dict[str, tuple[str, ...]] = {
    "check": (
        "自检失败通常是环境或依赖问题：先单独跑 `uv run jev-course check` 复现，",
        "并确认 `uv sync --locked` 已经装好 torch / transformers / peft。",
    ),
    "data": (
        "数据复制/校验失败：用 `uv run jev-course validate-data <数据目录>` 单独检查；",
        "保留原目录，确认数据路径正确；需要重新准备数据时，请另选一个空目录。"
    ),
    "train": (
        "训练失败最常见的原因：基座权重下载不完整（网络/磁盘）、显存或内存不足（可试 --device cpu），",
        "或依赖缺失（`uv sync --locked`）。日志尾部一般直接写着原因。",
    ),
    "eval": ("评估失败通常是 checkpoint 与数据不匹配：确认前面的训练步骤真的产出了 model.pt。",),
    "fit": ("温度拟合失败通常是 checkpoint 加载失败或校准划分读不出来；日志尾部有具体报错。",),
    "test": ("请按日志检查 temperature.json、测试数据和模型加载情况。",),
    "predict": ("重载预测失败通常是 checkpoint 加载失败或选项不合法（重复/太少）。",),
}

_GENERIC_HINT: tuple[str, ...] = (
    "先看失败步骤的日志尾部；修好之后换一个新的 --out 重跑（课程不会覆盖非空目录）。",
)


def _report_failure(out: Path, data_dir: Path) -> int:
    print()
    print(RULE)
    print("流程已停止，以下是失败步骤和日志。")
    print(RULE)
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        print(f"没有找到 {manifest_path}：失败发生在 run-all 之前，没有任何子进程启动。")
        print(_retry_hint(out, data_dir))
        return EXIT_FAILED
    manifest = _read_manifest(out)
    print(f"status：{manifest.get('status')}")
    if manifest.get("failure"):
        print(f"失败原因：{manifest['failure']}")
    print("阶段表：")
    for step in manifest.get("steps", []):
        seconds = step.get("seconds")
        timing = f"{seconds:8.2f} 秒" if seconds is not None else "（无计时）"
        print(f"  {str(step.get('name')):<9} exit={step.get('exit_code')}  {timing}")
    failing = [s for s in manifest.get("steps", []) if s.get("exit_code") not in (None, 0)]
    if failing:
        step = failing[-1]
        name = str(step.get("name"))
        log = step.get("log")
        log_path = Path(str(log)) if log else None
        if log_path is not None and not log_path.is_absolute():
            log_path = REPO_ROOT / log_path
        print(f"失败步骤：{name}（exit={step.get('exit_code')}）；日志：{log_path}")
        tail = _log_tail(log_path)
        if tail:
            print("日志尾部：")
            print(tail)
        hint = _STEP_HINTS.get(name, _GENERIC_HINT)
    else:
        print("未记录到非零退出的子进程，请查看上面的失败原因；可能发生在数据检查或步骤之间。")
        hint = _GENERIC_HINT
    print("怎么继续：")
    for line in hint:
        print(f"  {line}")
    print(_retry_hint(out, data_dir))
    return EXIT_FAILED


def _log_tail(log_path: Path | None, lines: int = 12) -> str:
    if log_path is None or not log_path.exists():
        return ""
    try:
        tail = log_path.read_text(encoding="utf-8").splitlines()[-lines:]
    except OSError:
        return ""
    return "\n".join(f"    {line}" for line in tail if line.strip())


def _retry_hint(out: Path, data_dir: Path) -> str:
    import shlex
    retry = out.with_name(out.name + "-retry")
    args = ["uv", "run", "jev-course", "guide", "--out", str(retry)]
    if data_dir.is_dir():
        args += ["--data-directory", str(data_dir)]
    return "  用新目录重新开始（现有数据仍会校验）：\n  " + shlex.join(args)


def _report_cancel(session: Session, out: Path, data_dir: Path) -> int:
    print()
    print(RULE)
    if session.stopped == "q":
        print("已停止：你在提示处输入了 q。")
    else:
        print("已停止：标准输入结束了，没有收到继续的许可。")
    print(RULE)
    print(f"停止位置：第 {session.lesson} 课（「{session.pending}」这一步没有执行）。")
    executed = _executed_steps(out)
    print("已执行的阶段：" + (", ".join(executed) if executed else "无"))
    print(f"数据目录：{data_dir}（保留着，可用 --data-directory {data_dir} 复用，不会重新下载）")
    if (out / "manifest.json").exists():
        print(
            f"{out} 里已经有运行记录（这次停止在 manifest 里记为 failed + GuideCancelled）；"
            "再次运行请换一个 --out。"
        )
        print(_retry_hint(out, data_dir))
    else:
        print(f"{out} 还没有被写入任何内容。")
    return EXIT_DONE if session.stopped == "q" else EXIT_NO_INPUT


# ---------------------------------------------------------------------------
# Lesson 9: reload the checkpoint, show real mistakes, ask your own question.
# ---------------------------------------------------------------------------


def _final_stage(session: Session, out: Path, device: str) -> int:
    # Imported here so the interactive start stays light and the heavy deps are
    # only needed once the model actually has to be loaded again.
    from .calibration import load_scorer, read_temperature, score_rows

    model_path = out / "model.pt"
    temperature_path = out / "temperature.json"
    test_path = out / "data" / "test.jsonl"

    session.header("重载：先看模型答错的题，再问自己的问题")
    print("这一步在课程进程里重新加载磁盘上的产物（没有训练、没有梯度）：")
    print(f"  checkpoint：{model_path}")
    print(f"  温度：{temperature_path}")
    print(f"  留出集：{test_path}")
    try:
        model, collator, _config, device_torch = load_scorer(model_path, device)
        temperature = read_temperature(temperature_path)
    except Exception as error:
        print(f"重载失败：{type(error).__name__}: {error}")
        print("已完成步骤的记录仍保存在运行目录。请根据错误信息检查依赖或文件，不必立即重新训练。")
        return EXIT_FAILED

    trainable = [p for p in model.parameters() if p.requires_grad]
    parameter_count = sum(p.numel() for p in trainable)
    print()
    print(f"可训练参数：{parameter_count:,} 个（只有适配器和评分头；基座权重不在这里面）")
    print(f"温度：{temperature:.6f}（温度只缩放概率，不改变 argmax，所以下面的对错与校准无关）")

    try:
        rows = load_examples(test_path)
        scored = score_rows(model, collator, test_path, device_torch, batch_size=EVAL_BATCH_SIZE)
        _print_mistakes(rows, scored, temperature)
    except Exception as error:
        print(f"错题展示失败（不影响已有产物）：{type(error).__name__}: {error}")

    _ask_loop(session, model, collator, temperature, device_torch)
    _closing(out)
    return EXIT_DONE


def _print_mistakes(rows: list[Any], scored: list[Any], temperature: float) -> None:
    import torch

    wrong = [
        (row, entry, int(entry.logits.argmax()))
        for row, entry in zip(rows, scored)
        if int(entry.logits.argmax()) != row.label
    ]
    shown = min(MISTAKES_SHOWN, len(wrong))
    print()
    print(f"留出集 {len(rows)} 题，重载后的模型答错 {len(wrong)} 题；先看前 {shown} 条：")
    for number, (row, entry, chosen) in enumerate(wrong[:shown], start=1):
        probabilities = torch.softmax(entry.logits / temperature, dim=-1)
        print(f"  {number}) 题干：{_clip(row.context, 96)}")
        for index, option in enumerate(row.options):
            marks = "  ← 模型选择" if index == chosen else ("  ← 正确答案" if index == row.label else "")
            print(f"        {float(probabilities[index]) * 100:5.1f}%  {option}{marks}")
        print(
            f"     模型选了「{row.options[chosen]}」（{float(probabilities[chosen]) * 100:.1f}%），"
            f"正确答案是「{row.options[row.label]}」（{float(probabilities[row.label]) * 100:.1f}%）"
        )


def _ask_loop(session: Session, model: Any, collator: Any, temperature: float, device: Any) -> None:
    from .calibration import permuted_choice_report, predict_report

    print()
    print("现在轮到你自己出题（模型只给「这几个候选里它认为最像的一个」，没有标准答案对照）：")
    print(f"  · 题干一行，最多保留 {CONTEXT_TOKENS} 个 token；token 是分词单位，不等同于汉字数")
    print(f"  · 候选答案每行一个，至少两个；选项超过 {OPTION_TOKENS} 个 token 也会被截断")
    print("  · 题目的空行 = 结束课程")
    while True:
        context = session.ask(PROMPT_QUESTION)
        if not context:
            break
        options: list[str] = []
        while True:
            answer = session.ask(PROMPT_OPTION)
            if not answer:
                break
            options.append(answer)
        if len(options) < 2:
            print(f"  只给了 {len(options)} 个候选答案，这条跳过；至少要两个不同的候选。")
            continue
        try:
            report = predict_report(model, collator, context, tuple(options), temperature, device)
        except Exception as error:  # a bad row or a backend limit must not end the lesson
            print(f"  这条不能打分：{_error_text(error)}")
            continue
        _print_prediction(report)
        try:
            check = permuted_choice_report(
                model, collator, context, tuple(options), temperature, device, report
            )
        except Exception as error:
            print(f"  顺序检查未能完成：{_error_text(error)}")
        else:
            print(
                f"  顺序检查：倒序后仍然是「{check['choice']}」"
                f"（最大概率漂移 {check['max_probability_drift']:.2e}）"
            )


def _error_text(error: BaseException) -> str:
    """Name the failure and keep going; a bad row must not end the lesson."""
    return f"{type(error).__name__}: {error}"


def _print_prediction(report: dict[str, Any]) -> None:
    ranked = sorted(report["probabilities"].items(), key=lambda item: item[1], reverse=True)
    print(f"  模型选择：{report['choice']}")
    for option, probability in ranked:
        print(f"    {probability * 100:5.1f}%  {option}")
    print(f"  概率和：{report['probability_sum']:.6f}（没有标准答案，这只是「几个候选之间的分布」）")


def _closing(out: Path) -> None:
    print()
    print(RULE)
    print("课程结束。")
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = _read_manifest(out)
        print(f"manifest：{manifest_path}（status={manifest.get('status')}）")
        for entry in manifest.get("artifacts", []):
            print(f"  {str(entry.get('label')):<18} {entry.get('path')}  {entry.get('bytes')} 字节")
        evaluation_path = out / "evaluation.json"
        if evaluation_path.exists():
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            before = evaluation["test_before"]
            after = evaluation["test_after"]
            print(
                f"留出集：准确率 {before['accuracy'] * 100:.2f}% → {after['accuracy'] * 100:.2f}%，"
                f"NLL {before['nll']:.4f} → {after['nll']:.4f}，"
                f"ECE {before['ece_10bins']:.4f} → {after['ece_10bins']:.4f}"
            )
    print(
        "这些指标只描述本次数据上的表现，不能直接与官方 Jev 或其他题库的成绩比较。"
        "开发配方使用英文 CommonsenseQA，相关测试题已在开发中反复查看；换任务后需要重新评估。"
    )
    print(
        f"之后想再提问而不重训：uv run jev-course predict {out / 'model.pt'} "
        f"--temperature {out / 'temperature.json'} --context … --option …"
    )


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def run_guide(out: Path, device: str = "mps", data_directory: Path | None = None) -> int:
    """Run the interactive Chinese course; see the module docstring for the contract."""
    out = _absolute(out)
    data_dir = _absolute(data_directory) if data_directory is not None else _default_data_dir(out)
    session = Session(out)
    try:
        _refuse_paths(out, data_dir)
        _require_device(device)
        _lesson_intro(session, out, device, data_dir, data_directory is not None)
        report = _stage_data(session, out, data_dir, data_directory is not None)
        before_step, on_step = _callbacks(session, out, dict(report["sizes"]), device)
        code = run_all(
            out,
            device=device,
            sizes=dict(report["sizes"]),
            # Real data is not generated from a seed; the external-copy report in
            # the manifest records "seed": null. data_directory is always set here,
            # so this value is never turned into a --seed argument.
            data_seed=None,
            train_seed=TRAIN_SEED,
            width=WIDTH,
            rank=RANK,
            context_tokens=CONTEXT_TOKENS,
            option_tokens=OPTION_TOKENS,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            predict_context=None,
            predict_options=None,
            encoder="lora",
            hf_model=BASE_MODEL,
            hf_revision=BASE_REVISION,
            data_directory=data_dir,
            eval_batch_size=EVAL_BATCH_SIZE,
            before_step=before_step,
            on_step=on_step,
        )
        if session.stopped is not None:
            return _report_cancel(session, out, data_dir)
        if code != 0:
            return _report_failure(out, data_dir)
        _report_success(out)
        return _final_stage(session, out, device)
    except GuideCancelled:
        return _report_cancel(session, out, data_dir)
    except GuideRefused as refusal:
        print(f"{refusal}")
        return EXIT_REFUSED
    except KeyboardInterrupt:
        print("已中断（KeyboardInterrupt）。已经写下的产物不会被删除。")
        return 130
    except Exception as error:
        print(f"意外错误：{type(error).__name__}: {error}")
        return _report_failure(out, data_dir)
