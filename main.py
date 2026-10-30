"""笔记无损重构工具 —— 5.5 人性化输出。

流程：
    1. 读取原文
    2. 提取图片 / 代码块 / 行内代码，替换成占位符
    3. 调 DeepSeek API（AI 只看到占位符）
    4. 检查占位符是否丢失
    5. 回填占位符，得到最终文本
    6. 两类校验（丢内容的硬性失败 / 只提示不拦截的贴心提示）
    7. 按结果分流保存到 notes-organized/ 或 notes-organized/_failed/
       文件名优先使用整理后的 H1 标题（自动去重，绝不覆盖；提取不到时回退原名）

用法:
    python main.py --file "data2.md"
    python main.py --dir "notes/"
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# Windows 控制台默认 GBK，强制 UTF-8 输出避免中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system-prompt.md"
GLOSSARY_PATH = PROJECT_ROOT / "glossary.md"
OUTPUT_DIR = PROJECT_ROOT / "notes-organized"
FAILED_DIR = OUTPUT_DIR / "_failed"
# 源文件 → 上次输出文件名 的映射，用于“同源覆盖、异源加后缀”
HISTORY_PATH = OUTPUT_DIR / ".history.json"

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
REQUEST_TIMEOUT = 300  # 秒，长笔记需要较长时间

# AI 漏掉 H1 时的督促补写提示（追加到系统提示词之后再调用一次）
H1_RETRY_PROMPT = (
    "你刚才的回复中遗漏了一级标题（#）。请阅读刚才的内容，"
    "为其提炼一个高度概括的标题，并重新输出完整内容。"
)
# H1 缺失时最多重试的次数（不含首次调用）
MAX_H1_RETRIES = 2

# ---------- 正则 ----------
# 代码块：``` 开头到 ``` 结尾（含语言标记），非贪婪匹配
CODE_BLOCK_PATTERN = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)
# 图片：![alt](path)，path 里允许出现 Windows 反斜杠和空格
IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# 占位符：[[IMG_001]] / [[CODE_001]] / [[INLINE_001]]
PLACEHOLDER_PATTERN = re.compile(r"\[\[(?:IMG|CODE|INLINE)_\d{3,}\]\]")
# 行内代码：`xxx`（不含反引号本身）
INLINE_CODE_PATTERN = re.compile(r"`[^`\n]+`")
# 引用块：> 开头
QUOTE_PATTERN = re.compile(r"^>.*$", re.MULTILINE)
# 数字 + 单位：112058 张、0.5、32GB、3.5%、100ms
# 只保护“真正的数据”：带度量单位、带小数、或两位以上的数字；
# 序号类量词（步/个/条/次/行…）不作为硬指标，避免 AI 重排步骤时误判。
NUMBER_UNIT_PATTERN = re.compile(
    r"\d+(?:\.\d+)?[ \t]*"
    r"(?:%|GB|MB|KB|TB|ms|s|min|h|px|fps|bit|byte|B|K|M|G|T"
    r"|张|帧|秒|分钟|小时|天|周|月|年|万|亿|倍|维|dB|Hz|GHz|MHz)"
    r"|\d+\.\d+"
    r"|\d{2,}"
)
# 全大写缩写：API、GPU、JSON（至少 2 个字母）
UPPERCASE_ABBR_PATTERN = re.compile(r"\b[A-Z]{2,}\b")
# 参数名：--force、-v、batch_size、learning_rate
PARAM_PATTERN = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*|[a-z][a-z0-9]*(?:_[a-z0-9]+)+)(?![\w-])")
# H1 标题
H1_PATTERN = re.compile(r"^# .+$", re.MULTILINE)
# Windows 文件名非法字符
ILLEGAL_FILENAME_PATTERN = re.compile(r'[\\/:*?"<>|]')
# 控制字符（换行、制表符等）
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
# Windows 保留设备名，不能直接作为文件名
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}
# 文件名最大长度（字符数，不含 .md 和 _1 后缀）
MAX_FILENAME_LENGTH = 50
# 任意标题
HEADING_PATTERN = re.compile(r"^(#{1,6}) .+$", re.MULTILINE)
# 列表项
LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+\S", re.MULTILINE)

# 人称残留检测（排除术语、代码、引用后）
PRONOUN_PATTERNS = [
    re.compile(r"我认为"),
    re.compile(r"我觉得"),
    re.compile(r"我的理解"),
    re.compile(r"在我看来"),
    re.compile(r"个人认为"),
    re.compile(r"我建议"),
    re.compile(r"我推荐"),
    re.compile(r"我们来看"),
    re.compile(r"我们逐字"),
    re.compile(r"让我们"),
    re.compile(r"你提到"),
    re.compile(r"你可以看到"),
    re.compile(r"你说"),
    re.compile(r"你问"),
    re.compile(r"希望这对你有帮助"),
]
# 术语内允许出现的“我/你/我们”，检测前先剔除
TERM_WHITELIST = ["自我注意力", "自我监督", "自我回归", "忘我", "无我", "本我", "超我"]


# ---------- 配置 ----------
def load_config() -> tuple[str, str, str]:
    """从 .env 读取配置，返回 (base_url, model, api_key)。"""
    load_dotenv(PROJECT_ROOT / ".env")

    base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()

    if not api_key or api_key == "your_api_key_here":
        raise SystemExit(
            "未找到有效的 DEEPSEEK_API_KEY。\n"
            "请复制 .env.example 为 .env，并填入你的真实 API Key。"
        )

    return base_url, model, api_key


def load_glossary() -> list[str]:
    """从 glossary.md 读取术语，返回词条列表。"""
    if not GLOSSARY_PATH.exists():
        return []
    terms = []
    for line in read_text(GLOSSARY_PATH).splitlines():
        line = line.strip()
        if line.startswith("- "):
            term = line[2:].strip()
            if term:
                terms.append(term)
    return terms


def read_text(path: Path) -> str:
    """读取文本文件，优先 UTF-8，失败回退 GBK。"""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk")


# ---------- 占位符保护 ----------
def protect(text: str) -> tuple[str, dict[str, str]]:
    """把代码块、行内代码、图片替换成占位符。

    顺序：代码块 → 行内代码 → 图片。
    返回 (替换后的文本, {占位符: 原始内容})。
    """
    store: dict[str, str] = {}

    def make_replacer(prefix: str):
        counter = {"n": 0}

        def replacer(match: re.Match) -> str:
            counter["n"] += 1
            key = f"[[{prefix}_{counter['n']:03d}]]"
            store[key] = match.group(0)
            return key

        return replacer

    text = CODE_BLOCK_PATTERN.sub(make_replacer("CODE"), text)
    text = INLINE_CODE_PATTERN.sub(make_replacer("INLINE"), text)
    text = IMAGE_PATTERN.sub(make_replacer("IMG"), text)
    return text, store


def check_placeholders(text: str, store: dict[str, str]) -> list[str]:
    """检查返回文本里哪些占位符丢了。"""
    return [key for key in store if key not in text]


def restore(text: str, store: dict[str, str]) -> str:
    """把占位符替换回原始内容。"""
    for key, value in store.items():
        text = text.replace(key, value)
    return text


# ---------- 关键词提取 ----------
def normalize(text: str) -> str:
    """归一化：小写 + 去掉所有空白，用于宽松比较。"""
    return re.sub(r"\s+", "", text).lower()


def extract_inline_code_contents(text: str) -> set[str]:
    """提取行内代码内容（去掉反引号）。"""
    return {m.group(0)[1:-1].strip() for m in INLINE_CODE_PATTERN.finditer(text)}


def extract_numbers_with_units(text: str) -> set[str]:
    """提取数字 + 单位。"""
    result = set()
    for m in NUMBER_UNIT_PATTERN.finditer(text):
        token = m.group(0).strip()
        if token:
            result.add(token)
    return result


def extract_uppercase_abbr(text: str) -> set[str]:
    """提取全大写缩写。"""
    return set(UPPERCASE_ABBR_PATTERN.findall(text))


def extract_params(text: str) -> set[str]:
    """提取参数名（--force、batch_size 等）。"""
    return set(PARAM_PATTERN.findall(text))


def extract_keywords(text: str, glossary: list[str]) -> set[str]:
    """提取关键词：行内代码 + 数字单位 + 全大写缩写 + 术语表词。"""
    keywords = set()
    keywords |= extract_inline_code_contents(text)
    keywords |= extract_numbers_with_units(text)
    keywords |= extract_uppercase_abbr(text)
    # 术语表词：只保留原文中真实出现的
    norm_text = normalize(text)
    for term in glossary:
        if normalize(term) in norm_text:
            keywords.add(term)
    return {k for k in keywords if k.strip()}


def coverage(keywords: set[str], output_text: str) -> tuple[float, list[str]]:
    """计算关键词覆盖率，返回 (覆盖率, 缺失列表)。"""
    if not keywords:
        return 1.0, []
    norm_out = normalize(output_text)
    missing = [k for k in keywords if normalize(k) not in norm_out]
    hit = len(keywords) - len(missing)
    return hit / len(keywords), missing


# ---------- 文本统计 ----------
def strip_code_and_quotes(text: str) -> str:
    """去掉代码块、行内代码、引用块，用于人称检测。"""
    text = CODE_BLOCK_PATTERN.sub("", text)
    text = INLINE_CODE_PATTERN.sub("", text)
    text = QUOTE_PATTERN.sub("", text)
    return text


def count_paragraphs(text: str) -> int:
    """统计段落数：非空、非标题、非列表项、非代码块的连续文本块。"""
    text = CODE_BLOCK_PATTERN.sub("", text)
    paragraphs = 0
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            in_block = False
            continue
        if stripped.startswith("#"):
            in_block = False
            continue
        if LIST_ITEM_PATTERN.match(line):
            in_block = False
            continue
        if not in_block:
            paragraphs += 1
            in_block = True
    return paragraphs


def count_list_items(text: str) -> int:
    """统计列表项数量。"""
    text = CODE_BLOCK_PATTERN.sub("", text)
    return len(LIST_ITEM_PATTERN.findall(text))


def count_headings(text: str) -> dict[str, int]:
    """统计各级标题数量。"""
    counts = {"h1": 0, "h2": 0, "h3": 0, "h4": 0, "h5": 0, "h6": 0}
    for m in HEADING_PATTERN.finditer(text):
        level = len(m.group(1))
        counts[f"h{level}"] += 1
    return counts


def find_pronoun_residue(text: str) -> list[str]:
    """检测人称残留，返回命中的模式列表。"""
    cleaned = strip_code_and_quotes(text)
    for term in TERM_WHITELIST:
        cleaned = cleaned.replace(term, "")
    hits = []
    for pattern in PRONOUN_PATTERNS:
        if pattern.search(cleaned):
            hits.append(pattern.pattern)
    return hits


# ---------- 校验 ----------
def validate(original: str, final: str, glossary: list[str]) -> dict:
    """执行校验，返回结果字典。

    hard_failures：AI 真丢了内容，进 _failed。
    tips：AI 的表达取舍（覆盖率、字数、列表化等），只用大白话提醒，绝不拦截。
    records：纯统计数据，供排查用。
    """
    result = {
        "hard_failures": [],
        "tips": [],
        "records": {},
    }

    # ===== 第一类：硬性失败（只管“丢了”，不管“多了”） =====
    orig_img = len(IMAGE_PATTERN.findall(original))
    out_img = len(IMAGE_PATTERN.findall(final))
    if out_img < orig_img:
        result["hard_failures"].append(f"图片数量变少：原文 {orig_img} → 输出 {out_img}")

    orig_code = len(CODE_BLOCK_PATTERN.findall(original))
    out_code = len(CODE_BLOCK_PATTERN.findall(final))
    if out_code < orig_code:
        result["hard_failures"].append(f"代码块数量变少：原文 {orig_code} → 输出 {out_code}")

    # 行内代码内容
    orig_inline = extract_inline_code_contents(original)
    norm_final = normalize(final)
    missing_inline = [c for c in orig_inline if normalize(c) not in norm_final]
    if missing_inline:
        preview = "、".join(f"`{c}`" for c in missing_inline[:5])
        more = f" 等 {len(missing_inline)} 个" if len(missing_inline) > 5 else ""
        result["hard_failures"].append(f"行内代码内容丢失：{preview}{more}")

    # 数字 + 单位
    orig_nums = extract_numbers_with_units(original)
    missing_nums = [n for n in orig_nums if normalize(n) not in norm_final]
    if missing_nums:
        preview = "、".join(missing_nums[:5])
        more = f" 等 {len(missing_nums)} 个" if len(missing_nums) > 5 else ""
        result["hard_failures"].append(f"数字+单位丢失：{preview}{more}")

    # 参数名
    orig_params = extract_params(original)
    missing_params = [p for p in orig_params if normalize(p) not in norm_final]
    if missing_params:
        preview = "、".join(missing_params[:5])
        more = f" 等 {len(missing_params)} 个" if len(missing_params) > 5 else ""
        result["hard_failures"].append(f"参数名丢失：{preview}{more}")

    # H1：缺失 H1 在调用 API 阶段就会重试，走到这里说明重试也没救回来
    orig_h1 = len(H1_PATTERN.findall(original))
    out_h1 = len(H1_PATTERN.findall(final))
    if out_h1 == 0:
        result["hard_failures"].append("H1 标题缺失：整理结果里没有一级标题")
    elif out_h1 != orig_h1 and not (orig_h1 == 0 and out_h1 == 1):
        if out_h1 > orig_h1:
            detail = "AI 多写了一级标题（#），大纲层级比原文多"
        else:
            detail = "AI 把正文里的一级标题降成了二级标题（##）"
        result["tips"].append(
            f"{detail}（一级标题 {orig_h1} 个 → {out_h1} 个）。"
            "内容都在，介意的话手动改一下 # 的数量即可。"
        )

    # ===== 软指标：一律翻译成大白话，只提醒、不拦截 =====
    keywords = extract_keywords(original, glossary)
    cov, missing_kw = coverage(keywords, final)
    if cov < 0.80:
        result["tips"].append(
            "AI 在整理时精简了部分词汇，可能有知识点或术语没保留"
            f"（关键词覆盖率 {cov:.0%}）。请您大致扫一眼，看看有没有漏掉关键知识点。"
        )
    elif cov < 0.90:
        result["tips"].append(
            "少数术语的写法可能和原文略有差异"
            f"（关键词覆盖率 {cov:.0%}）。一般不影响阅读，介意的话可以对照原文看一眼。"
        )

    # 专有名词缺失（全大写缩写 + 术语表）
    proper_nouns = extract_uppercase_abbr(original)
    norm_orig = normalize(original)
    for term in glossary:
        if normalize(term) in norm_orig:
            proper_nouns.add(term)
    missing_nouns = [n for n in proper_nouns if normalize(n) not in norm_final]
    if missing_nouns:
        preview = "、".join(missing_nouns[:5])
        more = f" 等 {len(missing_nouns)} 个" if len(missing_nouns) > 5 else ""
        result["tips"].append(
            f"有术语或英文缩写可能被换成了同义说法：{preview}{more}。"
            "如果这些词对您很重要，可以在 glossary.md 里登记，下次整理会强制保留。"
        )

    # 总字数
    ratio = len(final) / len(original) if original else 1.0
    if ratio < 0.70:
        result["tips"].append(
            "整理后的字数比原文少了一些"
            f"（约为原文的 {ratio:.0%}）。这通常是 AI 删掉了废话，"
            "但也请留意有没有丢失重要内容。"
        )

    # 人称残留
    pronoun_hits = find_pronoun_residue(final)
    if pronoun_hits:
        result["tips"].append(
            f"文中还留着“我/我们”这类口语口气（{len(pronoun_hits)} 处："
            f"{'、'.join(pronoun_hits[:5])}）。不影响内容，介意的话手动改成陈述句即可。"
        )

    # 列表化程度
    out_paras = count_paragraphs(final)
    out_lists = count_list_items(final)
    list_ratio = out_lists / out_paras if out_paras else 0.0
    if list_ratio > 2.0:
        result["tips"].append(
            "AI 把一些长段落拆分成了列表"
            f"（列表 {out_lists} 项 / 段落 {out_paras} 段）。"
            "阅读起来可能更清晰，也可能破坏了原本的段落连贯性，请确认一下。"
        )

    # ===== 第三类：只记录 =====
    orig_paras = count_paragraphs(original)
    orig_headings = count_headings(original)
    out_headings = count_headings(final)
    result["records"] = {
        "段落数": f"{orig_paras} → {out_paras}",
        "H1": f"{orig_headings['h1']} → {out_headings['h1']}",
        "H2": f"{orig_headings['h2']} → {out_headings['h2']}",
        "H3": f"{orig_headings['h3']} → {out_headings['h3']}",
        "H4": f"{orig_headings['h4']} → {out_headings['h4']}",
        "列表项": f"{count_list_items(original)} → {out_lists}",
        "字数": f"{len(original)} → {len(final)}（{ratio:.1%}）",
        "关键词覆盖率": f"{cov:.1%}（{len(keywords) - len(missing_kw)}/{len(keywords)}）",
    }

    return result


# ---------- 输出 ----------
# 把校验术语翻译成大白话，只在终端展示，不参与判定
FAILURE_HINTS = [
    ("图片数量", "有图片没被保留下来。"),
    ("代码块数量", "有代码块被合并或删掉了。"),
    ("行内代码内容", "反引号里的函数名、变量名被改了或删了。"),
    ("数字+单位", "数字或单位被改了，比如 100ms 变成 100。"),
    ("参数名", "命令行参数或变量名被改了，跑代码会报错。"),
    ("H1 标题缺失", "AI 反复遗漏一级标题，输出不可用，请重跑这一篇。"),
]


def explain_items(items: list[str], hints: list[tuple[str, str]]) -> list[str]:
    """按关键词把校验项翻译成大白话，去重后返回。"""
    details: list[str] = []
    for keyword, text in hints:
        if any(keyword in item for item in items) and text not in details:
            details.append(text)
    return details


def print_plain_hint(title: str, lines: list[str], details: list[str]) -> None:
    """打印提示段落，格式统一。"""
    print()
    print(title)
    for line in lines:
        print(f"   {line}")
    if details:
        print("   本次具体是：")
        for item in details:
            print(f"     · {item}")


def print_failure_hint(failures: list[str]) -> None:
    """解释“失败”代表什么、下一步怎么处理。"""
    print_plain_hint(
        "❌ 说明：这篇笔记在整理时丢了原有的内容（图片 / 代码 / 数字 / 参数名之一）。",
        [
            "为了避免您拿到残缺的笔记，它被放进了 notes-organized/_failed/，",
            "原文件没有被改动；您可以重跑这一篇，或先对比一下再手动挪回 notes-organized/。",
        ],
        explain_items(failures, FAILURE_HINTS),
    )


def print_report(result: dict, elapsed: float) -> None:
    """打印校验报告。"""
    print()
    print("=" * 60)
    print("校验报告")
    print("=" * 60)

    if result["hard_failures"]:
        print()
        print("【内容丢失】")
        for item in result["hard_failures"]:
            print(f"  ✗ {item}")

    if result["tips"]:
        print()
        print("【贴心提示（不影响使用，笔记已正常生成）】")
        for item in result["tips"]:
            print(f"  💡 {item}")

    if not result["hard_failures"] and not result["tips"]:
        print()
        print("  ✓ 全部检查通过，文字、图片、代码都没有丢失")

    if result["hard_failures"]:
        print_failure_hint(result["hard_failures"])

    print()
    print("【记录项】")
    for key, value in result["records"].items():
        print(f"  · {key}：{value}")
    print(f"  · 处理耗时：{elapsed:.1f} 秒")
    print("=" * 60)


def print_batch_summary(counts: dict[str, int], total: int) -> None:
    """批量处理收尾：数量汇总 + 常见疑问说明。"""
    print(
        f"\n本次共处理 {total} 篇，"
        f"✅成功 {counts['passed']} 篇，"
        f"❌失败 {counts['failed']} 篇。"
    )
    print()
    print("=" * 60)
    print("🎉 处理完成！您可以打开 notes-organized 文件夹查看成果了。")
    print("   - 出现的“💡 贴心提示”只是 AI 在表达上做了取舍，笔记可以直接使用。")
    if counts["failed"]:
        print("   - 失败的文件在 notes-organized/_failed/ 里，原文件没有被改动，重跑即可。")
    else:
        print("   - 这次没有失败的文件，全部都有可用的整理结果。")
    print("   祝您使用愉快！")
    print("=" * 60)


def extract_h1_title(text: str) -> Optional[str]:
    """从整理结果中提取第一个 H1 标题，去掉 # 标记和两端空格。"""
    match = H1_PATTERN.search(text)
    if not match:
        return None
    title = match.group(0).strip().lstrip("#").strip()
    # 兼容 "# 标题 ###" 这种 ATX 收尾写法，同时避免把 "C# 入门" 的 # 删掉
    title = re.sub(r"\s+#+\s*$", "", title).strip()
    return title or None


def replace_first_h1(text: str, title: str) -> str:
    """把第一个 H1 的标题替换成 title，其余内容保持不变。"""
    match = H1_PATTERN.search(text)
    if not match:
        return text
    return text[: match.start()] + f"# {title}" + text[match.end():]


def sanitize_filename(title: str) -> str:
    """把标题转换成 Windows 安全的文件名主体（不含扩展名）。

    非法字符替换成下划线，去掉控制字符，截断到 50 个字符。
    返回空字符串表示标题无法用于文件名。
    """
    name = ILLEGAL_FILENAME_PATTERN.sub("_", title)
    name = CONTROL_CHAR_PATTERN.sub("", name)
    name = re.sub(r"\s+", " ", name).strip()
    if name.lower().endswith(".md"):
        name = name[:-3].strip()
    name = name.rstrip(" .")  # Windows 不允许文件名以空格或点结尾
    name = name[:MAX_FILENAME_LENGTH].rstrip(" .")
    if not name:
        return ""
    if name.upper() in WINDOWS_RESERVED_NAMES:
        name = f"_{name}"
    return name


def load_history() -> dict[str, dict[str, str]]:
    """读取 .history.json（源文件名 → {output, sha256}），损坏时静默当作空映射。"""
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    history: dict[str, dict[str, str]] = {}
    for key, value in data.items():
        entry = normalize_history_entry(value)
        if entry is not None:
            history[str(key)] = entry
    return history


def normalize_history_entry(value: object) -> Optional[dict[str, str]]:
    """兼容旧格式（纯字符串）与新格式（{"output": ..., "sha256": ...}）。"""
    if isinstance(value, str):
        return {"output": value, "sha256": ""} if value else None
    if isinstance(value, dict):
        output = value.get("output")
        if not isinstance(output, str) or not output:
            return None
        digest = value.get("sha256")
        return {"output": output, "sha256": digest if isinstance(digest, str) else ""}
    return None


def save_history(history: dict[str, dict[str, str]]) -> None:
    """原子写回 .history.json，任何写入异常都不影响主流程。"""
    temp_path = HISTORY_PATH.with_name(HISTORY_PATH.name + ".tmp")
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(HISTORY_PATH)
    except OSError:
        pass


def remember_output(
    history: dict[str, dict[str, str]],
    source_name: str,
    out_path: Path,
    source_digest: str = "",
) -> None:
    """记录“源文件 → 输出文件 + 源文件内容哈希”（相对 notes-organized 的路径）。"""
    try:
        relative = out_path.relative_to(OUTPUT_DIR)
    except ValueError:
        relative = Path(out_path.name)
    history[source_name] = {
        "output": relative.as_posix(),
        "sha256": source_digest,
    }
    save_history(history)


def overwrite_file(path: Path, content: str) -> bool:
    """直接覆盖写入，失败（被占用等）返回 False。"""
    try:
        path.write_text(content, encoding="utf-8")
        return True
    except OSError:
        return False


def locked_stem_for(note_path: Path, source_digest: str) -> str:
    """源文件内容与上次完全一致时，返回上次的输出文件名主体（用于稳定文件名）。"""
    if not source_digest:
        return ""
    previous = load_history().get(note_path.name) or {}
    if previous.get("sha256") != source_digest:
        return ""
    return sanitize_filename(Path(previous.get("output", "")).stem)


def save_output(
    note_path: Path,
    content: str,
    failed: bool,
    source_digest: str = "",
) -> tuple[Path, bool]:
    """按分流结果保存，返回 (输出路径, 是否沿用了上次的文件名)。

    文件名优先使用整理结果的 H1 标题；同一源文件重新运行时直接覆盖上次的输出，
    只有“不同源文件撞了同一个标题”才加 _1、_2 后缀。
    若源文件内容哈希与上次一致，则锁定上次的文件名，避免 AI 标题抖动导致文件名跳跃。
    """
    target_dir = FAILED_DIR if failed else OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    history = load_history()
    source_name = note_path.name
    previous = history.get(source_name) or {}
    previous_output = previous.get("output", "")

    # 源文件一字未改：沿用上次的输出文件名（内容仍然照常写入）
    locked_stem = locked_stem_for(note_path, source_digest)

    title = extract_h1_title(content)
    stem = sanitize_filename(title) if title else ""
    if locked_stem:
        stem = locked_stem
    if not stem:
        stem = sanitize_filename(note_path.stem) or "untitled"

    # 同源重复运行：覆盖上一次生成的同一个文件，避免堆积 _1、_2
    previous_path: Optional[Path] = None
    stale_path: Optional[Path] = None
    if previous_output:
        candidate = OUTPUT_DIR / previous_output
        if candidate.suffix.lower() == ".md":
            if candidate.parent == target_dir:
                previous_path = candidate
            else:
                stale_path = candidate  # 上次落在另一边（失败↔通过），成功写入后清掉

    if previous_path is not None:
        wanted = target_dir / f"{stem}.md"
        if wanted == previous_path:
            if overwrite_file(previous_path, content):
                remember_output(history, source_name, previous_path, source_digest)
                return previous_path, bool(locked_stem)
        elif not wanted.exists():
            # 标题变了：改到新文件名并清掉旧文件，避免留下两份
            if overwrite_file(wanted, content):
                try:
                    previous_path.unlink()
                except OSError:
                    pass
                remember_output(history, source_name, wanted, source_digest)
                return wanted, bool(locked_stem)
        elif overwrite_file(previous_path, content):
            # 新标题已被别的源文件占用，保持原文件名就地覆盖
            remember_output(history, source_name, previous_path, source_digest)
            return previous_path, bool(locked_stem)

    # 用 "x" 模式独占创建，避免检查与写入之间被其他进程抢占导致覆盖
    index = 0
    while True:
        name = f"{stem}.md" if index == 0 else f"{stem}_{index}.md"
        out_path = target_dir / name
        try:
            with out_path.open("x", encoding="utf-8") as handle:
                handle.write(content)
            if stale_path is not None:
                try:
                    stale_path.unlink()
                except OSError:
                    pass
            remember_output(history, source_name, out_path, source_digest)
            return out_path, bool(locked_stem)
        except (FileExistsError, IsADirectoryError):
            index += 1


# ---------- API ----------
def call_deepseek(base_url: str, model: str, api_key: str,
                  system_prompt: str, user_content: str) -> str:
    """调用 DeepSeek Chat Completions 接口，返回 AI 输出的文本。"""
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        # 笔记重构要求高稳定性：0 温度可避免同一篇未修改的笔记每次生成同义词标题，
        # 从而让输出文件名与 .history.json 映射保持稳定。
        "temperature": 0,
        "stream": False,
    }

    print(f"正在调用 API（模型：{model}）……")
    response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)

    if response.status_code != 200:
        raise RuntimeError(
            f"API 调用失败：HTTP {response.status_code}\n{response.text[:1000]}"
        )

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise RuntimeError(f"API 返回格式异常：\n{data}")


# ---------- 主流程 ----------
def build_user_prompt(content: str) -> str:
    """把笔记正文包在明确的分隔符里，避免极短笔记被模型当成提问。"""
    return (
        "请整理下面这篇笔记，直接输出整理后的完整 Markdown，不要反问、不要解释。\n\n"
        "<<<笔记开始>>>\n"
        f"{content}\n"
        "<<<笔记结束>>>"
    )


def request_with_h1_retry(
    base_url: str,
    model: str,
    api_key: str,
    system_prompt: str,
    content: str,
) -> tuple[str, bool]:
    """调用 API，结果缺 H1 时追加严厉提示重试，返回 (AI 文本, 是否重试过)。

    重试用尽仍没有 H1 时抛 RuntimeError，交由 process_note 的异常出口统一报错。
    """
    retried = False
    for attempt in range(MAX_H1_RETRIES + 1):
        prompt = system_prompt
        user_content = build_user_prompt(content)
        if attempt > 0:
            prompt = f"{system_prompt}\n\n【补充要求·最高优先级】{H1_RETRY_PROMPT}"
            user_content = f"{H1_RETRY_PROMPT}\n\n{user_content}"
        result = call_deepseek(base_url, model, api_key, prompt, user_content)
        if not result or not result.strip():
            raise RuntimeError("API 返回内容为空")
        if extract_h1_title(result) is not None:
            return result, retried
        if attempt < MAX_H1_RETRIES:
            retried = True
            print()
            print("⚠️ AI 遗漏了标题，正在督促其重新提炼…")
    raise RuntimeError("AI 未能生成标题（重试后仍缺少一级标题）")


def process_note(
    note_path: Path,
    base_url: str,
    model: str,
    api_key: str,
    system_prompt: str,
    glossary: list[str],
) -> str:
    """处理单篇笔记并打印结果，返回 passed 或 failed。"""
    try:
        original = read_text(note_path)
        protected, store = protect(original)
        img_count = sum(1 for k in store if k.startswith("[[IMG_"))
        code_count = sum(1 for k in store if k.startswith("[[CODE_"))
        inline_count = sum(1 for k in store if k.startswith("[[INLINE_"))

        print(f"\n输入文件：{note_path}")
        print(f"原文字数：{len(original)}")
        print(
            f"已保护：图片 {img_count} 个，代码块 {code_count} 个，"
            f"行内代码 {inline_count} 个"
        )
        print(f"术语表：{len(glossary)} 个词条")

        start = time.time()
        result, retried = request_with_h1_retry(
            base_url, model, api_key, system_prompt, protected
        )
        elapsed = time.time() - start

        missing = check_placeholders(result, store)
        if missing:
            print()
            print("!" * 60)
            print(f"❌ AI 返回内容丢失了 {len(missing)} 个占位符：")
            for key in missing:
                preview = store[key].splitlines()[0][:60]
                print(f"  {key}  ← {preview}")
            print("图片或代码的占位符被弄丢了，回填时这部分内容会缺失，本篇会进 _failed。")
            print("!" * 60)

        final = restore(result, store)
        if retried:
            print(f"✅ 已成功提炼标题：{extract_h1_title(final)}")

        # 源文件未改动时锁定上次的文件名，并把 H1 对齐成该文件名，避免两者错位
        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
        locked_stem = locked_stem_for(note_path, digest)
        if locked_stem:
            final = replace_first_h1(final, locked_stem)

        report = validate(original, final, glossary)
        print_report(report, elapsed)

        failed = bool(report["hard_failures"])
        out_path, locked_name = save_output(note_path, final, failed, digest)
        if locked_name:
            print("   （源文件内容未改动，沿用上次的文件名）")

        if failed:
            print(f"❌ 原文件 {note_path.name} 已整理并重命名为：{out_path.name}")
            print(f"   错误原因：{'；'.join(report['hard_failures'])}")
            print(f"   已保存至：{out_path}")
            return "failed"

        print(f"✅ 原文件 {note_path.name} 已整理并重命名为：{out_path.name}")
        if report["tips"]:
            print(f"   笔记已正常生成，可直接使用（有 {len(report['tips'])} 条贴心提示见上）")
        else:
            print("   文字、图片、代码全部完整保留，放心使用")
        print(f"   已保存至：{out_path}")
        return "passed"
    except Exception as error:
        print(f"❌ 处理失败：{note_path.name}；错误原因：{type(error).__name__}: {error}")
        print("   这篇没有生成结果，原文件没有被改动。")
        print("   多半是网络或接口临时问题，稍后单独重跑这一篇即可。")
        return "failed"


def main() -> int:
    parser = argparse.ArgumentParser(description="笔记无损重构工具（5.4 批量处理版）")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", help="要处理的 .md 文件路径")
    input_group.add_argument("--dir", help="要批量处理的文件夹路径")
    args = parser.parse_args()

    if not SYSTEM_PROMPT_PATH.exists():
        print(f"系统提示词文件不存在：{SYSTEM_PROMPT_PATH}")
        return 1

    if args.file:
        note_path = Path(args.file)
        if not note_path.exists():
            print(f"文件不存在：{note_path}")
            return 1
        if not note_path.is_file():
            print(f"不是文件：{note_path}")
            return 1
        note_paths = [note_path]
    else:
        directory = Path(args.dir)
        if not directory.exists():
            print(f"文件夹不存在：{directory}")
            return 1
        if not directory.is_dir():
            print(f"不是文件夹：{directory}")
            return 1
        note_paths = sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() == ".md"
            ),
            key=lambda path: path.name.lower(),
        )
        print(f"共发现 {len(note_paths)} 篇笔记，即将开始处理，请稍候……")
        if not note_paths:
            print_batch_summary({"passed": 0, "failed": 0}, 0)
            return 0

    try:
        base_url, model, api_key = load_config()
        glossary = load_glossary()
        system_prompt = read_text(SYSTEM_PROMPT_PATH)
    except (OSError, SystemExit) as error:
        print(f"初始化失败：{error}")
        return 1

    counts = {"passed": 0, "failed": 0}
    for note_path in note_paths:
        status = process_note(
            note_path, base_url, model, api_key, system_prompt, glossary
        )
        counts[status] += 1

    if args.dir:
        print_batch_summary(counts, len(note_paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
