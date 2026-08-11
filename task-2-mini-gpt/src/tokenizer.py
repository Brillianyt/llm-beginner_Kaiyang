"""M1: 字节级 BPE tokenizer,中文按字符预切分。

设计要点:
- 预分词正则(GPT-2 风格 + 单个 CJK 字符):保证中文不被跨字合并
- BPE 训练:对预分词结果去重,以"词→频次"为迭代单位,效率比扫全文高
- 序列化:JSON 保存 vocab(id→bytes 字符串)/ merges(id 对列表)/ pre_tokenize_pattern

关键约定:
- encode → decode 必须严格 round-trip(eval 自检硬要求)
- vocab[0..255] = 256 个 byte token;256..259 = 4 个 special tokens;
  260+ = 学到的 merge token
- merges 按训练顺序排列,encode 时按 merge 顺序应用,新 id = 260 + rank
"""
import json
from collections import Counter, defaultdict

# 优先使用 regex(支持 \p{L} / \p{N}),否则用 re + 显式 ASCII 范围
try:
    import regex as _re
except ImportError:  # pragma: no cover
    import re as _re


# GPT-2 风格预分词 + 单个 CJK 字符兜底
#
# 关键陷阱:`\p{L}` 在 `regex` 模块里会把 CJK 字符也算成 Letter (Lo),
# 所以直接写 ` ?\p{L}+` 会把"床前明月光"当成一个 word,
# BPE 只能靠凑巧学到"每 3 个字节 = 一个汉字"的合并模式 —— 表面上 round-trip 通过,
# 但 vocab 被 UTF-8 字节巧合污染,语料换就会崩。
#
# 修复:在 `\p{L}+` 里加 `(?!\\p{Han})` 提前看,排除 CJK。
# 注意 `regex` 这版的 set subtraction 语法 `[\p{L}--\p{Han}]` 不工作,
# 提前看是唯一可靠的写法。
#
# 优先级从左到右:CJK 单字 → 缩写 → 非 CJK 字母 → 数字 → 非字母数字空白 → 空白
DEFAULT_PRE_TOKENIZE_PATTERN = (
    r"""\p{Han}"""
    r"""|'s|'t|'re|'ve|'m|'ll|'d"""
    r"""| ?(?:(?!\p{Han})\p{L})+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class BPETokenizer:
    """字节级 BPE tokenizer。

    Attributes:
        vocab_size: 当前词表大小(256 byte + 4 special + 若干 merge)
    """

    # 特殊 token 的固定 id(在 256 个 byte token 之后)
    UNK_ID: int = 256
    BOS_ID: int = 257
    EOS_ID: int = 258
    PAD_ID: int = 259
    FIRST_MERGE_ID: int = 260  # 学到的 merge token 从这里开始编号

    def __init__(self):
        self.vocab: dict[int, bytes] = {}                # id → bytes
        self.merges: list[tuple[int, int]] = []         # 按训练顺序的 merge 列表
        self.pre_tokenize_pattern: str = DEFAULT_PRE_TOKENIZE_PATTERN
        self._pre_tokenize_re = _re.compile(self.pre_tokenize_pattern)
        self._bytes_to_id: dict[bytes, int] = {}        # 反向索引,decode 时不需要,encode 时按 merge 顺序应用

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def _init_vocab(self) -> None:
        """256 byte token + 4 special token。"""
        self.vocab = {b: bytes([b]) for b in range(256)}
        self.vocab[self.UNK_ID] = b"<unk>"
        self.vocab[self.BOS_ID] = b"<bos>"
        self.vocab[self.EOS_ID] = b"<eos>"
        self.vocab[self.PAD_ID] = b"<pad>"
        self._bytes_to_id = {v: k for k, v in self.vocab.items()}

    @staticmethod
    def _word_to_byte_ids(word: str) -> list[int]:
        """预分词后的单个词 → UTF-8 字节 id 列表。"""
        return list(word.encode("utf-8"))

    @staticmethod
    def _merge_pair_in_word(
        word: tuple[int, ...],
        pair: tuple[int, int],
        new_id: int,
    ) -> tuple[int, ...]:
        """把 word 中所有 (pair[0], pair[1]) 替换为 new_id(从左到右、不重叠)。"""
        out: list[int] = []
        i, n = 0, len(word)
        while i < n:
            if i < n - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    def train(self, text: str, vocab_size: int = 4096, verbose: bool = False) -> "BPETokenizer":
        """在线训练 BPE(增量更新版,O(merges × 受影响 word 数))。

        Args:
            text: 训练语料(整段字符串)
            vocab_size: 目标词表大小(含 256 byte + 4 special + 学到的 merge)
            verbose: 是否打印每 500 步的 merge 信息

        Returns:
            self(便于链式调用)
        """
        self._init_vocab()

        target_merges = vocab_size - self.FIRST_MERGE_ID
        if target_merges < 0:
            raise ValueError(
                f"vocab_size={vocab_size} 小于最小值 {self.FIRST_MERGE_ID}"
            )

        # 1. 预分词 + 转 byte id 列表 + 去重(保留出现频次)
        words = self._pre_tokenize_re.findall(text)
        word_seqs: list[list[int]] = []
        word_counts: list[int] = []
        seq_to_idx: dict[tuple[int, ...], int] = {}
        for w in words:
            ids = tuple(self._word_to_byte_ids(w))
            if ids in seq_to_idx:
                word_counts[seq_to_idx[ids]] += 1
            else:
                seq_to_idx[ids] = len(word_seqs)
                word_seqs.append(list(ids))
                word_counts.append(1)

        # 2. 初始化 pair_stats 与 pair_to_words(增量更新所需)
        #    pair_stats[p] = sum over words containing p of word_count
        #    pair_to_words[p] = set of word indices containing p
        pair_stats: dict[tuple[int, int], int] = defaultdict(int)
        pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)
        for w_idx, seq in enumerate(word_seqs):
            cnt = word_counts[w_idx]
            for i in range(len(seq) - 1):
                p = (seq[i], seq[i + 1])
                pair_stats[p] += cnt
                pair_to_words[p].add(w_idx)

        # 3. 迭代 merge(增量更新)
        for step in range(target_merges):
            if not pair_stats:
                break

            # 取频次最高的 pair(若 tie,选字典序最小的,保证可复现)
            best_pair = max(pair_stats, key=lambda p: (pair_stats[p], -p[0], -p[1]))
            best_count = pair_stats[best_pair]
            if best_count <= 0:
                break

            new_id = self.FIRST_MERGE_ID + len(self.merges)
            new_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_id] = new_bytes
            self._bytes_to_id[new_bytes] = new_id

            # 仅处理包含 best_pair 的 word(从 pair_to_words 取出后即丢弃)
            affected_words = pair_to_words.pop(best_pair, set())
            for w_idx in affected_words:
                old_seq = word_seqs[w_idx]
                cnt = word_counts[w_idx]

                # a) 减去 old_seq 中所有 pair 的贡献
                for i in range(len(old_seq) - 1):
                    p = (old_seq[i], old_seq[i + 1])
                    pair_stats[p] -= cnt
                    if pair_stats[p] <= 0:
                        pair_stats.pop(p, None)
                    pair_to_words[p].discard(w_idx)

                # b) 在 old_seq 上合并 best_pair
                new_seq: list[int] = []
                i, n = 0, len(old_seq)
                while i < n:
                    if (
                        i < n - 1
                        and old_seq[i] == best_pair[0]
                        and old_seq[i + 1] == best_pair[1]
                    ):
                        new_seq.append(new_id)
                        i += 2
                    else:
                        new_seq.append(old_seq[i])
                        i += 1

                # c) 加上 new_seq 中新 pair 的贡献
                for i in range(len(new_seq) - 1):
                    p = (new_seq[i], new_seq[i + 1])
                    pair_stats[p] += cnt
                    pair_to_words[p].add(w_idx)

                word_seqs[w_idx] = new_seq

            # best_pair 的统计已在上方清零,主动 pop 一次确保
            pair_stats.pop(best_pair, None)
            self.merges.append(best_pair)

            if verbose and (step + 1) % 500 == 0:
                print(
                    f"  merge {step + 1}/{target_merges}: "
                    f"{best_pair} -> {new_id} (freq={best_count})"
                )

        if verbose:
            print(f"训练完成:vocab_size={self.vocab_size},merges={len(self.merges)}")
        return self

    # ------------------------------------------------------------------
    # 编码(快速版本:用 merge_rank 单 pass)
    # ------------------------------------------------------------------
    def _build_merge_rank(self) -> None:
        """merge_rank[pair] = 该 pair 在 merges 中的训练顺序(数字越小越优先合并)。

        encode 时不再遍历所有 7932 个 merge,而是单 pass 找当前词中最低 rank 的 pair 合并。
        """
        self._merge_rank = {pair: i for i, pair in enumerate(self.merges)}

    def _encode_word(self, word: str) -> list[int]:
        """单 pass 编码:反复找当前序列中 rank 最小的可合并 pair,合并掉。"""
        ids = self._word_to_byte_ids(word)
        if len(ids) < 2:
            return ids

        # 懒初始化 merge_rank(loaded 模型第一次 encode 时构造)
        if not hasattr(self, "_merge_rank"):
            self._build_merge_rank()

        merge_rank = self._merge_rank
        n_merges = len(self.merges)
        first_id = self.FIRST_MERGE_ID

        # 反复:找当前 ids 中 rank 最小的 pair,合并,直到没有可合并的
        # 每次 pass 内部扫描所有相邻对 = O(L),外层最多 L/2 次 pass = O(L²)
        # 但 L 是字节数(短词通常 ≤ 6),所以实际很快
        while len(ids) >= 2:
            best_rank = n_merges + 1   # 大于任何真实 rank,表示"未找到"
            best_idx = -1
            for i in range(len(ids) - 1):
                r = merge_rank.get((ids[i], ids[i + 1]), n_merges + 1)
                if r < best_rank:
                    best_rank = r
                    best_idx = i
            if best_idx < 0:
                break
            # 合并 best_idx 处
            new_id = first_id + best_rank
            ids = ids[:best_idx] + [new_id] + ids[best_idx + 2 :]

        return ids

    def encode(self, text: str) -> list[int]:
        """encode → decode 必须严格 round-trip(eval 自检硬要求)。"""
        if not text:
            return []
        words = self._pre_tokenize_re.findall(text)
        ids: list[int] = []
        for w in words:
            ids.extend(self._encode_word(w))
        return ids

    # ------------------------------------------------------------------
    # 解码
    # ------------------------------------------------------------------
    def decode(self, ids: list[int]) -> str:
        """id 列表 → bytes 拼接 → UTF-8 解码。

        对未知 id 用 UNK 兜底;UTF-8 解码失败用 errors='replace'。
        """
        chunks: list[bytes] = []
        for i in ids:
            chunk = self.vocab.get(i)
            if chunk is None:
                chunk = self.vocab[self.UNK_ID]
            chunks.append(chunk)
        raw = b"".join(chunks)
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """写 JSON 文件,字段:
        - tokenizer_type: "bpe_byte_level"
        - vocab_size: int
        - special_tokens: dict[str, int]
        - vocab: dict[str, int](token 字符串 → id,bytes 不可 JSON 化时用 <bytes:HEX>)
        - merges: list[[int, int]]
        - pre_tokenize_pattern: str
        """
        vocab_for_json: dict[str, int] = {}
        for token_id, token_bytes in self.vocab.items():
            try:
                token_str = token_bytes.decode("utf-8")
            except UnicodeDecodeError:
                token_str = f"<bytes:{token_bytes.hex()}>"
            vocab_for_json[token_str] = token_id

        data = {
            "tokenizer_type": "bpe_byte_level",
            "vocab_size": self.vocab_size,
            "special_tokens": {
                "<unk>": self.UNK_ID,
                "<bos>": self.BOS_ID,
                "<eos>": self.EOS_ID,
                "<pad>": self.PAD_ID,
            },
            "vocab": vocab_for_json,
            "merges": [[a, b] for a, b in self.merges],
            "pre_tokenize_pattern": self.pre_tokenize_pattern,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, path: str) -> "BPETokenizer":
        """从 JSON 加载。eval/run.py 的入口。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = cls()
        tok.pre_tokenize_pattern = data["pre_tokenize_pattern"]
        tok._pre_tokenize_re = _re.compile(tok.pre_tokenize_pattern)

        tok.vocab = {}
        for token_str, token_id in data["vocab"].items():
            if token_str.startswith("<bytes:") and token_str.endswith(">"):
                hex_str = token_str[len("<bytes:") : -1]
                tok.vocab[token_id] = bytes.fromhex(hex_str)
            else:
                tok.vocab[token_id] = token_str.encode("utf-8")
        tok._bytes_to_id = {v: k for k, v in tok.vocab.items()}
        tok.merges = [(a, b) for a, b in data["merges"]]
        return tok