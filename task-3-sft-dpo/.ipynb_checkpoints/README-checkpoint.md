# 任务三：指令微调与偏好对齐

> 主大纲见仓库根 [README](../README.md)；本目录是该任务的资源、自检与提交入口。

## 一句话目标

在 Qwen2.5-0.5B 上手写 LoRA 跑通 SFT + DPO 两阶段对齐：SFT 用 MOSS 中英双语对话数据，DPO 在 SFT 之上继续训练，自检三项全过，并能在同一指令上看到 base / SFT / DPO 的可观察差异。

## 任务情境

假装你接手一个 0.5B 的开源基座，组长要求把它从「会续写」调成「会按指令对话、还偏向更好的回答」。规则：

- 不许直接 `pip install peft` 调现成 LoRA，低秩注入自己写一遍
- chat template 套 Qwen 官方格式，loss 只算 assistant 部分
- 两到三周后汇报：SFT/DPO 训练日志 + base vs SFT vs DPO 的对比样例 + 你对 LoRA、loss masking、DPO 损失的理解

这就是本任务。

## 输入 / 输出

| | 内容 |
|---|---|
| **给你** | Qwen2.5-0.5B 基座（`data/download.py` 拉到 `models/Qwen2.5-0.5B`）/ MOSS-003-sft 中英对话数据（jsonl.zip）/ DPO 偏好数据自选（hiyouga/DPO-En-Zh-20k 等）/ PyTorch 2.7+ / 单卡 GPU（0.5B + LoRA 显存友好，8GB 可跑） |
| **交付** | 1. `ckpt/sft/`（SFT 后的 LoRA 权重目录） 2. `ckpt/dpo/`（DPO 后的 LoRA 权重目录） 3. `src/compare.py` 跑出的 base / SFT / DPO 对比样例 4. `eval/result.json`（自检结果） 5. 一段 200–500 字实验观察 |

## Definition of Done

必做 4 项，缺一不算完成：

- [ ] **M1** 手写 LoRA 低秩注入，自检 `lora_param_count` 通过（注入后可训参数占比 < 5%）
- [ ] **M2** 实现 Qwen chat template + loss masking，自检 `loss_masking` 通过（mock 多轮对话中 -100 占比落在 20%–90%，user/system 全 -100）
- [ ] **M3** 用 MOSS-003-sft 数据跑 SFT，产出非空 `ckpt/sft/`，自检 `sft_vs_base` 通过
- [ ] **M4** 在 SFT 之上跑 DPO，产出 `ckpt/dpo/`，并用 `src/compare.py` 在同一指令上对比 base / SFT / DPO 输出

加分（任选）：

- [ ] **S1** 全量微调 vs LoRA：显存占用与下游质量对比
- [ ] **S2** LoRA rank 消融（4 / 8 / 16 / 32）vs 质量
- [ ] **S3** 灾难性遗忘评估：C-Eval 子集上 base vs SFT
- [ ] **S4** SFT-only vs SFT+DPO 在偏好上的差异（带 reward margin 曲线）
- [ ] **S5** 贯通任务五：用 `moss-003-sft-plugin` 训一版带工具调用的 SFT 模型

## 实施步骤（建议节奏：2-3 周）

### 第 1-2 天：环境 + 模型 + 数据

```bash
pip install -r requirements.txt
python data/download.py
```

`data/download.py` 会把 Qwen2.5-0.5B 拉到 `models/Qwen2.5-0.5B`，并打印 SFT / plugin / DPO 数据的下载提示（自检按这个路径找基座）。按提示取数据：

```bash
# SFT 数据（推荐直接下 jsonl.zip，避免 dataset viewer / 自动 builder 解析大文件失败）
huggingface-cli download OpenMOSS-Team/moss-003-sft-data \
  moss-003-sft-no-tools.jsonl.zip --repo-type dataset --local-dir ./data/moss-sft

# DPO 偏好数据自选，如 hiyouga/DPO-En-Zh-20k（中英混合）
# 也可自行用 GPT-4 / Claude 给已有 SFT 数据打偏好标签
```

**常见坑**：

- 不设 `HF_ENDPOINT` 下载慢：境内可设 `HF_ENDPOINT=https://hf-mirror.com`
- 直接跑 dataset viewer / `load_dataset` 拉 MOSS 大文件容易解析失败，按提示下 jsonl.zip 自己读
- 自检按 `models/Qwen2.5-0.5B` 这个固定路径找基座，模型放别处会被判 `[跳过]` 而不是通过

### 第 3-6 天：手写 LoRA（M1）

**输入**：Qwen2.5-0.5B 的 `nn.Linear` 层
**输出**：`src/lora.py` 完整，能通过 `lora_param_count` 自检

实现 `inject_lora(model, target_modules, r, alpha)` 与 `merge_lora(model)`：在目标线性层旁挂低秩分支 A、B，forward 叠加 `scaling * B(A x)`，反向只更新 A、B。

**常见坑**：

- A、B 形状对调：A 是 `in×r`、B 是 `r×out`；初始化 A 用 kaiming、B 用零，否则一开始就改变输出
- scaling 忘了写成 `alpha / r`：换 rank 时等效学习率会跟着漂
- 原权重 W 没设 `requires_grad=False`：可训参数占比超 5%，`lora_param_count` 直接挂
- `merge_lora` 合并后没和未合并前向对齐、或没清理 LoRA 分支：推理结果会和训练时不一致
- 自检固定用 `target_modules=["q_proj","v_proj"], r=8, alpha=16` 调你的 `inject_lora`，参数名和位置要对得上

### 第 7-12 天：chat template + loss masking + SFT（M2 + M3）

**输入**：MOSS-003-sft 多轮对话
**输出**：`src/chat.py`、`train_sft.py`、非空 `ckpt/sft/`，通过 `loss_masking` 与 `sft_vs_base` 自检

实现内容：

1. `src/chat.py`：`format_messages` 套 Qwen 官方模板（`<|im_start|>` / `<|im_end|>`）；`build_labels` 给 user / system / 模板控制符打 -100，只对 assistant turn 算 loss
2. `train_sft.py`：在注入 LoRA 的模型上跑 next-token prediction，AdamW，保存 LoRA 权重到 `ckpt/sft/`

**常见坑**：

- loss mask 反了或漏了：`build_labels` 返回的 labels 必须和 `input_ids` 同形状，-100 占比落在 20%–90% 之间；全 0 或全 -100 都说明 mask 写错
- 多轮对话只算了最后一个 assistant turn：前面几轮 assistant 也该参与训练
- 把模板控制符（`<|im_start|>assistant` 这类）也算进 loss：模型会学着复读控制符
- SFT 数据没做 max_length 截断 + padding：长样本爆显存，或 batch 内长度不齐

### 第 13-18 天：DPO（M4）

**输入**：SFT 后的 LoRA 权重 + DPO 偏好数据（chosen / rejected 对）
**输出**：`ckpt/dpo/`、`src/compare.py` 的 base / SFT / DPO 对比

`train_dpo.py`：在 SFT 之上继续训练，需要一个 freeze 的 reference model 只跑 forward。DPO 损失写成 `-log σ(β·(log π/π_ref(chosen) - log π/π_ref(rejected)))`。

**常见坑**：

- reference model 没 freeze、或参与了反向：DPO 退化、显存翻倍；ref 只做 forward
- 一个 batch 漏掉 forward 次数：chosen / rejected 各要过一次，policy×2 + ref×2 共 4 次 forward
- DPO 直接从 base 起跑、跳过 SFT：偏好对齐没有「先会说话再分好坏」的基础，效果差
- 只看 loss 不看 reward margin：margin 不涨说明 chosen / rejected 没拉开，要回查数据或 β

### 第 19-21 天：对比 + 写报告

**输入**：base / SFT / DPO 三个模型
**输出**：`src/compare.py` 的输出 + 报告文字

`sft_vs_base` 只校验 `ckpt/sft/` 非空，输出质量得自己看：用 `src/compare.py` 在同一批指令上跑 base / SFT / DPO，对比指令遵循度和回答偏好，把结果附进报告。

## 实现约定

| 文件 | 必须导出 |
|---|---|
| `src/lora.py` | `inject_lora(model, target_modules, r, alpha) -> model`、`merge_lora(model) -> model` |
| `src/chat.py` | `format_messages(messages: List[dict]) -> str` 应用 Qwen chat template；`build_labels(input_ids, messages) -> labels` 做 loss masking |
| `ckpt/sft/` | SFT 后的 LoRA 权重目录 |
| `ckpt/dpo/` | DPO 后的 LoRA 权重目录 |

接口可以改，但改了请同步调整 `eval/run.py`。

## 自检

```bash
python eval/run.py
```

| 测试 | 通过标准 | 对应 DoD |
|---|---|---|
| `lora_param_count` | LoRA 注入后可训参数占比 < 5% | M1 |
| `loss_masking` | 对 mock 多轮对话，labels 中 -100 占比在 20%-90%（user/system 全 -100） | M2 |
| `sft_vs_base` | 同一指令上 SFT 和 base 输出有可观察差异（手动确认） | M3 |

> `lora_param_count` 与 `loss_masking` 都需要 `models/Qwen2.5-0.5B` 存在、`transformers` 已装，否则记 `[跳过]`；自检按固定参数 `target_modules=["q_proj","v_proj"], r=8, alpha=16` 调你的 `inject_lora`。
> `sft_vs_base` 只校验 `ckpt/sft` 非空；输出质量请手动跑 `src/compare.py` 对比 base 与 SFT 并附在提交里。

结果写入 `eval/result.json`，提交时附上。

## AI Tutor 反馈

把 [eval/tutor_prompt.md](eval/tutor_prompt.md) 整段贴给 Claude / Qwen / DeepSeek，连同你的代码。模型会按统一格式（必检 / 加分 / 优先级）给你针对性 review。

## 实验建议

- 全量 vs LoRA：显存与下游质量
- LoRA rank 消融（4/8/16/32）
- 灾难性遗忘评估（C-Eval 子集 base vs SFT）
- SFT-only vs SFT+DPO 在偏好上的差异

## 前置阅读（非必需）

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [DPO 论文](https://arxiv.org/abs/2305.18290)
- [HF TRL 文档](https://huggingface.co/docs/trl) / [PEFT 文档](https://huggingface.co/docs/peft)
- 实践书 v2《大语言模型与智能体》「监督微调与 LoRA」「偏好对齐：DPO」两节

## 提交

到 [nndl-discussion](https://github.com/nndl/nndl-discussion/discussions) 「llm-beginner 实践成果」分类发帖，附：

1. 你的 fork 仓库链接
2. `eval/result.json` 内容（贴文本即可）
3. DoD checklist 勾选状态
4. `src/compare.py` 跑出的 base / SFT / DPO 对比样例
5. 200-500 字实验观察：你做了哪些消融、看到了什么有意思的现象

## 时间

约 2-3 周。如果在 SFT 卡住，先用小批 MOSS 数据 + 小 rank 跑通 LoRA → chat template → SFT → DPO 整条 pipeline，再扩大数据和规模。
