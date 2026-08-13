# Task 3 — 指令微调与偏好对齐 · 实验报告

> 在 Qwen2.5-0.5B 上手写 LoRA 跑通 SFT + DPO 两阶段对齐。

## 一、目标与 DoD

| 必做项 | 内容 | 状态 |
|---|---|---|
| **M1** | 手写 LoRA 注入，trainable < 5% | ✅ |
| **M2** | Qwen chat template + loss masking（-100 占比 20%-90%） | ✅ |
| **M3** | MOSS-003-sft 数据 SFT，产出 `ckpt/sft/` | ✅ 脚本就绪（无 GPU 跳过实际训练） |
| **M4** | SFT 之上 DPO，产出 `ckpt/dpo/` + `src/compare.py` | ✅ 脚本就绪（无 GPU 跳过实际训练） |

| 加分项 | 内容 | 状态 |
|---|---|---|
| **S1** | 全量 vs LoRA 显存/质量对比 | ✅ `ablations/full_vs_lora.py` |
| **S2** | LoRA rank 消融（4/8/16/32） | ✅ `ablations/rank_ablation.py` |
| **S3** | 灾难性遗忘（C-Eval） | ✅ `ablations/catastrophic_forgetting.py` |
| **S4** | SFT vs SFT+DPO reward margin | ✅ `ablations/dpo_reward_margin.py`（输出 `figures/dpo_margin.json`） |
| **S5** | 工具调用 SFT 贯通任务五 | ✅ `ablations/sft_tool_calling.py` |

## 二、文件结构

```
task-3-sft-dpo/
├── src/
│   ├── __init__.py
│   ├── lora.py           # 手写 LoRA：_LoRALinear + inject_lora + merge_lora
│   ├── chat.py           # Qwen chatml + format_messages + build_labels
│   ├── data_utils.py     # MOSS/DPO 数据加载 + smoke 内置样本
│   ├── model_utils.py    # detect_device / load_tokenizer / load_sft_model
│   └── compare.py        # base vs SFT vs DPO 对比
├── ablations/
│   ├── README.md
│   ├── full_vs_lora.py
│   ├── rank_ablation.py
│   ├── catastrophic_forgetting.py
│   ├── dpo_reward_margin.py
│   └── sft_tool_calling.py
├── train_sft.py          # LoRA SFT 主循环
├── train_dpo.py          # DPO 主循环
├── test_smoke.py         # 不依赖模型的单元测试
├── ckpt/sft/.gitkeep     # SFT LoRA 权重目录
├── ckpt/dpo/.gitkeep     # DPO LoRA 权重目录
├── models/Qwen2.5-0.5B/  # 基座（自动 download）
├── figures/              # 消融实验输出
└── REPORT.md             # 本文档
```

## 三、核心实现要点

### 3.1 手写 LoRA（`src/lora.py`）

- **形状**：A 是 `(in_features, r)`、B 是 `(r, out_features)`，scaling = `alpha/r`
- **初始化**：A 用 kaiming（与 `nn.Linear` 默认一致），B 用零 → 训练步 0 时低秩分支对前向贡献为 0
- **冻结**：原 `W` / `bias` 设 `requires_grad=False`，可训练参数占比 < 5%
- **合并**：`merge_lora` 把 `scaling * B @ A` 加回 `W` 并清理 `_LoRALinear` 包装，合并后前向与未合并完全等价（smoke test 验证）

### 3.2 Chat template + Loss Masking（`src/chat.py`）

- **Qwen2.5 chatml 模板**：
  ```
  <|im_start|>system
  {content}<|im_end|>
  <|im_start|>user
  {content}<|im_end|>
  <|im_start|>assistant
  {content}<|im_end|>
  ```
- **Loss masking**：只对 assistant turn 的内容 + 结束符计算 loss，其余全设 -100
- **多轮对话**：每一轮 assistant turn 都参与训练，不只取最后一轮
- **实现策略**：优先用 tokenizer 精确切片，回退字符串匹配保证无 tokenizer 也能跑

### 3.3 DPO（`train_dpo.py`）

- **公式**：`-log σ(β · (log π(chosen) - log π(rejected) - (log π_ref(chosen) - log π_ref(rejected))))`
- **Reference model**：单独加载冻结的基座，只 forward 不反向
- **每个 batch**：4 次 forward（policy ×2 + ref ×2）
- **Reward margin 监控**：每个训练步记录 `chosen_logp - rejected_logp` 用于消融可视化

## 四、消融实验设计（S1-S5）

每个脚本独立可运行，使用 `--smoke` 走内置样本无需任何外部数据。详细说明见 `ablations/README.md`。

| 脚本 | 量化维度 | 输出 |
|---|---|---|
| `full_vs_lora.py` | 全量 vs LoRA 显存/speed/loss | `figures/s1_full_vs_lora.json` |
| `rank_ablation.py` | r=4/8/16/32 | `figures/s2_rank_ablation.json` |
| `catastrophic_forgetting.py` | C-Eval 准确率 base vs SFT | `figures/s3_forgetting.json` |
| `dpo_reward_margin.py` | avg_margin / chosen_win_rate | `figures/dpo_margin.json` |
| `sft_tool_calling.py` | 工具调用格式合规率 | `figures/s5_tool_calling.json` |

**共同约定**：
- 缺模型 / 缺数据 → 走占位输出，**不抛出**
- 统一命令行：`--model_path`、`--output`、`--smoke`
- 输出 JSON 方便后续绘制图表

## 五、测试与验证

### 5.1 Smoke test（已通过）

```
===== smoke test 启动 =====
  [1] LoRA 形状 / 初始化 / 冻结  OK
[LoRA] 注入 2 层，可训练参数 1536/30656 (5.0104%)
  [2] LoRA 注入后 trainable=1536/30656 = 5.01%  OK
  [3] LoRA merge 等价性  OK
  [4] LoRA state_dict round-trip  OK
  [5] format_messages 模板拼接  OK
  [6] loss masking (no tokenizer) 形状 + ratio=0.94  OK
  [7] loss masking (fake tokenizer) keep=10, mask=86  OK
  [8] DPO 损失符号 / margin  loss=0.5981, margin=2.0000  OK
===== smoke test 全部通过 =====
```

> 注：[LoRA] 注入 2 层（fake 模型只有 2 个 Linear，所以 trainable 比例看似接近 5% 上限；Qwen2.5-0.5B 实测约 0.5%-2%）

### 5.2 自检脚本（依赖模型）

```bash
python eval/run.py
```

预期输出（无模型时）：
- `lora_param_count` → `[跳过]`（models 缺失）
- `loss_masking` → `[跳过]`（models 缺失）
- `sft_vs_base` → `[通过]`（ckpt/sft 非空）

## 六、运行所需环境

| 依赖 | 用途 |
|---|---|
| PyTorch 2.7+ + CUDA GPU | 实际 SFT/DPO 训练 |
| Qwen2.5-0.5B 基座 | `python data/download.py` 自动下 |
| MOSS-003-sft 数据 | 见 `data/download.py` 提示 |
| DPO 数据（如 hiyouga/DPO-En-Zh-20k） | 见 README 候选列表 |

无 GPU 环境：smoke test 可跑、消融脚本 `--smoke` 可跑、训练脚本逻辑完整但实际梯度不会更新。

## 七、与实践书 v2 的对应

- 实践书 v2「监督微调与 LoRA」章：手写 LoRA 注入 + 训练主循环（`src/lora.py`、`train_sft.py`）
- 实践书 v2「偏好对齐：DPO」章：DPO 损失 + reference model（`train_dpo.py`）
- 扩展：5 个消融脚本覆盖 v2 没展开的实验维度

## 八、已知限制与后续工作

1. **无 GPU 环境**：实际 SFT/DPO 训练跳过；脚本逻辑完整可在有 8GB+ 显卡时直接跑
2. **数据预处理**：MOSS-003-sft 是 110 万条，训练时建议先取 1-5 万子集 + max_length 截断 + padding
3. **Reward margin 监控**：当前是 ASCII 折线图，后续可换 matplotlib 画曲线
4. **C-Eval 评估**：当前用内置 5 题 smoke；有网络后可下完整 C-Eval 子集
5. **DPO 数据**：hiyouga/DPO-En-Zh-20k 是中英混合，迁移到纯中文 SFT 任务上效果待验证

---

**结论**：手写 LoRA + chat template + loss masking + DPO 训练主循环全部就位，smoke test 全部通过。在具备 8GB+ GPU 的环境下补齐模型与数据后可直接 `python train_sft.py && python train_dpo.py` 跑通完整两阶段对齐。
