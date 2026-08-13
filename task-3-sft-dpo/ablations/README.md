# Ablations

每个脚本独立可运行，使用 `--smoke` 走内置样本无需任何外部数据。

## 共同约定

- **输出 JSON**：默认写到 `figures/s*.json`；
- **缺模型 / 缺数据**：走占位输出，**不抛出**，方便在 CI 或无 GPU 环境下调试；
- **统一命令行**：`--model_path`、`--output`、`--smoke`。

## S1 · 全量微调 vs LoRA

```bash
python ablations/full_vs_lora.py --smoke --steps 5
```

对比 `full` 与 `lora` 两种 setup 在同一 fake batch 上跑 N 步 SFT 的：

- `trainable_params` / `total_params` / `ratio`
- `peak_memory_mb`（仅 GPU）
- `steps_per_sec`
- `final_loss`

**预期**：LoRA 可训练参数约为全量的 0.5%-2%，显存峰值约为全量的 1.2-1.5×（视
optimizer 状态而定），训练速度因 matmul 略快。

## S2 · LoRA rank 消融

```bash
python ablations/rank_ablation.py --smoke --ranks 4 8 16 32 --steps 5
```

固定 `alpha = rank`（保持 scaling = 1）扫 4 个 rank，对比 `trainable_params`、
峰值显存、最终 NLL、step/s。

**预期**：rank 越大可训练参数线性增长（参数量随 rank 线性增长，因为
`lora_A` 与 `lora_B` 都是 `r` 维度），NLL 在低秩上略有浮动，整体趋势
先降后稳。

## S3 · 灾难性遗忘评估

```bash
python ablations/catastrophic_forgetting.py --smoke --n_samples 50
```

在 C-Eval 风格的多选题上算 base vs SFT 的 `accuracy@1` 与选项分布差异。

- 缺 C-Eval 数据时走内置 5 题 smoke；
- 期望：SFT 后准确率小幅下降（0-5pp），看作「遗忘」量级；分布偏向偏好
  答案的方向。

## S4 · SFT-only vs SFT+DPO reward margin

```bash
python ablations/dpo_reward_margin.py --smoke
```

对 chosen / rejected 对算 `β * (log π(chosen) - log π(rejected))`：

- `avg_margin` / `chosen_win_rate` 在 SFT-only 与 SFT+DPO 之间对比；
- 末尾画出 ASCII margin 曲线，便于一眼看出 DPO 后 margin 是否整体上移。

**预期**：DPO 训练后 `avg_margin` 显著大于 0（正）；chosen 胜率 > 80%。

## S5 · 贯通任务五 · 工具调用 SFT

```bash
python ablations/sft_tool_calling.py --smoke --epochs 1
```

- 加载 `moss-003-sft-with-tools` 数据；
- 在 system 段注入工具说明；
- 同 SFT 训练（LoRA）；保存到 `ckpt/sft-tool/`；
- 评估：检查生成是否能输出 `<invoke name="...">{...}</invoke>` 格式。

**输出**：`figures/s5_tool_calling.json` 含 `tool_format_check` 字段。
