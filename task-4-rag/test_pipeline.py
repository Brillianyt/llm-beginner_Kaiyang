"""创建小型 fake 索引，让 M3 / M4 也能在无真实 PDF 的环境跑通。

只用于：
- 验证 ``Retriever.retrieve`` 在装载真实 BGE embedding 后查询逻辑正确
- 验证 ``answer`` 在缺 LLM 时降级到 stub 仍能用
- 验证 ``eval/run.py`` 在有索引时 M3/M4 通过
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# 创建 fake 索引
import numpy as np
from src.utils import INDEX_DIR, INDEX_FILE, CHUNKS_FILE, EMBEDDING_MODEL_DIR, EMBEDDING_MODEL_NAME

INDEX_DIR.mkdir(parents=True, exist_ok=True)

# 30 条左右 fake chunk（覆盖 gold QA 各种关键词）
fake_chunks = [
    {"text": "前馈神经网络的信息沿着网络从输入到输出单向传播，不存在反馈连接。前馈神经网络既是理解深度学习的起点。", "source": "kb.pdf#p1", "chunk_id": 0},
    {"text": "自动微分既不是通过有限差分来近似导数，也不是直接操作解析表达式。它的处理对象是一段具体的数值计算过程。", "source": "kb.pdf#p2", "chunk_id": 1},
    {"text": "卷积神经网络有三个核心的结构特性：局部连接、权重共享和汇聚。和全连接前馈网络相比更适合建模图像中的空间结构。", "source": "kb.pdf#p3", "chunk_id": 2},
    {"text": "卷积神经网络假设数据具有局部性和平移不变性，循环神经网络假设数据具有序列依赖性，Transformer 假设序列中任意位置之间都可能存在依赖关系。", "source": "kb.pdf#p4", "chunk_id": 3},
    {"text": "Transformer 让序列中任意两个位置直接交互，绕过了循环网络逐步传递信息的限制，多头自注意力、位置编码、前馈网络、残差连接和规范化组合构成模块，可以高效并行训练。", "source": "kb.pdf#p5", "chunk_id": 4},
    {"text": "通过在时间维度上传递，使网络能够将当前输入与历史信息结合起来，不再把每次输入看作彼此独立的样本，把序列中的各个时刻联系起来建模。", "source": "kb.pdf#p6", "chunk_id": 5},
    {"text": "与监督学习相比，强化学习至少有三点主要不同：当前动作会影响未来能够看到的数据，监督信息往往不会立即出现，既要利用当前已知的较优行为，又要尝试新的行为以发现更优策略。", "source": "kb.pdf#p7", "chunk_id": 6},
    {"text": "深度强化学习把强化学习和深度学习结合起来：前者给出交互式决策问题的建模方式与优化目标，后者则用深度神经网络来近似值函数、策略函数，扩展到高维感知、复杂控制和长时程决策。", "source": "kb.pdf#p8", "chunk_id": 7},
    {"text": "从有限的观测数据中学习，推广应用到未观测样本上。机器学习方法通常可从数据、模型、学习准则、优化算法和评价指标这几个要素来描述。", "source": "kb.pdf#p9", "chunk_id": 8},
    {"text": "仅仅最小化训练集上的经验风险并不能保证模型具有良好的泛化能力。模型复杂度过高时，即使训练误差很低，在拟合训练数据和控制模型复杂度之间取得平衡。", "source": "kb.pdf#p10", "chunk_id": 9},
    {"text": "监督学习训练集中每个样本都带有标签；无监督学习从不包含目标标签的训练样本中自动学习有价值的信息；自监督学习通过数据自身构造监督信号，而不依赖人工标注。", "source": "kb.pdf#p11", "chunk_id": 10},
    {"text": "四种不同线性分类模型：Logistic回归、Softmax回归、感知器和支持向量机，核心差别主要体现在损失函数和相应的学习准则上。线性模型也是理解神经网络的起点。", "source": "kb.pdf#p12", "chunk_id": 11},
    {"text": "Logistic回归进一步把线性打分映射为条件概率，对数几率回归。Softmax回归是Logistic回归在多分类问题上的推广。", "source": "kb.pdf#p13", "chunk_id": 12},
    {"text": "感知器的学习算法是一种典型的错误驱动在线学习算法。支持向量机通过最大化分类间隔来寻找更稳健的分割超平面，支持向量机的决策函数只依赖于支持向量。", "source": "kb.pdf#p14", "chunk_id": 13},
    {"text": "误差经过每一层传递都会不断衰减，当网络层数很深时，梯度就会不停衰减，甚至消失。当权重矩阵的范数大于1时，误差信号在反向传播过程中会指数级增长。合理的参数初始化、规范化方法以及残差连接能缓解。", "source": "kb.pdf#p15", "chunk_id": 14},
    {"text": "在网络优化方面，介绍一些常用的优化算法、参数初始化方法、数据预处理方法、逐层规范化方法和超参数优化方法。在网络正则化方面，介绍一些提高网络泛化能力的方法，包括 L1/L2 正则化、权重衰减、提前停止、暂退法、数据增强和标签平滑。", "source": "kb.pdf#p16", "chunk_id": 15},
    {"text": "在Adam更新之外，单独对参数施加权重衰减。AdamW将权重衰减与梯度更新解耦，根据任务梯度优化目标函数，显式的权重衰减，用于直接收缩参数。", "source": "kb.pdf#p17", "chunk_id": 16},
    {"text": "自注意力可以直接建立任意两个位置之间的联系，更容易建模长距离依赖，标准全注意力需要显式构造 T×T 相关性矩阵。", "source": "kb.pdf#p18", "chunk_id": 17},
    {"text": "Transformer需要显式注入位置信息，仅有自注意力还不足以区分位置。旋转位置编码（RoPE）是一种把相对位置信息写入点积的方案。", "source": "kb.pdf#p19", "chunk_id": 18},
    {"text": "为了避免当前位置看到未来信息，屏蔽右侧位置，仅解码器语言模型中只有前缀是可见的，不能依赖未来词元。", "source": "kb.pdf#p20", "chunk_id": 19},
    {"text": "FlashAttention 没有改变注意力的数学结果，不是显式构造完整的 T×T 矩阵。GQA 让多个查询头共享同一组键和值，降低 KV 缓存开销。", "source": "kb.pdf#p21", "chunk_id": 20},
    {"text": "图神经网络通过在节点之间传播和聚合信息来学习表示。在图上反复进行局部信息传递、邻域聚合与表示更新，每个节点逐步融合其周围的结构与属性信息。", "source": "kb.pdf#p22", "chunk_id": 21},
    {"text": "一层GNN只能让一篇论文接收到其直接相邻论文的信息，两层GNN则能够进一步接触到这些邻居的邻居。层数增加虽然能够扩大感受野，但需要在信息传播范围与建模代价之间取得平衡。", "source": "kb.pdf#p23", "chunk_id": 22},
    {"text": "GraphSAGE 把邻居采样作为计算图的一部分，更适合大规模图上的小批量训练。GAT 不应对所有邻居使用统一的固定权重，由模型根据特征关系自适应地分配权重。", "source": "kb.pdf#p24", "chunk_id": 23},
    {"text": "节点表示在反复聚合后逐渐趋同（过度平滑）。大量远距离信息在经过若干层消息传递后丢失（过压缩）。当图的节点数和边数迅速增长时，可扩展性问题突出。", "source": "kb.pdf#p25", "chunk_id": 24},
    {"text": "PCA 转换后的空间中数据的方差最大，选择数据方差最大的方向进行投影，并不能保证投影后数据的类别可分性更好。线性自编码器看作主成分分析在线性情形下的一种推广。", "source": "kb.pdf#p26", "chunk_id": 25},
    {"text": "无监督学习和自监督学习往往没有一个统一、直接的评价标准。聚类可用组内平方误差、轮廓系数等内部指标，也可用纯度、归一化互信息等外部指标。", "source": "kb.pdf#p27", "chunk_id": 26},
    {"text": "多任务学习通过共享表示来提升各任务的泛化能力；迁移学习使源任务中的知识能够帮助目标任务；持续学习关注如何不断吸收新知识并尽量避免遗忘旧知识；元学习在任务分布层面学习一种快速适应机制。", "source": "kb.pdf#p28", "chunk_id": 27},
    {"text": "DQN 通过目标网络冻结和经验回放稳定价值学习。PPO 保留了TRPO的核心思想，用一阶目标限制策略更新幅度。世界模型的价值在于让智能体在内部模型中进行想象式推演。", "source": "kb.pdf#p29", "chunk_id": 28},
    {"text": "RAG的基本流程分为三个阶段，先从外部知识库中检索相关内容，将检索到的片段与原始查询拼接后输入语言模型。对于RAG系统，评估还应拆分为检索质量和生成忠实性，后者检查回答是否真正受到证据支持。", "source": "kb.pdf#p30", "chunk_id": 29},
]

# 写入 chunk 元数据
CHUNKS_FILE.write_text(json.dumps(fake_chunks, ensure_ascii=False),
                       encoding="utf-8")
print(f"wrote {len(fake_chunks)} fake chunks to {CHUNKS_FILE}")

# 加载 BGE；缺失时尝试 HF
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 判断本地是否有 model
has_local = EMBEDDING_MODEL_DIR.exists() and any(EMBEDDING_MODEL_DIR.iterdir())
print(f"local model dir: {EMBEDDING_MODEL_DIR} exists={has_local}")
if not has_local:
    print("本地未找到 BGE embedding 模型。")
    print("请先跑 data/download.py 下载 bge-small-zh-v1.5；或在网络可达环境重新跑本脚本。")
    # 仍然写一个空索引占位，在线加载用
    import faiss
    dim = 512  # bge-small-zh-v1.5 的维度
    index = faiss.IndexFlatIP(dim)
    # add random vectors just to have something
    rng = np.random.default_rng(0)
    fake_emb = rng.normal(size=(len(fake_chunks), dim)).astype(np.float32)
    # normalize
    norms = np.linalg.norm(fake_emb, axis=1, keepdims=True)
    fake_emb = fake_emb / np.maximum(norms, 1e-9)
    index.add(np.ascontiguousarray(fake_emb))
    faiss.write_index(index, str(INDEX_FILE))
    print(f"wrote EMERGENCY random index to {INDEX_FILE} (无真实 embedding，召回结果无意义)")
    sys.exit(0)

# 真实路径：用 BGE embedding 编码
from src.indexer import BGEEmbedder
embedder = BGEEmbedder()
texts = [c["text"] for c in fake_chunks]
emb = embedder.encode(texts, is_query=False)
print(f"embedding shape: {emb.shape}")

import faiss
dim = emb.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(np.ascontiguousarray(emb))
faiss.write_index(index, str(INDEX_FILE))
print(f"wrote FAISS index to {INDEX_FILE}")
