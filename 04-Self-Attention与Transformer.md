# 机器学习笔记 04 —— Self-Attention 与 Transformer

来源：李宏毅机器学习课 P12–P17（视频 BV1vL7E6dEqu）
整理时间：8/5

---

## 零、生词

看课遇到的英文都记在这儿：

Self-Attention 这边的：

- **Sequence**：序列，一串有顺序的数据（文字、语音、图都算）
- **Vector Set / Vector Sequence**：把序列里每个元素表示成一个向量，合起来就是向量集合
- **One-Hot Encoding**：独热编码，长度等于词汇表大小，只有一个位置是 1
- **Word Embedding**：词嵌入，给每个词一个带语义的向量
- **Self-Attention**：自注意力，每个输出都看整个输入序列
- **Query (q) / Key (k) / Value (v)**：查询 / 键 / 值——q 是当前"想问的问题"，k 是各位置的"索引标签"，v 是各位置真正携带的内容
- **Attention Score (α)**：注意力分数，q 和 k 的内积，衡量关联程度
- **Dot-Product Attention**：点积注意力，α = q·k，最常用
- **Multi-Head Attention**：多头注意力，好几组 Q/K/V 并行，各看各的相关性
- **Positional Encoding**：位置编码，给输入补上位置信息（自注意力本身不感知位置）
- **Truncated Self-Attention**：截断自注意力，只看一定范围，长序列能快些
- **Cross-Attention**：交叉注意力，Decoder 的 Q 去 Encoder 的 K/V 里查东西

Transformer 这边的：

- **Transformer**：就是 "Attention Is All You Need" 那篇提出的架构
- **Seq2Seq**：序列到序列，输入一串输出一串，输出多长由机器定
- **Encoder / Decoder**：编码器处理输入 / 解码器根据 Encoder 的输出生成目标序列
- **Residual Connection**：残差连接，输入直接加到输出上，给梯度留条"高速路"
- **Layer Normalization**：层归一化，同一个样本内部跨维度归一化，不依赖 batch
- **Masked Self-Attention**：掩码自注意力，只能看到当前位置和左边
- **Auto-Regressive (AT)**：自回归，输出一个个产生，依赖前面所有输出
- **Non-Auto-Regressive (NAT)**：非自回归，一次并行生成全部输出
- **Teacher Forcing**：训练时直接喂 Decoder 正确答案
- **Exposure Bias**：训练用正确答案、测试用自己输出，两边对不上导致的偏差
- **Beam Search**：束搜索，保留多条候选路径找全局更优
- **BLEU Score**：评估两个句子像不像的指标
- **BERT**：Transformer 的 Encoder 那部分

BatchNorm 和训练技巧：

- **Feature Normalization / Standardization**：特征归一化 / 标准化，让各维度均值 0 方差 1
- **Batch Normalization (BN)**：在一个 batch 内算 μ 和 σ 来归一化
- **Moving Average**：移动平均，训练时把 μ、σ 累积起来给测试用
- **Internal Covariance Shift**：内部协变量偏移，BN 最早的理论解释（后来被部分否定）
- **Scheduled Sampling**：训练时偶尔喂 Decoder 错误输入，练纠错
- **Copy Mechanism**：复制机制，Decoder 直接从输入里抄内容

Self-Attention 变体：

- **Longformer / BigBird / Linformer / Reformer / Performer / Synthesizer**：都是为了搞掉 O(N²) 的不同路子
- **Sinkhorn Sorting**：用另一个网络学"哪些位置需要算 attention"
- **Low-Rank**：低秩，矩阵里很多列冗余，能被别的列线性组合出来
- **Kernel Trick**：核技巧，把 exp(q·k) 拆成 φ(q)·φ(k)，好换计算顺序
- **LRA Benchmark**：长序列竞技场，比各种变体的速度-性能-内存权衡

---

## 一、问题的起点：输入变成了一排向量

之前学的东西，输入都是一个固定大小的向量。但现实里很多输入是**一排向量**，而且**长度不定**。

用的几种变成向量的方式：

- 文字：每个词 → Word Embedding，一句话就是个向量序列
- 语音：每 25ms 取一个窗口，每 10ms 滑一次，一秒就有 100 个向量
- 社交网络：每个节点（人）→ 一个 profile 向量
- 分子：每个原子 → 一个 One-Hot vector

## 二、序列标注，和 Self-Attention 的想法

**问题**：输入 N 个向量，要输出 N 个标签，一一对应，比如词性标注。

**先试笨办法**：把每个向量单独丢进 FC。结果同一个词永远得到同一个输出。像 "I saw a saw" 里两个 saw，一个动词一个名词，FC 分不出来。

**Self-Attention 的思路**：让每个输出向量**考虑整个序列的信息**。这样两个 saw 的上下文不一样，输出就能不一样。

## 三、Self-Attention 具体怎么算

### 算相关性（Attention Score）

对每个位置 i：

```
qⁱ = W^q × aⁱ   （Query）
kⁱ = W^k × aⁱ   （Key）
vⁱ = W^v × aⁱ   （Value）

αᵢⱼ = qⁱ · kⱼ       （内积，就是关联分数）
α'ᵢⱼ = softmax(αᵢⱼ) （归一化成权重）
```

### 加权求和

```
bⁱ = Σⱼ α'ᵢⱼ × vⱼ
```

> 注意：b¹ 是把整句话所有词的信息都考虑进来的。哪怕 a⁴ 离 a¹ 很远，只要 α'₁₄ 够大，v⁴ 就会被大量掺进 b¹。

Q、K、V 我自己是这样记的：Query 就像我往搜索引擎里敲的关键字，Key 像网页的索引标签，Value 是网页正文，Attention Score 就是搜索引擎给的匹配度。

## 四、换成矩阵来看（这样能并行算）

```
I = [a¹|a²|a³|a⁴]

Q = W^q × I    K = W^k × I    V = W^v × I

A = K^T × Q        （N×N 的 Attention Matrix）
A' = softmax(A)

O = V × A'         （最终输出）
```

整个 Self-Attention 里**真正要学的只有 W^q、W^k、W^v**，其他都是没参数的人为设定操作。

## 五、Multi-Head Self-Attention

原因很直白："相关性"不止一种——有的词是主语相关，有的是修饰相关。一个 Q 不够用，那就多来几组。

```
qⁱ → [qⁱ·¹, qⁱ·²]   （把 q 切成多个 head）
kⁱ → [kⁱ·¹, kⁱ·²]
vⁱ → [vⁱ·¹, vⁱ·²]

Head 1：只在 Q¹,K¹,V¹ 里做 Self-Attention
Head 2：只在 Q²,K²,V² 里做 Self-Attention

bⁱ = concat(bⁱ·¹, bⁱ·²) → 再过一个线性变换
```

## 六、位置编码（Positional Encoding）

问题在于 Self-Attention 不感知位置，a¹ 和 a⁴ 在它眼里没有"距离"。办法是给位置 i 加一个位置向量 eⁱ：

```
aⁱ ← aⁱ + eⁱ
```

eⁱ 怎么来，课上给了几种：三角函数生成（原论文那套 Sinusoidal）、当可学习参数学出来（Learned）、或者用 RNN 动态生成。

## 七、跟 CNN、RNN 比一比

### CNN 其实是 Self-Attention 的特例

- CNN 的 Receptive Field 是**人圈定**的
- Self-Attention 的 Receptive Field 是**自己学出来**的

实验结果比较有意思：数据少的时候 CNN 更好（弹性小，不容易过拟合）；数据一多（比如 3 亿张图），Self-Attention 更好（弹性大，吃得下更多数据）。

### Self-Attention vs RNN

|  | Self-Attention | RNN |
|---|---|---|
| 远距离信息 | 直接交互 | 得一步步传，可能丢 |
| 并行化 | 全部同时算 | 必须顺序算 |

## 八、Batch Normalization

### 为什么要归一化

输入特征各维度数值范围差太多，Error Surface 就崎岖不平，梯度下降很难收敛。标准化一下：

```
x̃ⁱ = (xⁱ - μⁱ) / σⁱ
```

让每个维度均值≈0、方差≈1，Error Surface 就平滑多了。

### 隐藏层也得做——这就是 BN

不光是输入层，**每一层的输入都做**，这就是 Batch Normalization。

```
μ, σ ← 从当前 Batch 里算
z̃ = (z - μ) / σ
ẑ = γ × z̃ + β      （γ、β 是可学习参数，初始 γ=1, β=0）
```

训练和测试不一样：

|  | 训练 | 测试 |
|---|---|---|
| μ, σ | 从当前 batch 算 | 用移动平均 μ_bar, σ_bar |

测试时可能就一个样本，根本算不出有意义的 batch 统计，所以必须用训练时攒下来的。

### BN 为什么有用

- Error Surface 更平滑
- 能开更大的 Learning Rate
- 连 Sigmoid 都能训得起来了
- 它最早那个解释（Internal Covariance Shift）后来被论文**部分否定**了

---

## 九、Transformer 整体与 Encoder

Seq2Seq 就是这么一个东西：

```
输入 Sequence → [Encoder] → [Decoder] → 输出 Sequence
                                    ↑
                          输出长度由机器自己决定
```

机器翻译、语音辨识、聊天机器人、QA 都是这个套路。

Encoder Block 长这样：

```
输入
 ↓
Multi-Head Self-Attention
 ↓
Residual Connection + Layer Normalization    ← 输入和输出直接相加
 ↓
Feed Forward Network
 ↓
Residual Connection + Layer Normalization
 ↓
输出

（这样重复 N 次）
```

Residual Connection 的本质就是 ResNet 那个：

```
输出 = 输入 + Self-Attention(输入)
       ↑
    "高速路"，让梯度直接穿过去
```

Layer Norm 和 Batch Norm 别搞混：

|  | Batch Norm | Layer Norm |
|---|---|---|
| 怎么归一化 | 跨样本，同一维度 | 同一样本，跨所有维度 |
| 依赖 Batch | 是 | 否 |
| Transformer 为什么用 LN |  | 序列长度不定，BN 不稳 |

## 十、Decoder

AT（自回归）Decoder 就是一个字一个字蹦：

```
输入 BEGIN → 输出词1
输入 BEGIN+词1 → 输出词2
输入 BEGIN+词1+词2 → 输出词3
...
直到输出 END（结束符）
```

Decoder 里有三个关键设计：

1. **Masked Self-Attention**：算 b² 的时候**不能看 a³、a⁴**，因为测试时它们还不存在。
2. **Cross-Attention**：Q 来自 Decoder，K、V 来自 Encoder。就是 Decoder 拿自己的"问题"去 Encoder 的输出里"找答案"。
3. **END Token**：词表末尾加个特殊符号，输出到它就停。

NAT Decoder 则是：一次喂进一堆 BEGIN，一次把整串输出全吐出来。

|  | AT | NAT |
|---|---|---|
| 速度 | 慢 | 快 |
| 性能 | 好 | 差（有多模态问题） |
| 用在哪 | 主要翻译 | 语音合成（FastSpeech） |

## 十一、Decoder 的几种训练技巧

| 技巧 | 解决什么 |
|---|---|
| Teacher Forcing | 训练时喂正确答案，收敛快，但会带来 Exposure Bias |
| Beam Search | 不只贪心选一个，保留多条候选路径 |
| Copy Mechanism | 让 Decoder 直接从输入里抄（人名、专有名词） |
| Scheduled Sampling | 训练时偶尔喂错误输入，增强纠错 |

## 十二、Self-Attention 变体：核心矛盾是 O(N²)

Self-Attention 的复杂度是 **O(N²)**，序列一长（比如图像 65536 个 pixel）就完全扛不住。

### 人设计的稀疏 Attention

- Local Attention：只盯邻居，小范围
- Stride Attention：跳着看
- Global Attention：留几个特殊 Token 收集全局信息
- Longformer = 上面三种的组合
- BigBird = Longformer + Random Attention

### 数据驱动的

- Reformer：用 LSH 聚类，只在同一个 cluster 里算 Attention
- Sinkhorn Sorting：拿个网络去学"哪些位置需要 Attention"

### 低秩——Linformer

Attention Matrix 是低秩的，那就只挑 K << N 个有代表性的 Key，矩阵从 N×N 缩成 **N×K**。

### 换矩阵乘法顺序

```
原始：O = V × softmax(K^T × Q)   → O(N²)
加速：O = (V × φ(K)^T) × φ(Q)   → O(N·d²)
```

关键就是 Kernel Trick：找一个 φ，让 exp(q·k) = φ(q)·φ(k)，于是可以先算 φ(K)^T·φ(V)。

## 十三、Kernel Trick 完整推导（Performer）

### 想加速的是什么

瓶颈就在那个 N×N 的注意力矩阵：

```
O = softmax(Q·Kᵀ) · V     ← Q·Kᵀ 是 N×N，N=65536 时 43 亿个数，直接爆
```

逐元素看第 i 个输出：

```
Oᵢ = Σⱼ exp(qᵢ·kⱼ)·vⱼ / Σⱼ exp(qᵢ·kⱼ)

问题根源：分母要求 qᵢ 跟每一个 kⱼ 配对 → 必须把所有 j 遍历一遍 → O(N)
```

### 核心：把 exp(q·k) 拆开

> 如果能拆成"q 的函数 × k 的函数"，求和顺序就能换！

```
exp(q·k) ≈ φ(q)·φ(k)     （随机特征映射 Random Feature Map）
```

φ 是这么造的（随机傅里叶特征）：

```
φ(x) = 1/√m · [ sin(x·w₁), cos(x·w₁), sin(x·w₂), cos(x·w₂), ..., sin(x·wₘ), cos(x·wₘ) ]

w₁..wₘ 是从标准正态分布采样的随机向量（采好就固定，不训练）
```

数学依据（Bochner 定理 + 随机傅里叶特征）：

```
E_w[sin(q·w)sin(k·w) + cos(q·w)cos(k·w)] = exp(-‖q-k‖²/2) ≈ exp(q·k)（只差个常数缩放）
```

采 m 个 w 取平均就能高精度近似。m 越大越准（一般 64~256），φ 的维度 d' = 2m。

> 直觉：exp(q·k) 像"两个人在同一间房里的亲近程度"，φ 就是给每个人拍张"特征照片"，照片像不像 ≈ 实际亲不亲近。有了照片就能先批量洗出来再比对，不用每次把人都叫到一起。

### 换顺序：先算 φ(K)ᵀ·φ(V)

代进去：

```
Oᵢ ≈ Σⱼ (φ(qᵢ)·φ(kⱼ))·vⱼ / Σⱼ (φ(qᵢ)·φ(kⱼ))
```

关键观察是 φ(qᵢ) 跟 j 无关，能提到求和号外面：

```
Oᵢ ≈ φ(qᵢ)·[Σⱼ φ(kⱼ)·vⱼ] / φ(qᵢ)·[Σⱼ φ(kⱼ)]
         └─────┬─────┘      └───┬───┘
       不依赖 i！          不依赖 i！
```

方括号里的求和跟 i 没关系，提前算一次，所有 i 共用：

```
矩阵形式（φ(K) 是 N×d'，V 是 N×d）：

先算（一次）：
  A = φ(K)ᵀ·V     → d'×d    （键值对的"浓缩摘要"）
  z = φ(K)ᵀ·1     → d'×1    （所有键的"总和"）

再算（每个 i）：
  Oᵢ = φ(qᵢ)·A / φ(qᵢ)·z
```

### 复杂度对比：O(N²) → O(N·d')

| 步骤 | 原始 | Performer |
|---|---|---|
| ① | Q·Kᵀ：N×d·d×N → N×N | φ(Q),φ(K)：O(N·d·d') |
| ② | softmax：O(N²) | φ(K)ᵀ·V：O(N·d'·d)，只算一次 |
| ③ | ·V：N×N·N×d | φ(Q)·A：O(N·d'·d) |
| 总复杂度 | O(N²·d) | O(N·d'·d)，线性了 |
| 内存 | O(N²)（40GB+） | O(N·d')（可以忽略） |

> 结论：原始必须生成 N×N 的"全连接关系图"；Performer 把关系图压成 d'×d 的摘要（φ(K)ᵀ·V），所有位置共享——**拿"全局统计"替掉"逐对计算"**。

### 有代价

| 好处 | 代价 |
|---|---|
| 复杂度 O(N)，长序列友好 | exp(q·k) 只是近似（m 有限） |
| 内存友好 | 精度略降，得调 m 平衡 |

### 推导合订本（背这个）

```
目标：Oᵢ = Σⱼ exp(qᵢ·kⱼ)vⱼ / Σⱼ exp(qᵢ·kⱼ)

Step1：随机特征映射近似   exp(qᵢ·kⱼ) ≈ φ(qᵢ)·φ(kⱼ)
Step2：代入              Oᵢ ≈ Σⱼ (φ(qᵢ)·φ(kⱼ))vⱼ / Σⱼ (φ(qᵢ)·φ(kⱼ))
Step3：φ(qᵢ) 提出来       Oᵢ ≈ φ(qᵢ)·[Σⱼ φ(kⱼ)vⱼ] / φ(qᵢ)·[Σⱼ φ(kⱼ)]
Step4：括号跟 i 无关       A = φ(K)ᵀ·V，z = φ(K)ᵀ·1（提前算好）
Step5：每个 i 查摘要      Oᵢ = φ(qᵢ)·A / φ(qᵢ)·z   → O(N·d'·d)
```

## 十四、最激进的 Synthesizer

它干脆不用 Q×K 生成 Attention Matrix，直接把那个矩阵当网络的可学习参数。结果性能居然没差太多——这块挺颠覆我对 Attention 的理解的。

## 十五、六集串起来看

```
P12-P13: Self-Attention
├── Q·K → Attention Score → 加权 V
├── Multi-Head：并行捕捉多种相关性
├── Positional Encoding：补位置信息
└── 矩阵形式：O = V × softmax(K^T×Q)

P14: Batch Normalization
├── 让 Error Surface 更平滑
├── 公式：ẑ = γ×(z-μ)/σ + β
└── 测试时用移动平均

P15-P16: Transformer
├── Encoder：Self-Attn + Residual + LayerNorm + FF
├── Decoder：Masked + Cross-Attention
├── AT：逐个生成  NAT：一次生成
└── 训练技巧：Teacher Forcing / Beam Search

P17: 变体 — 解决 O(N²) 瓶颈
├── 稀疏：Local/Stride/Global/BigBird
├── 低秩：Linformer
├── 换顺序：Performer (Kernel Trick)
└── 极端：Synthesizer（直接学 Attention Matrix）
```

---

## 复习

**Q1（P12 基础）**：为什么不能单独送每个词进 FC 做词性标注？
**A**：相同的输入（比如两个 saw）会得到完全一样的输出，没法按上下文区分词性。Self-Attention 靠考虑整句信息解决了这个。

**Q2**：Q、K、V 分别代表什么？
**A**：Q（查询）从当前位置发出"问题"；K（键）是每个位置的"索引"；V（值）是每个位置实际携带的信息。b¹ = Σ α'·v，α' 由 Q¹ 和各个 K 的内积决定。

**Q3（P13 进阶）**：为什么需要 Multi-Head？为什么需要 Positional Encoding？
**A**：Multi-Head 是因为"相关性"有多种形式（主语 / 宾语 / 修饰相关），需要多组 Q/K/V 并行捕捉。Positional Encoding 是因为 Self-Attention 不感知位置，得手动注进去。

**Q4**："CNN 是 Self-Attention 的特例"怎么理解？
**A**：CNN 的 Receptive Field 是人固定范围的；Self-Attention 的 Attention Score 是自动学会"关注哪个范围"的。数学上把参数设合适，Self-Attention 能做得跟 CNN 一样，所以 CNN 是受限版的 Self-Attention。

**Q5（P14 BatchNorm）**：BN 在训练和测试时行为有什么区别？
**A**：训练时 μ 和 σ 从当前 Batch 算；测试时用训练中累积的移动平均 μ_bar 和 σ_bar（因为测试可能只有一个样本）。

**Q6**：γ 和 β 的作用是什么？
**A**：γ（缩放）和 β（偏移）是可学习参数，初始 γ=1、β=0 就是不改变分布。后面网络要是觉得某维度不为 0 更好，可以自己学回来——相当于给网络恢复表达能力的"安全网"。

**Q7（P15 Encoder）**：Encoder Block 包含哪些组件？Residual Connection 起什么作用？
**A**：Self-Attention → Residual+LayerNorm → Feed Forward → Residual+LayerNorm。Residual Connection 把输入直接加到输出上，给梯度一条"高速路"，缓解深层梯度消失。

**Q8**：Layer Norm 和 Batch Norm 的本质区别？
**A**：BN 跨样本归一化（需要 batch 统计），LN 在同一样本内部归一化（不依赖 batch）。Transformer 用 LN 是因为序列变长时 BN 不稳定。

**Q9（P16 Decoder）**：为什么 Decoder 需要 Masked Self-Attention？
**A**：因为输出是逐个产生的，算 b² 的时候后面的 a³、a⁴ 还不存在。Mask 保证模型行为和测试时一致。

**Q10**：Cross-Attention 里 Q、K、V 分别来自哪里？
**A**：Q 来自 Decoder，K 和 V 来自 Encoder。Decoder 拿 Q 去 Encoder 的 K 里查，再用 V 的值做聚合。

**Q11（P17 变体）**：Self-Attention 的计算瓶颈是什么？Linformer 怎么解决？
**A**：瓶颈是 O(N²) 的 Attention Matrix。Linformer 利用低秩性质（矩阵里很多列冗余），只选 K<<N 个有代表性的 Key，把 N×N 矩阵降成 N×K。

**Q12**：Performer / Linear Transformer 加速的核心技巧是什么？
**A**：用 Kernel Trick 把 exp(q·k) 分解成 φ(q)·φ(k)，改变计算顺序——先算 φ(K)^T×φ(V)（跟 N 无关），再乘 φ(Q)。复杂度从 O(N²) 降到 O(N·d²)。

**Q13**：为什么 exp(q·k) 不能直接拆成 f(q)·g(k)？
**A**：因为 q、k 是**以内积形式纠缠**在指数里的，数学上没法精确分解。所以才用**随机特征映射**（随机傅里叶特征 φ）去近似：φ(q)·φ(k) 的期望 ≈ exp(q·k)。这是"近似"不是"精确"，用 m 个随机向量（64~256）的代价换线性复杂度。

**Q14**：为什么 φ(K)ᵀ·V 只需要算一次，所有 q 都能共享？
**A**：因为它不依赖查询下标 i。所有 q_i 共享同一份"键值摘要" A = φ(K)ᵀ·V（d'×d 的小矩阵）。这正是 O(N²) → O(N) 的关键：把"逐对计算"变成"全局统计 + 逐个查询"。代价是 exp(q·k) 的精度损失，得调 m 平衡速度和质量。
