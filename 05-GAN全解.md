# 机器学习笔记 05 —— 生成对抗网络 GAN

来源：李宏毅机器学习课 P18–P21（视频 BV1vL7E6dEqu）
整理时间：8/2

> 这是我自己边看课边整理的 GAN 笔记，公式和课上举的例子尽量都抄下来了，生词也都记在旁边，看不懂的地方先留着回头再看。

---

## 零、先把通用生词抄一遍

- **Distribution**：分布，描述随机变量取值的概率
- **Gaussian / Normal Distribution**：高斯分布 $N(\mu,\sigma^2)$，自然界最常见，GAN 里常用它给 z
- **Uniform Distribution**：均匀分布，每个取值概率一样
- **$P_{data}$**：真实数据分布；**$P_G$**：Generator 生成数据的分布
- **Sample**：采样，从分布里随机抽点
- **Manifold**：流形，高维空间里的低维结构；图片在像素空间里其实只摊成一张低维流形
- **Divergence**：散度，量两个分布差多远
- **KL Divergence**：KL 散度 $D_{KL}(P\|Q)=\sum P(x)\log\frac{P(x)}{Q(x)}$，不对称
- **JS Divergence**：JS 散度，KL 的对称版，原始 GAN 的目标函数跟它有关
- **Wasserstein Distance / Earth Mover's Distance**：推土机距离，把一个分布"搬"成另一个所需的最小平均搬运量
- **Loss Function**：损失函数，量预测和真实的差距
- **Objective Function**：目标函数，要最大化（或最小化）的那个
- **Cross Entropy**：交叉熵 $-\sum y_i\log\hat y_i$，分类常用
- **Binary Classifier**：二分类器，把输入分成真/假两类
- **Gradient Descent / Gradient Ascent**：梯度下降（最小化）/ 梯度上升（最大化）
- **Supervised / Unsupervised / Semi-Supervised Learning**：监督 / 无监督 / 半监督学习
- **Pre-train**：预训练，先用大量数据练出好的初始参数
- **Hyperparameter**：超参数，人手动设的（学习率、batch size、层数等）
- **Epoch / Iteration**：轮次 / 迭代，训练里的更新周期

---

## 一、P18 —— GAN 的基本概念

### 先记几个生词

- **GAN (Generative Adversarial Network)**：生成对抗网络，由 Generator 和 Discriminator 组成，靠对抗训练生成数据
- **Generator**：生成器，负责"生成"假数据，输入随机向量 z，输出假样本
- **Discriminator**：判别器，负责"辨真假"，输入一个样本，输出一个标量分数（越大越真）
- **Random Variable / Noise (z)**：随机变量 / 噪声，从简单分布（如高斯）里采的向量，当 Generator 的输入
- **Latent Vector**：隐向量，就是喂给 Generator 的那个 z，一般是低维的（50–100 维）
- **Neural Network**：神经网络，可学习的函数逼近器
- **CNN**：卷积神经网络，擅长图像；**Transformer**：自注意力网络，擅长序列；**RNN**：循环神经网络，处理序列；**Encoder / Decoder**：编解码器，序列到序列模型的两半
- **Unconditional Generation**：无条件生成，Generator 只收随机噪声 z，没有额外条件输入

### 为什么突然要学生成模型

前面学的网络清一色一个套路：

$$y = f(x)$$

给个 x，吐一个定死的 y。

但现实里好多问题，同一个输入本来就对应好几种对的输出。课上小精灵那个例子我印象最深：

> **Video Prediction**：小精灵在同一个转弯处，有时左转、有时右转。要还用传统 Supervised Learning，模型学到的是"又向左又向右"，画面直接糊掉，角色还会裂成两个。

所以碰到需要"创造力"的活儿，光一个确定输出不够了，模型得能吐出一个**分布（Distribution）**，而不是单个值。

需要创造力的地方还有：
- **画图**：同样说"红眼睛的角色"，每个人画出来都不一样
- **对话**：问"某某是谁"，正确答案有一大堆

### Generator 的想法：往输入里掺随机性

生成模型的核心招数就是给输入加个随机数：

$$y = f(x, z)$$

这里 z 是从一个**简单分布**（比如正态 $N(0,I)$）里采出来的随机向量。

> **为什么 z 的分布要"简单"？** 因为得能采样（sample）啊。正态、均匀这种我们知道公式、随手就能采。Generator 干的事，就是把简单的分布"揉"成复杂的分布。

这样每次采到不同的 z，输出就不一样，整体上 y 就变成一个复杂的分布。

### 先看无条件生成

这章先把 x 拿掉：

$$y = G(z)$$

- $z \sim N(0, I)$：低维正态向量（维度自己定，比如 50 或 100）
- G：Generator 这个网络
- y：高维输出（比如 64×64×3 的图 = 12288 维向量）

**目标**：让 Generator 吐出来的图看着就像真的二次元角色（或者别的目标数据）。

### 那 Discriminator 是干嘛的

麻烦在于：光有一个 Generator 没法训——它吐的东西没有标准答案啊。

GAN 的解法：再塞一个 **Discriminator（判别器）**。
- 输入：一张图 y
- 输出：一个标量分数 $D(y)\in\mathbb{R}$
- 分数越大 = 越像真的

Discriminator 自己也是个网络（图片的话一般用 CNN），架构随便定。

### 枯叶蝶和波波鸟

课上用演化来打比方：

| 演化里的概念 | GAN 里对应的 |
|---|---|
| 枯叶蝶（伪装者） | Generator（造假的图） |
| 波波鸟/比比鸟（天敌） | Discriminator（辨真假） |
| 枯叶蝶越来越像枯叶 | Generator 造的图越来越真 |
| 鸟越来越会分辨 | Discriminator 越来越狠 |

俩人在对抗里**一起进化**——写作敌人，念作朋友。

### 实际怎么训：两步循环

**训练是反复交替做的**：

```
Step 1: 固定 Generator，训 Discriminator
  - 从真图库采样真图（标签=1）
  - 从 Generator 采样假图（标签=0）
  - 让 Discriminator 做二分类（或者回归）

Step 2: 固定 Discriminator，训 Generator
  - 采 z → Generator 出假图 → Discriminator 打分
  - 更新 Generator 参数，让 Discriminator 打的分数越大越好
  - （G 和 D 串成一个大网络，但只更新 G 那部分）

反复迭代 Step 1 → Step 2 → Step 1 → Step 2 → ...
```

**串接网络的具体做法**：把 Generator（比如 5 层）和 Discriminator（比如 5 层）直接接成一个 10 层大网络：
- 输入：随机向量 z
- 中间某一层输出：图片（宽 = 像素数 × 3）
- 最终输出：Discriminator 的分数

用 Gradient Ascent 更新 Generator 的参数去抬高分数，**但 Discriminator 那部分参数不许动**（不然模型直接把 bias 调到天文数字就作弊了）。

### 训着训着长啥样

| 训练轮次 | 效果 |
|---|---|
| 100 次 | 基本是噪声 |
| 1000 次 | 冒出了眼睛 |
| 2000 次 | 冒出了嘴巴 |
| 5000 次 | 有点人脸样，水汪汪大眼 |
| 10000 次 | 形状清晰但有点水彩糊糊的 |
| 50000 次 | 还不错的二次元脸 |
| 最佳结果 | 高质量角色，还能做 interpolation（内插）出连续变化 |

### GAN 动物园 & 命名段子

- 变体超过 **500 种**，谁发明新的就在前面加个字母
- 字母很快不够用了——光 "SGAN" 就至少有 6 种
- 甚至有人被逼着把论文叫 "AVAGAN"，就因为 "VAEGAN" 已经被占了

---

## 二、P19 —— GAN 的理论和 WGAN

### 先记几个生词

- **WGAN (Wasserstein GAN)**：用 Wasserstein Distance 替代 JS Divergence 的 GAN 变体
- **1-Lipschitz Function**：满足 $\|f(x)-f(y)\|\le\|x-y\|$ 的函数，就是变化足够平滑的那种
- **Weight Clipping**：权重裁剪，WGAN 里把判别器参数硬限制在某个区间内
- **Gradient Penalty**：梯度惩罚，Improved WGAN 里用梯度范数惩罚代替 Weight Clipping
- **Spectral Normalization**：谱归一化，把网络每层的谱范数约束成 1，保证 1-Lipschitz

### 训练到底在优化啥

P18 学会了"怎么做"，这节回答"为啥这么做有用"。

**终极目标**：让 Generator 生成的分布 $P_G$ 跟真实数据分布 $P_{data}$ 越接近越好。

$$\text{目标：找到一个 } G^* = \arg\min_G \; \text{Div}(P_G, P_{data})$$

这里 $\text{Div}(P_G, P_{data})$ 就是两个分布之间的 **Divergence（散度 / 距离）**。

> Divergence 越大 → 两个分布越不像；Divergence 越小 → 越接近。

### 卡在哪：Divergence 根本算不出来

要是知道 $P_G$、$P_{data}$ 的完整公式，KL 直接怼上去就完了。问题是：
- $P_{data}$ 的公式不知道（手上只有采出来的图）
- $P_G$ 的公式也不知道（是个复杂网络转出来的分布）
- 高维连续分布上的积分，实操上基本别想算

**GAN 的魔法**：算不出来不要紧，**只要能分别从 $P_G$ 和 $P_{data}$ 采样**，就有办法估出 Divergence。

### Discriminator 和 Divergence 是一回事

Discriminator 的目标函数（Objective Function）：

$$V(G, D) = \mathbb{E}_{y \sim P_{data}}[\log D(y)] + \mathbb{E}_{y \sim P_G}[\log(1 - D(y))]$$

它要最大化这个函数：

$$D^* = \arg\max_D V(G, D)$$

也就是：真数据来的 y → D(y) 越大越好；Generator 来的 y → D(y) 越小越好。

**关键洞察**：$\max_D V(G, D)$ 这个最大值，跟 $P_G$、$P_{data}$ 之间的 **JS Divergence** 有关：

$$\max_D V(G, D) \approx \text{常数} - 2 \cdot \text{JS}(P_{data} \| P_G)$$

（详细证明在 Goodfellow 原始论文里，还得假设 Discriminator 容量无限大等等）

**直观理解**：

| $P_G$ 和 $P_{data}$ 的关系 | Discriminator 分得开吗 | $\max_D V$ |
|---|---|---|
| 很像、混在一起 | 很难分 | 值较小 |
| 很不像、分得很开 | 容易分 | 值较大 |

所以整个 GAN 的优化就写成一个式子：

$$\min_G \max_D V(G, D)$$

这就是 GAN 的 **Minimax 问题**：Generator 想最小化 Divergence，Discriminator 想最大化 V。

### JS Divergence 的致命伤

JS 有个大毛病：

> 两个分布要是没有重叠，JS Divergence 就恒等于常数 $\log 2$，不管它们离多远、多近！

**为啥 $P_G$ 和 $P_{data}$ 几乎不重叠？**

理由一，数据本身的特性。图片在高维空间里是低维流形（Manifold），$P_G$ 和 $P_{data}$ 都是高维里的"线"或"面"，两条随便的线几乎不可能相交。

理由二，采样不够。就算实际有重叠，我们也只采了有限个点，训 Discriminator 时它照样能画一条完美分界线（硬背都能 100% 正确率）。

> 注意：实操里最恐怖的现象——你要是用 Binary Classifier 当 Discriminator，训完正确率几乎总是 100%。Loss 根本不随 Generator 进步而变化，你压根不知道 Generator 有没有变好。**以前训 GAN 只能靠人眼盯着图看，发现崩了就砍掉重来。**

### 换个度量：Wasserstein Distance（推土机距离）

JS 不行，那换一种度量行不行？

**Wasserstein Distance**（又叫 Earth Mover's Distance / 推土机距离）：想象 P 是一堆土，Q 是目的地的形状，你开推土机把 P 的土搬成 Q 的形状，**最小平均搬运距离**就是它。

$$W(P, Q) = \inf_{\gamma \in \Pi(P,Q)} \mathbb{E}_{(x,y) \sim \gamma}[\|x - y\|]$$

其中 $\Pi(P, Q)$ 是所有能把 P 变成 Q 的联合分布。

**为啥它更好？** 对没有重叠的分布：
- JS Divergence：不管远近，始终 = $\log 2$（看不出区别）
- Wasserstein Distance：离得远 → 值大；离得近 → 值小（能反映渐进改善）

**演化类比**：从感光细胞到复杂的人眼，中间需要很多连续的小步骤。JS 要求一步到位，Wasserstein 允许每一步都有进展。

### WGAN

**WGAN** = 用 Wasserstein Distance 替代 JS Divergence 的 GAN。

WGAN 里 Discriminator 的目标函数变成：

$$\max_{D \in \text{1-Lipschitz}} \left\{ \mathbb{E}_{y \sim P_{data}}[D(y)] - \mathbb{E}_{y \sim P_G}[D(y)] \right\}$$

其中这个最大值恰好就等于 $W(P_{data}, P_G)$！

**关键约束**：D 必须是个 **1-Lipschitz 函数**，即变化足够平滑：

$$\|D(x) - D(y)\| \leq \|x - y\|$$

**为啥非加这个约束？** 不加的话，Discriminator 可以对真实数据给 $+\infty$、对生成数据给 $-\infty$，又绕回 JS 那个问题了。约束逼着 Discriminator "平滑"，两组数据近时分数不能差太多。

**怎么实现 Lipschitz 约束？**
1. **Weight Clipping**（原始 WGAN）：训完把参数硬裁到 $[-c, c]$。较粗糙但能用。
2. **Gradient Penalty**（Improved WGAN）：在真实数据和生成数据之间的随机插值点上，要求 Discriminator 梯度的范数接近 1，加个惩罚项来实现。
3. **Spectral Normalization**（当前效果最好）：把网络每层的谱范数归一化到 1，理论上满足 1-Lipschitz 条件。其中 **SNGAN（Spectral Normalization GAN）** 公认是效果最好的实现。

### GAN 难训的根本原因

就算有了 WGAN，GAN 还是不好训。根子在于 **Generator 和 Discriminator 得"旗鼓相当"**：
- Discriminator 没训好 → Generator 失去进步目标
- Generator 进步不了 → Discriminator 也没东西可学
- **只要一方停下来，另一方也跟着停**

这就是 "no pain, no gain" 在 GAN 里的体现，GAN 出名地难训。

---

## 三、P20 —— 怎么评估生成器 & 条件式 GAN

### 先记几个生词

- **Inception Score (IS)**：Inception 分数，用预训练分类器评估生成图的质量和多样性
- **FID (Fréchet Inception Distance)**：比较真实图和生成图在 Inception 网络特征空间里的分布距离
- **Quality**：质量，单张生成图逼不逼真
- **Diversity**：多样性，生成的一堆样本丰不丰富
- **Mode Collapse**：模式坍缩，Generator 只能产少数几种样本，缺多样性
- **Mode Dropping**：模式丢弃，生成数据只覆盖真实分布的一部分
- **Conditional GAN**：条件 GAN，带条件输入 x，能控制生成内容
- **VAE (Variational Autoencoder)**：变分自编码器，另一种生成模型，基于变分推断
- **Flow-based Model**：流模型，基于可逆变换的生成模型
- **Text-to-Image**：文字生成图像；**Image-to-Image / Pix2Pix**：图像到图像转换；**Style Transfer**：风格转换，改风格不改核心内容

### GAN 为啥还是这么难训

就算有了 WGAN，本质困难还在：

1. **Generator 和 Discriminator 互为前提**：一方不进步，另一方也停。任何一方的 Loss 偶尔不下降，整个训练就可能崩掉。
2. **GAN + 生成文字 = 地狱难度**：如果 Generator 是个 Sequence-to-Sequence 模型（比如 Transformer Decoder）：
   - 输出是离散 Token（取 $\arg\max$），参数微动一下，最大 Token 不变 → Discriminator 输出不变 → **梯度算不出来**！
   - 只能改用 **Reinforcement Learning** 训 Generator
   - GAN + RL = 大炸裂，极难训
   - 一直到 **ScratchGAN** 靠爆搜超参数 + 一堆技巧（超大 batch size、正则化等）才做到 From Scratch 训练
3. **但 GAN 仍是最好的图像生成模型**：难训归难训，VAE 和 Flow-based Model 出的图质量跟 GAN 差一大截，而且它们本身也不好训。

### 怎么评自己练出来的 Generator

有了 Generator，怎么客观评价它好不好？

#### 办法一：人工评

最直观，找人看图。问题是：不客观、不稳定、没法自动化。早年很多 GAN 论文只放图不放数字——"我觉得比前人好"就算结论了。

#### 办法二：针对具体任务评

比如作业里：跑一个人脸检测器，看 1000 张图能检测出多少张人脸。问题是只适合这个特定任务，不具一般性。

#### 办法三：用 Image Classifier 评 Quality

把生成的图丢进预训练的图像分类器：
- 输出概率分布**非常集中**（比如 99% 是"狗"）→ 说明图片质量好
- 输出概率分布**非常平坦**（各类概率差不多）→ 说明图片四不像

#### 办法四：评 Diversity（多样性）

把一大堆生成图丢进分类器，对所有输出概率分布取**平均**：
- 平均分布**太平坦**（各类概率均等）→ Diversity 高
- 平均分布太集中 → Diversity 低（比如所有图都被判成同一类 → Mode Collapse）

**Quality 和 Diversity 的评估是互斥的**：
- Quality：看单张图 → 越集中越好
- Diversity：看一堆图 → 平均越平坦越好

#### Inception Score (IS)

把两者合起来：

$$\text{IS} = \exp(\mathbb{E}_y[\text{KL}(p(c|y) \| p(c))])$$

其中 $p(c|y)$ 是单张图片的条件类别分布，$p(c)$ 是所有图片的边缘类别分布。
- Quality 高 → 每张图的 $p(c|y)$ 集中
- Diversity 高 → $p(c)$ 平坦

但 IS 在二次元人脸任务里不适用（Inception 网络把所有人脸都判成"人"类）。

#### FID (Fréchet Inception Distance)

现在最常用的指标：

1. 把真实图片和生成图片都丢进 Inception 网络
2. **不取最后的类别**，而是取进入 Softmax 之前的 Hidden Layer 输出（上千维的特征向量）
3. 假设两组特征向量都服从高斯分布
4. 算两个高斯分布之间的 **Fréchet Distance**

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

其中 $(\mu_r, \Sigma_r)$ 是真实图片特征的统计量，$(\mu_g, \Sigma_g)$ 是生成图片的。

**FID 越小越好。**

> 注意：假设是高斯分布可能不太准，而且得大量样本才能把分布估准。

#### Mode Collapse 和 Mode Dropping

**Mode Collapse（模式坍缩）**：
- Generator 翻来覆去就产生那几张图
- 比如训到后期，所有面孔变成同一张脸（只是发色不同）
- 原因：Discriminator 存在"盲点"，Generator 找到后一直薅

**Mode Dropping（模式丢弃）**：
- 更难被检测，生成数据只覆盖真实分布的一部分
- 比如：生成的图片人脸看着男女都有，但肤色都偏白——有种族歧视的倾向
- 连 BigGAN 都可能有这毛病

#### GAN vs VAE 谁更稳

Google 那篇 *"Are GANs Created Equal?"* 说：
- VAE 在不同随机种子下结果更稳定
- GAN 的 FID 方差很大（对随机种子敏感）
- 但 GAN 的最好结果远超 VAE（不在同一量级）
- 不同 GAN 变体之间的差异看起来不大（在那篇研究的统一网络架构下）

### Conditional GAN（条件式 GAN）

无条件生成虽然好玩，但我们更想**控制生成的内容**。

#### 长这样

$$y = G(x, z)$$

- x：条件（比如一段文字、一张图片）
- z：随机噪声
- y：生成结果

#### Text-to-Image

输入一段文字（比如 "red eyes"），Generator 产生对应的图：
- 每次采样不同的 z，产生不同但都符合描述的角色
- x（文字）可以先过 RNN 或 Transformer Encoder 编码再喂进 Generator

#### Discriminator 得这么改

**关键问题**：要是 Discriminator 只看 y（不看 x），Generator 会只产生清晰的图片，但**完全无视输入条件**。

**解法**：让 Discriminator 同时看 $(x, y)$：
- 图片质量好 + 与条件匹配 → 高分
- 图片质量差 或 与条件不匹配 → 低分

训练时需要三种数据：
1. **(真图, 真文字)** → 正例
2. **(假图, 真文字)** → 负例（Generator 生成的）
3. **(真图, 假文字)** → 负例（故意乱配的，用来强化"配不配"的判断）

#### Image-to-Image (Pix2Pix)

输入一张图片，输出另一张图片，例如：
- 设计图 → 实景照片
- 黑白图 → 上色
- 白天 → 夜景
- 有雾图 → 无雾图

纯 Supervised Learning 做 Image-to-Image 往往产生**模糊**结果（因为同一输入对应多种可能输出，模型取了"平均"）。加上 GAN 的 Discriminator 后，结果更清晰真实，但偶尔会"想象力过度丰富"（加进去输入里没有的东西）。

**最佳做法**：GAN Loss + Supervised Loss 一起用。

#### 别的 Conditional GAN 应用

- **声音 → 图片**：听狗叫声画狗、听瀑布声画瀑布
- **让静态图片动起来**：三星用 Conditional GAN 让蒙娜丽莎开口说话

---

## 四、P21 —— Cycle GAN

### 先记几个生词

- **Cycle GAN**：循环 GAN，用于不成对数据的风格转换，用两个 Generator + 循环一致性
- **Cycle Consistency**：循环一致性，Cycle GAN 的核心约束：$x \to G(x) \to F(G(x)) \approx x$
- **Style GAN**：能生成高质量、可控制风格人脸的进阶 GAN
- **Progressive GAN**：从低分辨率逐步生成高分辨率图片的 GAN
- **BigGAN**：大规模 GAN，能产生高质量且多样化的图片

### 没成对数据怎么办（Unpaired Data）

前面 Conditional GAN 要**成对数据**（paired data），每个 x 都有对上的目标 y。

但很多时候根本没成对数据：

> **例子：真人照片 → 二次元头像**
> 想给新垣结衣的照片配一张二次元版，得先拍照再请画师画——成本太高了。
> 这类问题叫**影像风格转换（Image Style Transfer）**。

我们手上是：
- 一堆 X domain 的图片（真人照片集）
- 一堆 Y domain 的图片（二次元头像集）
- 但 X 和 Y **不成对**

**目标**：训一个网络，输入 X domain 图片，输出 Y domain 风格的图片。

### 直接套原版 GAN 会翻车

最直接的想法——把原版 GAN 搬过来：
- Generator 输入从 Gaussian 改成从 X domain 采样
- Discriminator 判断输出像不像 Y domain

**问题**：Generator 完全可以**无视输入**！只要吐一张"看着像 Y domain"的图就能骗过去。输入新垣结衣，输出灰叶大小姐（八竿子打不着），Discriminator 照样给高分。

所以得**把输入和输出的关系绑死**。

### Cycle GAN 的核心：两个 Generator + 循环一致

#### 架构

| 组件 | 功能 |
|---|---|
| $G_{X \to Y}$（蓝色） | 把 X domain 图片转为 Y domain 风格 |
| $G_{Y \to X}$（橙色） | 把 Y domain 图片还原回 X domain 风格 |
| $D_Y$ | 判断图片是否像 Y domain |

#### Cycle Consistency（循环一致性）

核心约束：输入一张 X 图片，经过 $X \to Y$ 再经过 $Y \to X$，应该变回**和原来一模一样**的图片。

$$\mathcal{L}_{\text{cycle}}(G_{X \to Y}, G_{Y \to X}) = \mathbb{E}_{x \sim P_X}\left[\|G_{Y \to X}(G_{X \to Y}(x)) - x\|_1\right]$$

#### 为啥这招有效

假设输入是"戴眼镜的男生"：
- 要是 $G_{X \to Y}$ 随便输出灰叶（完全无关），$G_{Y \to X}$ 拿到灰叶根本不可能还原出"戴眼镜的男生"
- 为了满足 Cycle Consistency，$G_{X \to Y}$ 的输出必须**保留足够信息让 $G_{Y \to X}$ 能还原**

→ 输入和输出被迫建立关系！

### 完整结构一般是双向的

$$\begin{aligned}
&\text{正向：} X \xrightarrow{G_{X \to Y}} \hat{Y} \xrightarrow{G_{Y \to X}} \tilde{X} \approx X \quad (\text{Cycle Consistency})\\
&\text{反向：} Y \xrightarrow{G_{Y \to X}} \hat{X} \xrightarrow{G_{X \to Y}} \tilde{Y} \approx Y \quad (\text{Cycle Consistency})\\
&\text{对抗：} D_Y \text{ 判断 } \hat{Y} \text{ 是否像 } Y \text{ domain}\\
&\text{对抗：} D_X \text{ 判断 } \hat{X} \text{ 是否像 } X \text{ domain}
\end{aligned}$$

### 理论上其实有漏洞，但实战很少翻车

#### 理论问题

Cycle Consistency 并没有**保证**输入和输出在语义上对应。理论上 Generator 能学到很怪的转换：
- 看到眼镜 → 把眼镜抹掉 → 画一颗痣；反向 Generator 看到痣 → 画回眼镜
- 甚至：$G_{X \to Y}$ 把图片左右翻转，$G_{Y \to X}$ 再翻回来

这些"作弊"方案都能满足 Cycle Consistency，但根本不是我们要的风格转换！

#### 实际表现

**好消息**：这些理论问题在实践中**很少出现**。因为：
1. 神经网络本身有"惰性"——它更喜欢做简单的转换。把眼镜变成痣是复杂操作，保留眼镜简单多了。
2. 实际用起来，输入和输出往往**确实非常像**，只是风格改变了。

#### 同期独立发现

有意思的是，三个不同团队在几乎相同的时间（2017 年 3–4 月）独立提出了几乎一模一样的东西：
- **Cycle GAN**
- **Disco GAN**
- **Dual GAN**

同一个 idea 的不同名字。

### 真做出来啥样

#### 真人 → 二次元

- 韩国团队做了网页版：上传照片 → 输出二次元角色
- 效果：眼睛变大（典型二次元特征），但不总完美（比如一只眼大一只眼小）

#### 文字风格转换

用完全相同的 Cycle GAN 架构：
- 收一堆正面句子 + 一堆负面句子（从 PTT 推文/嘘文爬）
- 用两个 Sequence-to-Sequence 模型（如 Transformer）当 Generator
- 正面 → 负面 → 正面（Cycle Consistency）
- 要用 RL 训 Generator（输出是离散文字）

> 效果示例："肚子痛得厉害" → "生日快乐厉害"（机器学到了奇怪的映射：腹部毛病 → 生日快乐）

#### 无监督翻译（Unsupervised Translation）

- 一堆中文句子 + 一堆英文句子（不成对）
- Cycle GAN → 机器学会翻译

#### 无监督语音识别

- 一堆音频（没有文字标注）+ 一堆文字（没有对应音频）
- Cycle GAN 试着把音频转成文字

### StarGAN

Cycle GAN 只能在**两种风格**之间转。StarGAN 扩展成**多种风格**互转，更进阶的版本。

---

## 复习（自己出的题，答案在下面）

1. 为什么传统 Supervised Learning（定死的 $y=f(x)$）在 Video Prediction 上会翻车？用小精灵分裂那个例子说。
2. GAN 训练里，更新 Generator 时为啥 Discriminator 的参数必须冻住不动？
3. Generator 的输入 z 为啥常用 Normal Distribution？能用别的分布吗？
4. 说说 GAN 训练的两个步骤（Step 1 / Step 2），各自目标是啥？
5. $\min_G \max_D V(G, D)$ 里，$\max_D V(G, D)$ 实际是什么意思？
6. JS Divergence 在 $P_G$、$P_{data}$ 没有重叠时恒为 $\log 2$，这对训练有啥实际影响？
7. 用大白话解释 Wasserstein Distance（推土机距离），以及它为啥比 JS 更适合当 GAN 的度量？
8. WGAN 里 Discriminator 为啥必须满足 1-Lipschitz？说一种实现它的办法。
9. 什么是 Mode Collapse？它跟 Mode Dropping 有啥区别？
10. IS 和 FID 的核心区别是什么？FID 为啥更常用？
11. Conditional GAN 里，Discriminator 为啥必须同时接收 $(x, y)$，不能只看 $y$？
12. 用 GAN 生成文字（Sequence）为啥特别难？ScratchGAN 是怎么解决的？
13. 为啥原版 Conditional GAN 不能直接拿来做 Unpaired Data 的风格转换？
14. 解释 Cycle Consistency 怎么工作，它为啥能逼着输入和输出建立关系？
15. Cycle GAN 理论上有什么隐患？为啥实践里很少翻车？
16. Disco GAN、Dual GAN 和 Cycle GAN 有啥区别？

**答案：**

1. 同一个输入（比如转角处）可能对应好几种不同输出（左转或右转）。传统 Supervised Learning 只输出一个确定值，训练数据里左右转都有时，模型学出来就是"又向左又向右"，画面糊掉、角色裂成两个——既左又右反而**错得最离谱**。
2. 因为 Generator 和 Discriminator 是串成一个大网络来训的。Generator 的目标是让 Discriminator 输出越大越好。要是 Discriminator 的参数也能更新，模型直接把它最后一层的 bias 调到一个极大值（比如一百万）就能拿到巨大分数——这样 Generator 根本不用学会造好图。
3. 能用别的！选 Normal 只是因为：①知道它的公式，②容易从中采样。实际上任何"足够简单又能采样"的分布都行。经验上不同分布差别不大，因为 Generator 本身会把简单分布转换成复杂分布。原则就是：z 的分布只要能采样就行。
4. **Step 1**：固定 Generator → 从真实数据采真图、从 Generator 采假图 → 训 Discriminator 分真假（真图高分、假图低分）。目标：提高 Discriminator 的辨别能力。**Step 2**：固定 Discriminator → 采 z → Generator 造出假图 → Discriminator 打分 → 更新 Generator 参数让分数最大化。目标：提高 Generator 的"欺骗"能力。
5. 表示：在给定当前 Generator 的情况下，训出最优 Discriminator 后得到的 Objective Function 最大值。这个最大值与 $P_G$、$P_{data}$ 之间的 JS Divergence 相关——散度小则最大值小，散度大则最大值大。
6. 会导致梯度消失。因为 Generator 进步与否（$P_G$ 靠不靠近 $P_{data}$），Loss 看起来都一样，都是 $\log 2$。训练的人没法从 Loss 判断 Generator 有没有变好。实操里 Discriminator 的二分类正确率几乎总是 100%，只能靠**人眼看图片质量**来评估——这就是以前训 GAN 被叫"黑魔法"的原因。
7. **直观**：把 P 想象成一堆土，Q 是目标形状。Wasserstein Distance = 开推土机把土搬成 Q 的最小平均搬运距离。**好处**：就算两个分布没有重叠，它也能反映它们之间的距离（越近值越小、越远值越大），给 Generator 连续的梯度信号，让它知道"往哪挪一点"会有进步；而 JS 在没有重叠时恒为常数，给不了这种渐进指导。
8. **为啥要**：不加 Lipschitz 约束，Discriminator 会对真实数据输出 $+\infty$、对生成数据输出 $-\infty$，训练不收敛。约束要求 Discriminator 必须"平滑"——两组数据距离近时分数差异不能太大，这样 Wasserstein Distance 才能真反映分布间的距离。**实现办法**（写一种就行）：Weight Clipping（裁到 $[-c, c]$）/ Gradient Penalty（在真实与生成数据的插值点上惩罚梯度范数偏离 1）/ Spectral Normalization（把每层谱范数归一到 1）。
9. **Mode Collapse（模式坍缩）**：Generator 学会只产生很少几种（甚至一种）样本就能骗过 Discriminator，输出缺多样性，比如所有生成的人脸都变成同一张。**Mode Dropping（模式丢弃）**：生成数据只覆盖真实分布的一部分，但"缺失"不容易被发现，比如男女都有但肤色单一——看着多样性"够"，实际漏了部分真实模式。区别：Collapse 比较明显（一看就重复），Dropping 更隐蔽。
10. **IS**：基于 Inception 网络的**最终分类概率**，用单张图的集中度衡量 Quality、用边缘分布的平坦度衡量 Diversity。毛病是：要是所有图片都被分到同一个大类（比如"人"），IS 会判 Diversity 很低，哪怕图片之间其实有丰富变化。**FID**：用 Inception 网络**最后一层 Hidden Layer 的特征向量**（进 Softmax 之前），算真实图片和生成图片在这个特征空间里的分布距离。特征向量比最终分类保留了更多信息，即使是同一类别的图片（都是"人脸"），也会随肤色、发型等变化。FID 更常用就是因为对同类图片的多样性更敏感。
11. 要是 Discriminator 只看 y（生成的图片），Generator 会学到：只要造出**清晰的图片**就能骗过它，完全不管输入的 x（条件/文字）是什么。这样输出跟条件无关，不是我们想要的。让 Discriminator 同时看 $(x, y)$，它就能判断"图片是否与条件匹配"——逼着 Generator 产生既真实、又符合条件的结果。
12. **难在**：Generator 出句子时，每个位置要取概率最大的 Token（$\arg\max$）。参数微小变化时概率分布只微动，**最大 Token 不变** → 句子不变 → Discriminator 输出不变 → 梯度算不出来（Gradient Descent 失效）。只能改用 Reinforcement Learning，而 RL + GAN 极难训。**ScratchGAN 的解决**：靠**爆搜超参数** + 一堆技巧（极大 batch size ≈ 上千、Discriminator 加正则化、改 RL 算法等），在 Google 级算力下实现了 From Scratch 训练。
13. 因为 Conditional GAN 要靠成对数据来训 Discriminator 判断 $(x, y)$ 配不配。而 Unpaired 场景里只有一堆 X 图、一堆 Y 图，不知道哪个 x 对应哪个 y。要是只用原版 GAN（Discriminator 只看 y），Generator 会无视输入条件，输出跟输入无关——不是风格转换要的结果。
14. Cycle Consistency 要求：x 过 $G_{X \to Y}$ 转完，再过 $G_{Y \to X}$ 还原，结果要尽量接近原来的 x：$G_{Y \to X}(G_{X \to Y}(x)) \approx x$。**原理**：如果 $G_{X \to Y}$ 随便输出无关内容，$G_{Y \to X}$ 拿到后根本还原不出原来的 x（它也不知道原输入长啥样）。为了满足 Cycle Consistency，$G_{X \to Y}$ 的输出**必须保留足够信息**让 $G_{Y \to X}$ 能还原——输入和输出就被迫关联起来了。
15. **理论隐患**：Generator 可能学会"作弊"式的转换（左右翻转、眼镜 ↔ 痣），这些都能满足 Cycle Consistency，但不是我们想要的保语义的风格转换。**很少发生的原因**：神经网络天生有"惰性偏好"，爱学简单的转换。保留眼镜比"把眼镜变成痣"简单，风格化处理比发明新内容简单。所以实践里输入和输出往往真的非常像，只是风格变了。
16. **没有本质区别。** 它们是不同团队在几乎相同的时间（2017 年 3–4 月）独立提出的相同想法。三篇的核心概念——用两个 Generator 做双向转换 + Cycle Consistency——完全一致，只是名字和实现细节略有不同。这是科研中"多重独立发现"的有趣案例。
