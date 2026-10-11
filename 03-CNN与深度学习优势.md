# 机器学习笔记 03 —— CNN 与深度学习优势

来源：李宏毅机器学习课 P9–P11（视频 BV1vL7E6dEqu）
整理时间：7/20


---

## 零、生词

- **Convolutional Neural Network (CNN)**：卷积神经网络，专门做影像的架构
- **Tensor**：张量，维度 > 2 的矩阵（三维 = 宽 × 高 × 通道数）
- **Channel**：通道，彩色图的 RGB 就是 3 个通道
- **One-hot Vector**：独热向量，只有一个维度是 1，其余是 0
- **Fully Connected Network**：全连接网络，每个神经元都接到输入的所有维度
- **Receptive Field**：感受野（也叫守备范围），神经元只看图上一小块
- **Kernel Size**：卷积核大小，就是感受野的高和宽（比如 3×3）
- **Stride**：步长，感受野每次移动多远
- **Padding**：填充，超出范围就补零（Zero Padding）
- **Parameter Sharing**：参数共享，不同感受野的神经元共用同一组权重
- **Filter**：过滤器 / 卷积核，一组共享参数，用来侦查某个 pattern
- **Convolutional Layer**：卷积层 = Receptive Field + Parameter Sharing
- **Feature Map**：特征图，Filter 扫过图片后出来的一群数值
- **Pooling**：池化，把图缩小的操作，没有参数
- **Max Pooling**：最大池化，每组取最大值当代表
- **Flatten**：展平，把矩阵拉直成向量
- **Softmax**：归一化指数函数，输出分类的概率分布
- **Cross Entropy**：交叉熵，分类问题的标准 Loss
- **Data Augmentation**：数据增强，把训练图合理变换一下来增加多样性
- **Spatial Transformer**：空间变换层，专门处理缩放 / 旋转
- **Validation Set**：验证集，用来在多个候选模型里挑最好的
- **H_val**：候选模型的集合
- **Model Complexity**：模型复杂度，就是可选函数的候选数量（|H|）
- **Hyperparameter**：超参数，网络层数、神经元数这些要自己手动调
- **Trade-off**：两难 / 权衡，模型大小和泛化能力的矛盾
- **Deep Network**：深层网络，很多 Hidden Layer 堆起来
- **Shallow / Fat Network**：浅层 / 胖网络，只有一层 Hidden Layer 但神经元很多
- **Piecewise Linear Function**：分段线性函数
- **ReLU**：修正线性单元，x>0 → x，x≤0 → 0
- **Hard Sigmoid**：硬 S 型函数，阶梯状的简化 Sigmoid（两个 ReLU 叠出来）
- **Parity Check**：奇偶校验，判断序列里 1 的个数是奇数还是偶数
- **Exponential**：指数级，Deep 和 Shallow 的效率差就在这儿
- **End-to-End**：端到端，直接从输入到输出，中间不用人工模块

---

## 一、P9：为什么需要 CNN

### 影像分类的基本设定

- 输入：图片，比如 100×100
- 对机器来说，一张彩色图就是三维 Tensor：**宽 × 高 × 3（RGB 通道）**
- 100×100 彩色图 = **30,000 个数字**
- 输出：每个类别一个 **One-hot Vector**（类别数能到 1000+）

### 全连接网络的问题

直接把这 30,000 维向量丢进 Fully Connected Network：

- 第一层 1000 个神经元 → **参数量 = 1000 × 30,000 = 3 × 10⁷（3 千万！）**
- 参数太多 → 很容易 Overfitting

> 结论：得换个更好的架构。

## 二、CNN 的两个核心设计原则

### 观察一：不用看整张图 → Receptive Field

要侦测一个 pattern（鸟嘴、眼睛）只要看**小范围**：

- 每个神经元只看自己的 **Receptive Field**
- 比如 3×3×3 = 27 维输入，27 个 Weight + 1 个 Bias = **只有 28 个参数**

设计时要注意的三件事：

- Kernel Size：感受野的宽和高，常用 **3×3**
- Stride：感受野移动的步长，1 或 2
- Padding：超出范围时补值，一般用 **Zero Padding**

另外：
- 多个神经元可以守同一个 Receptive Field（比如 64 或 128 个）
- Receptive Field 之间**可以重叠**

### 观察二：同一 Pattern 出现在不同位置 → Parameter Sharing

- 鸟嘴可能出现在左上角，也可能出现在中间
- 如果每个 Receptive Field 都放一个独立侦测的神经元 → 参数冗余
- 办法：不同 Receptive Field 的神经元**共享同一组参数**

> 同一组共享参数 = **一个 Filter（卷积核）**

### 合起来看

```
Convolutional Layer = Receptive Field 概念 + Parameter Sharing
```

CNN 的 **Bias（归纳偏置）** 更大：弹性变小 → 专门为影像设计 → **不容易 Overfitting**；但也因此不适合没有影像特性的任务。

## 三、换个讲法：从 Filter 视角看

### Filter 怎么工作

Convolutional Layer 里有一排 Filter，每个 Filter 就是一个 3×3×channel 的 Tensor：

```
1. Filter 放在图片左上角 → 和对应区域做内积 (inner product)
2. 按 Stride 往右移 → 再做内积
3. 重复，直到扫完整张图片
```

- 每个 Filter 产生一群数字 → **Feature Map**
- 64 个 Filter → 64 个 Feature Map（也就是 64 个 channel）
- 新图片 = Feature Map（channel 数 = Filter 数）

### 多层 Convolution

第二层 Convolution：

- 输入 channel = 上一层的 Filter 数（比如 64）
- Filter 高度必须是 64（要匹配输入 channel）
- 层数越深，**同样 3×3 的 Filter 在原始影像上看到的范围越大**
  - 第 1 层 3×3 → 原始 3×3
  - 第 2 层 3×3 → 原始 **5×5**

## 四、Pooling（池化）

来自第三个观察：**缩略不影响辨识**。

- Pooling **没有参数**（比较接近 Activation Function，不算 Layer）
- **Max Pooling**：数字分组（比如 2×2），每组选最大值
- 作用：图片变小，**运算量减少**

不过这些年很多网络不用 Pooling 了：

```
Full Convolution Network：全用 Convolution，不用 Pooling
```

原因是运算能力够了，Pooling 可能影响精细特征的侦测。

## 五、一套经典 CNN 流程

```
输入 → [Convolution×N → Pooling]×M → Flatten → FC → Softmax → 输出
```

## 六、AlphaGo：CNN 下围棋

- 围棋是分类问题（19×19 = 361 个落子位置）
- 棋盘是 **19×19×48** 的 Tensor
- 为什么 CNN 适合围棋？
  1. 重要 pattern 只要看小范围（5×5 就能看出叫吃）
  2. 同样的 pattern 可能出现在任何位置

**AlphaGo 架构**：
- 第 1 层：Kernel 5×5，192 个 Filter，Stride=1，ReLU
- 第 2–12 层：Kernel 3×3，192 个 Filter，Stride=1，ReLU
- **没有 Pooling！**（围棋太精细，不能 subsampling）

> 结论：网络设计要看任务特性，不能盲目套用。应用之道，存乎一心。

## 七、CNN 的局限

CNN **没法直接处理放大 / 缩小（Scaling）和旋转（Rotation）**。

两个缓解办法：
- **Data Augmentation**：截取、放大、旋转训练图片
- **Spatial Transformer Layer**：专门处理几何变换的架构

---

## 八、P10：Validation 回一下顾

三个数据集各干各的活：

- D_train：训练模型参数
- D_val：从多个候选模型里挑最优的那个
- D_test：最后评估泛化能力

## 九、Validation 其实也是一种"训练"

```
H_val = {H₁, H₂, H₃}     ← 候选模型集合
```

Validation 干的事 = 在 **H_val** 里找一个 H，让它在 **D_val** 上的 Loss 最低。

这等于把 H_val 当模型、把 D_val 当训练数据，做一种很特殊的"训练"。

## 十、那 Validation 为什么也会 Overfitting

Overfitting 的条件是：**模型复杂度高** + **训练数据不够好**。

所以 H_val 太大 → 一样会 Overfitting！

极端情况是 Hyperparameter 搜索太激进的时候：
- 网络层数：1~10 层（10 种）
- 每层神经元数：1~1000（1000 种）
- 总候选模型：**1000¹⁰**（天文数字！）

→ H_val 复杂度太高 → Validation 结果就不可靠了。

> 注意：就算有 Validation Set，候选模型太多照样会 Overfitting，Validation 不是万能药。

---

## 十一、P11：机器学习的两难

- 大模型：理想 Loss L(h*, D_all) 低，但和现实的差距（泛化）大
- 小模型：泛化好，但理想 Loss 高

问题就是：能不能两样都拿到？

## 十二、先复习：为什么需要 Hidden Layer

- 一个 **Piecewise Linear Function** = 常数 + 一堆阶梯形函数
- 每个 Sigmoid 神经元 ≈ 一个 Hard Sigmoid
- **用 ReLU 实现**：两个 ReLU 叠起来 = 一个 Hard Sigmoid
- ReLU 够多 → 任何 Piecewise Linear → 逼近任何连续函数

## 十三、Deep vs Shallow：那个关键实验

质疑的人说：

> "一个 Hidden Layer 就能表示所有 Function，Deep 只是噱头 / 骗局。"

实验结果（同样参数量）：

- 1 层，3772 个神经元：参数 ≈ 5 层 × 2000，错误率 **22.5%**
- 5 层，每层 2000：参数 ≈ 1 层 × 3772，错误率 **17.2%**
- 1 层，16K 个神经元：参数远大于 Deep，还是不如一个小的 2 层网络

> 结论：同样参数量下，Deep 远优于 Shallow。**把网络变高优于变胖！**

## 十四、为什么 Deep 更好：一个构造性证明

- 2 个神经元一层 → 1 个"V 形"（谷）
- 多层叠起来 → **分段数指数增长**：

```
1 层 → 2¹ = 2 段
2 层 → 2² = 4 段
3 层 → 2³ = 8 段
k 层 → 2ᵏ 段
```

参数对比：

- Deep：**2k** 个神经元，就能做出 2ᵏ 段
- Shallow（1 层）：要 **2ᵏ** 个神经元，才能做出 2ᵏ 段

差距是 **Exponential（指数级）** 的。

换个说法：
- Deep：O(k) 个参数 → 表示 O(2ᵏ) 的复杂度
- Shallow：O(2ᵏ) 个参数 → 表示 O(2ᵏ) 的复杂度

## 十五、几个类比

- **逻辑电路**：Parity Check，两层需要 2^D 个门，Deep 只要 O(D)
- **编程**：不会把所有东西都写进 main()，而是函数嵌套
- **剪纸（窗花）**：对折几次再剪几刀 = 复杂图案

对折就相当于 Deep Learning 里的 Layer。

## 十六、破除一个误解

```
【误解】Deep = 大模型 → 需要大数据 → 容易 Overfitting

【真相】Deep 用更少参数就能完成同样任务
         → 模型更小 → 更不容易 Overfitting
         → 可以用更少数据训练！
```

> 结论："Deep Learning 真正的强项反而是**不容易 Overfitting**！"

适用条件：目标 Function 是**复杂且有规律**的 → Deep 优于 Shallow。影像、语音这些任务刚好满足。

## 十七、三集的知识地图

```
P9: CNN 架构
├── Receptive Field（不看整张图）
├── Parameter Sharing（共享参数 = Filters）
├── Pooling（缩小图片减运算）
└── AlphaGo（没有 Pooling 的 CNN）

P10: Validation 的陷阱
├── Validation = 一种特殊的 Training
├── H_val 太大 → Validation 也 Overfitting
└── 限制超参数搜索范围

P11: Deep Learning 的真正优势
├── Deep 用指数级更少的参数表达同样函数
├── 2k 神经元（Deep） vs 2ᵏ 神经元（Shallow）
├── 同样参数量下 Deep 远优于 Shallow
└── Deep 不容易 Overfitting！
```

## 十八、复习

**Q1**：为什么 Fully Connected Network 用在影像上会参数过多？算一下 100×100 彩色图输入到 1000 个神经元第一层的参数量。

**Q2**：CNN 的两个核心设计原则是什么？分别来自对影像的什么观察？

**Q3**：为什么多层 CNN 每层都用 3×3 Filter，深层还能侦测大范围 pattern？

**Q4**：AlphaGo 为什么不用 Pooling？

**Q5**：CNN 有什么重要局限？

**Q6**：为什么说 Validation 也是一种"训练"？

**Q7**：为什么用了 Validation Set 还会 Overfitting？

**Q8**：怎么避免 Validation Overfitting？

**Q9**：机器学习的"鱼与熊掌"两难是什么？Deep Learning 怎么解决？

**Q10**：为什么 k 层 Deep 网络能产生 2ᵏ 分段，而 Shallow 需要 2ᵏ 个神经元？

**Q11**：大家对 Deep Learning 的常见误解是什么？正确理解是什么？

**答案：**

1. 每张 100×100 彩色图 = 100×100×3 = 30,000 维。第一层 1000 个神经元 → 30,000×1000 = 3×10⁷（三千万）参数。参数太多，严重增加 Overfitting 风险。
2. **Receptive Field**：侦测 pattern 只需看小范围，不用看整张图；**Parameter Sharing**：同样的 pattern 可能出现在不同位置，所以不同感受野共用参数。
3. 第二层的 3×3 范围在原始影像上等于 5×5，层数越深有效范围越大。
4. 围棋太精细，subsampling 会破坏棋局。架构设计要看任务特性决定。
5. 没法直接处理缩放和旋转。用 Data Augmentation 或 Spatial Transformer Layer 缓解。
6. Validation 在候选集合 H_val 里找 D_val 上 Loss 最低的 Function，本质上和训练一样——在一组数据上找让 Loss 最低的函数。
7. H_val 太大时（比如超参数搜索里候选模型到 1000¹⁰），Validation 也可能 Overfitting。和 Training 的 Overfitting 一样，都取决于模型复杂度和数据量这两个因素。
8. (1) Validation Set 要够大；(2) 控制候选模型数量；(3) 用 K-fold Cross Validation；(4) 最后用独立 Test Set 验证。
9. 两难就是：大模型理想 Loss 低但泛化差，小模型泛化好但理想 Loss 高。Deep Learning 用深层结构以**更少参数**表达同样的函数——参数少 → 泛化好，深层 → 理想 Loss 低，两样都拿到。
10. Deep 里每加一层，"V 形"就叠到所有已有线段上，分段数翻倍，k 层 = 2ᵏ 段。Shallow（1 层）每个 ReLU 只贡献 1 个转折点，所以要 2ᵏ 个神经元。差距是**指数级**的。
11. 误解是"Deep = 大模型 → 需要大数据 → 容易 Overfitting"。真相是 Deep 用更少参数完成同样任务 → 模型更小 → 更不容易 Overfitting → 可以用更少数据训练。Deep 真正的强项反而是**不容易 Overfitting**。