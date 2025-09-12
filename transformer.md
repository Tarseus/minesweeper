# Minesweeper（扫雷）— 全局特征 + Decoder-Only 自回归方案

> 只参考 DGL 的**动态全局特征**思想，不用任何局部/窗口注意力；编码器采用 **Decoder-Only**（仅自注意力、无 cross-attn）结构。每一步对“整盘 + 若干全局 token”执行一次自注意力，随后用指针头在“未开集合”上选取一个格子 `OPEN(i)`，环境更新后进入下一步。

---

## 1) 输入表征

设棋盘大小 $H\\times W$，展平后 $N=H\\cdot W$，隐藏维度 $d$。对每个格子构造一个 token；另外附加 $n_g$ 个**全局 token**（可学习，随层动态更新）。

**离散嵌入（相加）：**
- `tile_id \\in \\{0..8, X, U, F\\}`：已翻开的数字 $0..8$、爆雷 $X$（可选）、未开 $U$、旗 $F$（可选）→ `Embedding(12, d)`  
- `row_id, col_id`：二维位置 → `Embedding(H,d)`、`Embedding(W,d)`

**数值特征（拼接后线性映射到 $d$）：**
- `frontier`：是否为“前沿未开”（邻域存在数字）  
- `adj_unknown`、`adj_flag`：8 邻域中未开/旗的数量，归一化到 $[0,1]$  
- `is_number`：是否数字格  
- `digit_value/8`：数字值归一化（非数字为 0）  
- `residual/8`：对数字格，$\\max(0,\\text{digit}-\\text{邻旗数})/8$，其余为 0

> 上述 6 个通道组成原始特征 $\\in\\mathbb{R}^{B\\times N\\times C}$，经 `Linear(C→d)` 投到 $d$ 并与离散嵌入相加。

**可选：二维相对位置偏置**  
共享的加性注意力偏置 $b\\in\\mathbb{R}^{T\\times T}$（$T=N+n_g$），细微地鼓励近邻交互（例如 $b_{ij}=-0.1\\log(1+\\lVert\\Delta r\\rVert_1+\\lVert\\Delta c\\rVert_1)$）；对涉及全局 token 的行/列置 0。

---

## 2) 编码器：Decoder-Only + 动态全局 token

- **拼接序列**：`[cells(1..N), globals(1..n_g)]`，每层都让二者通过**同一个自注意力**互相混合（无局部窗口、无 cross-attn）。  
- **结构**：$L$ 层 Pre-LN Transformer Block（自注意力 + MLP），可选 dropout/stochastic depth；最终再做 LayerNorm。  
- **动态全局**：全局 token 随层更新，聚合整盘信息；下游头通过 $\\bar G=\\mathrm{mean}(G')$ 使用它。

---

## 3) 自回归输出（OPEN-only 指针）

**指针打分（可加性指针）：**
$$
s_i=v^\\top\\tanh\\!\\left(W_eE_i+W_g\\bar G\\right)+u^\\top E_i+c,\\quad
\\bar G=\\mathrm{mean}(G')\\in\\mathbb{R}^d
$$

**掩码 softmax（只在未开集合 $S$ 上）：**
$$
\\pi(i\\mid o)=\\mathrm{softmax}\\big(s_i+\\log\\mathbb{1}[i\\in S]\\big)
$$

**附加头：**
- **价值**：$V(s)$ 由 $[\\mathrm{mean}(E),\\bar G]$ 经 MLP 得到；  
- **地雷概率**：$p_{\\text{mine}}(i)=\\sigma(w_m^\\top E_i)$（仅对未开格有意义）。

---

## 4) 训练目标

- **行为克隆（BC）**：教师给出任一安全格 $i^\\*\\in S$，$\\mathcal{L}_{\\text{BC}}=-\\log\\pi(i^\\*\\mid o)$；对“多解”用**均匀标签**平滑。  
- **强化学习（PPO）**：标准 PPO 策略损失 + 价值回归 + 熵正则，注意对**合法动作集合**归一化。  
- **一致性正则（可选）**：
  $$
  \\mathcal{L}_{\\text{cons}}=\\sum_{j\\in\\text{数字格}}\\Big\\lvert\\sum_{i\\in\\mathcal{N}(j)}p_{\\text{mine}}(i)-\\text{digit}(j)\\Big\\rvert
  $$
- **潜在形奖励（可选）**：$\\phi(s)=-\\sum_{j\\in\\text{数字格}}\\big\\lvert\\sum_{i\\in\\mathcal{N}(j)}\\hat p_i-\\text{digit}(j)\\big\\rvert$，用 $\\Delta\\phi$ 提升样本效率。

---

## 5) 推理循环（每步一次前向）

```text
while not terminal(board):
  logits, value, p_mine = model(board)
  logits[board != U] = -inf                  # 只允许未开
  i = argmax_or_sample(softmax(logits/τ))   # 自回归选格
  board = env.open(i)                        # 环境更新（含 0 扩散）
  # 可选：逻辑插旗 -> 从候选集合里剔除 p_mine>τ_flag 的格，但不更改环境
```

---

## 6) 形状与接口

- 输入：`board ∈ ℤ^{B×H×W}`，值域 `{0..8, 9(X), 10(U), 11(F)}`  
- 输出：`logits ∈ ℝ^{B×N}`（非 U 位置已置 `-inf`）、`value ∈ ℝ^{B}`、`p_mine ∈ [0,1]^{B×N}`  
- 复杂度：每层 $O((N+n_g)^2·d)$；无局部模块，代码更精简、数值更稳。

---

## 7) 推荐超参

- `d=256/384/512`，`L=8–12`，`heads=8`，`n_g=4–8`，`dropout=0–0.1`  
- PPO：$\\gamma=0.995$，GAE $\\lambda=0.95$，$\\epsilon=0.2$，$c_v=0.5$，$c_e=0.01$  
- 课程：从 $9\\times9$、低密度起步，逐步扩到 $30\\times16$ 及更高密度；8 种旋转/翻转增强
