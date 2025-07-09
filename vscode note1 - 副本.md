# 欢迎使用Vscode

> 记录一些vscode + Latex +SumatraPdf的杂例.

### 一些有用的快捷键

| 快捷键  |   功能 | 
| :-------- | ----------:|  
| Alt+Z     | 字体适应大小 | 
| Ctrl+Alt+J| 定位到pdf位置 | 
| Ctrl+Alt+B |   编译  |
|  eqt |  \begin{equation} |
|  dis | \displaystyle 将行内公式放大 | 
| Ctrl+Shift+B | 文本加粗Open Keyboard Shortcuts (JSON)   | 
| abf | 加粗$\mathbf{a}$ |
| Shift+Tab | 回到上一个对象 |
| xsb | x_{2} |
| xtp | x^{2} |
| align | 居中有(1)|
| aligned | 居中无(1)|
| gather | 只居中不对齐 |
| (Shift)+Enter | 搜索关键词下一个 |
| Fn + → | 跳转到最右边 |
| Ctrl+shift+N | google 进入无痕浏览 |
|||
|||

$$
\begin{gathered}
w \\ sxw
\end{gathered}
$$


快捷键`Ctrl + Shift + P`，再输入`Open Snippets Directory`，即可打开`all.hsnips`

### 未删除 lor 的原代码
``` JavaScript
priority 200
snippet `(?<!\\)(cap|cup|land|lor|lnot|oplus|ominus|otimes|sqcap|sqcup|vdash|models)` "logic operator" iAm
\\``rv = m[1]`` 
endsnippet
```




#### 本月想要完成的任务
- [ ] 记住代码（做数学笔记），查看计算机的第0课
- [ ] 矢量画图
- [ ] mathmatica 算抽象矩阵运算
- [ ] matlab 
- [ ] LaTeX 参考文献环境
- [ ] 入门python，熟练调用相关的库
- [ ] 神经网络 + 机器学习？



## ex7
``` python
input print(end1 + end2, end=' ')
print(end3)
output we a
# 以空格结尾，没有换行
```

<!-- ## Chapter 1. Error
Read and read.
We talk about Chapter 1-5.
direct methods / iterative methods
$\Vert x^k - x^ \ast \Vert $

Def (Round Error) $e = \Vert x^k - x^ \ast \Vert $
Def (Absolute Error) $e = \Vert x^k - x^ \ast \Vert $ -->


#### Motivation of $\sigma-$algebras
One of the key issues in Stochastic Calculus for finance is modelling the path of stock prices. In the real world stock prices can change extremely rapidly, but their changes are always discrete. That is, we can 'zoom in' on time and eventually see discrete jumps between stock prices over finite periods of time as new trades are made.

However because these changes are very rapid it is appropriate, mathematically, to model stock prices as if they change continuously. That is, if we attempt to 'zoom in' on time we will also see ever more 'wiggliness' of the underlying stock path. This is due to the fractal nature of Brownian motion, a common model for the evolution of stock prices in finance.

This implies that we need to consider uncountable sets of events if we are to begin discussing the concept of the probability of a stock price increasing or decreasing in a subsequent time increment.

However once we introduce uncountable sets and an attempt to 'measure' them in some fashion (of which assigning a probability to an event is an example) then we need to be sure that we can do so unambiguously.

Otherwise we may end up in a situation where we can legitimately (based on two separate mathematical proofs, say) assign a different value of probability to the same event.

Thus in order to unambiguously assign a probability to an event it is necessary to somehow exclude certain events from those which we are assigning probability to if they admit ambiguous probability values.

In essence we need to get rid of any sets that do not have a 'sensible' probability measure. This is where the concept of $\sigma-$algebras come in.

考虑不可数的事件时，我们没办法给每一个事件都赋予概率测度(如果是测度是$\sigma$可加+平移不变，则不能给[0,1]的全体幂集赋予测度，概率程度应该是同理)，因此我们只考虑Borel集的原像生成的$\sigma-$代数，则可以在该$\sigma-$代数上定义概率测度.