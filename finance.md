[toc]
## Leetcode 
**代码随想录**
### 数组

#### 3.滑动窗口
```python
# pseudo code
left = 0
ans = 0 # or bound
s = 0 # sum

for right, x in enumerate(nums):
  iterate for s and x
  while not cond:
    s -= nums[left]
    left += 1
  iterate for ans
return ans # if ...
```


209.长度最小的子数组
**求最短**
```python
left, n = 0, len(nums)
s, ans = 0, n+1 # ans > n

for right, x in enumerate(nums): # x: nums[right]
  s += x # s: nums[left] +...+ nums[right]
  while s >= target:
    ans = min(ans, right-left+1) # s satisfy condition
    s -= nums[left] # minus nums[left]
    left += 1

return ans if ans <= n else 0
```

**越短越合法**
713. 乘积小于 K 的子数组
```python
# if prod[left, right] < k
left, ans = 0, 0
n = len(nums)
prod = 1

if k <= 1: # else prod >= k will get error
  return 0

for right, x in enumerate(nums):
  prod *= nums[right]
  while prod >= k:
    prod /= nums[left]
    left += 1
  ans += right - left + 1
return ans
```

3. 无重复字符的最长子串
```python
left = 0
ans = 0
cnt = Counter() # hashmap char int

for right, c in enumerate(s):
  cnt[c] += 1 # don't use dict
  while cnt[c] > 1: # guarantee left <= right
    cnt[s[left]] -= 1
    left += 1
  ans = min(ans, right - left + 1)
return ans
```






#### 4.二分查找

判断是否满足条件(蓝), 
```python
# pseudo code
l, r = -1, len(nums) # 即满足条件的左右端, 可更改
while l+1 != r: 
  m = (l+r)// 2
  if m ** 2 <= x:
    l = m
  else:
    r = m
  return l or r # 取决于输出最大值or最小值
```






### 双指针法
#### 快慢指针
27.移除元素

```python
# pseudo code



```




### 哈希表
三大结构：list, set, dict
#### 242.有效的字母异位词
- 使用List
时间复杂度 O(n), 空间复杂度 O(1) (固定长度24).
idea: 将字母存到数组中，记录每个字母出现数量.
```python
# pseudo code


```























- 期权，期货(有期限)
沪金2503
- 期货：有期限，约定在未来特定时间和地点，以预先确定的价格买卖特定数量的标的物。(买方必须买入，卖方必须卖出)
- 期权：有期限，买方有权利，卖方有义务
  - 买方成本 = 期权费（权利金）
  - 期权费不可退回
  - 卖方需要缴纳保证金
- 现货：即时交易

- 上证50(IH)：上海证券交易所挑选的50只大型股票，行业龙头+保险+银行
- 沪深300(IF)：上海和深圳证券交易所挑选的300只大型股票

情绪价值
- 中证500(IC)：中证指数公司挑选的500只中型股票
- 中证1000(IM)：中证指数公司挑选的1000只小型股票

策略
- 做多：买入，低买高卖
- 做空：卖出，高卖低买

- 多头：看涨
- 空头：看跌


- 衍生品赌方向风险极大
- 教科书基本都是期货？



$ \Gamma= \frac{\partial^2 P}{\partial S^2}$

$ \Delta= \frac{\partial P}{\partial S}$ # 看对挣钱

$ \Theta= \frac{\partial P}{\partial t}$

$ Vega= \frac{\partial P}{\partial \sigma}$

$ \rho= \frac{\partial P}{\partial r}$

$ \lambda= \frac{\partial P}{\partial \lambda}$

pyQt
tkinter

backtrader:
- 回测
- 实盘


‌波指（波动率指数）：衡量资产价格波动的剧烈程度
- 波起：买；波落：卖

中性对冲


衍生品交易是资金盘，需要顺势

‌CTA策略（Commodity Trading Advisor Strategy）‌是一种通过量化模型和算法对期货、期权等交易品种的价格走势进行分析和预测，以获取收益的量化交易策略. CTA策略不依赖于对具体商品或市场的基本面分析，而是侧重于利用价格波动和市场趋势来进行交易.


- 多头头寸：买入+持有资产





期货特性：浮盈加仓，一把烧光








苹果期权：开盘7585 - 收盘7732 (日内)

标的价格：期权、期货、差价合约（CFD）等金融衍生品所挂钩的基础资产的市场价格

价差 = 标的价格 - 行权价

价差小：虚实变化的可能性大
权利金小：倍数大

对付大幅下跌：卖 * 1，买 * 2

债券 -- 利率债 T, TS, TL(30年国债)
期权是一个对冲风险管理工具

## 策略
1. 双买（买入跨式组合 / Long Straddle）
定义：同时买入相同行权价、相同到期日的看涨期权（Call）和看跌期权（Put）。

- 适用场景
预期标的价格会大幅波动，但不确定方向（如财报、重大新闻前）。
押注波动率上升（隐含波动率IV可能上涨）。



2. 双卖（卖出跨式组合 / Short Straddle）
定义：同时卖出相同行权价、相同到期日的看涨期权（Call）和看跌期权（Put）。

- 适用场景
预期标的价格波动小，横盘震荡。
赚取时间价值衰减（Theta收益），赌隐含波动率下降。


