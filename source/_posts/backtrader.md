---
title: backtrader
date: 2024-07-25 12:18:39
categories: [Economics&Finance, Investment]
tags: [quant, investment]
---

# Summary

[Backtrader](https://www.backtrader.com/) 是一个可同时用于量化交易回测（基于事件）与实盘交易的开源框架。

# 控制器

在 Backtrader 中，控制器叫做 `Cerebro` ，是一个调度中心，控制着数据、策略、执行与评估部分。

```python
import backtrader as bt

# 实例化 Cerebro
cerebro = bt.cerebro()

# TODO: 获取数据实例
# 将数据添加进控制器中
cerebro.adddata(data)

# TODO: 编写策略类
# 将策略添加进控制器中
cerebro.addstrategy(XXXStrategy)

# TODO: 编写头寸类
# 设置交易的头寸
cerebro.addsizer(XXXSizer)

# TODO: 获取经纪商实例
# 设置交易的经纪商
# 如果不写默认用 BackBroker ，这是一个本地回测用的虚拟券商
cerebro.setbroker(xxxbroker)

# TODO: 编写评估指标类
# 将评估指标添加进控制器
cerebro.addanalyzer(XXX)

# TODO: 编写观察器类
# 将观察器添加进控制器，观察器和画图相关
cerebro.addobserver(XXX)

# 运行控制器
res = cerebro.run()
# 画图
cerebro.plot()
```

整个过程的模拟都是由 `cerebro` 对象去完成的。

# 数据

## DataFeeds

在 Backtrader 中， `datafeeds` 是一个用于导入数据的模块，可以将表格数据转化为一个 `DataSeries` 对象。

### DataSeries

`DataSeries` 是一个表格的形式的类，这个表格由 7 列组成：

- datetime - 用于记录一个时段的开始时间。
- open - 此时段的开盘价。
- high - 此时段的最高价。
- low - 此时段的最低价。
- close - 此时段的收盘价。
- volume - 此时段的成交量。
- openinterest - 此时段未平仓的合约量。

创建 DataFeeds 的方法有很多种，比较常用的就是通过 pandas 的 DataFrame 创建。

```python
data = bt.feeds.PandasData(dataname=df)
```

在创建 `DataFeeds` 的时候，其实不需要保证原始数据里面具有上面提到的 7 列。假设出现这种情况，会在 `cerebro` 运行时报错。

### LineSeries & Lines

看源码可以发现 `DataSeries` 类继承了 `LineSeries` 类（可以看作一个没有指定这些列的类），而 `datetime`, `open`... 这些就是 `Lines` 。

因此可以理解为 `LineSeries` 相当于 pandas 中的 DataFrame ，而 `Lines` 则相当于 pandas 中的 Series 。

# 策略

## Strategy

### 交易策略

```python
class ExampleStrategy(bt.Strategy):
    
    params = (
        (...,...),
    )
    
    def log(self, txt: str, dt=None):
        """
        设定策略打印日志的格式
        """
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}, {txt}")
        
	def __init__(self):
        """
        初始化属性、计算指标等
        """
        pass
    
    def start(self):
        """
        在开始前调用
        """
        pass
    
    def prenext(self):
        """
        `nextstart()` 之前策略准备阶段
        """
        pass
    
    def nextstart(self):
        """
        `next()` 之前，策略运行的第一个时点
        
        只运行一次
        """
        pass
    
    def next(self):
        """
        策略正常运行阶段，每根 Bar 都调用
        """
        pass
    
    def stop(self):
        """
        策略结束，对应最后一根 Bar 结束

        告知系统回测已完成，可以进行策略重置和回测结果整理了
        """
        pass
    
   
	# 以下都是设定打印日志的具体内容
    def notify_order(self, order):
        """
        通知订单信息
        """
        pass

    def notify_trade(self, trade):
        """
        通知交易信息
        """
        pass
    
    def notify_cashvalue(self, cash, value):
        """
        通知当前资金和总资产
        """
        pass
    
    def notify_fund(self, cash, value, fundvalue, shares):
        """
        返回当前资金、总资产、基金价值、基金份额
        """
        pass
    
    def notify_store(self, msg, *args, **kwargs):
        """
        返回供应商发出的信息通知
        """
        pass
    
    def notify_data(self, data, status, *args, **kwargs):
        """
        返回数据相关的通知
        """
        pass
    
    def notify_timer(self, timer, when, *args, **kwargs):
        """
        返回定时器的通知
        
        定时器可以通过函数 `add_time()` 在 `__init__()` 里添加
        """
        pass
```

制定好策略之后，需要将策略添加进 `cerebro` ：

```python
cerebro.addstrtegy(ExampleStrategy)
```

### 交易信号

除了编写 `Strategy` 类来编写策略，还可以编写 `SingalStrategy` 类来编写策略。这两者的区别在于 `SignalStrategy` 类**不需要调用交易函数（ buy, sell ）**，`cerebro` 会自动将 `SingalStrategy` 里的信号 `signal` 转换为交易指令。

```python
# 注意这里是继承的 `Indicator`
class ExampleSignal(bt.Indicator):
    
    # 声明 xxx 线，交易信号放在 xxx 这个 Lines 上
    lines = ('signal',)
    params = (xxx,)
    
    def __init__(self):
        self.lines.signal = ...
```

这里 `lines.signal` 的取值决定了信号类型：

- `signal` 大于 0 - 对应多头 long 信号
- `signal` 小于 0 - 对应空头 short 信号
- `signal` 等于 0 - 不发指令

制定好信号后，需要将信号添加进 `cerebro` ：

```python
cerebro.add_signal(
    sigtype = ,		# 配置触发信号时的交易类型
    sigcls = ,		# 传入交易信号类
    ...				# 传入交易信号参数
)
```

其中可以配置的交易类型有以下等：

```python
# 多空开仓信号
# 多头信号会先平仓空头，再开仓多头
# 空头信号会先平仓多头，再开仓空头
bt.SIGNAL_LONGSHORT

# 多头开仓信号
# 多头信号开仓，空头信号平仓
bt.SIGNAL_LONG

# 空头开仓信号
# 空头信号开仓，多头信号平仓
bt.SIGNAL_SHORT

# 多头平仓信号
# 空头信号平仓
bt.SIGNAL_LONGEXIT

# 空头平仓信号
# 多头信号平仓
bt.SIGNAL_SHORTEXIT
```

由于交易信号指标通常只是技术指标之间进行加减得到，在技术指标完全已知的情况下，很容易连续不断地生成交易信号。这里会有两种特殊情况：

- 订单累计 - 即使已经在市场上，信号也会产生新的订单，进而增加头寸。
- 订单并发 - 新订单会并行执行，而不是等待其他订单执行完毕。

```python
# 是否允许订单累计发生
cerebro.signal_accumulate(True)
# 是否允许订单并发发生
cerebro.signal_concurrency(True)
```

## Indicator

### 预计算指标

如果想要导入预计算指标，可以通过继承 `DataSeries` 的方式添加 `Lines` （在数据部分已经提到，一个 `Lines` 对象就是一列，这里相当于添加指标列）。

```python
class PandasDataExtend(bt.feeds.PandasData):

    lines = (
        'indicator1',
        'indicator2',
    )
    
    # -1 表示自动按列名匹配数据
    params = (
        ('indicator1', -1),
        ('indicator2', -1),
    )
    
# data 是一个 dataframe
# calculate_indicator1, calculate_indicator1 都是计算指标的函数
data['indicator1'] = data.apply(lambda x: calculate_indicator1(x), axis=1)
data['indicator2'] = data.apply(lambda x: calculate_indicator2(x), axis=1)
datafeed = PandasDataExtend(dataname=data)
cerebro.add(datafeed)
...
```

### 实时计算指标

#### 原生指标

[Indicators - Reference - Backtrader](https://www.backtrader.com/docu/indautoref/)

```python
bt.indicator.XXX
```

#### TA-lib 库指标

[Indicators - ta-lib - Reference - Backtrader](https://www.backtrader.com/docu/talibindautoref/)

```python
bt.talib.XXX
```

#### 自定义指标

- `__init__()` - **对整条 `Lines` 进行计算，运算结果也以整条 `Lines` 返回。**也就是说，并不是只有最开始的数据点。
- `next()` - 对数据点进行运算。每个 bar （ DataFeeds 的每行）会运行一次。
- `once()` - 这个方法只运行一次，但是需要从头到尾循环计算指标。

```python
class ExampleIndicator(bt.Indicator):
    
    # 指标暂存的数据
    lines = (
        'temp1',
        'temp2',
    )
    
    # 指标的可选参数
    params = (
        ('x', 10),
    )
    
    def __init__(self):
        """
        可选
        """
        pass
    
    def next(self):
        """
        可选
        """
        pass
    
    def once(self):
        """
        可选
        """
        pass
    
    # 画图用的，如果不需要画图可以不设置
    plotinfo = ...
	plotlines = ...
```

## ？Sizer

## Order

### 订单类型

- Order.Market - 市价单。回测时按**下一个 Bar 的开盘价**成交。
- Order.Close - 也是市价单。只不过回测时按**下一个 Bar 的收盘价**成交。
- Order.Limit - 限价单。对于买方来说，如果下一个 Bar 的开盘价更低，以下一个 Bar 的开盘价成交；如果开盘价更高，但是订单价格限定在最高最低价之间，那么以订单价格成交；否则不成交。
- Order.Stop - 止损单。一旦达到或超过特定的止损价格后，止损单即变为市价单来买或买证券或商品。止损单不担保以特定的价格执行。
- Order.StopLimit - 和止损单类似。一旦达到或超过特定价格后启动，不过是以限价单的方式启动。
- Order.StopTrail - 跟踪止损单。一种会自动调整价格的止损单。
- Order.StopTrailLimit - 跟踪止损限价单，启动后以限价单的方式执行。

### 创建订单

```python
"""
这里的 self 都是指策略
"""

# 买入/做多 long
self.order = self.buy(
	data = xxx,				# 默认为当前策略的第一个数据集 self.datas[0] 创建订单
    size = xxx,				# 默认调用 getsizer() 获取头寸
    price = xxx,			# 适用于限价单，止损单和止损限价单
    plimit = xxx,			# limit price ，仅适用于 StopLimit
    exectype = Order.XXX,	# 订单类型，默认市价单
    valid = xxx,			# 订单有效期，默认撤单前有效
    tradeid = xxx,			# 交易编号
)

# 卖出/做空 short
self.order = self.sell(...)	# 和 buy 参数一样

# 平仓 cover
self.order = self.close(...)# 和 buy 参数一样

# 目标下单函数
# 按目标数量下单，多退少补
self.order = self.order_target_size(target=size)
# 按目标金额和当前金额的情况决定下单方式
self.order = self.order_target_value(target=value)
# 按当前账户总资产目标百分比下单，多退少补
self.order = self.order_target_percent(target=percent)

# 组合订单
self.order = self.buy_bracket()
self.order = self.sell_bracket()
```

`buy_bracket()` 用于long side 的交易场景，买入证券后，在价格下跌时，希望通过止损单卖出证券，限制损失；在价格上升时，希望通过限价单卖出证券，及时获利，通过 `buy_bracket()` 可以同时提交上述 3 个订单，而无需繁琐的调用 3 次常规交易函数。

`sell_bracket()` 用于short side 的交易场景，卖出证券做空后，在价格上升时，希望通过止损单买入证券，限制损失；在价格下降时，希望通过限价单买入证券，及时获利，`sell_bracket()` 也是一次同时提交上述 3 个订单 。

只当在主订单执行后，止损单和止盈单才会被激活，而且是同时激活；如果主订单被取消，止盈单和止损单也会被取消；在止盈单和止损单激活之后，如果取消两者中的任意一个，那另外一个也会被取消。

### 取消订单

```python
# 通过 broker 取消
broker.canel(order)
```

### 关联订单

通过在 `buy()` 或者 `sell()` 中给 `oco` 参数传入订单实例来实现订单的关联。

### 订单状态

- Order.Created - 订单已经被创建。
- Order.Submitted - 订单已经被传递给经纪商。
- Order.Accepted - 订单已经被经纪商接收。
- Order.Partial - 订单已部分成交。
- Order.Complete - 订单已成交。
- Order.Rejected - 订单已被经纪商拒绝。
- Order.Margin - 执行该订单需要追加保证金。
- Order.Cancelled - 确认订单已经被撤销。
- Order.Expired - 订单已到期。

# 执行

## Broker

具体到不同的券商，会有不同的初始本金以及交易费用（包括佣金、平台费、SEC 收取的费用等等）。对于回测来讲，还可以考虑滑点（ slippage ）以及市场影响（ market impact ）。这些统称为交易条件。

这些可以通过两种方式进行设置：

1. 将交易条件作为参数的方式实例化 `Broker` 类，生成新的实例，然后在 `cerebro` 里设置这个实例。
2. 调用 `broker` 对象的 `set_xxx` 方法去修改交易条件（通过 `get_xxx` 可以查看交易条件）。

### 交易费用

`BackBroker` 中有一个 `commission` 参数，用来全局设置交易手续费。

#### 默认设置模板

```python
broker.setcommission(
    commission=...,		# 佣金费率，根据 commtype 确定百分比还是固定
    margin=None,		# 期货保证金，只有 `stocklike=False` 时生效
    mult=...,			# 计算期货保证金的乘数
    commtype=None,		# 佣金类型 `COMM_PERC` 为百分比 `COMM_FIXED` 为固定 `None` 根据 margin 取值确定
    percabs=True,		# 百分比格式 `True` 表示 0.xx 这种格式 `False` 表示 xx% 这种格式
    stocklike=False,	# 是否为股票模式
    interest=...,		# 空头头寸地年化利息
    interest_long=False,# 多头头寸地年化利息
    leverage=...,		# 杠杆比率
    automargin=False,	# 自动计算保证金
    name=None,			# 取值为 `None` 默认作用于全部数据集
)
```

这种方式其实不太灵活，还是建议自定义设置。

#### 自定义设置

要自定义交易费用设置，需要写自己的 `Commission` 类：

```python
class ExampleCommission(bt.CommInfoBase):
    
    # 对应 setcommission 中的参数，也可以增添新的全局参数
    params = (
        (xxx, xxx),
    )
    
    def _getcommission(self, size, price, pseudoexec):
        """
        自定义交易费用计算方式
        """
        pass
    
    def get_margin(self, price):
        """
        自定义佣金计算方式
        """
        pass
```

然后将自定义的交易费用类**实例化**，再传递给 `broker` ：

```python
example_comm = ExampleCommission(...)
# `name` 参数用于指定该交易费用适用标的
broker.addcommissioninfo(example_comm, name='xxx')
```

### 回测

#### 资金与持仓管理

```python
# 设置初始资金
broker.set_cash(xxx)

# 获取当前可用资金
broker.get_cash()

# 添加资金
# 正数表示添加，负数表示减少
broker.add_cash()

# 获取当前总资产
broker.get_value()

# 获取当前持仓情况， getposition() 需要指定具体数据集
# 持仓数量
broker.getposition(data).size
# 持仓成本价
broker.getposition(data).price
```

#### 滑点管理

滑点是**回测**里应该考虑的额外成本，因此只有 `BackBroker` 实现了相关方法。

```python
# 以百分比的方式估计滑点及市场影响
broker.set_slippage_perc(...)

# 以固定金额估计滑点及市场影响
broker.set_slippage_fixed(...)
```

有关滑点的其他参数如下：

- `slip_open` - bool - 是否对开盘价做滑点处理。
- `slip_match` - bool - 是否将滑点处理后的新成交价与成交当天的价格区间最低价和最高价之间做匹配，如果为 True ，则根据新成交价重新匹配调整价格区间，确保订单能被执行；如果为 False ，则不会与价格区间做匹配，订单不会执行，但会在下一日执行一个空订单。
- `slip_out` - bool - 如果新成交价高于最高价或低于最高价，是否以超出的价格成交，如果为 True ，则允许以超出的价格成交；如果为 Fasle ，实际成交价将被限制在最高最低价格区间内；默认取值为 False 。
- `slip_limit` - bool - 是否对限价单执行滑点，如果为 True ，即使 slip_match 为 Fasle ，也会对价格做匹配，确保订单被执行；如果为 Fasle ，则不做价格匹配；默认取值为 True 。

#### 成交量限制

这里的成交量限制并不是分配仓位，而是一种对流动性的估计。

```python
# 设置最大固定股数成交量 xxx
filler = bt.broker.fillers.FixedSize(size=xxx)

# 设置最大固定百分比成交量 xxx
# 百分比为当前 Bar 总成交量的百分比
filler = bt.broker.fillers.FixedBarPerc(perc=xxx)

# 设置最大流动百分比的成交量 xxx
# 其中 minmov 是可选参数，默认为 None
filler = bt.broker.fillers.BarPointPerc(perc=xxx1， minmov=xxx2)

broker.set_filler(filler=filler)
```

前两种 `FixedSize` 与 `FixedBarPerc` 都挺好理解，第三种 `BarPointPerc` 的计算过程为：
$$
\begin{equation}
\begin{aligned}
parts&=\frac{high-lowe+minmov}{minmov}\\
volume_{max}&=\frac{volume_{bar}}{parts}\times perc\\
&=\frac{minmov}{high-low+minmov}\times volume_{bar} \times perc
\end{aligned}
\end{equation}
$$
可以看到实际上就是在当前 Bar 中，如果最低价最高价价差比较大，对成交量的限制就比较严格；如果最低价最高价价差比较小，限制则比较宽松。

#### 交易时机管理

Backtrader 默认使用当日收盘后下单，次日以开盘价成交这种模式。但是也有当日下单，以当日价格（开盘价/收盘价）成交的选项，这种情况下是有可能使用未来数据的（比如下单的时候并不知道收盘价，但是却用收盘价来计算了订单数量和金额）。

```python
# 以当日开盘价下单
broker.set_coo(True)

# 以当日收盘价下单
broker.set_coc(True)
```

但是值得注意的是，这里以当日开盘价下单其实意思是将下单时间放到下个交易日（本来是当前交易日下单，下个交易日成交）。

这里需要将交易的逻辑写在 `next_open()`, `nextstart_open()`, `prenext_open()` 中，而不是 `next()` 中。

### ？实盘



# 评估

## Analyzer

### 内置分析器

对于要评估的指标，在运行 `cerebro` 之前需要添加：

```python
# 添加分析器（指标）
cerebro.addanalyzer(
    bt.analyzers.XXX,
    _name="_XXX",
）

# 运行
result = cerebro.run()
    
# 提取结果
xxx_result = result[0].analyzers._XXX.get_analysis()
```

### 自定义分析器

```python
class ExampleAnalyzer(bt.Analyzer):
    
    params = (
        (xxx, xxx)
    )
    
    def __init__(self):
        """
        初始化属性、计算指标等
        """
        pass
    
    # 以下发生时点和自定义 `Strategy` 一样
    def start(self):
        pass
    
    def prenext(self):
        pass
    
    def nextstart(self):
        pass
    
    def next(self):
        pass
    
    def stop(self):
    
    # 与自定义 `Strategy` 一样的信息打印函数
    def notify_order(self, order):
        '''通知订单信息'''
        pass

    def notify_trade(self, trade):
        '''通知交易信息'''
        pass
    
    def notify_cashvalue(self, cash, value):
        '''通知当前资金和总资产'''
        pass
    
    def notify_fund(self, cash, value, fundvalue, shares):
        '''返回当前资金、总资产、基金价值、基金份额'''
        pass
    
    def get_analysis(self):
        pass
```

## ？Observer

