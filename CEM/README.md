CEM是2006年的算法，慢一点的半小时就能读完论文，
CEM是用来玩俄罗斯方块的游戏。

俄罗斯方块的状态很少，按照 Bertsekas, D. P. and Tsitsiklis, J. N. (1996).里的方法，只有22个(知道状态很少就行)。

现在我们知道，策略是可以用神经网络表示的。CEM采用单层神经网络，即一层W。
我们知道，W是个向量，但凡改一个参数，都是一个新的策略。而与最优策略相近的那些策略，效果也差不到哪里去。
于是萌生了一个想法：能否通过控制生成W和b的均值和方差，来定位最优策略？
正好，我们可以用各个状态的Ws累加作为策略W的评分。
具体做法如下：

1. 初始化一个标准正态分布Φ~N(0,1)。
2. 根据Φ，随机一批W，可能是100个。
3. 给每个state(记为s)计算价值V(s)=Ws。
4. 挑出前20个V最大的W，计算均值和方差，作为新的W正态分布。
5. 回到2。

收敛性未知，模型简单，但效果巨好。
据说很多项目测试可行性时，都用这个方法。

优点：不用求导
缺点：state一多就失灵了。state一多，相应的W参数就多，有时候一层不好拟合，还得多加几层，参数更多了，参数一多，对分布的计算是极大的压力。
所以复杂任务CEM处理不了.