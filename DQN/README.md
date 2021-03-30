环境：
python3.8.8
tensorflow-gpu 2.4.1
gym 0.18.0

main.py是玩gym里经典小游戏的DQN基础模型。
运行时短暂的未响应是正常现象。

针对DQN的改进：
1. target network
最初训练DQN时，我们是用on-policy的，即每次都用同样的网络生成Q(s_t,a_t)和maxQ(s_t+1,a_t+1).数学上，E(max(Q))>=max(E(Q))，而我们想最大化目标函数，且用很多个采样，就相当于是在求maxQ下的期望E。我们想做的是让Q的期望最大，并不是让Q最大，所以Q总是高估。为了让网络正常，我们选择用历史的Q求maxQ，。当然，每步内的训练成果也得学习到Q表上，所以就记那个固定的Q表网络为target network, 训练的Q表照常训练，只不过每k步就把最新的训练Q表更新到target network上。

2. double Q-learning:
这个设计想解决初始Q值估计不准的问题，target network虽然提供了稳定的action决策来源，但终究没有提供最可靠的Q值，可靠的Q值应该来源于最新训练的Q表。
所以该double Q-learning在取maxQ(s,a)时，a从target network中获得，Q从最新的Q网络中获得。公式看起来更方便：y = r + Gamma*Q(s,argmax_(a)Q'(s,a))

3. PRIORITIZED EXPERIENCE REPLAY(2016)
2016年的论文。DQN并不是每次都从头生成新的训练数据，而是建一个队列deque，把所有经历过的（state，action, next_state, reward, done(代表游戏是否结束)）作为一个时刻的元组存到里面。装满了就把最老的数据剔除。而训练的数据是从deque里随机抽几十例。这里最好先把代码读懂，语言描述实在匮乏。
这些数据里当然有些数据是更有教育意义的，也就是时序差分(TD-error)的值更大。该值(TD-error)的计算方法在论文第5页算法流程第11行。给其加上绝对值，重命名为priority.
根据priority，加上经典的归一化函数转换为概率。
这确实可以加快我们训练的效率，但终究是揠苗助长，改变数据均匀分布就等于人为产生了bias偏差。所以在这里再引入一个压制其效果的方法，在论文3.4ANNEALING THE BIAS.该公式让P大的时候w小，以压制过于强效的训练数据。在β=1时相当于原始的均匀采样，所以β会在训练时逐渐线性缩小到1.

4. Dueling Network(2016)
同2016年论文。在atira游戏上，因为不是简单的gym上的小游戏，所以肯定要用CNN捕捉画面的。
传统DQN直接在卷积后接一个MLP输出Q值，但Dueling Network把卷积维度提高一维，把多出来的一维单独拿出来算了个单值V(s)，再结合传统Q值的层再接一层网络，输出维度不变，仍是action数量的大小。
作者臆想那个单值V是所有action的共有价值，而增值价值是由动作触发的，价值总和是两部分的综合计算。
解释的有些牵强，但看在效果好的出奇的份上，且这个改动对任何已有模型都可以方便添加，故值得记录。

5. Noisy Network（2017）
在atari游戏上，如果按画面帧作为state,那可是有数不清个state，因此即使action就那几个，也没法让网络有机会在所有state下尝试所有action，进而可能没机会尝试到某一state下的最优action。这是网络训练的老毛病，找到了一个局部最优，就很难再跑出来。
经典做法是在选择action时按概率来，而不是直接选最优。你可以根据上一时刻的Q值，计算每个action的归一化概率。
本篇论文给出另一种做法。既然我是按照Q最优选动作，那我加上一个扰动，让Q值不稳定，就有机会让次一级的Q值跑到第一，然后被选择。
具体就是在Q值网络的w和b上加高斯噪声。
(题外话)论文的14页就是加不加Noisy的效果得分，第一列是人类得分参考。我在里面看到一款叫pitfall的游戏，迷宫寻宝类，这种需要记地图而且流程极长难以探索的游戏，让所有网络都没办法。而像atlantis这种画面固定的简单飞船射击游戏，得分比人类高的出奇。
加噪声不是随便加的。如果事先知道VAE网络，你就能知道如何用网络训练均值U和方差θ，再结合标准正态分布，获得一个某正态分布的随机采样，而且该分布由被网络定义，所以又能参与训练，寻找到符合环境的分布。
对于w和b，该论文就用这种方法对二者采样，靠训练U和θ控制二者的值。对应公式在第3页公式6.

6. Rainbow: Combining Improvements in Deep Reinforcement Learning
看名字就知道，这篇论文做了个大融合，把所有好用的方法都用上，然后起个名叫rainbow。
正因如此，这篇论文算是个综述，可以从里面找到很多经典论文。
例如下一篇。

7. Distributional Reinforcemet Learning with Quantile Regression
强化学习最初建模就是把对未来的reward看作期望。而本篇把reward看作某个分布。
直接看最后一页效果，可以看到QR-DQN(就是本论文模型)有0型和1型。二者在几款游戏上的得分是之前模型的好几倍。说明该模型在某些场景下是好用的。
【TODO】