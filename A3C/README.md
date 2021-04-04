A3C:Asynchronous Methods for Deep Reinforcement Learning(2016)

A3C全称Asynchronous Advantage Actor-Critic，异步 优势 演员-批评家。
异步，意为该算法可以异步执行，异步是一种并行，有一种off-policy的感觉。
优势，意为优势函数A,这在以前的论文中已经有体现了。
演员-批评家，这是sutton在2000年的论文中的框架，actor网络给出动作，critic网络给出动作评判。

简介：AC，A2C，A3C，ACKTR.建议阅读下面给出的blog。
1. AC就是sutton的actor-critic。actor用于计算策略的动作分布，critic计算V(s)，两者共同作用于目标函数J(theta).
2. A3C是本文。
3. A2C是A3C的一个同步版本，A2就是因为把异步的字样拿掉了。官网描述它是deterministic variant of A3C[TODO]。A3C实现时，由于异步的原因，很难知道什么时候更新参数合适。
尝试同步后，发现竟然"gives equal performance".这个结论记录在了openAI的blog里。
blog地址： https://openai.com/blog/baselines-acktr-a2c/
4. ACKTR是加速采样效率的版本，也在上面的blog中，采样效率比A2C和TRPO更快，但比起A2C有计算量的增加，增加的不多，"only slightly more computation"。


A3C用另一种方法解决了on-policy的不稳定问题，而且还不是用off-policy或回放机制。
A3C把actor-critic复制好几份，然后各自进行下一次的on-policy的采样和训练，再把更新传给本体，本体整合后再把最新网络给各个复制。
以结果来看，这种做法防止了单次on-policy采样导致的探索性下降进而局部最优的情况，而且速度快多了。当然，这主要是并行的功劳。



