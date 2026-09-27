import networkx as nx
import numpy as np
import random
import json

# 生成具有周期性边界的方格网络（N*N）
def square_lattice(N):
    G = nx.generators.lattice.grid_graph(dim=[1, N, N])
    # 构造周期性边界
    for i in range(N):
        G.add_edge((0, i, 0), (N - 1, i, 0))
        G.add_edge((i, 0, 0), (i, N - 1, 0))
    return G

def choose_nei(G, x, reputation):  # 以更高的概率选出声誉高的邻居
    nei = list(G.neighbors(x))
    re_list = []
    re_list_ = []
    for i in nei:
        re_list.append(reputation[i][-1])
    for i in range(len(re_list)):
        re_list_.append(re_list[i] / sum(re_list))
    ran = random.uniform(0, 1)
    index = 0
    s = re_list_[index]
    while index < len(nei):
        if 0 <= ran < s:
            break
        index += 1
        s += re_list_[index]
    return nei[index]

n = 10000
ETime = 10000
K = 0.1  # 噪音因素
lam_0 = 0.015  # PDG变为SDG的速率
mu_1 = 0.03  # SDG变为PDG的速率
lam_1 = 0.01  # SDG变为SHG的速率
mu_2 = 0.02  # SHG变为SDG的速率
avg = 5
# G = nx.barabasi_albert_graph(n, 3)  # 无标度网络
# G = square_lattice(32)    # 方格网络
G = nx.watts_strogatz_graph(n, 4, 0.15)  # 小世界网络
result = {}
rlist = np.arange(0, 1.05, 0.05)
blist = np.arange(1, 2.05, 0.05)
for r in rlist:
    r_f = []
    for b in blist:
        pom_0 = [[1, 0], [b, 0]]  # PDG
        pom_1 = [[1, 1 - r], [1 + r, 0]]  # SDG
        pom_2 = [[1, -r], [r, 0]]        # SHG
        fc = 0
        for average in range(avg):
            f_c = []
            state = {}  # 玩家状态字典 (PDG为0，SDG为1, SHG为2)
            strategy = {}  # 玩家策略字典
            next_change = {}  # 玩家状态改变时间
            next_changer_flag = {}  # 玩家在进行SDG时决定改变为PDG或SHG的标记
            reputation = {}  # 玩家的声誉
            max_re = 4
            delta = 0.04  # 声誉改变步长
            fun_f = lambda x: (len(x) - sum(list(map(lambda x: x[-1], list(x.values()))))) / len(x)  # 根据策略字典计算合作者比例
            # 初始化
            for i in G.nodes:
                next_change[i] = random.expovariate(lam_0)
                state[i] = 0
                strategy[i] = [0 <= random.uniform(0, 1) < 0.5]  # True为Defector,False为Cooperator
                reputation[i] = [random.normalvariate(2, 0.6)]  # 初始化每个玩家的声誉
            f_c.append(fun_f(strategy))
            num_pdg = n  # 进行PDG个数为1000
            num_sdg = 0  # 进行SDG个数为0
            num_shg = 0  # 进行SHG个数为0
            tnow = 0
            t_update = 0  # 玩家在整数时间更新策略
            pdg_list = [n]  # PDG个数
            sdg_list = [0]  # SDG个数
            shg_list = [0]  # SHG个数
            flag = 0
            cmd = 0  # 从0后开始统计从众者个数
            while tnow < ETime:
                print("\r" + "avg={},r={:.2f},b={:.2f},fc={:.3f},num_pdg ={},num_sdg={},num_shg={},t:{:.2f}/{:.2f}".format(
                        average, r, b, f_c[-1], pdg_list[-1], sdg_list[-1], shg_list[-1], tnow, ETime), end='')
                while t_update < tnow:  # 玩家在整数时间同步更新策略（开始博弈）
                    # 计算每个时间节点玩家的收益
                    Payoff = {}
                    for x in G.nodes:
                        Ux = 0
                        x_nei = list(G.neighbors(x))
                        if state[x] == 0:  # 玩家进行PDG
                            for j in x_nei:
                                Ux += pom_0[strategy[x][t_update]][strategy[j][t_update]]
                        elif state[x] == 1:  # 玩家进行SDG
                            for j in x_nei:
                                Ux += pom_1[strategy[x][t_update]][strategy[j][t_update]]
                        elif state[x] == 2:  # 玩家进行SHG
                            for j in x_nei:
                                Ux += pom_2[strategy[x][t_update]][strategy[j][t_update]]
                        Payoff[x] = Ux
                    # 更新每个玩家的策略
                    for x in G.nodes:
                        # x_nei = list(G.neighbors(x))
                        # y = random.choice(x_nei)  # 随机选出一个邻居节点
                        y = choose_nei(G, x, reputation)  # 以高概率选出声誉较高的一个邻居
                        s_change = strategy[y][t_update]
                        ran = 1 / (1 + np.exp((Payoff[x] - Payoff[y]) / K))
                        if 0 <= random.uniform(0, 1) < ran:
                            strategy[x].append(strategy[y][t_update])
                        else:
                            strategy[x].append(strategy[x][t_update])
                    # 更新每个玩家的声誉
                    for x in G.nodes:
                        if strategy[x][-1] == False:  # 玩家为合作者，声誉增加
                            if (reputation[x][-1] + delta) >= max_re:
                                reputation[x].append(max_re)
                            else:
                                reputation[x].append(reputation[x][-1] + delta)
                        elif strategy[x][-1] == True:  # 玩家为背叛者，声誉减少
                            if (reputation[x][-1] - delta) <= 0:
                                reputation[x].append(delta / 2)
                            else:
                                reputation[x].append(reputation[x][-1] - delta)
                    t_update += 1
                    f_c.append(fun_f(strategy))
                    # 统计cmd后的进行SDG个数变化情况
                    if tnow >= cmd:
                        pdg_list.append(num_pdg)
                        sdg_list.append(num_sdg)
                        shg_list.append(num_shg)
                # 玩家状态转移
                changer = min(next_change.items(), key=lambda x: x[1])[0]  # 找出下一个改变状态的玩家
                tnow = next_change[changer]
                if state[changer] == 1:  # 玩家当前在进行SDG，即将改变状态
                    if next_changer_flag[changer] == 'shg':  # 玩家变为SHG
                        state[changer] = 2
                        next_change[changer] += random.expovariate(mu_2)  # 节点下一次变为进行SDG的时间
                        num_shg += 1  # 进行SHG个数加1
                        num_sdg -= 1  # 进行SDG个数减1
                    elif next_changer_flag[changer] == 'pdg':  # 玩家变为PDG
                        state[changer] = 0
                        next_change[changer] += random.expovariate(lam_0)  # 节点下一次变为进行SDG的时间
                        num_pdg += 1  # 进行PDG个数加1
                        num_sdg -= 1  # 进行SDG个数减1
                    else:
                        print('error!')
                elif state[changer] == 0:  # 玩家当前在进行PDG，即将改变状态
                    state[changer] = 1  # 玩家只能变为SDG
                    shg_time = random.expovariate(lam_1)  # 变为SHG的时间
                    pdg_time = random.expovariate(mu_1)  # 变为PDG的时间
                    if pdg_time > shg_time:  # 玩家变为SHG
                        next_change[changer] += shg_time  # 节点下一次变为进行SHG的时间
                        next_changer_flag[changer] = 'shg'
                    else:
                        next_change[changer] += pdg_time  # 节点下一次变为进行PDG的时间
                        next_changer_flag[changer] = 'pdg'
                    num_sdg += 1  # 进行SDG个数加1
                    num_pdg -= 1  # 进行PDG个数减1
                elif state[changer] == 2:  # 玩家当前在进行SHG，即将改变状态
                    state[changer] = 1  # 玩家只能变为SDG
                    shg_time = random.expovariate(lam_1)  # 变为SHG的时间
                    pdg_time = random.expovariate(mu_1)  # 变为PDG的时间
                    if pdg_time > shg_time:  # 玩家变为SHG
                        next_change[changer] += shg_time  # 节点下一次变为进行SHG的时间
                        next_changer_flag[changer] = 'shg'
                    else:
                        next_change[changer] += pdg_time  # 节点下一次变为进行PDG的时间
                        next_changer_flag[changer] = 'pdg'
                    num_sdg += 1  # 进行SDG个数加1
                    num_shg -= 1  # 进行PDG个数减1
            fc += np.mean(f_c[-500:])
        r_f.append(fc / avg)
    result['r = {:.2f}'.format(r)] = r_f