import numpy as np
from random import randrange
from scipy.optimize import linprog

class Result:
    def __init__(self):
        self.obj, self.ass, self.scc, self.ins = None, None, None, None

    def __str__(self):
        if not self.scc:
            return 'Fail! It may due to invalid instance'
        
        return 'Success!\n'\
            f'Objective Value: {self.obj}\n'\
            'Configurations:\n' + '\n'.join([
                f'Machine {x[0]}, Weight {x[2]}, Config ' + np.binary_repr(x[1]).rjust(self.ins.getTuple()[1], '0') for x in self.ass
            ])

class Instance:
    def __init__(self):
        self.M, self.J, self.p, self.en = None, None, None, None

    def assign(self, M, J, p, en):
        self.M, self.J, self.p, self.en = M, J, p, en
        return self
    
    def getTuple(self):
        return self.M, self.J, self.p, self.en
    
    def random(self):
        while True:
            self.M = randrange(2, 6)
            self.J = max(self.M + 1, randrange(3, 10))
            self.p = np.random.randint(10, size = self.J) + 1
            prob = 0.5
            self.en = np.random.choice(a = [False, True], size = (self.M, self.J), p = [prob, 1 - prob])
            if self.__valid():
                break
        return self

    def __valid(self):
        for i in range(self.J):
            for j in range(self.M):
                if self.en[j, i]:
                    break
            else:
                return False
        return True

    def read(self, fname):
        with open(fname, 'r') as f:
            self.M, self.J = list(map(int, f.readline().split()))
            self.p = np.array(list(map(int, f.readline().split())))
            self.en = np.zeros((self.M, self.J), dtype = bool)
            for i in range(self.M):
                for j, v in enumerate(list(map(int, f.readline().split()))):
                    if v == 1:
                        self.en[i, j] = True
        return self

    def __str__(self):
        return f'# of Machines: {self.M}\n'\
            f'# of Jobs: {self.J}\n'\
            f'Processing times: {self.p}\n'\
            'Enable Matrix:\n' + '\n'.join([
                f'Machine {i}: ' + ''.join(['1' if x else '0' for x in self.en[i]]) for i in range(self.M)
            ])


class ConfigLP:
    def __init__(self):
        pass

    def solve(self, instance, decimals = 3, tol = 1e-5, eps = 1e-6):
        self.M, self.J, self.p, self.en = instance.getTuple()
        self.__transform()
        x = linprog(
            self.c, A_ub = self.A_ub, b_ub = self.b_ub, A_eq = self.A_eq, b_eq = self.b_eq, bounds = self.bounds,
            options = None if tol is None else {'tol' : tol}
        )
        
        y = Result()
        y.ins = instance
        y.scc = x.success
        if x.success:
            sol = x.x if decimals is None else np.around(x.x, decimals = decimals)
            y.obj = x.fun if decimals is None else np.around(x.fun, decimals = decimals)
            y.ass = []
            for i, v in enumerate(self.vars):
                if sol[i] > eps:
                    y.ass.append((v[0], v[1], sol[i]))

        return y


    def __transform(self):
        M, J = self.M, self.J
        self.vars = []

        dis = np.zeros(M, dtype = np.int32)
        for i in range(M):
            for j in range(J):
                if not self.en[i, j]:
                    dis[i] |= (1 << j)
        for i in range(M):
            for j in range(1, 1 << J):
                if dis[i] & j == 0:
                    self.vars.append((i, j))
        
        n = len(self.vars)
        self.c = np.array([self.__cost(x[1]) for x in self.vars], dtype = np.int32)
        self.b_ub = np.ones(self.M, dtype = np.int32)
        self.A_ub = np.zeros((self.M, n), dtype = np.int32)
        for i in range(self.M):
            for j in range(n):
                if self.vars[j][0] == i:
                    self.A_ub[i, j] = 1
        
        self.b_eq = np.ones(self.J, dtype = np.int32)
        self.A_eq = np.zeros((self.J, n), dtype = np.int32)
        for i in range(J):
            for j in range(n):
                if (self.vars[j][1] & (1 << i)) != 0:
                    self.A_eq[i, j] = 1
        
        self.bounds = np.concatenate([np.zeros(n)[:, np.newaxis], np.ones(n)[:, np.newaxis]], axis = 1)

    def __cost(self, bm):
        c = np.array([self.p[i] if (bm & (1 << i) != 0) else 0 for i in range(self.J)])
        return (np.sum(c * c) + np.sum(c) * np.sum(c)) // 2
        

class NaiveDP:
    def __init__(self):
        pass

    def solve(self, instance, max = 1000000007):
        self.M, self.J, self.p, self.en = instance.getTuple()

        M, J = self.M, self.J
        c = [self.__cost(i) for i in range(1 << J)]
        en = np.zeros(M, dtype = np.int32)
        for i in range(M):
            for j in range(J):
                if self.en[i, j]:
                    en[i] |= (1 << j)
        
        conf = np.zeros((M + 1, 1 << J), dtype = np.int32)
        dp = np.zeros((M + 1, 1 << J), dtype = np.int32)
        dp[0, 1:] = max

        buf, cnt = np.zeros(J, dtype = np.int32), 0

        for i in range(M):
            for j in range(1 << J):
                dp[i + 1, j] = max

                aval, cnt = j & en[i], 0
                for k in range(J):
                    if (aval & (1 << k)) != 0:
                        buf[cnt] = k
                        cnt += 1
                
                for k in range(1 << cnt):
                    cur = 0
                    for l in range(cnt):
                        if (k & (1 << l)) != 0:
                            cur |= 1 << buf[l]
                
                    if dp[i, j ^ cur] + c[cur] < dp[i + 1, j]:
                        dp[i + 1, j] = dp[i, j ^ cur] + c[cur]
                        conf[i + 1, j] = cur

        y = Result()
        if dp[M, (1 << J) - 1] == max:
            y.scc = False
            return y

        y.scc = True
        y.obj = dp[M, (1 << J) - 1]
        ass = np.zeros((M, J), dtype = np.int32)
        cur = (1 << J) - 1
        for i in range(M - 1, -1, -1):
            for j in range(J):
                if (conf[i + 1][cur] & (1 << j)) != 0:
                    ass[i, j] = 1
            cur ^= conf[i + 1][cur]

        y.ass = []
        y.ins = instance
        for i in range(M):
            conf = 0
            for j in range(J):
                if ass[i, j] == 1:
                    conf |= 1 << j
            y.ass.append((i, conf, 1))
        return y


    def __cost(self, bm):
        c = np.array([self.p[i] if (bm & (1 << i) != 0) else 0 for i in range(self.J)])
        return (np.sum(c * c) + np.sum(c) * np.sum(c)) // 2