from scipy.optimize import linprog
import numpy as np

class Result:
    def __init__(self, x, obj, T) -> None:
        self.x, self.obj, self.T = x, obj, T

class LPInstance:
    def __init__(self) -> None:
        # A: processing time matrix, shape=mx(2n)
        # b: pre-assigned length, len=m
        # T: makespan limit, len=m
        # cf: conflict jobs, shape=(2n)x(2n)
        #     True if first implies second
        #     
        # order is [x_1^+, x_1^-, x_2^+, x_2^-, ..., x_n^+, x_n^-]
        self.A, self.b, self.cf = None, None, None
    
    def set(self, A, b, cf):
        self.A, self.b, self.cf = np.array(A, dtype=np.float16), np.array(b, dtype=np.float16), np.array(cf, dtype=bool)

    def getN(self):
        return self.A.shape[1] // 2

    def getM(self):
        return self.A.shape[0]

    def getConflictNum(self):
        return np.sum(self.cf)

    def solve(self):
        l, r = 0, 100
        while r - l > 1e-5:
            mid = (l + r) / 2
            res = self._solve_single(mid)
            if res.x is None:
                l = mid
            else:
                r = mid
        res = self._solve_single(r)
        return Result(res.x, res.fun, r)

    def obj(self, x):
        return np.max(np.matmul(self.A, x) + self.b)

    def _solve_single(self, T):
        n = self.getN()

        # objective vector
        c = np.zeros(2 * n, dtype=np.float16)

        # upper bound
        a_ub, b_ub = self._costructUpperBound(T)

        # naive bound
        bounds = np.array([(0, None)] * (2 * n))

        # print(a_ub.shape, b_ub.shape, c.shape, bounds.shape)

        res = linprog(c, a_ub, b_ub, bounds=bounds)
        # print(res.x)
        return res

    def _costructUpperBound(self, T):
        n, m, n_cf = self.getN(), self.getM(), self.getConflictNum()

        # container
        a_ub = np.zeros((m + n + n_cf, 2 * n), dtype=np.float16)

        # first m x 2n is A it self
        a_ub[0:m, :] = self.A

        # from m to m + n is x_ij + x_i'j' <= 1
        for i in range(n):
            a_ub[m + i, 2 * i : 2 * i + 2] = -1

        # last is conflicts
        for i, p in enumerate(self._parseConflict()):
            v1, v2 = p
            a_ub[m + n + i, v1] = 1
            a_ub[m + n + i, v2] = -1

        # container
        b_ub = np.zeros(m + n + n_cf, dtype=np.float16)

        # first m is T-b
        b_ub[0:m] = T * np.ones(m) - self.b

        # from m to m + n is x_ij + x_i'j' <= 1
        b_ub[m:m + n] = -1

        # the last is conflicts, so 0

        return a_ub, b_ub

    def _parseConflict(self):
        n = self.getN()
        lst = []
        for i in range(2 * n):
            for j in range(2 * n):
                if self.cf[i, j]:
                    lst.append([i, j])

        return lst


def main():
    ins = LPInstance()
    ins.set(
        A=[
            [2, 1, 4, 3, 6, 5],
            [3, 2, 5, 4, 7, 6],
            [3, 4, 5, 6, 7, 8]
        ],
        b=[3, 2, 1],
        cf=[
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ]
    )

    res= ins.solve()

    print(res.T)
    print(res.x)
    print(ins.obj(np.array([1, 0, 1, 0, 1, 0])))
    

if __name__ == '__main__':
    main()