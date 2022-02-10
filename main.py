import R__pjCj as JS
import numpy as np
import time

def example1():
    ins = JS.Instance().assign(
        4, 6, 
        np.array([3, 3, 1, 1, 1, 1]),
        np.array([
            [True, False, True, False, True, False],
            [True, False, False, True, False, True],
            [False, True, True, True, False, False],
            [False, True, False, False, True, True],
        ])
    )

    print('The Instance')
    print(ins, '\n')

    lp = JS.ConfigLP().solve(ins)
    print('The LP Solution')
    print(lp, '\n')

    dp = JS.NaiveDP().solve(ins)
    print('The Integral Solution')
    print(dp)

def example2():
    while True:
        ins = JS.Instance().random()

        lp = JS.ConfigLP().solve(ins)
        dp = JS.NaiveDP().solve(ins)

        r = dp.obj / lp.obj
        if r > 1:
            print('The Instance')
            print(ins, '\n')
            
            print('The LP Solution')
            print(lp, '\n')

            print('The Integral Solution')
            print(dp, '\n')

            print('The Ratio', r)
            break

        time.sleep(0.1)

def example3():
    ins = JS.Instance().read('in.txt')
    
    print('The Instance')
    print(ins, '\n')

    lp = JS.ConfigLP().solve(ins)
    print('The LP Solution')
    print(lp, '\n')

    dp = JS.NaiveDP().solve(ins)
    print('The Integral Solution')
    print(dp)


if __name__ == '__main__':
    # example1()
    # example2()
    example3()
