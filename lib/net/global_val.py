import numpy as np


def init():
    global setting
    setting = {"t_dtype": np.int64, "l": 64, "small_ring": 31, "l_D": 16, "l_I": 3, "trunc": "need_only"}


def set(arg, value):
    setting[arg] = value


def get(arg):
    return setting[arg]


def init_with_return_1():
    global setting
    """1. 明文int"""
    setting = {"name": "plain", "workers": 1, "secret_form": None, "t_dtype": np.int64, "l": 64,
               "small_ring": None, "l_D": 16, "l_I": None, "trunc": "need_only"}    # 不存在截断错误和误差的概念
    return setting


def init_with_return_2():
    global setting
    """2. 密态两方trivial share"""
    setting = {"name": "cenia", "workers": 2, "secret_form": "trivial", "t_dtype": np.int64, "l": 64,
               "small_ring": 31, "l_D": 16, "l_I": 2, "trunc": "need_only"}
    return setting


if __name__ == "__main__":
    # gl.init()  # 初始化全局变量管理模块
    # gl.set('val_a', 80)  # 设置变量值 val_a = 80    # 可以在其他文件中改变全局变量的值，或增加键值对
    # val_b = gl.get('t_dtype')  # a文件需要用val_a的值，也需要添加这行代码来得到它的值

    settting = init_with_return_2()   # settting有两行变量

    # init_with_return()   # 这样去调用settting就可以存在了

    pass

