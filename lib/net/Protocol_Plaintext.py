"""secure 2PC protocols via trivial shares"""

import timeit
import time
import numpy as np
import math


import global_val as gl


setting = gl.init_with_return_1()


def share_split(x, l=setting["l"], cls="I"):    # (cls= F or I)
    """若cls == "F"，则需要先将浮点数转换为整数；再进行数据拆分（秘密共享）"""
    if cls == "F":
        x = get_enlarge_int(x)
    return [x]       # 将秘密x表示为由share组成的列表


def share_reconstruct(x, cls="I"):    # (cls= F or I)
    """先进行整数格式下的秘密重构；若cls == "F"，则表明需要得到对应的浮点数；"""
    x = x[0]
    if cls == "F":
        x = get_float(x)
    return x


def get_float(x, l_D=setting["l_D"]):
    """
    :param x:   int64:
    :param l_D: enlarge factor
    :return:   float, get the original float number （缩小2^l_D）
    """
    y = x / (2**l_D)
    return y.astype(np.float64)


def get_middle_int(x, l_D=setting["l_D"], round_type="around"):
    """
    :param x:   uint64
    :param l_D: enlarge factor
    :return:   int64, get the middle int64 number (because of mul operation)
    """
    if round_type == "around":
        # y = (np.around(x / (2 ** l_D))).astype(setting["t_dtype"])    # 四舍五入取整（中间结果是浮点数，会表示不精确）
        c3, c_reminder = np.divmod(x, 2 ** l_D)
        y = c3 + np.where(c_reminder >= (2 ** l_D) // 2, 1, 0)  # 四舍五入取整（直接得到整数结果）
    elif round_type == "floor":
        y = x // (2 ** l_D)  # 同np.floor_divide函数 （向下取整）
    elif round_type == "zero":
        y = np.where(x >= 0, x // (2 ** l_D), (x - 1) // (2 ** l_D) + 1)  # x - 1 是避免"000000"，例如-6212480883362000000（感觉是向零取整）
    elif round_type == "ceil":
        raise NotImplementedError
    else:
        raise ValueError("Invalid argument: round_type")
    return y.astype(setting["t_dtype"])


def get_public_div(x, number=1, round_type="floor"):
    """
    :param x:   uint64
    :param l_D: enlarge factor
    :return:   int64, get the middle int64 number (because of mul operation)
    """
    if round_type == "floor":
        y = x // number    # 同np.floor_divide函数 （向下取整）
    elif round_type == "around":
        raise NotImplementedError
    elif round_type == "zero":
        y = np.where(x >= 0, x // number, (x - 1) // number + 1)    # （向零取整）
    elif round_type == "ceil":
        raise NotImplementedError
    else:
        raise ValueError("Invalid argument: round_type")
    return y.astype(setting["t_dtype"])


def get_enlarge_int(x, l_D=setting["l_D"], round_type="floor"):
    """
    :param x:   float
    :param l_D: enlarge factor
    :return:   int64, get the enlarge int64 number (enlarge and round)
    """
    if round_type == "floor":
        y = x * (2**l_D)             # 直接截断，丢掉所有小数位
    else:
        raise ValueError("Invalid argument: round_type")

    # if type(y) is not np.ndarray:
    #     y = np.array(y)
    return y.astype(setting["t_dtype"])


def SDot(x, y, if_trunc='not'):
    """
        # secure matrix multiplication
        # python3 int64: range (-2**63, 2**63-1)
        if_trunc: not表示不需要截断操作；need_acc表示需要被截断部分对高位的影响（进位或借位）；need_only表示对共享分别直接截断
        :param factor: enlarge factor
        :return: z1 + z2 = (x1 + x2) * (y1 + y2) = x * y
        note: f1 & f2 with "l_D"
        """
    x_1 = x[0]
    y_1 = y[0]
    z = np.einsum('...ij,jk...->...ik', x_1, y_1)

    if if_trunc == 'not':
        return [z]
    else:
        return [get_middle_int(z)]


def SMul(x, y, if_trunc='not'):
    """
    # secure multiplication
    # python3 int64: range (-2**63, 2**63-1)
    if_trunc: not表示不需要截断操作；need_acc表示需要被截断部分对高位的影响（进位或借位）；need_only表示对共享分别直接截断
    :param factor: enlarge factor
    :return: z1 + z2 = (x1 + x2) * (y1 + y2) = x * y
    note: f1 & f2 with "l_D"
    """
    x_1 = x[0]
    y_1 = y[0]
    z = x_1 * y_1

    if if_trunc == 'not':
        return [z]
    else:
        return [get_middle_int(z)]


def SComp_v2(x, y, direct="great_equal"):
    """
    （direct="great_equal"）secure comparison about ">=" and "<"，则令v = 2(x-y) + 1；
    （direct="less_equal"）secure comparison about ">" and "<="，则令v = 2(x-y) - 1；
    hide the sign of input by selected share alpha, and hide the value of input by random positive number r.
    :param x1; x2; y1; y2: with factor
    :param factor:
    :return: z = z1 + z2 without "factor".
    （direct="great_equal"）if z = 0, then x >= y; if z = 1, then x < y.
    （direct="less_equal"）if z = 0, then x > y; if z = 1, then x <= y.
    """
    x1 = x[0]
    y1 = y[0]

    if direct == "great_equal":
        z = np.where(x1 - y1 >= 0, 0, 1)
    elif direct == "less_equal":
        z = np.where(x1 - y1 > 0, 0, 1)
    else:
        raise ValueError("Invalid argument: direct")
    return [z.astype(setting["t_dtype"])]


def Strunc_any(x, number=1, if_trunc='need_only'):
    """
    return x // number  (number可以是数，也可以是同维数组, number is public)
    """
    x1 = x[0]
    z = get_public_div(x1, number)
    return [z]


def SExp_sim(x, l_D=setting["l_D"]):
    """
    secure exponent
    误差与l_I有关，且增长很快
    :param x1; x2: with factor
    :param l_D: l_D表示扩大的小数位数，l_I表示整数位数
    :return: z1 + z2 = e^(x1 + x2) with "factor"
    """
    x1 = x[0]
    c1 = get_float(x1)
    a1 = np.exp(c1)
    z1 = get_enlarge_int(a1)
    return [z1]


def SLog(x, l_D=setting["l_D"]):
    """
    secure logarithm
    :param x1; x2: with factor
    :param factor:
    :return: z1 + z2 = log(ax) - log(a) = log(x) with "factor"
    """
    x1 = x[0]
    f1 = np.log(x1) - l_D * math.log(2)
    z1 = get_enlarge_int(f1)
    return [z1]


def SDiv(x, y, l_D=setting["l_D"]):
    """
    secure division
    :param x1; x2; y1; y2: with factor
    :param factor:
    :return: z = z1 + z2 = (a*y)/(a*x) = y/x with "factor".
    """
    x1 = x[0]
    y1 = y[0]
    s1 = y1 / x1
    z1 = get_enlarge_int(s1)
    return [z1]





# def SRec(x1, x2, l_D=16, l=64, small_ring=32, if_trunc='not'):
#     """
#     secure division
#     :param x1; x2; y1; y2: with factor
#     :param factor:
#     :return: z = z1 + z2 = (a* (2^l_D))/(a*x) = 1/x with "factor".
#     """
#     # online
#     a1 = np.random.randint(-2 ** (small_ring - 1), 2 ** (small_ring - 1) - 1, x1.shape, dtype=np.int64)
#     a2 = np.random.randint(-2 ** (small_ring - 1), 2 ** (small_ring - 1) - 1, x2.shape, dtype=np.int64)
#     p1, p2 = a1 * (2 ** l_D), a2 * (2 ** l_D)
#     q1, q2 = SMul(x1, x2, a1, a2, l_D=l_D, if_trunc=if_trunc)
#     q = q1 + q2  # re-share g
#
#     # 后面可以进一步对这个“溢出截断错误”的处理方法进一步优化
#     pp1 = np.random.randint(-2 ** (small_ring - 1), 2 ** (small_ring - 1) - 1, x1.shape, dtype=np.int64)
#     delta = p1 - pp1    # avoid p1+p2 over/underflow, result in the incorrect trunc or narrow
#     pp2 = p2 + delta    # 秘密共享pp1和pp2的范围包含明文分布的范围，就可以保证安全性？并且，可以防止pp1和pp2之和溢出？
#
#     s1 = pp1 / q
#     s2 = pp2 / q
#     z1 = get_enlarge_int(s1, l_D=l_D)
#     z2 = get_enlarge_int(s2, l_D=l_D)
#     return z1, z2









if __name__ == "__main__":
    np.random.seed(0)
    shape_vec = (1, 100000)
    l_x = 32           # used for calculating the probability of truncation failure
    l = 64             # the size of ring
    l_D = 16           # trunc bits in binary system, l_D < l_x + 1 < l
    l_I = 3  # 整数比特
    # factor = 6         # trunc bits in decimal system
    small_ring = int(l / 2) - 1  # used for comparison protocol

    _x = np.random.uniform(-2 ** (l_I - 1), 2 ** (l_I - 1) - 1, shape_vec)    # 明文浮点数
    _y = np.random.uniform(-2 ** (l_I - 1), 2 ** (l_I - 1) - 1, shape_vec)
    _y[0, 0] = _x[0, 0]

    x_enlarge = get_enlarge_int(_x)    # 扩大后的整数（考虑int型的整数域上，而不是正整数环上）
    y_enlarge = get_enlarge_int(_y)
    x = get_float(x_enlarge)        # 2^16 = 65536, 意味着十进制第五位可能会有影响（x和_x）
    y = get_float(y_enlarge)

    x1 = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, shape_vec, dtype=np.int64)   # 整数share
    x2 = x_enlarge - x1
    y1 = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, shape_vec, dtype=np.int64)
    y2 = y_enlarge - y1

    # """Testing for Strunc"""
    # g1, g2 = Strunc([x1, y1], l_D=setting["l_D"])
    # g_sec = g1 + g2
    # g_ori = get_middle_int(x1 + y1)
    # error_strunc = g_sec - g_ori
    #
    # """Testing for Strunc_any"""
    # number = np.random.randint(-2 ** (small_ring - 1), 2 ** (small_ring - 1) - 1, shape_vec, dtype=np.int64)
    # g1, g2 = Strunc_any([x1, y1], number=number)
    # g_sec = g1 + g2
    # g_ori = get_public_div(x1 + y1, number=number)
    # error_strunc_any = g_sec - g_ori


    pass
