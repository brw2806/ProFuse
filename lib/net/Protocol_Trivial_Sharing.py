"""secure 2PC protocols via trivial shares"""

import timeit
import time
import numpy as np
import math
import global_val as gl


setting = gl.init_with_return_2()


def share_split(x, l=setting["l"], cls="I"):    # (cls= F or I)
    """若cls == "F"，则需要先将浮点数转换为整数；再进行数据拆分（秘密共享）"""
    if cls == "F":
        x = get_enlarge_int(x)
    x_1 = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, x.shape, dtype=setting["t_dtype"])  # 整数share
    x_2 = x - x_1
    return [x_1, x_2]       # 将秘密x表示为由share组成的列表


def share_reconstruct(x, cls="I"):    # (cls= F or I)
    """先进行整数格式下的秘密重构；若cls == "F"，则表明需要得到对应的浮点数；"""
    x = x[0] + x[1]
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
    return y.astype(setting["t_dtype"])


def SDot(x, y, l=setting["l"], if_trunc='not'):
    """
        # secure matrix multiplication
        # python3 int64: range (-2**63, 2**63-1)
        if_trunc: not表示不需要截断操作；need_acc表示需要被截断部分对高位的影响（进位或借位）；need_only表示对共享分别直接截断
        :param factor: enlarge factor
        :return: z1 + z2 = (x1 + x2) * (y1 + y2) = x * y
        note: f1 & f2 with "l_D"
        """
    x_1, x_2 = x[0], x[1]
    y_1, y_2 = y[0], y[1]

    # offline
    a_1 = np.random.randint(-2**(l-1), 2 ** (l-1) - 1, x_1.shape, dtype=setting["t_dtype"])  # T
    a_2 = np.random.randint(-2**(l-1), 2 ** (l-1) - 1, x_2.shape, dtype=setting["t_dtype"])
    b_1 = np.random.randint(-2**(l-1), 2 ** (l-1) - 1, y_1.shape, dtype=setting["t_dtype"])
    b_2 = np.random.randint(-2**(l-1), 2 ** (l-1) - 1, y_2.shape, dtype=setting["t_dtype"])
    t0 = time.perf_counter()
    # c = (a_1 + a_2) @ (b_1 + b_2)
    c = np.einsum('...ij,jk...->...ik', a_1 + a_2, b_1 + b_2)
    c_1 = np.random.randint(-2**(l-1), 2 ** (l-1) - 1, c.shape, dtype=setting["t_dtype"])
    c_2 = c - c_1

    # online
    alpha = (x_1 - a_1) + (x_2 - a_2)         # S1 & S2
    beta = (y_1 - b_1) + (y_2 - b_2)          # S1 & S2
    # z_1 = c_1 + alpha @ b_1 + a_1 @ beta      # S1
    # z_2 = c_2 + alpha @ b_2 + a_2 @ beta + alpha @ beta   # S2
    z_1 = c_1 + np.einsum('...ij,jk...->...ik', alpha, b_1) + np.einsum('...ij,jk...->...ik', a_1, beta)
    z_2 = c_2 + np.einsum('...ij,jk...->...ik', alpha, b_2) + np.einsum('...ij,jk...->...ik', a_2, beta)\
              + np.einsum('...ij,jk...->...ik', alpha, beta)

    if if_trunc == 'need_acc':
        return Strunc([z_1, z_2])
    elif if_trunc == 'need_only':
        return [get_middle_int(z_1), get_middle_int(z_2)]
    else:
        return [z_1, z_2]


def SMul(x, y, l=setting["l"], l_D=setting["l_D"], if_trunc='not'):
    """
    # secure multiplication
    # python3 int64: range (-2**63, 2**63-1)
    if_trunc: not表示不需要截断操作；need_acc表示需要被截断部分对高位的影响（进位或借位）；need_only表示对共享分别直接截断
    :param factor: enlarge factor
    :return: z1 + z2 = (x1 + x2) * (y1 + y2) = x * y
    note: f1 & f2 with "l_D"
    """
    x_1, x_2 = x[0], x[1]
    y_1, y_2 = y[0], y[1]

    # offline
    a_1 = np.random.randint(-2**(l-1), 2 ** (l-1) - 1, x_1.shape, dtype=setting["t_dtype"])  # T
    a_2 = np.random.randint(-2**(l-1), 2 ** (l-1) - 1, x_2.shape, dtype=setting["t_dtype"])
    b_1 = np.random.randint(-2**(l-1), 2 ** (l-1) - 1, x_1.shape, dtype=setting["t_dtype"])
    b_2 = np.random.randint(-2**(l-1), 2 ** (l-1) - 1, x_2.shape, dtype=setting["t_dtype"])
    c = (a_1 + a_2) * (b_1 + b_2)
    c_1 = np.random.randint(-2**(l-1), 2 ** (l-1) - 1, x_1.shape, dtype=setting["t_dtype"])
    c_2 = c - c_1

    # online
    alpha = (x_1 - a_1) + (x_2 - a_2)         # S1 & S2
    beta = (y_1 - b_1) + (y_2 - b_2)          # S1 & S2
    z_1 = c_1 + b_1 * alpha + a_1 * beta      # S1
    z_2 = c_2 + b_2 * alpha + a_2 * beta + alpha * beta   # S2

    if if_trunc == 'need_acc':
        return Strunc([z_1, z_2], l_D=l_D)
    elif if_trunc == 'need_only':
        return [get_middle_int(z_1, l_D=l_D), get_middle_int(z_2, l_D=l_D)]
    else:
        return [z_1, z_2]


def STMA(u, l=setting["l"], l_D=setting["l_D"], if_trunc='not'):
    """
    secure multiplicative share into additive share
    if_trunc: not表示不需要截断操作；need_acc表示需要被截断部分对高位的影响（进位或借位）；need_only表示对共享分别直接截断
    :param u1; u2
    :return: f1 + f2 = u1 * u2
    note: f1 & f2 with "2 * l_D"
    """
    u1, u2 = u[0], u[1]

    # offline
    a1 = np.random.randint(-2**(l-1), 2 ** (l-1) - 1, u1.shape, dtype=setting["t_dtype"])    # T
    a2 = np.random.randint(-2**(l-1), 2 ** (l-1) - 1, u2.shape, dtype=setting["t_dtype"])
    c = a1 * a2
    c1 = np.random.randint(-2**(l-1), 2 ** (l-1) - 1, c.shape, dtype=setting["t_dtype"])
    c2 = c - c1

    # online
    p1 = u1 - a1     # S1
    p2 = u2 - a2     # S2
    f1 = c1 + a1 * p2            # S1
    f2 = c2 + a2 * p1 + p1 * p2  # S2

    if if_trunc == 'need_acc':
        return Strunc([f1, f2], l_D=l_D)
    elif if_trunc == 'need_only':
        return [get_middle_int(f1, l_D=l_D), get_middle_int(f2, l_D=l_D)]
    else:   # if_trunc='not'
        return [f1, f2]


def SMul_3(x, y, z, l=setting["l"], if_trunc='not'):
    """
    # secure multiplication (3-fan-in)
    # python3 int64: range (-2**63, 2**63-1)
    if_trunc: not表示不需要截断操作；need_acc表示需要被截断部分对高位的影响（进位或借位）；need_only表示对共享分别直接截断
    :param factor: enlarge factor
    :return: w1 + w2 = (x1 + x2) * (y1 + y2) * (z1 + z2) = x * y * z
    note: f1 & f2 with "l_D"
    """
    x_1, x_2 = x[0], x[1]
    y_1, y_2 = y[0], y[1]
    z_1, z_2 = z[0], z[1]

    # offline
    a_1 = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, x_1.shape, dtype=setting["t_dtype"])  # T
    a_2 = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, x_2.shape, dtype=setting["t_dtype"])
    b_1 = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, y_1.shape, dtype=setting["t_dtype"])
    b_2 = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, y_2.shape, dtype=setting["t_dtype"])
    c_1 = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, z_1.shape, dtype=setting["t_dtype"])
    c_2 = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, z_2.shape, dtype=setting["t_dtype"])
    d = (a_1 + a_2) * (b_1 + b_2)
    e = (b_1 + b_2) * (c_1 + c_2)
    f = (a_1 + a_2) * (c_1 + c_2)
    g = (a_1 + a_2) * (b_1 + b_2) * (c_1 + c_2)
    d_1 = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, d.shape, dtype=setting["t_dtype"])
    e_1 = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, e.shape, dtype=setting["t_dtype"])
    f_1 = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, f.shape, dtype=setting["t_dtype"])
    g_1 = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, g.shape, dtype=setting["t_dtype"])
    d_2 = d - d_1
    e_2 = e - e_1
    f_2 = f - f_1
    g_2 = g - g_1

    # online
    _x = (x_1 - a_1) + (x_2 - a_2)         # S1 & S2
    _y = (y_1 - b_1) + (y_2 - b_2)         # S1 & S2
    _z = (z_1 - c_1) + (z_2 - c_2)         # S1 & S2
    w_1 = _x * _y * c_1 + a_1 * _y * _z + _x * b_1 * _z + _x * e_1 + _y * f_1 + _z * d_1 + g_1 + _x * _y * _z
    w_2 = _x * _y * c_2 + a_2 * _y * _z + _x * b_2 * _z + _x * e_2 + _y * f_2 + _z * d_2 + g_2

    if if_trunc == 'need_acc':
        return Strunc(w_1, w_2)
    elif if_trunc == 'need_only':
        return [get_middle_int(w_1, l_D=2*setting["l_D"]), get_middle_int(w_2, l_D=2*setting["l_D"])]
    else:
        return [w_1, w_2]


def SComp_v1(x, y, l=setting["l"], small_ring=setting["small_ring"], direct="great_equal"):
    """
    （direct="great_equal"）secure comparison about ">=" and "<"，则令v = 2(x-y) + 1；
    （direct="less_equal"）secure comparison about ">" and "<="，则令v = 2(x-y) - 1；
    hide the sign and value of input by random number r, msb(x) = msb(v) = msb(vr) ^ msb(r)
    #  int32: [-2147483648, 2147483647], i.e., [-2^31, 2^31 - 1]
    # uint32: [0, 4294967295], i.e., [0, 2^32 - 1]
    #  int64: [-9223372036854775808, 9223372036854775807], i.e., [-2^63, 2^63 - 1]
    # r (int64): [-2^63, 2^63 - 1] (avoid the MSB of x1 and x2 are the same, avoid overflow and underflow)
    # x , y (int64): [-2^31, 2^31 - 1]   <---
    # x - y (int64): [-2^32 + 1, 2^32 - 1]
    # v = 2 * (x - y) + 1 (int64): [-2^33 + 3, 2^33 - 1]
    # v * r (int64): [-2^63, 2^63 - 1] (required)
    # r (int64): [min, max] =. [-2 ** 30, 2 ** 30](set)
    :param x1; x2; y1; y2: with factor
    :param factor:
    :return: z = z1 + z2 without "factor".
    （direct="great_equal"）if z = 0, then x >= y; if z = 1, then x < y.
    （direct="less_equal"）if z = 0, then x > y; if z = 1, then x <= y.
    """
    x1, x2 = x[0], x[1]
    y1, y2 = y[0], y[1]

    # offline
    r = np.random.randint(-2 ** (small_ring - 1), 2 ** (small_ring - 1) - 1, x1.shape, dtype=setting["t_dtype"])
    r = np.where(r == 0, r + 1, r)     # r = r1 + r2 != 0
    r1 = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, r.shape, dtype=setting["t_dtype"])
    r2 = r - r1
    r_msb = (np.where(r >= 0, 0, 1)).astype(setting["t_dtype"])     # r_msb = r_msb1 ^ r_msb2
    r_msb1 = np.random.randint(0, 2, r_msb.shape, dtype=setting["t_dtype"])
    r_msb2 = r_msb ^ r_msb1

    # online
    if direct == "great_equal":
        v1 = 2 * (x1 - y1) + 1   # v = 2(x-y)+1 != 0; msb(v1 + v2) = msb(x-y)
    elif direct == "less_equal":
        v1 = 2 * (x1 - y1) - 1
    else:
        raise ValueError("Invalid argument: direct")
    v2 = 2 * (x2 - y2)

    g1, g2 = SMul([v1, v2], [r1, r2])    # hide sign and value
    g = g1 + g2      # re-share g (g != 0)
    g_msb = (np.where(g > 0, 0, 1)).astype(setting["t_dtype"])    # msb(x) = msb(v) = msb(g) ^ msb(r)
    x_msb1_xor = g_msb ^ r_msb1       # boolean share
    x_msb2_xor = r_msb2
    x_part1, x_part2 = STMA([x_msb1_xor, x_msb2_xor])
    z1 = x_msb1_xor - 2 * x_part1     # additive share
    z2 = x_msb2_xor - 2 * x_part2
    return [z1, z2]


def SComp_v2(x, y, l=setting["l"], small_ring=setting["small_ring"], direct="great_equal"):
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
    x1, x2 = x[0], x[1]
    y1, y2 = y[0], y[1]

    # offline
    r = np.random.randint(1, 2 ** (small_ring - 1) - 1, x1.shape, dtype=setting["t_dtype"])  # r = r1 + r2 > 0
    r1 = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, r.shape, dtype=setting["t_dtype"])
    r2 = r - r1
    alpha_int = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, x1.shape, dtype=setting["t_dtype"])
    alpha = (np.where(alpha_int >= 0, 1, -1)).astype(setting["t_dtype"])   # alpha = alpha1 + alpha2
    alpha1 = np.random.randint(-1, 2, alpha.shape, dtype=setting["t_dtype"])
    alpha2 = alpha - alpha1
    # online
    if direct == "great_equal":
        v1 = 2 * (x1 - y1) + 1   # v = 2(x-y)+1 != 0; msb(v1 + v2) = msb(x-y)
    elif direct == "less_equal":
        v1 = 2 * (x1 - y1) - 1
    else:
        raise ValueError("Invalid argument: direct")
    v2 = 2 * (x2 - y2)

    # g1, g2 = SMul([v1, v2], [r1, r2])
    # h1, h2 = SMul([g1, g2], [alpha1, alpha2])
    h1, h2 = SMul_3([v1, v2], [r1, r2], [alpha1, alpha2])   # v * r * alpha不能溢出！！！

    h = h1 + h2     # re-share h (h != 0)
    out1 = np.where(h > 0, alpha1, -alpha1)     # if out = 1, x >= y; if out = -1, x < y
    out2 = np.where(h > 0, alpha2, -alpha2)
    z1 = (-out1 + 1) // 2  # if z = 0, x >= y; if z = 1, x < y
    z2 = (-out2 + 1) // 2
    return [z1, z2]


def Strunc(x, l=setting["l"], l_D=setting["l_D"]):
    """
    return x // (2**l_D)
    # secure truncation with carry and borrow flag
    :param x1 & x2 with "l_D"
    :return: z1 + z2 = (x1 + x2)//(2**l_D) != x1//(2**l_D) + x2//(2**l_D), z1 & z2 with "l_D"
    1: There is a bit error in truncation respectively (trunc error)
    solution for issue 1: judge the carry, borrow, or neither
    2: special case:(int64), (trunc failure)
    underflow: -9223366723005406287 + -9223371938372875789 = 5412331269540
    overflow: 9223366723005406287 + 9223371938372875789 = -5412331269540
    solution for issue 2: comparison over ring来获得细的溢出标志; or, 利用只可能同号溢出，武断的将截断前的输入刷新为异号
    """
    x1, x2 = x[0], x[1]

    # 1. 截断误差（末尾一个比特）
    trans1 = np.mod(np.abs(x1), 2 ** l_D) * np.where(x1 >= 0, 1, -1)  # s1表示x1的被截断部分
    _x1 = x1 - trans1
    _x2 = x2 + trans1  # 转换为可以整除2^l_D的share; 发送s1被认为是安全的，因为小于2^l_D的部分会被截断，仅泄露一个比特

    # 2. 截断错误（share的和产生溢出）
    s1 = np.mod(np.abs(x1), 2 ** int(l/2))
    s2 = np.mod(np.abs(x2), 2 ** int(l/2))
    l1, l2 = SComp_v2([np.array([2 ** 32 - 1], dtype=setting["t_dtype"]), 0], [s1, s2])  # 1(2 ** 32 - 1 < s)表示低32位有向高位产生进位（1个比特）

    # 如果忽略低32比特的进位（仅1比特），可以节省一个比较协议，误差为1/(2^small_ring)
    w1 = np.abs(x1) // (2 ** int(l/2))
    w2 = np.abs(x2) // (2 ** int(l/2))
    h1, h2 = SComp_v2([np.array([2 ** (int(l/2) - 1) - 1], dtype=setting["t_dtype"]), 0], [w1 + l1, w2 + l2])  # 1(2 ** 63 - 1 < |x1|+|x2|) 表示绝对值之和产生溢出

    # 若sign1 = np.where(x1 >= 0, 0, 1), sign2 = np.where(x2 >= 0, 0, 1)    # sign1^sign2 = 0 表示x1和x2同号
    # 显然若(h1+h2)∩(sign1^sign2^1)=1表示溢出，若为0表示不溢出（现在不区分上溢和下溢）,然后再采用sign1或sign2区分溢出类型
    # 这种方式复杂度较高，下面采用一种新的方式，主要体现在符号结果选择1和-1分别表示正和负，而不是0和1,可以直接区分上溢和下溢
    sign1 = np.where(x1 >= 0, 1, -1)
    sign2 = np.where(x2 >= 0, 1, -1)
    d1, d2 = SMul([h1, h2], [sign1, sign2])    # (-2, 0, 2) (下溢，不溢出，上溢)
    _d1, _d2 = (d1 + 1) // 2, d2 // 2          # (-1, 0, 1) (下溢，不溢出，上溢)

    # （1）在下溢情况下，分开截断的值要小于合并后截断的值，为了消去不该保留的高位负值，需要加上这部分的相反数，_d = _d1 + _d2 = -1，即加上2 ** (64 - l_D)
    # （2）在上溢情况下，分开截断的值要大于合并后截断的值，为了消去不该保留的高位正值，需要加上这部分的相反数，_d = _d1 + _d2 = 1，即减去2 ** (64 - l_D)
    z1 = get_middle_int(_x1) - (2 ** (l - l_D)) * _d1
    z2 = get_middle_int(_x2) - (2 ** (l - l_D)) * _d2
    return [z1, z2]


def Strunc_any(x, l=setting["l"], number=1, if_trunc='need_only'):
    """
    return x // number  (number可以是数，也可以是同维数组, number is public)
    """
    x1, x2 = x[0], x[1]

    if if_trunc == 'need_acc':  # 考虑截断损失（组合flow_with_comp和bit_without_comp！！！）
        # 2. 截断错误（share的和产生溢出）
        s1 = np.mod(np.abs(x1), 2 ** int(l/2))
        s2 = np.mod(np.abs(x2), 2 ** int(l/2))
        l1, l2 = SComp_v2([np.array([2 ** int(l/2) - 1], dtype=setting["t_dtype"]), 0], [s1, s2])  # 1(2 ** 32 - 1 < s)表示低32位有向高位产生进位（1个比特）

        # 如果忽略低32比特的进位（仅1比特），可以节省一个比较协议，误差为1/(2^small_ring)
        w1 = np.abs(x1) // (2 ** int(l/2))
        w2 = np.abs(x2) // (2 ** int(l/2))
        h1, h2 = SComp_v2([np.array([2 ** (int(l/2) - 1) - 1], dtype=setting["t_dtype"]), 0], [w1 + l1, w2 + l2])  # 1(2 ** 63 - 1 < |x1|+|x2|) 表示绝对值之和产生溢出

        # 若sign1 = np.where(x1 >= 0, 0, 1), sign2 = np.where(x2 >= 0, 0, -1)    # sign1^sign2 = 0 表示x1和x2同号
        # 显然若(h1+h2)∩(sign1^sign2^1)=1表示溢出，若为0表示不溢出（现在不区分上溢和下溢）,然后再采用sign1或sign2区分溢出类型(实际上不太好区分)
        # 这种方式复杂度较高，下面采用一种新的方式，主要体现在符号结果选择1和-1分别表示正和负，而不是0和1,可以直接区分上溢和下溢
        sign1 = np.where(x1 >= 0, 1, -1)
        sign2 = np.where(x2 >= 0, 1, -1)
        d1, d2 = SMul([h1, h2], [sign1, sign2])  # (-2, 0, 2) (下溢，不溢出，上溢)
        _d1, _d2 = (d1 + 1) // 2, d2 // 2  # (-1, 0, 1) (下溢，不溢出，上溢)
        flow_label = _d1 + _d2  # 公开是否溢出的标志不会泄露隐私？会不会影响share的随机性？

        # xx1 + xx2 = x1 + x2 - (2 ** 64) * flow_label， 由于python numpy不能表示超过int64的数，所以将运算平均分配到各个share上
        xx1 = x1 - (2 ** (l - 2)) * flow_label - (2 ** (l - 2)) * flow_label
        xx2 = x2 - (2 ** (l - 2)) * flow_label - (2 ** (l - 2)) * flow_label

        # 1. 截断误差（末尾一个比特）
        trans1 = np.mod(np.abs(xx1), number) * np.where(xx1 >= 0, 1, -1)  # s1表示x1的被截断部分
        _x1 = xx1 - trans1
        _x2 = xx2 + trans1  # 转换为可以整除2^l_D的share; 发送s1被认为是安全的，因为小于2^l_D的部分会被截断，仅泄露一个比特

        z1 = get_public_div(_x1, number)
        z2 = get_public_div(_x2, number)
    else:
        z1 = get_public_div(x1, number)
        z2 = get_public_div(x2, number)
    return [z1, z2]


def flow_with_comp(x, l=setting["l"]):
    """
    （1. 绕环溢出错误问题）输入x1和x2，通过溢出标志来判断x1+x2是否溢出，（1上溢、0不溢出、-1下溢）；
    输出相加不溢出的xx1和xx2；
    完全避免溢出错误！
    """
    x1, x2 = x[0], x[1]

    # 2. 截断错误（share的和产生溢出）
    s1 = np.mod(np.abs(x1), 2 ** int(l/2))
    s2 = np.mod(np.abs(x2), 2 ** int(l/2))
    l1, l2 = SComp_v2([np.array([2 ** int(l/2) - 1], dtype=setting["t_dtype"]), 0], [s1, s2])  # 1(2 ** 32 - 1 < s)表示低32位有向高位产生进位（1个比特）

    # 如果忽略低32比特的进位（仅1比特），可以节省一个比较协议，误差为1/(2^small_ring)
    w1 = np.abs(x1) // (2 ** int(l/2))
    w2 = np.abs(x2) // (2 ** int(l/2))
    h1, h2 = SComp_v2([np.array([2 ** (int(l/2) - 1) - 1], dtype=setting["t_dtype"]), 0], [w1 + l1, w2 + l2])  # 1(2 ** 63 - 1 < |x1|+|x2|) 表示绝对值之和产生溢出

    # 若sign1 = np.where(x1 >= 0, 0, 1), sign2 = np.where(x2 >= 0, 0, -1)    # sign1^sign2 = 0 表示x1和x2同号
    # 显然若(h1+h2)∩(sign1^sign2^1)=1表示溢出，若为0表示不溢出（现在不区分上溢和下溢）,然后再采用sign1或sign2区分溢出类型(实际上不太好区分)
    # 这种方式复杂度较高，下面采用一种新的方式，主要体现在符号结果选择1和-1分别表示正和负，而不是0和1,可以直接区分上溢和下溢
    sign1 = np.where(x1 >= 0, 1, -1)
    sign2 = np.where(x2 >= 0, 1, -1)
    d1, d2 = SMul([h1, h2], [sign1, sign2])  # (-2, 0, 2) (下溢，不溢出，上溢)
    _d1, _d2 = (d1 + 1) // 2, d2 // 2        # (-1, 0, 1) (下溢，不溢出，上溢)
    flow_label = _d1 + _d2                   # 公开是否溢出的标志不会泄露隐私？会不会影响share的随机性？

    # xx1 + xx2 = x1 + x2 - (2 ** 64) * flow_label，由于python numpy不能表示超过int64的数，所以将运算平均分配到各个share上
    xx1 = x1 - (2 ** (l - 2)) * flow_label - (2 ** (l - 2)) * flow_label
    xx2 = x2 - (2 ** (l - 2)) * flow_label - (2 ** (l - 2)) * flow_label
    return [xx1, xx2]


def flow_without_comp(x, small_ring=setting["small_ring"]):
    """
    （1. 绕环溢出错误问题）输入x1和x2，通过重新刷新share的范围，大概率会令x1+x2不会发生溢出；
    输出相加不溢出的xx1和xx2
    有小概率出现溢出错误！
    """
    x1, x2 = x[0], x[1]
    xx1 = np.random.randint(-2 ** (small_ring - 1), 2 ** (small_ring - 1) - 1, x1.shape, dtype=setting["t_dtype"])
    delta = x1 - xx1    # avoid p1+p2 over/underflow, result in the incorrect trunc or narrow
    xx2 = x2 + delta    # 秘密共享xx1和xx2的范围不小于明文分布的范围，就可以保证安全性？并且，可以防止pp1和pp2之和溢出？
    return [xx1, xx2]


def bit_with_comp(x, number=1):
    """
    （2. 末尾比特误差问题）输入x1和x2，通过判断被截断的两个share之和是否产生进位，来对本地截断的结果加上损失的进位值；
    使得分开截断的结果与明文截断的结果一致。
    """
    # 1. 截断误差（末尾一个比特）
    number_sign = np.where(number >= 0, 1, -1)
    x1, x2 = x[0] * number_sign, x[1] * number_sign
    number = np.abs(number)     # 将除数number的sign加到输入share上，令除数大于0

    """
    # 1. 方法一
    ss1 = np.mod(np.abs(x1), number) * np.where(x1 >= 0, 1, -1)    # s1表示x1的被截断部分，符号与x1相同,(-number, number)
    ss2 = np.mod(np.abs(x2), number) * np.where(x2 >= 0, 1, -1)
    ss = ss1 + ss2      # 取值范围在(-2 * number, 2 * number)
    # 相当于(ss // number)，由于ss1和ss2可正可负，故进、借位包含4种情况（分布区间）：分别为借2bit、借1bit、不进不借、进1bit
    # 这样做比较复杂，需要执行多次比较运算；
    # 实际上，我们可以在本地截断运算中，使被截断的部分（余数）恒为非负数，则只需要考虑两种情况：1进位和0不进位
    carry_ss = np.select([ss < -number, (ss >= -number) & (ss < 0), (ss >= 0) & (ss < number)], [-2, -1, 0], default=1)
    x1_div = get_public_div(x1 - ss1, number)         # 整除结果
    x2_div = get_public_div(x2 - ss2, number)
    error_ss = (x1_div + x2_div + carry_ss) - out_ori    # 误差为0
    #
    # 2. 方法二
    x1_div = get_public_div(x1, number)     # 向下取整的除法结果
    x2_div = get_public_div(x2, number)
    s1 = x1 - x1_div * number     # 被截断的部分（余数），s1和s2恒为正数
    s2 = x2 - x2_div * number
    s = s1 + s2
    # 被截断的余数s1和s2恒为非负数，则只需要考虑两种情况：1进位和0不进位
    carry_s = np.where(s >= number, 1, 0)    # s // number
    error_s = (x1_div + x2_div + carry_s) - out_ori
    """

    x1_div = get_public_div(x1, number)  # 向下取整的除法结果
    x2_div = get_public_div(x2, number)
    s1 = x1 - x1_div * number  # 被截断的部分（余数），s1和s2恒为正数
    s2 = x2 - x2_div * number

    # 被截断的余数s1和s2恒为非负数，则只需要考虑两种情况：1进位和0不进位
    carry1, carry2 = SComp_v2([s1, s2], [number, 0])    # if s>=number, carry=0; else(0<=s<number), carry=1
    carry1, carry2 = 1 - carry1, -carry2       # if s>=number, carry=1（产生进位）; else(0<=s<number), carry=0（不进位）

    z1 = x1_div + carry1
    z2 = x2_div + carry2
    return [z1, z2]


def bit_without_comp(x, number=1):
    """
    （2. 末尾比特误差问题）输入x1和x2，通过将任一输入share被截断的部分传输给另一方；
    来达到一个目的，一方的share能够整除分母，另一方的share执行正常截断；
    使得分开截断的结果与明文截断的结果一致。
    """
    # 1. 截断误差（末尾一个比特）
    x1, x2 = x[0], x[1]
    trans1 = np.mod(np.abs(x1), number) * np.where(x1 >= 0, 1, -1)  # s1表示x1的被截断部分
    _x1 = x1 - trans1
    _x2 = x2 + trans1  # 转换为可以整除2^l_D的share; 发送s1被认为是安全的，因为小于2^l_D的部分会被截断，仅泄露一个比特

    z1 = get_public_div(_x1, number)
    z2 = get_public_div(_x2, number)
    return [z1, z2]


def Strunc_complete(x, number=1, flow_label="without", bit_label="without"):
    # setting["t_dtype"](number)用于将可能是元素值的输入转换为类似numpy的格式
    # part = (((2 ** (l - 2)) // setting["t_dtype"](number)) * (2 ** 2)).astype(setting["t_dtype"])  # 不能整除怎么办？

    if (flow_label == "without") & (bit_label == "without"):
        y1, y2 = flow_without_comp(x)
        z1, z2 = bit_without_comp([y1, y2], number=number)
    elif (flow_label == "without") & (bit_label == "with"):
        y1, y2 = flow_without_comp(x)
        z1, z2 = bit_with_comp([y1, y2], number=number)
    elif (flow_label == "with") & (bit_label == "without"):
        y1, y2 = flow_with_comp(x)
        z1, z2 = bit_without_comp([y1, y2], number=number)
    elif (flow_label == "with") & (bit_label == "with"):
        y1, y2 = flow_with_comp(x)
        z1, z2 = bit_with_comp([y1, y2], number=number)
    else:
        raise ValueError("Invalid argument: flow_label or flow_label")
    return [z1, z2]


def SExp_sim(x, l_D=setting["l_D"], l_I=setting["l_I"], if_trunc='need_only'):
    """
    当x1和x2很小时，优先使用SExp_sim，开销较低。（例如，softmax中的输入减去最小值后）
    secure exponent
    误差与l_I有关，且增长很快
    :param x1; x2: with factor
    :param l_D: l_D表示扩大的小数位数，l_I表示整数位数
    :return: z1 + z2 = e^(x1 + x2) with "factor"
    """
    x1, x2 = x[0], x[1]
    _x1 = np.random.randint(-2 ** (l_D + l_I - 1), 2 ** (l_D + l_I - 1) - 1, x1.shape, dtype=setting["t_dtype"])
    s1 = x1 - _x1  # S1
    _x2 = x2 + s1  # S2   # _x1 + _x2 = x1 + x2，避免后面的指数运算溢出，只要随机共享的范围大于明文的范围，就可保证安全性？
    c1, c2 = get_float(_x1), get_float(_x2)
    a1, a2 = np.exp(c1), np.exp(c2)
    e1, e2 = get_enlarge_int(a1), get_enlarge_int(a2)
    z1, z2 = STMA([e1, e2], if_trunc=if_trunc)  # S1 & S2
    return [z1, z2]


def SExp(x1, x2, base=math.e, l_I=2, l_D=16, l=64, small_ring=32, if_trunc='not'):
    """
    secure exponent
    :param x1; x2: with factor
    :param l_D: l_D表示扩大的小数位数，l_I表示整数位数
    :return: z1 + z2 = e^(x1 + x2) with "factor"
    """
    A = math.floor((2 ** l_I) * (math.log(base, 2))) + 1   # 使整数部分不小于0，以进行整环中的共享计算

    _x1 = np.random.randint(-2 ** (l_D + l_I - 1), 2 ** (l_D + l_I - 1) - 1, x1.shape, dtype=np.int64)
    s1 = x1 - _x1  # S1   # 假设x1+x2在环（small_ring）内，直接本地截断x1和x2会遇到因为绕环导致的截断错误。
    _x2 = x2 + s1  # S2   # _x1 + _x2 = x1 + x2，避免后面的指数运算溢出，只要随机共享的范围大于明文的范围，就可保证安全性？

    _x1 = _x1 * (math.log(base, 2))    # 将以base为底的幂函数转换为以2为底
    _x2 = _x2 * (math.log(base, 2))

    x1_int = get_middle_int(_x1, l_D=l_D)       # 整数部分
    x2_int = get_middle_int(_x2, l_D=l_D)
    x1_frac = get_float(_x1, l_D=l_D) - x1_int  # 小数部分
    x2_frac = get_float(_x2, l_D=l_D) - x2_int

    x1_int_pow = np.power(2, x1_int + A)
    x2_int_pow = np.power(2, x2_int + A)
    z1_int, z2_int = STMA([x1_int_pow, x2_int_pow], l_D=A, if_trunc='need_only')  #

    x1_pow = np.power(2, x1_frac)
    x2_pow = np.power(2, x2_frac)
    x1_pow_enlarge = get_enlarge_int(x1_pow, l_D=l_D)
    x2_pow_enlarge = get_enlarge_int(x2_pow, l_D=l_D)
    z1_frac, z2_frac = STMA([x1_pow_enlarge, x2_pow_enlarge], l_D=l_D, if_trunc='need_only')

    z1, z2 = SMul([z1_int, z2_int], [z1_frac, z2_frac], l_D=A, if_trunc='need_only')  # 这里截断会有一些误差
    return [z1, z2]


# def SExp_combine(x1, x2, base=2, l_I=2, l_D=16, l=64, small_ring=32, if_trunc='not'):
#     """
#     secure exponent
#     :param x1; x2: with factor
#     :param l_D: l_D表示扩大的小数位数，l_I表示整数位数
#     :return: z1 + z2 = e^(x1 + x2) with "factor"
#     """
#     A = math.floor((2 ** l_I) * (math.log(base, 2))) + 1    # 使整数部分不小于0，以进行整环中的共享计算
#
#     _x1 = np.random.randint(-2 ** (l_D + l_I - 1), 2 ** (l_D + l_I - 1) - 1, x1.shape, dtype=np.int64)
#     s1 = x1 - _x1  # S1   # 假设x1+x2在环（small_ring）内，直接本地截断x1和x2会遇到因为绕环导致的截断错误。
#     _x2 = x2 + s1  # S2   # _x1 + _x2 = x1 + x2，避免后面的指数运算溢出，只要随机共享的范围大于明文的范围，就可保证安全性？
#
#     _x1 = _x1 * (math.log(base, 2))    # 将以base为底的幂函数转换为以2为底
#     _x2 = _x2 * (math.log(base, 2))
#
#     x1_int = get_middle_int(_x1, l_D=l_D)       # 整数部分
#     x2_int = get_middle_int(_x2, l_D=l_D)
#     x1_frac = get_float(_x1, l_D=l_D) - x1_int  # 小数部分
#     x2_frac = get_float(_x2, l_D=l_D) - x2_int
#
#     x1_int_pow = np.power(2, x1_int + A)
#     x2_int_pow = np.power(2, x2_int + A)
#     x1_pow = np.power(2, x1_frac)
#     x2_pow = np.power(2, x2_frac)
#     x1_pow_enlarge = get_enlarge_int(x1_pow, l_D=l_D)
#     x2_pow_enlarge = get_enlarge_int(x2_pow, l_D=l_D)
#
#     x1 = x1_int_pow * x1_pow_enlarge
#     x2 = x2_int_pow * x2_pow_enlarge
#     z1, z2 = STMA(x1, x2, l_D=l_D + 2 * A, if_trunc=if_trunc)
#     return z1, z2


def SDiv(x, y, l_D=setting["l_D"], small_ring=setting["small_ring"], if_trunc='not'):
    """
    secure division
    :param x1; x2; y1; y2: with factor
    :param factor:
    :return: z = z1 + z2 = (a*y)/(a*x) = y/x with "factor".
    """
    x1, x2 = x[0], x[1]
    y1, y2 = y[0], y[1]

    # online
    a1 = np.random.randint(-2 ** (small_ring - 1), 2 ** (small_ring - 1) - 1, x1.shape, dtype=setting["t_dtype"])
    a2 = np.random.randint(-2 ** (small_ring - 1), 2 ** (small_ring - 1) - 1, x2.shape, dtype=setting["t_dtype"])
    p1, p2 = SMul([y1, y2], [a1, a2])
    q1, q2 = SMul([x1, x2], [a1, a2])
    q = q1 + q2  # re-share g

    pp1, pp2 = flow_with_comp([p1, p2])   # 避免溢出截断错误
    s1, s2 = pp1 / q, pp2 / q
    z1 = get_enlarge_int(s1)
    z2 = get_enlarge_int(s2)
    return z1, z2


def SRec(x, l_D=setting["l_D"], l=setting["l"], small_ring=setting["small_ring"], if_trunc='not'):
    """
    secure division
    :param x1; x2; y1; y2: with factor
    :param factor:
    :return: z = z1 + z2 = (a* (2^l_D))/(a*x) = 1/x with "factor".
    """
    x1, x2 = x[0], x[1]

    # online
    a1 = np.random.randint(-2 ** (small_ring - 1), 2 ** (small_ring - 1) - 1, x1.shape, dtype=np.int64)
    a2 = np.random.randint(-2 ** (small_ring - 1), 2 ** (small_ring - 1) - 1, x2.shape, dtype=np.int64)
    p1, p2 = a1 * (2 ** l_D), a2 * (2 ** l_D)
    q1, q2 = SMul([x1, x2], [a1, a2])
    q = q1 + q2  # re-share g

    # 后面可以进一步对这个“溢出截断错误”的处理方法进一步优化
    pp1 = np.random.randint(-2 ** (small_ring - 1), 2 ** (small_ring - 1) - 1, x1.shape, dtype=np.int64)
    delta = p1 - pp1    # avoid p1+p2 over/underflow, result in the incorrect trunc or narrow
    pp2 = p2 + delta    # 秘密共享pp1和pp2的范围包含明文分布的范围，就可以保证安全性？并且，可以防止pp1和pp2之和溢出？

    s1 = pp1 / q
    s2 = pp2 / q
    z1 = get_enlarge_int(s1)
    z2 = get_enlarge_int(s2)
    return [z1, z2]


def SLog(x, l_D=setting["l_D"], l=setting["l"], small_ring=setting["small_ring"], if_trunc='not'):
    """
    secure logarithm
    :param x1; x2: with factor
    :param factor:
    :return: z1 + z2 = log(ax) - log(a) = log(x) with "factor"
    """
    x1, x2 = x[0], x[1]
    # offline
    a1 = np.random.randint(1, 2 ** (int(small_ring / 2)) - 1, x1.shape, dtype=setting["t_dtype"])  # T
    a2 = np.random.randint(1, 2 ** (int(small_ring / 2)) - 1, x2.shape, dtype=setting["t_dtype"])
    c = a1 * a2
    c1 = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, x1.shape, dtype=setting["t_dtype"])
    c2 = c - c1
    # online
    d1, d2 = SMul([x1, x2], [c1, c2])  # d cannot over/underflow
    d = d1 + d2    # re-share d
    f1 = np.log(d) - l_D * math.log(2) - np.log(a1)    # 扩大了2^l_D倍，要减去
    f2 = -np.log(a2)
    z1 = get_enlarge_int(f1)
    z2 = get_enlarge_int(f2)
    return z1, z2




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
