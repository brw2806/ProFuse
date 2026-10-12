"""Network layers."""
import time
import numpy as np
from numpy.lib.stride_tricks import as_strided
import math

import Protocol_Plaintext as plain
import Protocol_Trivial_Sharing as cipher

import global_val as gl
setting = gl.init_with_return_2()


def im2col_parallel(img, k_h, k_w, s_h, s_w):
    """利用np.lib.stride_tricks.as_strided函数实现并行化的Img2Col算法"""
    batch, channel, height, width = img.shape
    out_h = (height - k_h) // s_h + 1  # 输出尺寸：高
    out_w = (width - k_w) // s_w + 1  # 输出尺寸：宽
    shape = (batch, channel, out_h, out_w, k_h, k_w)
    strides = (img.strides[0], img.strides[1], img.strides[2] * s_h, img.strides[3] * s_w, img.strides[2], img.strides[3])
    patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    col = patches.reshape(batch, channel, out_h * out_w, k_h * k_w)
    col = np.transpose(col, axes=(0, 1, 3, 2)).reshape(batch, channel * k_h * k_w, out_h * out_w)
    return col, batch, out_h, out_w


def Conv2d_f(inputs, weight_value, bias_value=None, stride=(1, 1), padding=(0, 0)):
    # Input: [batch, in_ch, height, width]
    # Output: [batch, out_ch, height, width]
    # Weight: [out_ch, in_ch, k_h, k_w]
    # Bias: [out_ch, ]

    out_c, in_c, k_h, k_w = weight_value.shape
    s_h, s_w = stride

    # 1.填充inputs, [batch_sz, in_ch, in_h + k_height - 1, in_w + k_width - 1]
    if padding != (0, 0):
        inputs = np.pad(inputs, pad_width=[(0, 0), (0, 0), padding, padding])

    # 2.根据im2col扩展输入（考虑卷积核尺寸和步长），out_h和out_w与卷积核高宽、填充尺寸、步长有关
    col, batch, out_h, out_w = im2col_parallel(inputs, k_h, k_w, s_h, s_w)  # [batch_sz, in_ch * k_h * k_w, out_h * out_w]
    col = col.transpose(0, 2, 1)  # [batch_sz, out_h * out_w, in_ch * k_h * k_w]

    # 3.执行矩阵乘法
    W = weight_value.reshape(out_c, -1).T   # reshape卷积核权重, [in_ch * k_h * k_w, out_ch]
    Z = col @ W    # [batch_sz, out_h * out_w, out_ch]
    if bias_value is not None:  # 是否需要偏置
        Z = Z + bias_value   # 卷积层的前向计算结果（输出）

    # 4.reshape为网络定义的形式, [batch_sz, out_ch, out_h, out_w]
    Z = Z.transpose(0, 2, 1).reshape(batch, out_c, out_h, out_w)
    return Z


def Conv2d(inputs, weight_value, bias_value=None, stride=(1, 1), padding=(0, 0)):
    # Input: [batch, in_ch, height, width]
    # Output: [batch, out_ch, height, width]
    # Weight: [out_ch, in_ch, k_h, k_w]
    # Bias: [out_ch, ]

    inputs = inputs[0]
    weight_value = weight_value[0]
    if bias_value is not None:
        bias_value = bias_value[0]

    out_c, in_c, k_h, k_w = weight_value.shape
    s_h, s_w = stride

    # 1.填充inputs, [batch_sz, in_ch, in_h + k_height - 1, in_w + k_width - 1]
    if padding != (0, 0):
        inputs = np.pad(inputs, pad_width=[(0, 0), (0, 0), padding, padding])

    # 2.根据im2col扩展输入（考虑卷积核尺寸和步长），out_h和out_w与卷积核高宽、填充尺寸、步长有关
    col, batch, out_h, out_w = im2col_parallel(inputs, k_h, k_w, s_h, s_w)  # [batch_sz, in_ch * k_h * k_w, out_h * out_w]
    col = col.transpose(0, 2, 1)  # [batch_sz, out_h * out_w, in_ch * k_h * k_w]

    # 3.执行矩阵乘法
    W = weight_value.reshape(out_c, -1).T   # reshape卷积核权重, [in_ch * k_h * k_w, out_ch]
    Z = plain.SDot([col], [W], if_trunc='need_only')   # [batch_sz, out_h * out_w, out_ch]
    if bias_value is not None:  # 是否需要偏置
        out = Z[0] + bias_value   # 卷积层的前向计算结果（输出）

    # 4.reshape为网络定义的形式, [batch_sz, out_ch, out_h, out_w]
    out = out.transpose(0, 2, 1).reshape(batch, out_c, out_h, out_w)
    return [out]


def SecConv2d(inputs, weight_value, bias_value=None, stride=(1, 1), padding=(0, 0)):
    # Input: [batch, in_ch, height, width]
    # Output: [batch, out_ch, height, width]
    # Weight: [out_ch, in_ch, k_h, k_w]
    # Bias: [out_ch, ]

    inputs_1, inputs_2 = inputs[0], inputs[1]
    weight_value_1, weight_value_2 = weight_value[0], weight_value[1]
    if bias_value is not None:
        bias_value_1, bias_value_2 = bias_value[0], bias_value[1]

    out_c, in_c, k_h, k_w = weight_value_1.shape
    s_h, s_w = stride

    # 1.填充inputs, [batch_sz, in_ch, in_h + k_height - 1, in_w + k_width - 1]
    if padding != (0, 0):
        inputs_1 = np.pad(inputs_1, pad_width=[(0, 0), (0, 0), padding, padding])
        inputs_2 = np.pad(inputs_2, pad_width=[(0, 0), (0, 0), padding, padding])

    # 2.根据im2col扩展输入（考虑卷积核尺寸和步长），out_h和out_w与卷积核高宽、填充尺寸、步长有关
    col_1, batch, out_h, out_w = im2col_parallel(inputs_1, k_h, k_w, s_h, s_w)  # [batch_sz, in_ch * k_h * k_w, out_h * out_w]
    col_2, batch, out_h, out_w = im2col_parallel(inputs_2, k_h, k_w, s_h, s_w)
    col_1 = col_1.transpose(0, 2, 1)  # [batch_sz, out_h * out_w, in_ch * k_h * k_w]
    col_2 = col_2.transpose(0, 2, 1)

    # 3.执行矩阵乘法
    W_1 = weight_value_1.reshape(out_c, -1).T   # reshape卷积核权重, [in_ch * k_h * k_w, out_ch]
    W_2 = weight_value_2.reshape(out_c, -1).T

    Z = cipher.SDot([col_1, col_2], [W_1, W_2], if_trunc='need_only')  # [batch_sz, out_h * out_w, out_ch]
    if bias_value is not None:  # 是否需要偏置
        out_1 = Z[0] + bias_value_1   # 卷积层的前向计算结果（输出）
        out_2 = Z[1] + bias_value_2

    # 4.reshape为网络定义的形式, [batch_sz, out_ch, out_h, out_w]
    out_1 = out_1.transpose(0, 2, 1).reshape(batch, out_c, out_h, out_w)
    out_2 = out_2.transpose(0, 2, 1).reshape(batch, out_c, out_h, out_w)
    return [out_1, out_2]


def BN2d_f(inputs, mean, var, gamma, beta, epsilon=1e-5, momentum=0.1):
    # Input: [batch, in_ch, height, width]
    # Output: [batch, out_ch, height, width]
    # mean, var, gamma, beta:  [in_ch, ] = [out_ch, ]

    # standardize
    X_center = inputs - mean[:, None, None]
    std = np.sqrt(var + epsilon)[:, None, None]
    X_norm = X_center / std

    # linear
    output = gamma[:, None, None] * X_norm + beta[:, None, None]
    return output


def ReLU_f(input):    #明文ReLU
    bit = np.where(input >= 0, 1, 0)
    output = input * bit
    return output


def ReLU(input):    #明文ReLU
    input = input[0]
    bit = np.where(input >= 0, 1, 0)
    output = input * bit
    return [output]


def SecReLU(input):     # 密文ReLU
    input_1, input_2 = input[0], input[1]

    sign_1, sign_2 = cipher.SComp_v2([input_1, input_2], [0, 0])                 # @S1 & S2
    output_1, output_2 = cipher.SMul([input_1, input_2], [1 - sign_1, -sign_2])  # @S1 & S2
    return [output_1, output_2]


def AvgPool2d_f(x, stride=2):
    # Input: [batch, in_ch, height, width]
    # Output: [batch, out_ch, height // stride, width // stride]
    out = x.reshape(x.shape[0], x.shape[1], x.shape[2] // stride, stride, x.shape[3] // stride, stride)
    out = out.mean(axis=(3, 5))
    return out


def MaxPool2d(X, stride=2):  # X代表待池化图像，池化区域为2×2
    X = X[0]
    # 这个最大池化代码只支持步长和池化尺寸均为2
    # 将池化区域拆分为四个子区域，然后两两进行三轮比较，分别输出最大的分量
    X = X.transpose(2, 3, 0, 1)  # 必须转置，因为按奇偶下标拆分子矩阵是按维度顺序
    X_0_0 = X[::2, ::2]        # ::2取奇数位，1::2取偶数位
    X_0_1 = X[::2, 1::2]
    X_1_0 = X[1::2, ::2]
    X_1_1 = X[1::2, 1::2]

    x = np.where(X_0_0 - X_0_1 > 0, X_0_0, X_0_1)
    x = np.where(x - X_1_0 > 0, x, X_1_0)
    x = np.where(x - X_1_1 > 0, x, X_1_1)
    x = x.transpose(2, 3, 0, 1)
    return [x]


def SecMaxPool(X):  # X代表子图像1，Y代表子图像2，池化区域为2×2
    # 将池化区域拆分为四个子区域，然后两两进行三轮比较，分别输出最大的分量
    X, Y = X[0], X[1]

    X = X.transpose(2, 3, 0, 1)    # 必须转置，因为按奇偶下标拆分子矩阵是按维度顺序
    Y = Y.transpose(2, 3, 0, 1)

    X_0_0 = X[::2, ::2]          # ::2取奇数位，1::2取偶数位
    X_0_1 = X[::2, 1::2]
    X_1_0 = X[1::2, ::2]
    X_1_1 = X[1::2, 1::2]

    Y_0_0 = Y[::2, ::2]
    Y_0_1 = Y[::2, 1::2]
    Y_1_0 = Y[1::2, ::2]
    Y_1_1 = Y[1::2, 1::2]

    sign_1, sign_2 = cipher.SComp_v2([X_0_0, Y_0_0], [X_0_1, Y_0_1])                # @S1 & S2 & S3
    _x, _y = cipher.SMul([sign_1, sign_2], [X_0_1 - X_0_0, Y_0_1 - Y_0_0])  # @S1 & S2 & S3
    x = _x + X_0_0
    y = _y + Y_0_0

    sign_1, sign_2 = cipher.SComp_v2([x, y], [X_1_0, Y_1_0])                # @S1 & S2 & S3
    _x, _y = cipher.SMul([sign_1, sign_2], [X_1_0 - x, Y_1_0 - y])  # @S1 & S2 & S3
    x = _x + x
    y = _y + y

    sign_1, sign_2 = cipher.SComp_v2([x, y], [X_1_1, Y_1_1])  # @S1 & S2 & S3
    _x, _y = cipher.SMul([sign_1, sign_2], [X_1_1 - x, Y_1_1 - y])  # @S1 & S2 & S3
    x = _x + x
    y = _y + y

    x = x.transpose(2, 3, 0, 1)
    y = y.transpose(2, 3, 0, 1)
    return [x, y]


def FC_f(input, wight, bias=None):
    """
    :param input: size of (batch_size, in_channels)
    :param wight: size of (out_channels, in_channels)
    :param bias:
    :return:
    """
    output = np.dot(input, wight.transpose())
    if bias is not None:
        output = output + bias
    return output


def FC(input, wight, bias=None):
    """
    :param input: size of (batch_size, in_channels)
    :param wight: size of (out_channels, in_channels)
    :param bias:
    :return:
    """
    output = plain.SDot(input, [wight[0].transpose()], if_trunc='need_only')
    if bias is not None:
        output = output[0] + bias[0]
    return [output]


def SecFC(input, wight, bias=None):
    """
    :param input: size of (batch_size, in_channels)
    :param wight: size of (out_channels, in_channels)
    :param bias:
    :return:
    """
    output = cipher.SDot(input, [wight[0].transpose(), wight[1].transpose()], if_trunc='need_only')
    if bias is not None:
        output_1 = output[0] + bias[0]
        output_2 = output[1] + bias[1]
    return [output_1, output_2]


def Sigmoid(x):    #每行---概率
    # Sigmoid(x) = \frac{1}{1 + \exp(-x)}
    y = 1 / (1 + np.exp(-x))  #概率
    return y


def SecSigmoid(x, base=math.e, l_I=2, l_D=16, l=64, if_trunc='not'):             # 每行---概率
    x_1, x_2 = x[0], x[1]

    # a_1, a_2 = SExp(-x_1, -x_2)       # (rois_num, cls_num)
    # y_1, y_2, y_3 = SDiv(2 ** 16, 0, 1 + a_1, a_2)
    exp_r = cipher.SExp(-x_1, -x_2, base=base, l_I=l_I, if_trunc=if_trunc)  # (rois_num, cls_num)
    y = cipher.SRec([(2 ** l_D) + exp_r[0], exp_r[1]])
    return y


def Tanh(x):
    # tanh(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)} = \frac{2}{1 + \exp(-2x)} - 1
    y = 2 / (1 + np.exp(-2 * x)) - 1
    return y


def SecTanh(x, base=math.e, l_I=2, l_D=setting["l_D"], l=64, if_trunc='not'):             # 每行---概率
    """存在局限性，秘密x=x1+x2不能太大，不然指数运算会产生溢出"""
    x_1, x_2 = x[0], x[1]

    a_1, a_2 = cipher.SExp(-2 * x_1, -2 * x_2, base=base, l_I=l_I, if_trunc=if_trunc)  # (rois_num, cls_num)
    y_1, y_2 = cipher.SRec([(2 ** l_D) + a_1, a_2])
    out_1 = 2 * y_1 - (2 ** l_D)
    out_2 = 2 * y_2
    return out_1, out_2


def SecTanh_neg(rip_cipher):
    """
    将特征值x映射至[-1, 1]，因为特征值有正有负，且数值很大
    由于特征值正向很大的话，会导致指数运算结果溢出
    考虑Tanh函数(2/(1+e^{-2x}) - 1)的奇函数性质，可以将 -2x 转换为负数来计算指数运算，然后将相应的Tanh计算结果取相反数
    """
    msb_cipher = cipher.SComp_v2(rip_cipher, [np.array(0), np.array(0)])  # MSB函数：rip>=0, msb=0; rip<0, msb=1
    sign_cipher = [-2 * msb_cipher[0] + 1, -2 * msb_cipher[1]]  # Sign函数：rip>=0, msb=1; rip<0, msb=-1
    rip_pos_cipher = cipher.SMul(rip_cipher, sign_cipher)  # rip_pos = rip * sign = np.abs(rip)

    a1 = np.random.randint(cipher.get_enlarge_int(np.array(-1), l_D=setting["l_D"]), cipher.get_enlarge_int(np.array(1), l_D=setting["l_D"]),
                           rip_pos_cipher[0].shape, dtype=np.int64)
    delta = rip_pos_cipher[0] - a1
    a2 = rip_pos_cipher[1] + delta  # [a1, a2], 限制rip_pos的share范围，防止计算溢出

    rip_pos_sh_cipher = [cipher.get_float(a1), cipher.get_float(a2)]  # 未乘2^{l_D}，秘密值x>=0，share均大于-1

    rip_neg2_cipher = [rip_pos_sh_cipher[i] * (-2) for i in range(2)]  # share of {-2x} < 2恒成立
    exp_mss_float_cipher = [np.exp(rip_neg2_cipher[i]) for i in range(2)]  # e^{-2x}, multiplicative share
    exp_mss_cipher = [cipher.get_enlarge_int(exp_mss_float_cipher[i]) for i in range(2)]  # e^{-2x} * 2^{2 * l_D}
    exp_cipher = cipher.STMA(exp_mss_cipher, if_trunc='need_only')  # e^{-2x} * 2^{l_D} additive share

    rec_cipher = cipher.SRec(
        [cipher.get_enlarge_int(np.array(1)) + exp_cipher[0], exp_cipher[1]])  # rec = 1 / (1 + e^{-2x})

    rip_pos_tan_cipher = [2 * rec_cipher[0] - cipher.get_enlarge_int(np.array(1)), 2 * rec_cipher[1]]  # tanh函数计算结果的绝对值
    rip_tan_cipher = cipher.SMul(rip_pos_tan_cipher, sign_cipher)  # rip_tan = rip_tan_pos * sign (4096, 24)

    return rip_tan_cipher


if __name__ == '__main__':
    np.random.seed(0)
    bt = 64
    ch = 1
    sh = 32
    shape_vec = (bt, ch, sh, sh)
    l_x = 32  # used for calculating the probability of truncation failure
    l = 64  # the size of ring
    l_D=setting["l_D"]  # trunc bits in binary system, l_D < l_x + 1 < l
    l_I = 2  # 整数比特
    small_ring = int(l / 2)  # used for comparison protocol

    value = 200
    l_I = math.log2(value) + 2

    _x = np.random.uniform(-value, value, shape_vec)  # 明文浮点数
    _y = np.random.uniform(-2 ** (l_I - 1), 2 ** (l_I - 1) - 1, shape_vec)
    x_enlarge = cipher.get_enlarge_int(_x)  # 扩大后的整数（考虑int型的整数域上，而不是正整数环上）
    y_enlarge = cipher.get_enlarge_int(_y)
    x = cipher.get_float(x_enlarge)  # 2^16 = 65536, 意味着十进制第五位可能会有影响（x和_x）
    y = cipher.get_float(y_enlarge)

    x1 = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, shape_vec, dtype=np.int64)  # 整数share
    x2 = x_enlarge - x1
    y1 = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, shape_vec, dtype=np.int64)
    y2 = y_enlarge - y1


    tanh_plain = Tanh(x)

    # tanh_sec1, tanh_sec2 = SecTanh([x1, x2], l_I=l_I)
    # tanh_sec = cipher.get_float(tanh_sec1 + tanh_sec2)
    # error_1 = tanh_sec - tanh_plain

    tanh_sec_neg1, tanh_sec_neg2 = SecTanh_neg([x1, x2])
    tanh_sec_neg = cipher.get_float(tanh_sec_neg1 + tanh_sec_neg2)

    error = tanh_sec_neg - tanh_plain

    print(value)
    # print('tanh1:', np.min(error_1), np.max(error_1))
    print('tanh:', np.min(error), np.max(error))

    print(1)


    # conv_weight = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, (ch, ch, 1, 1), dtype=np.int64)
    # conv_bias = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, (1, ch), dtype=np.int64)
    # t0_conv = time.perf_counter()
    # conv = Conv2d(x_enlarge, conv_weight, conv_bias)
    # t1_conv = time.perf_counter()
    # conv_1 = Conv2d(x1, conv_weight, conv_bias)
    # conv_2 = Conv2d(x2, conv_weight)
    # t2_conv = time.perf_counter()
    #
    # print('size:  ', (sh, sh))
    # print('plaintext_conv:  ', (t1_conv - t0_conv) * 1000)
    # print('secure_conv:  ', (t2_conv - t1_conv) * 1000)

    # t0_relu = time.perf_counter()
    # f_ori = ReLU(x)
    # t1_relu = time.perf_counter()
    # f_1, f_2 = SecReLU_v1(x1, x2)
    # t2_relu_v1 = time.perf_counter()
    # f_1, f_2 = SecReLU(x1, x2)
    # t2_relu_v2 = time.perf_counter()
    # f_1, f_2 = SecReLU_secmpnn(x1, x2)
    # t2_relu_secmpnn = time.perf_counter()
    # f_1, f_2 = SecReLU_sonic(x1, x2)
    # t2_relu_sonic = time.perf_counter()
    # f = f_1 + f_2
    # f_sec = get_float(f)
    # error_relu = f_sec - f_ori
    #
    # print('plaintext_relu:  ', (t1_relu - t0_relu) * 1000)
    # print('secure_relu_v1:  ', (t2_relu_v1 - t1_relu) * 1000)
    # print('secure_relu_v2:  ', (t2_relu_v2 - t2_relu_v1) * 1000)
    # print('secure_relu_secmpnn:  ', (t2_relu_secmpnn - t2_relu_v2) * 1000)
    # print('secure_relu_sonic:  ', (t2_relu_sonic - t2_relu_secmpnn) * 1000)

    # t0_pool = time.perf_counter()
    # f_ori = MaxPool2d(x)
    # t1_pool = time.perf_counter()
    # f_1, f_2 = SecMaxPool_v1(x1, x2)
    # t2_pool_v1 = time.perf_counter()
    # f_1, f_2 = SecMaxPool(x1, x2)
    # t2_pool_v2 = time.perf_counter()
    # f_1, f_2 = SecMaxPool_secmpnn(x1, x2)
    # t2_pool_secmpnn = time.perf_counter()
    # f_1, f_2 = SecMaxPool_sonic(x1, x2)
    # t2_pool_sonic = time.perf_counter()
    # f = f_1 + f_2
    # f_sec = get_float(f)
    # error_maxpool = f_sec - f_ori
    #
    # print('plaintext_pool:  ', (t1_pool - t0_pool) * 1000)
    # print('secure_pool_v1:  ', (t2_pool_v1 - t1_pool) * 1000)
    # print('secure_pool_v2:  ', (t2_pool_v2 - t2_pool_v1) * 1000)
    # print('secure_pool_secmpnn:  ', (t2_pool_secmpnn - t2_pool_v2) * 1000)
    # print('secure_pool_sonic:  ', (t2_pool_sonic - t2_pool_secmpnn) * 1000)

    # x_enlarge_shape = x_enlarge[:, 0, :, :].reshape(x_enlarge.shape[0], -1)
    # x1_shape = x1[:, 0, :, :].reshape(x1.shape[0], -1)
    # x2_shape = x2[:, 0, :, :].reshape(x2.shape[0], -1)
    # fc_weight = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, (sh * sh, sh * sh), dtype=np.int64)
    # fc_bias = np.random.randint(-2 ** (l - 1), 2 ** (l - 1) - 1, (1, sh * sh), dtype=np.int64)
    # t0_fc = time.perf_counter()
    # fc = FC(x_enlarge_shape, fc_weight, fc_bias, l_D=l_D)
    # t1_fc = time.perf_counter()
    # fc_1 = FC(x1_shape, fc_weight, fc_bias, l_D=l_D)
    # fc_2 = FC(x2_shape, fc_weight, l_D=l_D)
    # t2_fc = time.perf_counter()
    #
    # print('plaintext_fc:  ', (t1_fc - t0_fc) * 1000)
    # print('secure_fc:  ', (t2_fc - t1_fc) * 1000)

    # x_shape = x[:, 0, :, :].reshape(x.shape[0], -1)
    # x_shape = x_shape[:, 0:10]
    # x1_shape = x1_shape[:, 0:10]
    # x2_shape = x2_shape[:, 0:10]
    # print(x_shape.shape, x1_shape.shape, x2_shape.shape)

    # t0_sigmoid = time.perf_counter()
    # f_ori = Sigmoid(x_shape)
    # t1_sigmoid = time.perf_counter()
    # f_1, f_2 = SecSigmoid(x1_shape, x2_shape, base=math.e, l_I=l_I, if_trunc='need_acc')
    # t2_sigmoid = time.perf_counter()
    # f = f_1 + f_2
    # f_sec = get_float(f)
    # error_sigmoid = f_sec - f_ori

    # x_shape_1 = np.random.uniform(-2 ** (l_I - 1), 2 ** (l_I - 1) - 1, x_shape.shape)
    # x_shape_2 = x_shape - x_shape_1
    # from smc_function import smc_functions
    # iteration_round = 1
    # obj = smc_functions(iteration_round=iteration_round, precision=0.01)
    # t1_sigmoid_smc = time.perf_counter()
    # ff_1, ff_2 = obj.Deri_SSigmoid(x_shape_1, x_shape_2)
    # t2_sigmoid_smc = time.perf_counter()
    # print('plaintext_sigmoid:  ', (t1_sigmoid - t0_sigmoid)*1000)
    # print('secure_sigmoid:  ', (t2_sigmoid - t1_sigmoid)*1000)
    # # print('secure_sigmoid_smc:  ', (t2_sigmoid_smc - t1_sigmoid_smc) * 1000 + 2.223 * iteration_round)

    pass