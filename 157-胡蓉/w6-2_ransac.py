# -*- coding: utf-8 -*-
# @Author : Ruby
# @File : w6-2_ransac.py
import numpy as np
import scipy as sp
import scipy.linalg as sl


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """
    输入:
        data - 样本点
        model - 假设模型:事先自己确定
        n - 生成模型所需的最少样本点
        k - 最大迭代次数
        t - 阈值:作为判断点满足模型的条件
        d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
    输出:
        bestfit - 最优拟合解（返回nil,如果未找到）

    iterations = 0
    bestfit = nil #后面更新
    besterr = something really large #后期更新besterr = thiserr
    while iterations < k
    {
        maybeinliers = 从样本中随机选取n个,不一定全是局内点,甚至全部为局外点
        maybemodel = n个maybeinliers 拟合出来的可能符合要求的模型
        alsoinliers = emptyset #满足误差要求的样本点,开始置空
        for (每一个不是maybeinliers的样本点)
        {
            if 满足maybemodel即error < t
                将点加入alsoinliers
        }
        if (alsoinliers样本点数目 > d)
        {
            %有了较好的模型,测试模型符合度
            bettermodel = 利用所有的maybeinliers 和 alsoinliers 重新生成更好的模型
            thiserr = 所有的maybeinliers 和 alsoinliers 样本点的误差度量
            if thiserr < besterr
            {
                bestfit = bettermodel
                besterr = thiserr
            }
        }
        iterations++
    }
    return bestfit
    """
    iterations = 0
    bestfit = None
    besterr = np.inf  # 设置默认值
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])  # 随机打乱x的索引号并取出前n个数据的数据索引号及剩余索引号
        print('test_idxs = ', test_idxs)
        maybe_inliers = data[maybe_idxs]  # 获取size(maybe_idxs)行数据(Xi,Yi)
        test_points = data[test_idxs]  # 若干行(Xi,Yi)数据点
        maybemodel = model.fit(maybe_inliers)  # 拟合模型
        test_err = model.get_error(test_points, maybemodel)  # 计算误差:平方和最小
        print('test_err = ', test_err < t)
        also_idxs = test_idxs[test_err < t]  # 取出误差平方和小于t对应数据的索引号
        print('also_idxs = ', also_idxs)
        also_inliers = data[also_idxs, :]  # 获取内群数据
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))

        print('d = ', d)
        if (len(also_inliers) > d):  # 当内群数量大于d时，重新拟合曲线后计算新的误差
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # 样本连接
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)  # 平均误差作为新的误差
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新局内点索引,将新点加入
        iterations += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


def random_partition(n, n_data):
    """return n random rows of data and the other len(data) - n rows"""
    all_idxs = np.arange(n_data)  # 获取n_data下标索引
    np.random.shuffle(all_idxs)  # 打乱下标索引
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


class LinearLeastSquareModel:
    # 最小二乘求线性解,用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        """
        sl.lstsq(A, B)返回值
        x：包含最小二乘解的二维数组。
        residuals：残差的平方和。残差指的是真实值与观测值的差。
        rank：矩阵a的秩。
        s：矩阵a的奇异值。
        """
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = np.vstack([[data[:, i], data[:, i] ** 0] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        x, resids, rank, s = sl.lstsq(A, B)  # residues:残差和
        return x  # 返回最小平方和向量

    def get_error(self, data, model):
        A = np.vstack([[data[:, i], data[:, i] ** 0] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        Bc = sp.dot(A, model)  # 计算的y值,B_fit = model.k*A + model.b
        err_per_point = np.sum((B - Bc) ** 2, axis=1)  # sum squared error per row
        return err_per_point


def test():
    # 生成理想数据
    data_num = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    A_exact = 20 * np.random.random((data_num, n_inputs))  # 随机生成0-20之间的500个数据:行向量
    K_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率
    b_fit = 10 * np.random.random(size=(n_inputs, n_outputs))
    B_exact = sp.dot(A_exact, K_fit) + b_fit  # y = x * k +b

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1行向量,代表Yi

    if 1:
        # 添加"局外点",随机取100个索引值，改变其x,y
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])  # 获取索引0-499
        np.random.shuffle(all_idxs)  # 将all_idxs打乱
        outlier_idxs = all_idxs[:n_outliers]  # 100个0-500的随机局外点
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  # 加入噪声和局外点的Xi
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 加入噪声和局外点的Yi
    # setup model
    all_data = np.hstack((A_noisy, B_noisy))  # 形式([Xi,Yi]....) shape:(500,2)500行2列
    input_columns = range(n_inputs)  # 数组的第一列x:0
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1
    debug = False
    # debug = True
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)  # 类的实例化:用最小二乘生成已知模型
    A = np.hstack([all_data[:, input_columns], all_data[:, input_columns] ** 0])  # 以列拼接

    linear_fit, resids, rank, s = sp.linalg.lstsq(A, all_data[:, output_columns]) # 直接用最小二乘生成

    # run RANSAC 算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 500, 7e3, 300, debug=debug, return_all=True)

    if 1:
        import pylab

        A_sorted = A[:, 0].reshape(len(A[:, 0]), 1)  # 500x1的数组

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'rx',
                       label="RANSAC data")
        else:
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

        pylab.plot(A[:, 0],
                   np.dot(A, ransac_fit)[:, 0],
                   label='RANSAC fit')
        pylab.plot(A_sorted[:, 0],
                   (np.dot(A_sorted, K_fit) + b_fit)[:, 0],
                   label='exact system')
        pylab.plot(A_sorted[:, 0],
                   np.dot(A, linear_fit)[:, 0],
                   label='linear fit')
        pylab.legend()
        pylab.show()


if __name__ == "__main__":
    test()
