计算建模 Lab6 RANSAC和最小二乘(OLS)对直线和曲线的拟合
# 单条直线拟合

## OLS (Ordinary Least Square)

## RANSAC

# 一系列平行直线拟合

已知直线斜率相同，故可先采用单条直线拟合的算法求出拟合的斜率.

确定斜率后即可算出对应各点截距，问题转化为一维的截距拟合.

OLS 对应结果即为均值，RANSAC 只需线性扫描即可.

# 正弦曲线拟合

$y = \sin(2\pi x)$

用 m 阶多项式拟合，由于参数个数较多（m 个），RANSAC 时间复杂度过高，故仅实现 OLS 拟合.

## OLS

建议改变 L2 正则项系数 $\lambda$ ，outlier 样本个数等参数多次测试.

# 圆拟合

## OLS

## RANSAC