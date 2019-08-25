# Import the packages
import numpy as np
from numpy import mean
from math import sqrt
from scipy import stats
from scipy.stats import ttest_rel
from scipy.stats import t


def dependent_ttest(data1, data2, alpha):
    # calculate means
    mean1, mean2 = mean(data1), mean(data2)
    # number of paired samples
    n = len(data1)
    # sum squared difference between observations
    d1 = sum([(data1[i]-data2[i])**2 for i in range(n)])
    # sum difference between observations
    d2 = sum([data1[i]-data2[i] for i in range(n)])
    # standard deviation of the difference between means
    sd = np.sqrt((d1 - (d2**2 / n)) / (n - 1))
    # standard error of the difference between the means
    sed = sd / sqrt(n)
    # calculate the t statistic
    t_stat = (mean1 - mean2) / sed
    # degrees of freedom
    df = n - 1
    # calculate the critical value
    cv = t.ppf(1.0 - alpha, df)
    # calculate the p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
    # return everything
    return t_stat, df, cv, p


def main():
    # Define samples
    N = 390

    # Teste A
    testA = []
    a = open("H-TesteA.txt", "r")
    a1 = a.readlines()
    for x in a1:
        testA.append(float(x))
    testA = np.asarray(testA)

    # Teste C
    testC = []
    c = open("H-TesteD.txt", "r")
    c1 = c.readlines()
    for x in c1:
        testC.append(float(x))
    testC = np.asarray(testC)

    # test A - test C
    testAminusC = []
    for i in range(testA.size):
        testAminusC.append(testA[i]-testC[i])
    testAminusC = np.asarray(testAminusC)

    # calculate the t test for one sample
    meanA = 0
    t_stat, p = stats.ttest_1samp(testAminusC, meanA)
    print('t=%.3f, p=%.10f' % (t_stat, p))

    # calculate the t test for 2 samples
    # alpha = 0.05
    # t_stat, df, cv, p = dependent_ttest(testA, testC, alpha)
    # print('t=%.3f, df=%d, cv=%.3f, p=%.3f' % (t_stat, df, cv, p))
    # # interpret via critical value
    # if abs(t_stat) <= cv:
    #     print('Accept null hypothesis that the means are equal.')
    # else:
    #     print('Reject the null hypothesis that the means are equal.')
    # # interpret via p-value
    # if p > alpha:
    #     print('Accept null hypothesis that the means are equal.')
    # else:
    #     print('Reject the null hypothesis that the means are equal.')


if __name__ == '__main__':
    main()