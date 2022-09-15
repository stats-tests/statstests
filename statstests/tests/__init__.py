import numpy as np
from scipy import stats
   
def shapiro_francia(array):

    """
    The statistical test of Shapiro-Francia considers the squared correlation between the ordered sample values and the (approximated) expected ordered quantiles from the standard normal distribution. The p-value is computed from the formula given by Royston (1993).
    This function performs the Shapiro-Francia test for the composite hypothesis of normality, according to Thode Jr. (2002).

    References:
    Royston, P. (1993). A pocket-calculator algorithm for the Shapiro-Francia test for non-normality: an application to medicine. Statistics in Medicine, 12, 181-184.
    Thode Jr., H. C. (2002). Testing for Normality. Marcel Dekker, New York.   
    """
   
    def ppoints(n, a):
        try:
            n = float(len(n))
        except TypeError:
            n = float(n)
        return (np.arange(n) + 1 - a)/(n + 1 - 2*a)

    x = np.sort(array)
    n = x.size
    y = stats.norm.ppf(ppoints(n, a = 3/8))
    W, _ = np.square(stats.pearsonr(x, y))
    u = np.log(n)
    v = np.log(u)
    mu = -1.2725 + 1.0521 * (v - u)
    sig = 1.0308 - 0.26758 * (v + 2/u)
    z = (np.log(1 - W) - mu)/sig
    pval = stats.norm.sf(z)
   
    dic = {}

    dic["method"] = "Shapiro-Francia normality test"
    dic["statistics W"] = W
    dic["statistics z"] = z
    dic["p-value"] = pval

    for key, value in dic.items():
        print(key, ' : ', value)

    return dic