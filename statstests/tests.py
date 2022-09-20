import numpy as np
from scipy import stats
import pandas as pd  # manipulação de dado em formato de dataframe
import statsmodels.api as sm  # biblioteca de modelagem estatística
import statsmodels.formula.api as smf
   
def shapiro_francia(array):

    r"""

    The statistical test of Shapiro-Francia considers the squared 
    correlation between the ordered sample values and the (approximated) 
    expected ordered quantiles from the standard normal distribution. 

    The p-value is computed from the formula given by Royston (1993).
    This function performs the Shapiro-Francia test for the composite 
    hypothesis of normality, according to Thode Jr. (2002).

    Example

    .. ipython:: python

        import pandas as pd
        import statsmodels.api as sm
        from statstests.datasets import bebes
        from statstests.tests import shapiro_francia

        # import bebes dataset
        df = bebes.get_data()

        # Estimate and fit model
        model = sm.OLS.from_formula('comprimento ~ idade', df).fit()

        # Print summary
        print(model.summary())

        # Print statistics of the normality test
        shapiro_francia(model.resid)

    The statistical test of Shapiro-Francia considers the squared 
    correlation between the ordered sample values and the (approximated) 
    expected ordered quantiles from the standard normal distribution. 

    The p-value is computed from the formula given by Royston (1993).
    This function performs the Shapiro-Francia test for the composite 
    hypothesis of normality, according to Thode Jr. (2002).

    References
    ----------
    .. [1] Royston, P. (1993). A pocket-calculator algorithm for the Shapiro-Francia test for non-normality: an application to medicine. Statistics in Medicine, 12, 181-184.
    .. [2] Thode Jr., H. C. (2002). Testing for Normality. Marcel Dekker, New York.

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

def overdisp(model, data):
    """
    Overdisp test for Statsmodels GLM Poisson model
    
    """

    # dictionary that identifies the type of the inputed model
    models_types = {
        "<class 'statsmodels.genmod.generalized_linear_model.GLM'>": "GLM"}

    try:
        # identify model type
        model_type = models_types[str(type(model.model))]
    except:
        raise Exception("The model is not yet supported...",
                        "Suported types: ", list(models_types.values()))

    # dictionary that identifies the family type of the inputed glm model
    glm_families_types = {
        "<class 'statsmodels.genmod.families.family.Poisson'>": "Poisson"}

    try:
        # identify family type
        glm_families_types[str(type(model.family))]

    except:
        raise Exception("This family is not supported...",
                        "Suported types: ", list(glm_families_types.values()))

    formula = model.model.data.ynames + " ~ " + \
        ' + '.join(model.model.data.xnames[1:])

    df = pd.concat([model.model.data.orig_endog.astype("int"),
                    model.model.data.orig_exog], axis=1)

    # adjust column names with special characters from categorical columns
    df.columns = df.columns.str.replace('[', '', regex=True)
    df.columns = df.columns.str.replace('.', '_', regex=True)
    df.columns = df.columns.str.replace(']', '', regex=True)

    # adjust formula with special characters from categorical columns
    formula = formula.replace("[", "")
    formula = formula.replace('.', "_")
    formula = formula.replace(']', "")

    print("Estimating model...: \n", model_type)

    df = df.drop(columns=["Intercept"])

    if model_type == "Poisson":
        model = smf.glm(formula=formula, data=df,
                        family=sm.families.Poisson()).fit()

    # find lambda
    df['lmbda'] = model.fittedvalues

    # creating ystar
    df['ystar'] = (((data[model.model.data.ynames]-df['lmbda'])**2)
                   - data[model.model.data.ynames])/df['lmbda']

    # ols estimation
    modelo_auxiliar = sm.OLS.from_formula("ystar ~ 0 + lmbda", df).fit()

    print(modelo_auxiliar.summary2(), "\n")

    print("==================Result======================== \n")
    print(f"p-value: {modelo_auxiliar.pvalues[0]} \n")

    if modelo_auxiliar.pvalues[0] > 0.05:
        print("Indicates equidispersion at 95% confidence level")
    else:
        print("Indicates overdispersion at 95% confidence level")