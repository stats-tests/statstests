import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
from statsmodels.discrete.discrete_model import Poisson, NegativeBinomial
from scipy.stats import norm
from typing import Union


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
    y = stats.norm.ppf(ppoints(n, a=3/8))
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


def vuong_test(m1: Union[Poisson, NegativeBinomial], m2: Union[ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP]):
    """    
    Module to perform Vuong test for identification of zero inflation in count data regression models.

    The new Python command vuong_test of the package statstests.tests reports the results of the Voung test.

    The Vuong statistical test specifies that the Vuong (1989) test of ZIP or ZINB versus Poisson or negative binomial, 
    respectively, be reported. This test statistic has a standard normal distribution with large values favoring 
    ZIP or ZINB models over Poisson or negative binomial regression models, respectively.

    .. ipython:: python

        import pandas as pd
        import statsmodels.api as sm
        from statstests.datasets import corruption
        from statstests.tests import vuong_test
        from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP, ZeroInflatedPoisson
        from statsmodels.discrete.discrete_model import Poisson, NegativeBinomial
        import warnings
        warnings.filterwarnings('ignore')

        # import corruption dataset
        df = corruption.get_data()

        #Definição da variável dependente (voltando ao dataset 'df_corruption')
        y = df.violations

        #Definição das variáveis preditoras que entrarão no componente de contagem
        x = df[['staff','post','corruption']]
        X = sm.add_constant(x)

        X = pd.get_dummies(X, columns=['post'], drop_first=True)
        X["post_yes"] = X["post_yes"].astype("int")

        # Estimação do modelo poisson
        modelo_poisson = Poisson(endog=y, exog=X).fit()

        #Parâmetros do modelo_poisson
        print(modelo_poisson.summary())

        # Estimação do modelo poisson
        modelo_bneg = NegativeBinomial(endog=y, exog=X, loglike_method='nb2').fit()

        #Parâmetros do modelo_poisson
        print(modelo_bneg.summary())

        #Definição das variáveis preditoras que entrarão no componente de contagem
        x1 = df[['staff','post','corruption']]
        X1 = sm.add_constant(x1)

        #Definição das variáveis preditoras que entrarão no componente logit (inflate)
        x2 = df[['corruption']]
        X2 = sm.add_constant(x2)

        #Se estimarmos o modelo sem dummizar as variáveis categórias, o modelo retorna
        #um erro
        X1 = pd.get_dummies(X1, columns=['post'], drop_first=True, dtype='int')

        #Estimação do modelo ZIP pela função 'ZeroInflatedPoisson' do pacote
        #'Statsmodels'

        #Estimação do modelo ZIP
        #O argumento 'exog_infl' corresponde às variáveis que entram no componente
        #logit (inflate)
        # modelo_zip = ZeroInflatedPoisson(y, 
        #                                 X1, 
        #                                 exog_infl=X2,
        #                                 inflation='logit').fit(maxiter=1000000000)

        # #Parâmetros do modelo
        # print(modelo_zip.summary())

        # vuong_test(modelo_poisson, modelo_zip)

    """

    supported_models = [ZeroInflatedPoisson,
                        ZeroInflatedNegativeBinomialP,
                        Poisson,
                        NegativeBinomial]

    if type(m1.model) not in supported_models:
        raise ValueError(f"Model type not supported for first parameter. List of supported models: (ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP, Poisson, NegativeBinomial) from statsmodels discrete collection.")

    if type(m2.model) not in supported_models:
        raise ValueError(f"Model type not supported for second parameter. List of supported models: (ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP, Poisson, NegativeBinomial) from statsmodels discrete collection.")

    # Extração das variáveis dependentes dos modelos
    m1_y = m1.model.endog
    m2_y = m2.model.endog

    m1_n = len(m1_y)
    m2_n = len(m2_y)

    if m1_n == 0 or m2_n == 0:
        raise ValueError("Could not extract dependent variables from models.")

    if m1_n != m2_n:
        raise ValueError("Models appear to have different numbers of observations.\n"
                         f"Model 1 has {m1_n} observations.\n"
                         f"Model 2 has {m2_n} observations.")

    if np.any(m1_y != m2_y):
        raise ValueError(
            "Models appear to have different values on dependent variables.")

    m1_linpred = pd.DataFrame(m1.predict(which="prob"))
    m2_linpred = pd.DataFrame(m2.predict(which="prob"))

    m1_probs = np.repeat(np.nan, m1_n)
    m2_probs = np.repeat(np.nan, m2_n)

    which_col_m1 = [list(m1_linpred.columns).index(x) if x in list(
        m1_linpred.columns) else None for x in m1_y]
    which_col_m2 = [list(m2_linpred.columns).index(x) if x in list(
        m2_linpred.columns) else None for x in m2_y]

    for i, v in enumerate(m1_probs):
        m1_probs[i] = m1_linpred.iloc[i, which_col_m1[i]]

    for i, v in enumerate(m2_probs):
        m2_probs[i] = m2_linpred.iloc[i, which_col_m2[i]]

    lm1p = np.log(m1_probs)
    lm2p = np.log(m2_probs)

    m = lm1p - lm2p

    v = np.sum(m) / (np.std(m) * np.sqrt(len(m)))

    pval = 1 - norm.cdf(v) if v > 0 else norm.cdf(v)

    print("Vuong Non-Nested Hypothesis Test-Statistic (Raw):")
    print(f"Vuong z-statistic: {v}")
    print(f"p-value: {pval}")
