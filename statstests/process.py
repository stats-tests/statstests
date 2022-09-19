import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def stepwise(model, pvalue_limit: float=0.05):
    """

    Stepwise process for GLM models

    >>> import pandas as pd
    >>> import statsmodels.api as sm
    >>> from statstests.process import stepwise
    >>> model = sm.OLS.from_formula('retorno ~ disclosure + endividamento + ativos + liquidez', df).fit()
    >>> print(model.summary())
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                retorno   R-squared:                       0.833
    Model:                            OLS   Adj. R-squared:                  0.827
    Method:                 Least Squares   F-statistic:                     147.9
    Date:                Mon, 19 Sep 2022   Prob (F-statistic):           3.35e-45
    Time:                        21:26:20   Log-Likelihood:                -401.07
    No. Observations:                 124   AIC:                             812.1
    Df Residuals:                     119   BIC:                             826.2
    Df Model:                           4                                         
    Covariance Type:            nonrobust                                         
    =================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    Intercept         6.0506      4.080      1.483      0.141      -2.028      14.129
    disclosure        0.1067      0.048      2.227      0.028       0.012       0.202
    endividamento    -0.0882      0.051     -1.723      0.087      -0.190       0.013
    ativos            0.0035      0.001      5.134      0.000       0.002       0.005
    liquidez          1.9762      0.396      4.987      0.000       1.191       2.761
    ==============================================================================
    Omnibus:                       35.509   Durbin-Watson:                   2.065
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):                7.127
    Skew:                          -0.136   Prob(JB):                       0.0283
    Kurtosis:                       1.858   Cond. No.                     2.94e+04
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.94e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.

    Examples

    >>> stepwise(modelo_empresas)
    Regression type: OLS 
    Estimating model...: 
    retorno ~ disclosure + endividamento + ativos + liquidez
    Discarding atribute "endividamento" with p-value equal to 0.08749071283026402 
    Estimating model...: 
    retorno ~ disclosure + ativos + liquidez
    Discarding atribute "disclosure" with p-value equal to 0.06514029954310786 
    Estimating model...: 
    retorno ~ ativos + liquidez
    No more atributes with p-value higher than 0.05
    Atributes discarded on the process...: 
    {'atribute': 'endividamento', 'p-value': 0.08749071283026402}
    {'atribute': 'disclosure', 'p-value': 0.06514029954310786}
    Model after stepwise process...: 
    retorno ~ ativos + liquidez 
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                retorno   R-squared:                       0.823
    Model:                            OLS   Adj. R-squared:                  0.820
    Method:                 Least Squares   F-statistic:                     282.1
    Date:                Mon, 19 Sep 2022   Prob (F-statistic):           2.76e-46
    Time:                        21:29:20   Log-Likelihood:                -404.37
    No. Observations:                 124   AIC:                             814.7
    Df Residuals:                     121   BIC:                             823.2
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     -2.5348      2.341     -1.083      0.281      -7.169       2.100
    ativos         0.0040      0.001      7.649      0.000       0.003       0.005
    liquidez       2.7391      0.258     10.637      0.000       2.229       3.249
    ==============================================================================
    Omnibus:                       23.591   Durbin-Watson:                   1.926
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):                5.887
    Skew:                          -0.087   Prob(JB):                       0.0527
    Kurtosis:                       1.947   Cond. No.                     1.65e+04
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.65e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.
    """

    # dictionary that identifies the type of the inputed model
    models_types = {"<class 'statsmodels.regression.linear_model.OLS'>": "OLS",
                    "<class 'statsmodels.discrete.discrete_model.Logit'>": "Logit",
                    "<class 'statsmodels.genmod.generalized_linear_model.GLM'>": "GLM"}

    try:
        # identify model type
        model_type = models_types[str(type(model.model))]
    except:
        raise Exception("The model is not yet supported...",
                        "Suported types: ", list(models_types.values()))

    print("Regression type:", model_type, "\n")

    try:

        formula = model.model.data.ynames + " ~ " + \
            ' + '.join(model.model.data.xnames[1:])

        df = pd.concat([model.model.data.orig_endog,
                       model.model.data.orig_exog], axis=1)

        # adjust column names with special characters from categorical columns
        df.columns = df.columns.str.replace('[', '', regex=True)
        df.columns = df.columns.str.replace('.', '_', regex=True)
        df.columns = df.columns.str.replace(']', '', regex=True)

        # adjust formula with special characters from categorical columns
        formula = formula.replace("[", "")
        formula = formula.replace('.', "_")
        formula = formula.replace(']', "")

        atributes_discarded = []

        while True:

            print("Estimating model...: \n", formula)

            if model_type == 'OLS':

                # return OLS model
                model = sm.OLS.from_formula(formula=formula, data=df).fit()

            elif model_type == 'Logit':

                # return Logit model
                model = sm.Logit.from_formula(formula=formula, data=df).fit()

            elif model_type == 'GLM':

                # dictionary that identifies the family type of the inputed glm model
                glm_families_types = {"<class 'statsmodels.genmod.families.family.Poisson'>": "Poisson",
                                      "<class 'statsmodels.genmod.families.family.NegativeBinomial'>": "Negative Binomial",
                                      "<class 'statsmodels.genmod.families.family.Binomial'>": "Binomial"}

                # identify family type
                family_type = glm_families_types[str(type(model.family))]

                print("\n Family type...: \n", family_type)

                if family_type == "Poisson":
                    model = smf.glm(formula=formula, data=df,
                                    family=sm.families.Poisson()).fit()
                elif family_type == "Negative Binomial":
                    model = smf.glm(formula=formula, data=df,
                                    family=sm.families.NegativeBinomial()).fit()
                elif family_type == "Binomial":
                    model = smf.glm(formula=formula, data=df,
                                    family=sm.families.Binomial()).fit()

            atributes = model.model.data.xnames[1:]

            # find atribute with the worst p-value
            worst_pvalue = (model.pvalues.iloc[1:]).max()
            worst_atribute = (model.pvalues.iloc[1:]).idxmax()

            # identify if the atribute with the worst p-value is higher than p-value limit
            if worst_pvalue > pvalue_limit:

                # exclude atribute higher than p-value limit from atributes list
                atributes = [
                    element for element in atributes if element is not worst_atribute]

                # declare the new formula without the atribute
                formula = model.model.data.ynames + \
                    " ~ " + ' + '.join(atributes)

                # append the atribute to the atributes_discarded list
                atributes_discarded.append(
                    {'atribute': worst_atribute, 'p-value': worst_pvalue})

                print(
                    '\n Discarding atribute "{}" with p-value equal to {} \n'.format(worst_atribute, worst_pvalue))

            else:

                # print that the loop is finished and there are not more atributes to discard
                print(
                    '\n No more atributes with p-value higher than {}'.format(pvalue_limit))

                break

        # print model summary after stepwised process
        print("\n Atributes discarded on the process...: \n")

        # print all the discarded atributes from the atributes_discarded list
        [print(item) for item in atributes_discarded]

        print("\n Model after stepwise process...: \n", formula, "\n")
        print(model.summary())

        # return stepwised model
        return model

    except Exception as e:
        raise e