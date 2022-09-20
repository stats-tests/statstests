import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def stepwise(model, pvalue_limit: float=0.05):
    
    r"""
    
    Stepwise process for Statsmodels regression models

    Usage example

    .. ipython:: python

        import statsmodels.api as sm
        from statstests.datasets import empresas
        from statstests.process import stepwise

        # import empresas dataset
        df = empresas.get_data()

        # Estimate and fit model
        model = sm.OLS.from_formula("retorno ~ disclosure + endividamento + ativos + liquidez", df).fit()

        # Print summary
        print(model.summary())

        # Stepwise process
        stepwise(model, pvalue_limit=0.05)

    Parameters
    ----------
    model : Statsmodels model

    pvalue_limit : float

    Returns
    -------
    model : Stepwised model

    References
    ----------
    .. [1] Reference

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