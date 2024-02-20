def test_replace():
    """
        Teste para validar alteração
    """
    import pytest
    import pandas as pd
    df = pd.DataFrame({'[col.1]': [1, 2], '[col.2]': [3, 4]})
    df.columns = df.columns.str.replace('\[', '', regex=True)
    df.columns = df.columns.str.replace('\.', '_', regex=True)
    df.columns = df.columns.str.replace('\]', '', regex=True)

    df_expected = pd.DataFrame({'col_1': [1, 2], 'col_2': [3, 4]})    
    assert df.columns.all() == df_expected.columns.all()
