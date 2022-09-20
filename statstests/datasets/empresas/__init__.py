from statstests.datasets import util

def get_data():
    data = util.load_csv(__file__, 'empresas.csv')
    return data