from statstests.datasets import util


def get_data():
    data = util.load_csv(__file__, 'corruption.csv')
    return data
