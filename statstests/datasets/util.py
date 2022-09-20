from os.path import abspath, dirname, exists, expanduser, join
import pandas as pd

def load_csv(base_file, csv_name, sep=',', convert_float=False):
    """Standard simple csv loader"""
    filepath = dirname(abspath(base_file))
    filename = join(filepath,csv_name)
    engine = 'python' if sep != ',' else 'c'
    float_precision = {}
    if engine == 'c':
        float_precision = {'float_precision': 'high'}
    data = pd.read_csv(filename, sep=sep, engine=engine, **float_precision)
    if convert_float:
        data = data.astype(float)
    return data