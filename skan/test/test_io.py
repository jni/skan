import numpy as np
import pandas as pd

from skan import io

from skimage._shared._tempfile import temporary_file


def test_write_excel_tables():
    num_sheets = np.random.randint(1, 4)
    num_cols = np.random.randint(1, 5, size=num_sheets)
    num_rows = np.random.randint(20, 40, size=num_sheets)
    tables = []
    for m, n in zip(num_rows, num_cols):
        columns = [f'column{i}' for i in range(n)]
        data = np.random.random((m, n))
        tables.append(pd.DataFrame(data=data, columns=columns))
    sheet_names = [f'sheet {i}' for i in range(num_sheets)]
    kwargs = dict(zip(sheet_names, tables))
    kwargs['config'] = {'image files': ['image1.tif', 'image2.tif'],
                        'image format': 'fei',
                        'threshold radius': 5e-8}
    with temporary_file(suffix='.xlsx') as file:
        io.write_excel(file, **kwargs)
        tables_in = [pd.read_excel(file, sheet_name=name, index_col=0)
                     for name in sheet_names]
        config_in_df = pd.read_excel(file, sheet_name='config')
        config_in = dict(zip(config_in_df['parameters'],
                             config_in_df['values']))
    for table, table_in in zip(tables, tables_in):
        assert list(table.columns) == list(table_in.columns)
        np.testing.assert_allclose(table_in.values, table.values)
    for key, val in kwargs['config'].items():
        str(val) == str(config_in[key])
