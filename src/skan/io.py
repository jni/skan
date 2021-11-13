import pandas as pd


def _params_dict_to_dataframe(d):
    s = pd.Series(d)
    s.index.name = 'parameters'
    f = pd.DataFrame({'values': s})
    return f


def write_excel(filename, **kwargs):
    """Write data tables to an Excel file, using kwarg names as sheet names.
    
    Parameters
    ----------
    filename : str
        The filename to write to.
    kwargs : dict
        Mapping from sheet names to data.
    """
    writer = pd.ExcelWriter(filename)
    for sheet_name, obj in kwargs.items():
        if isinstance(obj, dict):
            obj = _params_dict_to_dataframe(obj)
        if isinstance(obj, pd.DataFrame):
            obj.to_excel(writer, sheet_name=sheet_name)
    writer.save()
    writer.close()
