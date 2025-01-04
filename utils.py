import copy
import os
from typing import Callable, Union, Any, List, Dict
from pathlib import Path
import dill


def ensureDir(file_path):
    directory = os.path.dirname(file_path)
    if directory != '':
        if not os.path.exists(directory):
            os.makedirs(directory)


def serialize(filename: Union[Path, str], obj: Any):
    # json only works for nested dicts
    ensureDir(filename)
    file = open(filename, 'wb')
    # dill can dump lambdas, and dill also dumps the class and not only the contents
    dill.dump(obj, file)
    file.close()


def deserialize(filename: Union[Path, str]):
    # json only works for nested dicts
    file = open(filename, 'rb')
    result = dill.load(file)
    file.close()
    return result


def cached(f: Callable) -> Callable:
    def result_f(*args, **kwargs):
        filename = '_'.join(
            [f.__name__] + ['=' + str(arg) for arg in args] + [f'{key}={value}' for key, value in
                                                               kwargs.items()]) + '.pkl'
        file_path = Path('results') / filename
        if os.path.isfile(file_path):
            return deserialize(file_path)
        else:
            result = f(*args, **kwargs)
            serialize(file_path, result)
            return result

    return result_f


def join_dicts(*dicts):
    # Attention: arguments do not commute since later dicts can override entries from earlier dicts!
    result = copy.copy(dicts[0])
    for d in dicts[1:]:
        result.update(d)
    return result


def map_nested(obj: Union[List, Dict, Any], f: Callable, dim: int):
    """
    dim=0 will apply f to obj directly, dim=1 to all elements in obj, etc.
    """
    if dim <= 0:
        return f(obj)
    elif isinstance(obj, dict):
        return {key: map_nested(value, f, dim-1) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [map_nested(value, f, dim-1) for value in obj]


def shift_dim_nested(obj: Union[List, Dict], dim1: int, dim2: int):
    # in a nested combination of lists and dicts, shift the indexing dimension dim1 to dim2
    # example: if d = {'a': [{'b': 1}, {'b': 2}]}, dim1 = 1, dim2 = 2, then the result should be
    # {'a': {'b': [1, 2]}}

    if dim1 < 0 or dim2 < 0:
        raise ValueError(f'expected dim1 >= 0 and dim2 >= 0, but got {dim1=} and {dim2=}')
    # if dim2 <= dim1:
    #     raise ValueError(f'expected dim2 > dim1, but got {dim1=} and {dim2=}')

    if dim1 > 0 and dim2 > 0:
        if isinstance(obj, dict):
            return {key: shift_dim_nested(value, dim1-1, dim2-1) for key, value in obj.items()}
        else:
            # assume that value is a list
            return [shift_dim_nested(value, dim1-1, dim2-1) for value in obj]
    elif dim1 > 1:
        # dim1 > dim2, shift backwards
        return shift_dim_nested(shift_dim_nested(obj, dim1, dim1 - 1), dim1 - 1, dim2)
    elif dim2 > 1:
        # dim2 > dim1, shift forwards
        return shift_dim_nested(shift_dim_nested(obj, dim1, dim1 + 1), dim1 + 1, dim2)
    else:
        # switch dimensions 0 and 1
        if isinstance(obj, dict):
            first = next(iter(obj.values()))
            if isinstance(first, dict):
                # swap two dicts
                return {key2: {key1: obj[key1][key2] for key1 in obj} for key2 in first}
            else:
                # assume it is a list
                return [{key1: obj[key1][i] for key1 in obj} for i in range(len(first))]
        else:
            first = obj[0]
            if isinstance(first, dict):
                return {key2: [obj[i][key2] for i in range(len(obj))] for key2 in first}
            else:
                # assume it is a list
                return [[obj[i][j] for i in range(len(obj))] for j in range(len(first))]
            pass
        pass