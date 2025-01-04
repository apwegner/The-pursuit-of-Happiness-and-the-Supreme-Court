import pathlib
from enum import unique
from pathlib import Path
from typing import Callable, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from pandas.core.dtypes.common import is_float_dtype
from pytabkit.models.alg_interfaces.alg_interfaces import AlgInterface
from pytabkit.models.alg_interfaces.catboost_interfaces import CatBoostSubSplitInterface
from pytabkit.models.alg_interfaces.ensemble_interfaces import CaruanaEnsembleAlgInterface
from pytabkit.models.alg_interfaces.lightgbm_interfaces import LGBMSubSplitInterface
from pytabkit.models.alg_interfaces.nn_interfaces import NNAlgInterface
from pytabkit.models.alg_interfaces.sub_split_interfaces import SingleSplitWrapperAlgInterface
from pytabkit.models.alg_interfaces.xgboost_interfaces import XGBSubSplitInterface
from pytabkit.models.sklearn.default_params import DefaultParams
from pytabkit.models.sklearn.sklearn_base import AlgInterfaceClassifier
from pytabkit.models.sklearn.sklearn_interfaces import Ensemble_TD_Classifier
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import utils


def date_to_float(series: pd.Series) -> pd.Series:
    return series.dt.year + (series.dt.dayofyear - 1) / (series.dt.is_leap_year.apply(lambda x: 366 if x else 365))


@utils.cached
def load_dataset() -> pd.DataFrame:
    print(f'loading dataset')
    filename = 'data/10_with_vandv.xlsx'
    df = pd.read_excel(filename)
    return df


@utils.cached
def get_preprocessed_dataset(part_as_target: bool = False, only_majority_opinions: bool = False) -> pd.DataFrame:
    # print(f'preprocessing for {part_as_target=}')
    df = load_dataset()
    columns = list(df.columns)
    columns_richter = [columns[i] for i in range(len(columns)) if
                       columns.index('Story') <= i <= columns.index('keine Richter')]
    columns_state = [columns[i] for i in range(len(columns)) if
                     columns.index('Nebraska') <= i <= columns.index('West Virginia')]
    columns_ignore = ['ID', 'Name', 'Fallnummer', 'Bereich', 'Scope']
    df = df.drop(columns=columns_richter + columns_state + columns_ignore)
    # df['keine Richter'] = df['keine Richter'].fillna(0.0).astype(bool)  # column dropped

    df['date_float'] = date_to_float(df['Datum der Entscheidung'])
    df = df.drop(columns='Datum der Entscheidung')
    # don't do one-hot encoding

    if only_majority_opinions:
        df = df.loc[df['Teil des Urteils'] == 'M']
        # print(df['Teil des Urteils'])
        # print(df['date_float'])
        df = df.drop(columns='Teil des Urteils')
        # print(f'{len(df)=}')

    # H and poH as one target column
    df['Target'] = ['+'.join((['H'] if H else []) + (['poH'] if poH else [])) for H, poH in zip(df['H'], df['poH'])]
    df = df.drop(columns=['H', 'poH'])
    if part_as_target:
        df['Target'] = [target + '|' + part for target, part in zip(df['Target'], df['Teil des Urteils'])]
        df = df.drop(columns='Teil des Urteils')

    for col in df.columns:
        if not is_float_dtype(df[col]):
            df[col] = df[col].astype('category')

    return df.reset_index(drop=True)


def get_idx_splits(n_samples: int, use_backtesting: bool, n_folds: int, n_repeats: int,
                   backtest_test_size: Optional[int] = None, seed: int = 0):
    rng = np.random.default_rng(seed=seed)

    splits = []
    if not use_backtesting:
        for _ in range(n_repeats):
            perm = rng.permutation(n_samples)
            split_points = [0] + [round(n_samples * i / n_folds) for i in range(1, n_folds + 1)]
            # print(f'{split_points=}')
            for i in range(n_folds):
                start = split_points[i]
                stop = split_points[i + 1]
                splits.append((np.concatenate([perm[:start], perm[stop:]], axis=0), perm[start:stop]))
    else:
        assert backtest_test_size is not None
        for repeat_idx in range(n_repeats):
            offset = round(backtest_test_size * repeat_idx / n_repeats)
            for fold_idx in range(n_folds):
                n_total = n_samples - offset - fold_idx * backtest_test_size
                n_train = n_total - backtest_test_size
                assert n_train >= 1
                splits.append((np.arange(n_train), np.arange(n_train, n_total)))

    return splits


@utils.cached
def get_splits(use_backtesting: bool, n_folds: int, n_repeats: int, backtest_test_size: Optional[int] = None,
               only_after_1865: bool = False, only_majority_opinions: bool = False) -> List[Tuple[np.ndarray, np.ndarray]]:
    # returns (train_idxs, test_idxs) tuples
    # print(f'get_splits with {use_backtesting=}, {n_folds=}, {n_repeats=}, {backtest_test_size=}, {only_after_1865=}')
    df = get_preprocessed_dataset(part_as_target=False, only_majority_opinions=only_majority_opinions)  # it's the same for both variants
    unique_dates_sorted = np.sort(np.asarray(df['date_float'].unique()))

    if only_after_1865:
        unique_dates_sorted = unique_dates_sorted[unique_dates_sorted >= 1866.0]

    n_samples = len(unique_dates_sorted)
    # print(f'{n_samples=}')

    seed = 0
    while True:
        idx_splits = get_idx_splits(n_samples, use_backtesting=use_backtesting, n_folds=n_folds, n_repeats=n_repeats,
                                    backtest_test_size=backtest_test_size, seed=seed)
        # print(f'{idx_splits[0]=}')

        def convert_idxs(idxs: np.ndarray) -> np.ndarray:
            dates = unique_dates_sorted[idxs]
            return np.asarray([i for i in range(len(df)) if df['date_float'].iloc[i] in dates], dtype=np.int32)

        splits = [(convert_idxs(train), convert_idxs(val)) for train, val in idx_splits]

        # check if all classes are present in train and val, otherwise try again with a different seed
        success = True
        for train, val in splits:
            if len(df.iloc[train]['Target'].unique()) < 3 or len(df.iloc[val]['Target'].unique()) < 3:
                success = False

        if success:
            return splits

        seed += 1
        print(f'Retrying with {seed=}')


class Ensemble_TD_Classifier_2(AlgInterfaceClassifier):
    def __init__(self, device: Optional[str] = None, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_cv: int = 1, n_refit: int = 0, val_fraction: float = 0.2, n_threads: Optional[int] = None,
                 tmp_folder: Optional[Union[str, pathlib.Path]] = None, verbosity: int = 0):
        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.val_fraction = val_fraction
        self.n_threads = n_threads
        self.tmp_folder = tmp_folder
        self.verbosity = verbosity

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        td_interfaces = [
            SingleSplitWrapperAlgInterface(
                [LGBMSubSplitInterface(**DefaultParams.LGBM_TD_CLASS, val_metric_name='cross_entropy', allow_gpu=False)
                 for i in range(n_cv)]),
            SingleSplitWrapperAlgInterface(
                [XGBSubSplitInterface(**DefaultParams.XGB_TD_CLASS, val_metric_name='cross_entropy', allow_gpu=False)
                 for i in range(n_cv)]),
            SingleSplitWrapperAlgInterface([CatBoostSubSplitInterface(**DefaultParams.CB_TD_CLASS,
                                                                      val_metric_name='cross_entropy', allow_gpu=False)
                                            for i in range(n_cv)]),
            NNAlgInterface(**utils.join_dicts(DefaultParams.RealMLP_TD_CLASS,
                                              dict(use_ls=False, val_metric_name='cross_entropy'))),
        ]
        return CaruanaEnsembleAlgInterface(td_interfaces, val_metric_name='cross_entropy')

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


@utils.cached
def train_model(split_idx: int, use_backtesting: bool, n_folds: int, n_repeats: int,
                model_type: str = 'autogluon',
                time_limit: float = 120.0, num_bag_sets: int = 1,
                backtest_test_size: Optional[int] = None,
                only_after_1865: bool = False, part_as_target: bool = False,
                only_majority_opinions: bool = False):
    df = get_preprocessed_dataset(part_as_target=part_as_target, only_majority_opinions=only_majority_opinions)
    splits = get_splits(use_backtesting=use_backtesting, n_folds=n_folds, n_repeats=n_repeats,
                        backtest_test_size=backtest_test_size, only_after_1865=only_after_1865,
                        only_majority_opinions=only_majority_opinions)
    train_idxs, test_idxs = splits[split_idx]
    train = df.iloc[train_idxs]
    test = df.iloc[test_idxs]
    if model_type == 'autogluon':
        clf = TabularPredictor(label='Target', eval_metric='log_loss')
        clf.fit(train, presets='best_quality', time_limit=time_limit, num_bag_sets=num_bag_sets)
        print('Test leaderboard:')
        print(clf.leaderboard(test))
    elif model_type == 'dummy':
        clf = DummyClassifier()
        clf.fit(train.drop(columns='Target'), train['Target'])
    elif model_type == 'ensemble-td':
        clf = Ensemble_TD_Classifier_2(n_cv=5)
        print(f'{train.dtypes=}')
        clf.fit(train.drop(columns='Target'), train['Target'].to_numpy())
    elif model_type == 'rf':
        # clf = StringToCategoryWrapper(RandomForestClassifier(n_estimators=500))
        clf = Pipeline(
            [('one_hot', OneHotEncoder(handle_unknown='ignore')), ('rf', RandomForestClassifier(n_estimators=1000, n_jobs=-1))])
        clf.fit(train.drop(columns='Target'), train['Target'].to_numpy())
    else:
        raise ValueError(f'Unknown model_type "{model_type}"')
    return clf
