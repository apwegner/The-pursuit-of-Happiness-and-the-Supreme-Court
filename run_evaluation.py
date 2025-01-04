import numbers
import time
from typing import Optional, Dict, List

import sklearn
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import matplotlib.pyplot as plt

import utils
from main import train_model, get_preprocessed_dataset, get_splits

import scipy.stats as ss
import pandas as pd
import numpy as np

from plotting import create_horizontal_barplot


def cramers_corrected_stat(x, y):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    # taken from https://stackoverflow.com/a/54625589
    result = -1
    if len(x.value_counts()) == 1:
        print("First variable is constant")
    elif len(y.value_counts()) == 1:
        print("Second variable is constant")
    else:
        conf_matrix = pd.crosstab(x, y)

        if conf_matrix.shape[0] == 2:
            correct = False
        else:
            correct = True

        chi2 = ss.chi2_contingency(conf_matrix, correction=correct)[0]

        n = sum(conf_matrix.sum())
        phi2 = chi2 / n
        r, k = conf_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        result = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    return round(result, 6)


def eval_preds(classes: List, y_pred_proba: np.ndarray, y_test: pd.Series) -> Dict[str, float]:
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    # y_true = np.asarray(y_test.map(class_to_index).astype(int))
    y_true = y_test.map(class_to_index)

    y_pred_proba = np.asarray(y_pred_proba)

    # print(f'{y_true=}, {y_pred_proba=}')

    y_true_oh = np.zeros_like(y_pred_proba)
    y_true_oh[np.arange(y_pred_proba.shape[0]), y_true] = 1.0

    results = dict()
    results['accuracy'] = accuracy_score(y_true, np.argmax(y_pred_proba, axis=-1))
    results['error'] = 1.0 - results['accuracy']
    results['brier'] = np.mean(np.sum((y_pred_proba - y_true_oh) ** 2, axis=-1))
    results['logloss'] = log_loss(y_true, y_pred_proba, labels=[0, 1, 2])
    results['auroc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', labels=[0, 1, 2])
    results['auroc_ovo'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovo', labels=[0, 1, 2])
    return results


def eval_clf(clf: sklearn.base.BaseEstimator, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    return eval_preds(classes=clf.classes_, y_pred_proba=clf.predict_proba(x_test), y_test=y_test)


@utils.cached
def get_perm_importances(use_backtesting: bool, n_folds: int, n_repeats: int,
                         model_type: str = 'autogluon',
                         time_limit: float = 120.0, num_bag_sets: int = 1,
                         backtest_test_size: Optional[int] = None,
                         only_after_1865: bool = False, part_as_target: bool = False,
                         n_shuffles: int = 10,
                         only_majority_opinions: bool = False) -> tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    # returns permutation_importances (indexed by metric_name, col_name) and
    # a kind of permutation correlation importances (indexed by encoded_col_name)
    df = get_preprocessed_dataset(part_as_target=part_as_target, only_majority_opinions=only_majority_opinions)
    splits = get_splits(use_backtesting=use_backtesting, n_folds=n_folds, n_repeats=n_repeats,
                        backtest_test_size=backtest_test_size, only_after_1865=only_after_1865,
                        only_majority_opinions=only_majority_opinions)

    all_results = []
    all_perm_results = []
    all_perm_corr_results = []

    rng = np.random.default_rng(seed=0)

    for split_idx, (train_idxs, test_idxs) in enumerate(splits):
        print(f'Evaluating split {split_idx}')
        clf = train_model(split_idx, use_backtesting=use_backtesting, n_folds=n_folds, n_repeats=n_repeats,
                          model_type=model_type, time_limit=time_limit, num_bag_sets=num_bag_sets,
                          backtest_test_size=backtest_test_size,
                          only_after_1865=only_after_1865, part_as_target=part_as_target,
                          only_majority_opinions=only_majority_opinions)
        df_test = df.iloc[test_idxs]
        x_test = df_test.drop(columns='Target')
        y_test = df_test['Target']
        results = eval_clf(clf, x_test, y_test)
        all_results.append(results)

        poh_idx = list(clf.classes_).index('poH')
        h_idx = list(clf.classes_).index('H')

        print(f'{clf.classes_=}, {poh_idx=}, {h_idx=}')

        perm_results = dict()
        perm_corr_results = dict()

        x_test_list = []
        for column in x_test.columns:
            print(f'Shuffling column {column}')
            col_results = []
            for shuffle_idx in range(n_shuffles):
                x_test_copy = x_test.copy()
                x_test_copy[column] = rng.permutation(x_test_copy[column].values)
                x_test_list.append(x_test_copy)
            perm_results[column] = col_results

        x_test_concat = pd.concat(x_test_list)
        y_pred_proba_concat = clf.predict_proba(x_test_concat)

        loop_idx = 0
        for column in x_test.columns:
            print(f'Evaluating column {column}')
            col_results = []
            start_idx = loop_idx * len(x_test)
            for shuffle_idx in range(n_shuffles):
                col_results.append(eval_preds(clf.classes_,
                                              y_pred_proba_concat[
                                              loop_idx * len(x_test):(loop_idx + 1) * len(x_test)],
                                              y_test))
                loop_idx += 1
            end_idx = loop_idx * len(x_test)

            perm_results[column] = col_results

            # compute prediction correlations of permuted (encoded) variable with target (P(poH) - P(H))
            x_test_column = x_test_concat.iloc[start_idx:end_idx][[column]]
            x_test_column_enc = one_hot_encode_non_binary_categories(x_test_column)
            y_pred_column = y_pred_proba_concat[start_idx:end_idx]
            y_pred_column_target = y_pred_column[:, poh_idx] - y_pred_column[:, h_idx]

            # print(x_test_column_enc.head())

            for col_name in x_test_column_enc.columns:
                print(f'Evaluating correlations for column {col_name}')
                if len(x_test_column_enc[col_name].unique()) > 1:
                    corr = x_test_column_enc[col_name].corr(pd.Series(y_pred_column_target))
                    if not np.isnan(corr):
                        perm_corr_results[col_name] = corr
                    else:
                        print(f'Got a NaN value')

        all_perm_results.append(perm_results)
        all_perm_corr_results.append(perm_corr_results)

    all_results = utils.shift_dim_nested(all_results, 0, 1)
    all_results = {key: np.mean(value) for key, value in all_results.items()}

    # indexing was (fold, column, shuffle_idx, metric_name)
    # change it to (column, metric_name, fold, shuffle_idx)
    all_perm_results = utils.shift_dim_nested(utils.shift_dim_nested(all_perm_results, 0, 1), 3, 1)
    all_perm_results = utils.map_nested(all_perm_results, np.mean, dim=2)

    # indexing was (fold, column), we change it to (column, fold) and then average over the fold dimension
    # do it like this because some keys might not appear in all folds
    perm_corr_keys = set(sum([list(r.keys()) for r in all_perm_corr_results], []))
    all_perm_corr_results = {key: [r[key] for r in all_perm_corr_results if key in r] for key in perm_corr_keys}
    # all_perm_corr_results = utils.shift_dim_nested(all_perm_corr_results, 0, 1)
    all_perm_corr_results = utils.map_nested(all_perm_corr_results, np.mean, dim=1)

    # reverse permutation importances for metrics where higher is better
    col_names = list(all_perm_results.keys())
    reverse_metrics = ['accuracy', 'auroc_ovr', 'auroc_ovo']
    perm_importances = {metric_name: {col_name:
                                          (-1 if metric_name in reverse_metrics else 1) * (
                                                  all_perm_results[col_name][metric_name] - all_results[
                                              metric_name]) for col_name in col_names
                                      } for metric_name in all_results}

    return perm_importances, all_perm_corr_results


def eval_perm_importances(use_backtesting: bool, n_folds: int, n_repeats: int,
                          model_type: str = 'autogluon',
                          time_limit: float = 120.0, num_bag_sets: int = 1,
                          backtest_test_size: Optional[int] = None,
                          only_after_1865: bool = False, part_as_target: bool = False,
                          n_shuffles: int = 10,
                          only_majority_opinions: bool = False):
    start_time = time.time()
    perm_importances, perm_correlations = get_perm_importances(use_backtesting, n_folds, n_repeats, model_type,
                                                               time_limit, num_bag_sets,
                                                               backtest_test_size, only_after_1865, part_as_target,
                                                               n_shuffles,
                                                               only_majority_opinions)

    filename_suffix = f'{model_type}_folds-{n_folds}_repeats-{n_repeats}_time-{time_limit:g}'
    filename_suffix = filename_suffix + f'_nbs-{num_bag_sets}_btts-{backtest_test_size}_oa1865-{only_after_1865}'
    filename_suffix = filename_suffix + f'_pat-{part_as_target}_nshuffle-{n_shuffles}_only-majority-{only_majority_opinions}.pdf'

    brier_perm_imp = list(perm_importances['brier'].values())
    perm_imp_keys = list(perm_importances['brier'].keys())
    top_idxs = np.argsort(brier_perm_imp)[::-1][:20]
    top_keys = [perm_imp_keys[i] for i in top_idxs]
    max_brier_imp = np.max(brier_perm_imp)

    matching_keys = {key: [k for k in perm_importances['brier'].keys() if key.startswith(k)] for key in
                     perm_correlations.keys()}
    for key, value in matching_keys.items():
        if len(value) != 1:
            raise ValueError(f'Expected one value for key {key}, but got {value}')

    matching_keys = {key: value[0] for key, value in matching_keys.items()}

    # filter to only use columns who are in the top permutation importance columns
    perm_correlations = {key: value for key, value in perm_correlations.items() if matching_keys[key] in top_keys}

    norm_corr_imp = [max(0.0, 0.1 + 0.9 * perm_importances['brier'][matching_keys[key]] / max_brier_imp) for key in
                     perm_correlations.keys()]

    create_horizontal_barplot(list(perm_correlations.keys()),
                              list(perm_correlations.values()),
                              alphas=norm_corr_imp,
                              filename=f'perm_correlations_{filename_suffix}',
                              title=r'Permutation correlations with prediction $P(\mathrm{poH}) - P(\mathrm{H})$',
                              xlabel=r'Pearson correlation coefficient between permuted variable and prediction $P(\mathrm{poH}) - P(\mathrm{H})$',
                              ylabel='',
                              sort_by='abs')


    for metric_name in perm_importances:
        create_horizontal_barplot(list(perm_importances[metric_name].keys()),
                                  list(perm_importances[metric_name].values()),
                                  filename=f'perm_importances_{metric_name}_{filename_suffix}',
                                  title=f'Permutation importances ({metric_name})',
                                  xlabel='Permutation importance', ylabel='',
                                  sort_by='id')

    end_time = time.time()
    print(f'Time: {end_time - start_time:g} s')


def eval_model_simple(use_backtesting: bool, n_folds: int, n_repeats: int,
                      model_type: str = 'autogluon',
                      time_limit: float = 120.0, num_bag_sets: int = 1,
                      backtest_test_size: Optional[int] = None,
                      only_after_1865: bool = False, part_as_target: bool = False,
                      only_majority_opinions: bool = False):
    print(f'Evaluating model {model_type}')

    start_time = time.time()
    df = get_preprocessed_dataset(part_as_target=part_as_target, only_majority_opinions=only_majority_opinions)
    splits = get_splits(use_backtesting=use_backtesting, n_folds=n_folds, n_repeats=n_repeats,
                        backtest_test_size=backtest_test_size, only_after_1865=only_after_1865,
                        only_majority_opinions=only_majority_opinions)

    all_results = []

    for split_idx, (train_idxs, test_idxs) in enumerate(splits):
        # print(f'Evaluating split {split_idx}')
        clf = train_model(split_idx, use_backtesting=use_backtesting, n_folds=n_folds, n_repeats=n_repeats,
                          model_type=model_type, time_limit=time_limit, num_bag_sets=num_bag_sets,
                          backtest_test_size=backtest_test_size,
                          only_after_1865=only_after_1865, part_as_target=part_as_target,
                          only_majority_opinions=only_majority_opinions)
        df_test = df.iloc[test_idxs]
        x_test = df_test.drop(columns='Target')
        y_test = df_test['Target']
        results = eval_clf(clf, x_test, y_test)
        all_results.append(results)

    all_results = utils.shift_dim_nested(all_results, 0, 1)
    all_results = {key: np.mean(value) for key, value in all_results.items()}

    print(f'{all_results=}')

    end_time = time.time()
    print(f'Time: {end_time - start_time:g} s')


def create_time_plot():
    df = get_preprocessed_dataset(part_as_target=False)
    df = df.sort_values(by=['date_float'])

    for target_value in ['H', 'H+poH', 'poH']:
        plt.plot(df['date_float'], np.cumsum(df['Target'] == target_value), label=target_value)

    plt.axvline(1866.0, color='k', linestyle='--', label='1866')
    plt.xlabel('Year')
    plt.ylabel('Total number of decisions')

    plt.title('Cumulative incidences of decisions')

    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/time_plot.pdf')
    plt.close()


def create_time_plot_xiv():
    df = get_preprocessed_dataset(part_as_target=False)
    df = df.sort_values(by=['date_float'])

    plt.plot(df['date_float'], np.cumsum(df['Amd. XIV U. S. Const.'] == 1), label='Amd. XIV U. S. Const.')

    plt.axvline(1866.0, color='k', linestyle='--', label='1866')
    plt.xlabel('Year')
    plt.ylabel('Total number of decisions')

    plt.title('Cumulative incidences of usage of Amd. XIV U. S. Const.')

    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/time_plot_xiv.pdf')
    plt.close()


def analyze_cramer_v():
    df = get_preprocessed_dataset(part_as_target=False)
    df = df.drop(columns='date_float')

    columns = []
    v_values = []
    for column in df.columns:
        if column != 'Target':
            columns.append(column)
            v_values.append(cramers_corrected_stat(df[column], df['Target']))

    # sort by descending v-value
    perm = np.argsort(v_values)[::-1]
    columns = [columns[i] for i in perm]
    v_values = [v_values[i] for i in perm]

    print(f'Cramers V for different columns, descending:')
    for c, v in zip(columns, v_values):
        print(f'{c}: {v:g}')

    columns_nonzero = [c for c, v in zip(columns, v_values) if v > 1e-8]
    v_values_nonzero = [v for c, v in zip(columns, v_values) if v > 1e-8]

    df_cramer = pd.DataFrame({
        'Column': columns,
        "Cramer's V": v_values
    })

    df_cramer.to_csv('plots/cramer_v.csv')

    create_horizontal_barplot(columns_nonzero, v_values_nonzero, filename='cramer_v.pdf',
                              xlabel="Cramer's V association with target", ylabel='',
                              title="Associations with target variable")


def one_hot_encode(df: pd.DataFrame, col_names: List[str]) -> pd.DataFrame:
    df = df.copy()
    new_columns = dict()
    for col_name in col_names:
        for value in pd.unique(df[col_name]):
            new_columns[f'{col_name} = {value}'] = (df[col_name] == value).astype(int)

    df = df.drop(columns=col_names)
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    return df


def one_hot_encode_non_binary_categories(df: pd.DataFrame):
    return one_hot_encode(df,
                          [col for col in df.columns if len(df[col].unique()) > 2
                           and not all([isinstance(v, numbers.Number) for v in df[col].unique()])
                           # and df[col].dtype.name == 'category'
                           ])


def analyze_correlations():
    df = get_preprocessed_dataset(part_as_target=False)
    for column in df.columns:
        if all([isinstance(v, int) for v in df[column].unique()]):
            df[column] = df[column].astype(int)
        # if len(df[column].unique()) > 2 and not all([isinstance(v, int) for v in df[column].unique()]):
        #     print(f'{column=}')
    target = df['Target']
    target = target.map(lambda v: {'H': -1, 'H+poH': 0, 'poH': 1}[v])
    df = df.drop(columns=['Target'])
    df = one_hot_encode_non_binary_categories(df)

    columns = []
    corr_values = []
    for column in df.columns:
        if column != 'Target':
            columns.append(column)
            corr_values.append(df[column].corr(target))

    # sort by descending v-value
    perm = np.argsort(corr_values)[::-1]
    columns = [columns[i] for i in perm]
    corr_values = [corr_values[i] for i in perm]

    print(f'Correlations for different columns, descending:')
    for c, v in zip(columns, corr_values):
        print(f'{c}: {v:g}')

    df_cramer = pd.DataFrame({
        'Column': columns,
        "Correlation": corr_values
    })

    df_cramer.to_csv('plots/correlations.csv')

    create_horizontal_barplot(columns, corr_values, filename='correlations.pdf', xlabel="Correlation with (poH - H)",
                              ylabel='',
                              title="Correlations with (poH - H)", sort_by='abs')


def run_evals():
    create_time_plot()
    create_time_plot_xiv()
    analyze_cramer_v()
    analyze_correlations()

    for only_after_1865 in [True, False]:
        for only_majority_opinions in [True, False]:
            eval_perm_importances(use_backtesting=False, n_folds=5, n_repeats=2, model_type='rf',
                                  only_after_1865=only_after_1865, only_majority_opinions=only_majority_opinions)


if __name__ == '__main__':
    run_evals()
