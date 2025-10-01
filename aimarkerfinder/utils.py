import gc
import logging

from math import ceil
import os
import random
import signal
import sys
import sympy
from sympy.utilities.lambdify import lambdify
import numpy as np
import pandas as pd
from tabulate import tabulate
import torch

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

from kan import KAN
from kan.utils import SYMBOLIC_LIB

from openpyxl.utils import get_column_letter
from openpyxl.formula.translate import Translator
from openpyxl.formatting.rule import ColorScaleRule

from aimarkerfinder.classes import AEA, AEAClassificator
from aimarkerfinder.inflection_point_v1 import inflection_point_in_sorted_Y

import umap

plt.rcParams['svg.fonttype'] = 'none'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
logging.info(f"Device: {device}")

SYMBOLIC_LIB['sigmoid'] = (lambda x: torch.sigmoid(
    x), lambda x: 1 / (1 + sympy.exp(-x)))
SYMBOLIC_LIB['cosh'] = (lambda x: torch.cosh(x), lambda x: sympy.cosh(x), 5)


def read_data(input, extension=None, separator=',', decimalseparator='.'):
    if extension is None:
        _, filename = os.path.split(input)
        _, extension = os.path.splitext(filename)
    if extension == ".csv":
        df = pd.read_csv(input, header=0, sep=separator,
                         decimal=decimalseparator)
    elif extension == ".tsv":
        df = pd.read_csv(input, header=0, sep="\t", decimal=decimalseparator)
    elif extension == ".xlsx" or extension == ".xls":
        df = pd.read_excel(input, header=0)
    elif extension == ".ods":
        df = pd.read_excel(input, header=0, engine="odf")
    else:
        logging.error("Incorrect input format")
        sys.exit(-1)
    return df


def init_seed(seed):
    if seed is None:
        seed = random.randint(0, 1000000000)
    logging.info(f"Set seed number = {seed}")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def save_heatmap(filename, df_X, Ystr, yColName='Target'):
    df_X[yColName] = Ystr.to_list()
    df_X = df_X.sort_values(by=[yColName], ascending=False)
    Ystr = df_X.pop(yColName)

    Yuniq = Ystr.unique()
    rgb_list = plt.cm.get_cmap('tab10', len(Yuniq))
    cmap = [mcolors.rgb2hex(rgb_list(i)) for i in range(rgb_list.N)]
    lut = dict(zip(Yuniq, cmap))
    row_colors = Ystr.map(lut)

    # Рисуем полноразмерную карту
    plot = sns.clustermap(df_X, z_score=1, cmap="bwr",
                          figsize=((df_X.shape[1] + 1) // 2, df_X.shape[0] // 5), vmin=-3, vmax=3,
                          row_colors=row_colors, row_cluster=False, col_cluster=True)

    plot.ax_heatmap.set_ylabel("")
    plot.ax_heatmap.set_xticklabels(
        plot.ax_heatmap.get_xmajorticklabels(), fontsize=12, rotation=90)

    col = plot.ax_col_dendrogram.get_position()
    plot.ax_col_dendrogram.set_position([col.x0, col.y0, col.width, col.height * 0.3])

    # plot.cax.set_visible(False)
    plot.ax_cbar.set_ylabel("z score", size=12)
    plot.ax_cbar.tick_params(labelsize=12)
    col = plot.ax_cbar.get_position()
    plot.ax_cbar.set_position([col.x0, col.y0 * 0.8, col.width, col.height])

    plot.ax_row_colors.tick_params(labelsize=12)

    plt.savefig(f"{filename}.pdf", bbox_inches='tight')
    plt.savefig(f"{filename}.svg", bbox_inches='tight')
    plt.clf()

    # Рисуем компактную карту
    plot = sns.clustermap(df_X, z_score=1, cmap="bwr",
                          figsize=(12, 12), vmin=-3, vmax=3,
                          row_colors=row_colors, row_cluster=False, col_cluster=True)

    plot.ax_heatmap.set_ylabel("")
    plot.ax_heatmap.set_xticklabels(
        plot.ax_heatmap.get_xmajorticklabels(), fontsize=12, rotation=90)

    col = plot.ax_col_dendrogram.get_position()
    plot.ax_col_dendrogram.set_position([col.x0, col.y0, col.width, col.height * 0.3])

    # plot.cax.set_visible(False)
    plot.ax_cbar.set_ylabel("z score", size=12)
    plot.ax_cbar.tick_params(labelsize=12)
    col = plot.ax_cbar.get_position()
    plot.ax_cbar.set_position([col.x0, col.y0 * 0.8, col.width, col.height])

    plot.ax_row_colors.tick_params(labelsize=12)

    plt.savefig(f"{filename}.compact.pdf", bbox_inches='tight')
    plt.savefig(f"{filename}.compact.svg", bbox_inches='tight')
    plt.clf()


def umap_picture(X, Target, name):
    components = min(X.columns.size, 3)
    columns = [f"UM{k + 1}" for k in range(components)]
    reducer = umap.UMAP(n_components=components)
    umap_features = reducer.fit_transform(X)
    # print(umap_features.shape)
    umap_df = pd.DataFrame(
        data=umap_features,
        columns=columns)
    umap_df['target'] = Target.to_list()

    sns.set_theme(font_scale=1.25)

    g = sns.pairplot(
        data=umap_df,
        hue='target'
    )
    g.figure.set_size_inches(12, 10)

    plt.savefig(f'{name}.umap_results.pdf', bbox_inches='tight')
    plt.savefig(f'{name}.umap_results.svg', bbox_inches='tight')
    plt.clf()

    return umap_df


def pca_picture(X, Target, name):
    components = min(X.columns.size, 3)
    columns = [f"PC{k + 1}" for k in range(components)]
    pca = PCA(n_components=components)
    pca_features = pca.fit_transform(X)
    logging.info(pca.explained_variance_ratio_)
    pca_df = pd.DataFrame(
        data=pca_features,
        columns=columns)
    pca_df['target'] = Target.to_list()

    sns.set_theme(font_scale=1.25)

    g = sns.pairplot(
        data=pca_df,
        hue='target'
    )
    g.figure.set_size_inches(12, 10)

    plt.savefig(f'{name}.pca_results.pdf', bbox_inches='tight')
    plt.savefig(f'{name}.pca_results.svg', bbox_inches='tight')
    plt.clf()

    return pca_df


def plot_scores(features: pd.Series, name: str):
    plt.plot(features.index.to_list(), features.to_list())
    plt.xticks(rotation=90)
    plt.savefig(f'{name}.pdf', bbox_inches='tight')
    plt.savefig(f'{name}.svg', bbox_inches='tight')
    plt.clf()


def repeat_dataset(df: pd.DataFrame, multiplier: int) -> pd.DataFrame:
    multiplier = max(1, multiplier)
    return df.loc[df.index.repeat(multiplier)].reset_index(drop=True)


def add_noise(df: pd.DataFrame, noise_factor: float) -> pd.DataFrame:
    shape = df.shape
    std_vector = df.std().tolist()
    noise = np.random.normal(
        [0] * shape[1], np.array(std_vector, float) * noise_factor, shape
    )
    noisy_df = df + noise

    return noisy_df


def add_grouped_noise(df: pd.DataFrame, target_column: str, data_columns: list, noise_factor: float):
    """
    Добавляет шум к данным, подчиняющийся нормальному распределению с нулевым средним и дисперсией,
    равной дисперсии образцов в каждой группе.

    :param df: DataFrame с данными.
    :param target_column: Название столбца, по которому производится группировка.
    :param data_columns: Список названий столбцов, к которым добавляется шум.
    :return: DataFrame с добавленным шумом.
    """
    for column in data_columns:
        df[column] = df[column].astype('float64')

    for _, group in df.groupby(target_column):
        for column in data_columns:
            # Вычисляем дисперсию для текущей группы и столбца
            std = group[column].std()
            # Генерируем шум с нулевым средним и вычисленной дисперсией
            noise = np.random.normal(0.0, std * noise_factor, size=len(group))
            # Добавляем шум к значениям в группе
            df.loc[group.index, column] += noise.astype(df[column].dtype)

    return df


def get_std_vector(df: pd.DataFrame) -> list:
    return df.std().tolist()


def select_indexes_by_inflection_point(s_data: pd.Series, opt_shift: int = 0) -> list:
    index = inflection_point_in_sorted_Y(s_data.to_list())
    index = max(1, index) + opt_shift  # хотя бы 1 элемент должен остаться
    return s_data.iloc[0:index].index.to_list()


def select_indexes_by_derivation(s_data: pd.Series) -> list:
    indexes = s_data.index.to_list()
    prev = indexes.pop(0)
    selected_indexes = [prev]
    m = 0
    k = 0
    m_idx = -1
    for idx in indexes:
        if s_data[idx] == 0:
            val = 0
        else:
            val = (s_data[prev] - s_data[idx]) / s_data[idx]
        # print(f"idx = {prev}  val = {val}")
        if val > m:
            m = val
            m_idx = k
        prev = idx
        k += 1
    if m_idx >= 0:
        selected_indexes = selected_indexes + indexes[:m_idx]
    return selected_indexes


def select_indexes_by_derivation_gt_1(s_data: pd.Series, opt_shift: int = 0) -> list:
    indexes = s_data.index.to_list()
    prev = indexes.pop(0)
    selected_indexes = [prev]
    k = 0
    while k < len(indexes) and s_data[indexes[k]] > 1.0:
        selected_indexes.append(indexes[k])
        k += 1
    if k > 0:
        prev = indexes[k - 1]
        indexes = indexes[k:]
    m = 0
    k = 0
    m_idx = -1
    for idx in indexes:
        if s_data[idx] == 0:
            val = 0
        else:
            val = (s_data[prev] - s_data[idx]) / s_data[idx]
        if val > m:
            m = val
            m_idx = k
        prev = idx
        k += 1
    if m_idx >= 0:
        m_idx += opt_shift
        selected_indexes = selected_indexes + indexes[:m_idx]
    return selected_indexes


def get_norm_feature_importances(X, Ycl, selected_indexes, bottleneck=0):
    params = {'n_estimators': np.arange(
        10, 201, 20), 'max_depth': np.arange(3, 15, 2), 'random_state': [0]}
    rfc_grid = GridSearchCV(RandomForestClassifier(),
                            param_grid=params, n_jobs=-1)
    rfc_grid.fit(StandardScaler().fit_transform(X), Ycl)

    model = rfc_grid.best_estimator_
    feature_importances = pd.Series(model.feature_importances_,
                                    index=selected_indexes).sort_values(ascending=False)

    _, counts = np.unique(Ycl, return_counts=True)
    splits = min(5, min(counts))
    if splits > 1:
        scores = cross_val_score(model, StandardScaler().fit_transform(X), Ycl,
                                 cv=StratifiedKFold(
                                     n_splits=splits, shuffle=True, random_state=54367),
                                 n_jobs=-1)
    else:
        scores = np.array([1.0])

    median = feature_importances.median()
    if median == 0.0:
        median = feature_importances.loc[lambda x: x > 0.0].min()
    power = 2
    if bottleneck > 0 and feature_importances.size < bottleneck:
        power = (bottleneck + feature_importances.size) / bottleneck
    return feature_importances / median * scores.mean()**power


def get_important_features(norm_feature_importances: pd.Series, opt_shift: int = 0):
    indexes = select_indexes_by_inflection_point(norm_feature_importances, opt_shift)
    # indexes = select_indexes_by_derivation_gt_1(norm_feature_importances, opt_shift)
    if (norm_feature_importances[indexes[0]] >= 1.0):
        res = norm_feature_importances[indexes]
    else:
        res = norm_feature_importances.loc[lambda x: x >= 1.0]
    return res


# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    # for every Linear layer in a model..
    if isinstance(m, (torch.nn.Linear)):
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def weights_init_normal(m):
    # for every Linear layer in a model
    if isinstance(m, (torch.nn.Linear)):
        n = m.in_features
        # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0, 1.0 / np.sqrt(n))
        # m.bias.data should be 0
        m.bias.data.fill_(0)


def weights_init_xavier_uniform(m):
    # for every Linear layer in a model
    if isinstance(m, (torch.nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def weights_init_xavier_normal(m):
    # for every Linear layer in a model
    if isinstance(m, (torch.nn.Linear)):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


def get_bottleneck(df_X):
    pca = PCA(0.99)
    pca.fit(df_X)
    return pca.n_components_ + ceil(0.01 * pca.n_components_)


def learn_AEAClassificator(df_X, df_Xm, df_Xm_noise, Y):
    # Нормировка данных
    scaler = MinMaxScaler().fit(pd.concat([df_X, df_Xm_noise]))
    X_norm = scaler.transform(df_Xm_noise)
    X_out = scaler.transform(df_Xm)
    X_eval = scaler.transform(df_X)

    X_train, X_test, X_train_out, X_test_out, Y_train, Y_test = train_test_split(
        X_norm, X_out, Y, test_size=0.3)

    del scaler
    gc.collect

    input_size = df_X.columns.size
    bottleneck = get_bottleneck(df_X)
    cl_size = bottleneck // 2
    model = AEAClassificator("encoder",
                             input_size,
                             bottleneck,
                             cl_size,
                             len(Y[0])
                             ).to(device)
    # model.apply(weights_init_uniform_rule)
    logging.info(
        f"Input_size = {input_size}  bn = {bottleneck}  cl = {cl_size}  out = {len(Y[0])}")

    model.train_model(X_train, X_train_out, Y_train, X_test, X_test_out, Y_test,
                      device=device, epochs=2000)

    # check and display the accuracy of the trained model
    with torch.no_grad():
        model.load()
        model.eval()

        att, encoded, _, output = model.run(
            torch.from_numpy(X_eval).float().to(device))

    mse = root_mean_squared_error(X_eval, output)
    r_sq = r2_score(X_eval, output, multioutput="variance_weighted")

    logging.info(
        f"MSE: {mse:4f}, "
        f"R^2: {r_sq:4f}"
    )

    return att, encoded, model, bottleneck


def learn_AEA(df_X, df_Xm, df_Xm_noise):
    # Нормировка данных
    scaler = MinMaxScaler().fit(pd.concat([df_X, df_Xm_noise]))
    X_norm = scaler.transform(df_Xm_noise)
    X_out = scaler.transform(df_Xm)
    X_eval = scaler.transform(df_X)

    X_train, X_test, X_train_out, X_test_out = train_test_split(
        X_norm, X_out, test_size=0.3)

    del scaler
    gc.collect

    input_size = df_X.columns.size
    bottleneck = get_bottleneck(df_X)
    model = AEA("encoder",
                input_size,
                bottleneck
                ).to(device)
    # model.apply(weights_init_uniform_rule)
    logging.info(
        f"Input_size = {input_size}  bn = {bottleneck}")

    model.train_model(X_train, X_train_out, X_test, X_test_out,
                      device=device, epochs=2000)

    # check and display the accuracy of the trained model
    with torch.no_grad():
        model.load()
        model.eval()

        att, encoded, output = model.run(torch.from_numpy(X_eval).float().to(device))

    mse = root_mean_squared_error(X_eval, output)
    r_sq = r2_score(
        X_eval,
        output,
        multioutput="variance_weighted",
    )

    logging.info(
        f"MSE: {mse:4f}, "
        f"R^2: {r_sq:4f}"
    )

    return att, encoded, model, bottleneck


def step(df_X, df_Xm, df_Xm_noise, Y, Ycl):

    att, _, _, bottleneck = learn_AEAClassificator(df_X, df_Xm, df_Xm_noise, Y)

    df_att = pd.DataFrame(att, columns=df_X.columns)
    column_means = df_att.mean(skipna=True).sort_values(ascending=False)

    # Выбираем колонки, важные для классификации и сжатия
    selected_indexes = select_indexes_by_derivation(column_means)
    # selected_indexes = select_indexes_by_inflection_point(column_means)

    logging.debug(f"Encoder select indexes = {selected_indexes}")

    if len(selected_indexes) > 0:
        norm_feature_importances = get_norm_feature_importances(
            df_X[selected_indexes], Ycl, selected_indexes, bottleneck)
        logging.debug(
            f"Classifier norm. features = {norm_feature_importances}")

        xgb_selected_indexes = get_important_features(
            norm_feature_importances).index.to_list()
        logging.debug(f"Classifier selected indexes = {xgb_selected_indexes}")
        return xgb_selected_indexes
    else:
        return selected_indexes


def Select_features(df_X, df_Xm, df_Xm_noise, Y, Ycl, min_empty_steps=2):
    feature_indexes = []
    feature_df_X = pd.DataFrame()
    feature_df_Xm_noise = pd.DataFrame()

    empty_step = 0
    k = 0
    while empty_step < min_empty_steps:
        n_f_i = step(df_X, df_Xm, df_Xm_noise, Y, Ycl)

        if len(n_f_i) > 0:
            feature_df_X = pd.concat(
                [feature_df_X, df_X[n_f_i]], axis=1, ignore_index=False)
            feature_df_Xm_noise = pd.concat(
                [feature_df_Xm_noise, df_Xm_noise[n_f_i]], axis=1, ignore_index=False)
            df_X = df_X.drop(n_f_i, axis=1)
            df_Xm = df_Xm.drop(n_f_i, axis=1)
            df_Xm_noise = df_Xm_noise.drop(n_f_i, axis=1)
            feature_indexes += n_f_i
            empty_step = 0
        else:
            empty_step += 1

        k += 1

        gc.collect

    if len(feature_indexes) > 0:
        # Score features
        norm_feature_importances = get_norm_feature_importances(
            feature_df_X, Ycl, feature_indexes)
        return feature_df_Xm_noise[norm_feature_importances.index], feature_df_X[norm_feature_importances.index], norm_feature_importances
    else:
        return feature_df_Xm_noise, feature_df_X, pd.Series(feature_indexes, index=feature_indexes)


def formular_accuracy(formulas, X, y):
    batch = X.shape[0]
    correct = 0
    if batch > 0:
        bin_formulas = list(map(lambda x: lambdify(expr=x, args=sorted(list(
            x.free_symbols), key=lambda sym: int(str(sym)[2:])), modules=["math"]), formulas))
        for i in range(batch):
            logit = []
            for formular in bin_formulas:
                args = np.array(X[i].detach().cpu())
                logit.append(formular(*args))
            correct += (np.argmax(logit)) == (np.argmax(y[i]))
    else:
        for i in range(batch):
            logit = []
            for formular in formulas:
                f = formular
                for k in range(X.shape[1]):
                    f = f.subs(f"x_{k + 1}", X[i, k])
                logit.append(np.array(f).astype(np.float64))
            correct += (np.argmax(logit)) == (np.argmax(y[i]))
    return correct / batch


def kan_model_accuracy(model, X, y):
    res = model(X)
    batch = res.shape[0]
    correct = 0
    for i in range(batch):
        logit = res[i]
        correct += (torch.argmax(logit)) == (torch.argmax(y[i]))
    return correct / batch


# KAN learning
def KAN_learning(train_input, train_label, test_input, test_label, feature_indexes,
                 min_accuracy=0.999, min_best_result_repeated=3,
                 seed_range=range(0, 100), min_seed_steps=0, use_formular_accuracy=False, force_formular_prune=False):

    def handler(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, handler)

    dataset = {}
    dataset['train_input'] = train_input.to(device)
    dataset['train_label'] = train_label.to(device)
    dataset['test_input'] = test_input.to(device)
    dataset['test_label'] = test_label.to(device)

    best_acc = 0.0
    best_formular = []
    best_count = 0
    step = 0

    for k in seed_range:
        try:
            logging.info(f"Seed = {k}")
            model = KAN(width=[len(feature_indexes), max(2, int(len(feature_indexes) / 3)), train_label[0].size()[0]],
                        grid=5,
                        k=3,
                        scale_base_mu=1, scale_base_sigma=2,
                        noise_scale=0.3,
                        seed=k,
                        auto_save=False,
                        symbolic_enabled=True,
                        device=device
                        )
            model.update_grid_from_samples(dataset)
            model.fit(dataset, opt="LBFGS", steps=200,
                      loss_fn=torch.nn.CrossEntropyLoss(),
                      lamb=0.01, lamb_entropy=10.0,
                      reg_metric='edge_forward_spline_n'
                      )

            lib = ['x',
                   'x^2', 'x^3', 'x^4',
                   'abs',
                   '1/x', '1/x^2', '1/x^3', '1/x^4',
                   'exp', 'log', 'sqrt',
                   'tanh', 'sin', 'tan', '1/sqrt(x)',
                   'gaussian'
                   # , 'cosh','sigmoid',
                   ]
            if use_formular_accuracy:
                model.auto_symbolic(lib=lib, r2_threshold=0.0, verbose=0)
                formulas = model.symbolic_formula()[0]
                accuracy = formular_accuracy(formulas,
                                             dataset['test_input'].detach(
                                             ).cpu(),
                                             dataset['test_label'].detach().cpu()
                                             )
            else:
                accuracy = kan_model_accuracy(
                    model, dataset['test_input'], dataset['test_label'])

            logging.info(f'train acc of the formula: {accuracy}')

            if accuracy > best_acc:
                best_acc = accuracy
                best_count = 1
                best_model = model
            elif accuracy == best_acc:
                best_count += 1

            step += 1
            if step > min_seed_steps:
                if accuracy > min_accuracy or best_count >= min_best_result_repeated:
                    break
        except KeyboardInterrupt:
            break
        except ValueError as ex:
            logging.warning(ex)

    # Try to prune model
    try:
        best_model.saveckpt("tmp_copy")
        prune_model = KAN.loadckpt("tmp_copy").prune(
            node_th=1e-2, edge_th=3e-2)
        os.remove("tmp_copy_cache_data")
        os.remove("tmp_copy_config.yml")
        os.remove("tmp_copy_state")

        if use_formular_accuracy:
            prune_model.auto_symbolic(lib=lib, r2_threshold=0.0, verbose=0)
            formulas = prune_model.symbolic_formula()[0]
            prune_accuracy = formular_accuracy(formulas,
                                               dataset['test_input'].detach(
                                               ).cpu(),
                                               dataset['test_label'].detach(
                                               ).cpu()
                                               )
        else:
            prune_accuracy = kan_model_accuracy(
                prune_model, dataset['test_input'], dataset['test_label'])
        logging.info(f"Accuracy after prune: {prune_accuracy}")
        if force_formular_prune or (prune_accuracy >= best_acc):
            best_model = prune_model
            best_acc = prune_accuracy
    except Exception as e:
        logging.warning("Can't prune KAN model")
        logging.exception(e)

    if not use_formular_accuracy:
        best_model.auto_symbolic(lib=lib, r2_threshold=0.0, verbose=0)
    best_formular = best_model.symbolic_formula()[0]
    # Дообучение модели, если r2_threshold > 0 и не все функции определены
    # best_model.fit(dataset, opt="LBFGS", steps=200,
    #                  loss_fn=torch.nn.CrossEntropyLoss(),
    #                  lamb=0.01, lamb_entropy=10.0,
    #                  reg_metric='edge_forward_spline_n')
    # best_model.auto_symbolic(lib=lib, r2_threshold=0.5, verbose=1)
    # best_formular = best_model.symbolic_formula()[0]
    # print(best_formular)

    if not use_formular_accuracy:
        accuracy = formular_accuracy(best_formular,
                                     dataset['test_input'].detach().cpu(),
                                     dataset['test_label'].detach().cpu()
                                     )
        logging.info(f"Best formular accuracy: {accuracy}")
    else:
        accuracy = kan_model_accuracy(
            best_model, dataset['test_input'], dataset['test_label'])
        logging.info(f"Best model accuracy: {accuracy}")
    return best_acc, best_formular


def convert_sympy_to_excel(sympy_formula):
    """
    Конвертирует формулу sympy в формулу Excel.

    Args:
        sympy_formula (sympy.Expr): Формула sympy, которую нужно конвертировать.

    Returns:
        str: Формула Excel, соответствующая входной формуле sympy.
    """
    # Замена операторов и функций sympy на их аналоги в Excel
    replacements = {
        '**': '^',
        'sin': 'SIN',
        'cos': 'COS',
        'tan': 'TAN',
        'log': 'LOG',
        'exp': 'EXP',
        'sqrt': 'SQRT'
    }

    # Преобразование формулы sympy в строку
    excel_formula = str(sympy_formula)

    # Замена символов и функций sympy на их аналоги в Excel
    for sympy_symbol, excel_symbol in replacements.items():
        excel_formula = excel_formula.replace(sympy_symbol, excel_symbol)

    # Добавление знака равенства в начало формулы Excel
    excel_formula = f'={excel_formula}'

    return excel_formula


def save_result_table(feature_df_X, yColName, Ystr, rf_results, c45_formular, kan_formulars, writer):
    outdf = feature_df_X.copy()
    outdf[yColName] = Ystr
    k = 1
    for f in kan_formulars:
        outdf[f"KAN F{k}"] = None
        num = outdf.columns.get_loc(f"KAN F{k}")
        for kk in range(feature_df_X.shape[1]):
            Letter = get_column_letter(kk + 1)
            f = f.subs(sympy.Symbol(f"x_{kk + 1}"), sympy.Symbol(f"{Letter}2"))
        exF = convert_sympy_to_excel(f)
        outdf[f"KAN F{k}"] = [Translator(exF, origin=f"{get_column_letter(num + 1)}2").translate_formula(
            f"{get_column_letter(num + 1)}{i + 2}") for i in range(len(outdf))]
        k += 1

    frange = "%s2:%s2" % (get_column_letter(outdf.columns.get_loc("KAN F1") + 1),
                          get_column_letter(outdf.columns.get_loc(f"KAN F{len(kan_formulars)}") + 1))
    resF = f"=MATCH(MAX({frange}), {frange}, 0)"
    outdf["KAN result"] = None
    outdf["KAN result"] = [Translator(resF, origin=f"{get_column_letter(num + 1)}2").translate_formula(
        f"{get_column_letter(num + 1)}{i + 2}") for i in range(len(outdf))]

    outdf["RF result"] = rf_results + 1

    outdf["C4.5 result"] = None
    num = outdf.columns.get_loc("C4.5 result")
    outdf["C4.5 result"] = [Translator(c45_formular, origin=f"{get_column_letter(num + 1)}2").translate_formula(
        f"{get_column_letter(num + 1)}{i + 2}") for i in range(len(outdf))]

    outdf.to_excel(writer, index=False, sheet_name=yColName)

    num = outdf.columns.get_loc("KAN result")
    writer.book[yColName].conditional_formatting.add(f'{get_column_letter(num + 1)}2:{get_column_letter(num + 1)}{len(outdf) + 2}',
                                                     ColorScaleRule(start_type='min', start_color='FF0000', end_type='max', end_color='00FF00'))

    num = outdf.columns.get_loc("RF result")
    writer.book[yColName].conditional_formatting.add(f'{get_column_letter(num + 1)}2:{get_column_letter(num + 1)}{len(outdf) + 2}',
                                                     ColorScaleRule(start_type='min', start_color='FF0000', end_type='max', end_color='00FF00'))

    num = outdf.columns.get_loc("C4.5 result")
    writer.book[yColName].conditional_formatting.add(f'{get_column_letter(num + 1)}2:{get_column_letter(num + 1)}{len(outdf) + 2}',
                                                     ColorScaleRule(start_type='min', start_color='FF0000', end_type='max', end_color='00FF00'))


def random_forest_classification(X_train, Y_train, X_test, Y_test):
    # Создаем и обучаем классификатор методом случайного леса
    params = {'n_estimators': np.arange(
        10, 201, 20), 'max_depth': np.arange(3, 15, 2), 'random_state': [0]}
    grid = GridSearchCV(RandomForestClassifier(),
                        param_grid=params, n_jobs=-1)
    # grid.fit(X_train, Y_train)
    grid.fit(X_test, Y_test)

    model = grid.best_estimator_

    # Предсказываем значения для тестовой выборки
    Y_pred = model.predict(X_test)

    metrics = calculate_classification_metrics(model, X_test, Y_test, Y_pred)

    return metrics, Y_pred


class Node:
    def __init__(self, condition=None, left=None, right=None, class_label=None):
        self.condition = condition
        self.left = left
        self.right = right
        self.class_label = class_label

    def is_leaf(self):
        return self.class_label is not None


def parse_tree(tree_str):
    lines = tree_str.strip().split('\n')
    root = None
    stack = []

    for line in lines:
        line = line.rstrip()
        if not line:
            continue

        # Определяем уровень вложенности
        indent_part = line.split('|--- ')[0]
        current_level = indent_part.count('|   ')

        # Парсим содержимое строки
        content = line.split('|--- ')[1].strip()
        is_leaf = content.startswith('class: ')

        # Создаем узел
        if is_leaf:
            class_label = content.split(': ')[1]
            node = Node(class_label=class_label)
        else:
            condition = content.replace(' >  ', '>').replace(' <= ', '<=')  # Чистим пробелы
            node = Node(condition=condition)

        # Находим родителя
        while stack and stack[-1][0] > current_level:
            stack.pop()

        if not stack:
            root = node
            stack.append((current_level, node))
        else:
            # Привязываем к родителю
            parent = stack[-1][1]
            if parent.left is None:
                parent.left = node
            else:
                parent.right = node

        stack.append((current_level, node))

    return root  # Игнорируем фиктивный корень


def generate_excel_formula(root):
    def generate(node):
        if node is None:
            return '"Ошибка: отсутствует ветвь"'
        if node.is_leaf():
            return int(node.class_label) + 1
        else:
            left = generate(node.left)
            right = generate(node.right)
            if node.right is not None:
                return f'IF({node.condition}, {left}, {right})'
            else:
                return f'{left}'
    return generate(root)


def c45_classification(X_train, Y_train, X_test, Y_test):

    params = {'criterion': ['entropy'],
              'max_depth': np.arange(1, 21).tolist()[0::2] + [None],
              'min_samples_split': np.arange(2, 11).tolist()[0::2],
              'max_leaf_nodes': np.arange(3, 26).tolist()[0::2] + [None],
              'max_features': ['sqrt'],
              'random_state': np.arange(0, 10).tolist()}

    grid = GridSearchCV(DecisionTreeClassifier(),
                        param_grid=params, n_jobs=-1)
    # grid.fit(X_train, Y_train)
    grid.fit(X_test, Y_test)
    # print(grid.best_params_)

    model = grid.best_estimator_

    # Ypred = model.predict(X_test)
    metrics = calculate_classification_metrics(model, X_test, Y_test)

    tree_rules = export_text(model, max_depth=model.get_depth(), decimals=5, feature_names=[f"{get_column_letter(kk + 1)}2" for kk in range(X_test.shape[1])])
    root = parse_tree(tree_rules)
    excel_formula = f"={generate_excel_formula(root)}"
    # print(tree_rules)
    # print(excel_formula)
    return metrics, excel_formula


def calculate_classification_metrics(model, x, y, y_pred=None):
    """
    Calculate various classification metrics.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True labels.
    - y_pred: array-like of shape (n_samples,)
        Predicted labels by the classifier.

    Returns:
    - metrics_dict: dict
        Dictionary containing calculated metrics.
    """

    # Calculate CV score
    _, counts = np.unique(y, return_counts=True)
    splits = min(5, min(counts))
    scores = cross_val_score(
        model, x, y,
        cv=StratifiedKFold(n_splits=splits, shuffle=True, random_state=54367),
        n_jobs=-1)
    cv_score = scores.mean()

    if y_pred is None:
        y_pred = model.predict(x)

    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)

    # Calculate precision
    precision = precision_score(y, y_pred, average='weighted')

    # Calculate recall
    recall = recall_score(y, y_pred, average='weighted')

    # Calculate F1 score
    f1 = f1_score(y, y_pred, average='weighted')

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y, y_pred)

    # Generate classification report
    class_report = classification_report(y, y_pred, output_dict=True)

    # Create a dictionary to store all metrics
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cv_score': cv_score,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }

    return metrics_dict


def metrics_to_string(metrics):
    """
    Print the classification metrics in a readable format.

    Parameters:
    - metrics: dict
        Dictionary containing calculated metrics.
    """
    # Print overall metrics
    results = ["Overall Metrics:"]
    overall_metrics = [
        ['Accuracy', f"{metrics['accuracy']:.4f}"],
        ['Precision', f"{metrics['precision']:.4f}"],
        ['Recall', f"{metrics['recall']:.4f}"],
        ['F1 Score', f"{metrics['f1_score']:.4f}"],
        ['CV Score', f"{metrics['cv_score']:.4f}"]
    ]
    results.append(tabulate(overall_metrics, headers=['Metric', 'Value'], tablefmt='grid'))

    # Print confusion matrix
    conf_matrix = metrics['confusion_matrix']
    classes = list(range(len(conf_matrix)))
    conf_matrix_data = [[str(j) for j in i] for i in conf_matrix]
    results.append("\nConfusion Matrix:")
    results.append(tabulate(conf_matrix_data, headers=[''] + [f'Predicted {c}' for c in classes], showindex=[f'True {c}' for c in classes], tablefmt='grid'))

    # Print classification report
    class_report = metrics['classification_report']
    report_data = []
    for label, values in class_report.items():
        if isinstance(values, dict):
            row = [label] + [f"{v:.4f}" for v in values.values()]
            report_data.append(row)
        else:
            report_data.append([label, f"{values:.4f}"])

    results.append("\nClassification Report:")
    headers = ['Class'] + list(next(iter(class_report.values())).keys())
    results.append(tabulate(report_data, headers=headers, tablefmt='grid'))

    return "\n".join(results)
