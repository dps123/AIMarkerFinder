#!/usr/bin/env python3

import logging

import gc
import os
import sys
import shlex
import argparse
import pickle

import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from imblearn.over_sampling import RandomOverSampler

import torch

from aimarkerfinder.utils import KAN_learning, Select_features, add_grouped_noise, c45_classification, get_important_features, get_norm_feature_importances, init_seed, learn_AEAClassificator, metrics_to_string, pca_picture, plot_scores, random_forest_classification, read_data, repeat_dataset, save_heatmap, save_result_table, umap_picture

if 'DEBUG' in os.environ and os.environ['DEBUG'] == '1':
    log_level = logging.DEBUG
else:
    log_level = logging.INFO


def data_prepare(df, yColName, balance=False, random_seed=None):
    dfw = df.loc[df[yColName].notnull()].copy()
    dfw[yColName] = dfw[yColName].astype('str')

    if balance:
        ros = RandomOverSampler(sampling_strategy='not majority', random_state=random_seed)
        X, Y = ros.fit_resample(dfw.select_dtypes(include='number'), dfw[yColName])
        dfw = X.copy()
        dfw[yColName] = Y

    df_X = dfw.select_dtypes(include='number')
    mult_factor = max(1, max(3000, df_X.shape[1] * 10) // dfw.shape[0])
    logging.debug(f"Samples size = {df_X.shape[0]}")
    logging.debug(f"Multiplication factor = {mult_factor}")

    df_mult = repeat_dataset(dfw, mult_factor)
    df_Xm = df_mult.select_dtypes(include='number')
    df_Ym = df_mult[[yColName]]

    if mult_factor > 0:
        add_grouped_noise(df_mult, yColName, df_X.columns, 0.25)
        df_Xm_noise = df_mult.select_dtypes(include='number')
    else:
        df_Xm_noise = df_Xm

    # Классы для Нейронной сети
    OH_encoder = OneHotEncoder(handle_unknown='ignore')
    Y = OH_encoder.fit_transform(df_Ym).toarray()
    Yeval = OH_encoder.transform(dfw[[yColName]]).toarray()

    Ystr = dfw[yColName]
    Le = LabelEncoder()
    Ycl = Le.fit_transform(Ystr)
    Yclm_noise = Le.transform(df_mult[yColName])

    return df_X, df_Xm, df_Xm_noise, Ystr, Y, Yeval, Ycl, Yclm_noise


def main():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input", metavar="input", help="Input file (csv, tsv, xls, xlsx)", action="store", nargs=None, const=None, default=None, type=None, choices=None)
    parser.add_argument("-CC", metavar=None, dest="classcolumn", help="Columns with classification data (default: all non-numeric fields)", action="store", nargs="*", const=None, default=None, type=str)
    parser.add_argument("-f", dest="forcelearning", help="Force the learning process", action="store_true", default=False)
    parser.add_argument("-pca", dest="pca_res", help="The use of hidden layer for solving classification task", action="store_true", default=False)
    parser.add_argument("-pca-only", dest="pca_only", help="Make PCA and UMAP analysis for input data", action="store_true", default=False)
    parser.add_argument("-s", dest="separator", help="CSV field separator (default: '%(default)s')", action="store", default=",", type=str)
    parser.add_argument("-ds", dest="decimalseparator", help="CSV decimal separator (default: '%(default)s')", action="store", default=".", choices=[".", ","], type=str)
    parser.add_argument("-balance", dest="balance", help="Balance dataset", action="store_true", default=None)
    parser.add_argument("-mes", metavar=None, dest="minemptystep", help="Minimum empty steps for autoencoder before stopping (default: %(default)s)", action="store", nargs=None, const=None, default=2, type=int, choices=None)
    parser.add_argument("-seed", metavar=None, dest="seed", help="Seed number for autoencoder leaning (default: %(default)s)", action="store", nargs=None, const=None, default=None, type=int, choices=None)
    parser.add_argument("-use-kan-only", dest="kanonly", help="Use only KAN model", action="store_true", default=None)
    parser.add_argument("-kan-mss", metavar=None, dest="minseedstep", help="Minimum KAN learning steps before stopping (default: %(default)s)", action="store", nargs=None, const=None, default=0, type=int, choices=None)
    parser.add_argument("-kan-range-begin", metavar=None, dest="minrange", help="Range for KAN learning starts from this value (default: %(default)s)", action="store", nargs=None, const=None, default=0, type=int, choices=None)
    parser.add_argument("-kan-range-end", metavar=None, dest="maxrange", help="Range for KAN learning stops at this value (default: %(default)s)", action="store", nargs=None, const=None, default=100, type=int, choices=None)
    parser.add_argument("-kan-min-best-results", metavar=None, dest="minbestresrepeate", help="Minimum best result repeat count before stop (default: %(default)s)", action="store", nargs=None, const=None, default=3, type=int, choices=None)
    parser.add_argument("-kan-disable-formular-accuracy", dest="use_formular_accuracy", help="Not use formular accuracy", action="store_false", default=True)
    parser.add_argument("-kan-formular-prune", dest="force_formular_prune", help="Force formular prune", action="store_true", default=None)
    parser.add_argument("-optimum-shift", metavar=None, dest="opt_shift", help="Shift index for optimal featureset (default: %(default)s)", action="store", nargs=None, const=None, default=0, type=int, choices=None)
    args = parser.parse_args()

    workDir, filename = os.path.split(args.input)
    name, extension = os.path.splitext(filename)

    if workDir == '':
        workDir = '.'

    # configure logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    name_suffix = ""
    if args.pca_res:
        name_suffix = ".pca"
    if args.pca_only:
        name_suffix = ".pca_only"

    logging.basicConfig(
        format="%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])" if log_level == logging.DEBUG else "%(levelname)s (%(asctime)s): %(message)s",
        datefmt="%d/%m/%Y %I:%M:%S %p",
        level=log_level,
        filename=f"{workDir}/{name}{name_suffix}.log",
        filemode="w",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))

    logging.info(f"Run cmd: {shlex.join([os.path.basename(sys.argv[0])] + sys.argv[1:])}")

    df = read_data(args.input, extension, args.separator, args.decimalseparator)

    if args.classcolumn is None or len(args.classcolumn) == 0:
        ycols = []
        for yColName in df.select_dtypes(exclude='number').columns.tolist():
            dfnn = df.loc[df[yColName].notnull()]
            if dfnn[yColName].unique().size < dfnn[yColName].size:
                ycols.append(yColName)
    else:
        ycols = args.classcolumn

    seed = init_seed(args.seed)

    if args.pca_only:
        for yColName in ycols:
            logging.info(f"Select '{yColName}' classification field")
            df_X, df_Xm, df_Xm_noise, Ystr, Y, _, _, _ = data_prepare(df, yColName, args.balance, seed)

            umap_picture(df_X, Ystr, f"{workDir}/{name}.{yColName}.original")
            pca_picture(df_X, Ystr, f"{workDir}/{name}.{yColName}.original")

    elif args.pca_res:
        for yColName in ycols:
            logging.info(f"Select '{yColName}' classification field")
            df_X, df_Xm, df_Xm_noise, Ystr, Y, _, _, _ = data_prepare(df, yColName, args.balance, seed)

            _, encoded, model, _ = learn_AEAClassificator(df_X, df_Xm, df_Xm_noise, Y)
            del df_Xm
            df_Xm = None
            del df_Xm_noise
            df_Xm_noise = None
            del Y
            Y = None
            gc.collect

            model.save_to(f"{workDir}/{name}.{yColName}.best_model.pt")

            umap_picture(df_X, Ystr, f"{workDir}/{name}.{yColName}.original")
            pca_picture(df_X, Ystr, f"{workDir}/{name}.{yColName}.original")
            del df_X
            df_X = None
            gc.collect

            df_encoded = pd.DataFrame(encoded)
            umap_picture(df_encoded, Ystr, f"{workDir}/{name}.{yColName}")
            resDf = pca_picture(df_encoded, Ystr, f"{workDir}/{name}.{yColName}")
            resDf.to_csv(f"{workDir}/{name}{name_suffix}.{yColName}.tsv", index=False, sep="\t")

            df_encoded[yColName] = Ystr
            df_encoded.to_csv(f"{workDir}/{name}.embedings.{yColName}.tsv", index=False, sep="\t")

    else:
        with pd.ExcelWriter(f"{workDir}/{name}.results.xlsx", engine="openpyxl") as writer:
            sheetCount = 0
            for yColName in ycols:
                logging.info(f"Select '{yColName}' classification field")
                if args.kanonly:
                    feature_df_X, df_Xm, feature_df_Xm_noise, Ystr, Y, Yeval, Ycl, Yclm_noise = data_prepare(df, yColName, args.balance, seed)
                    features = get_norm_feature_importances(feature_df_X, Ycl, feature_df_X.columns)
                else:
                    if os.path.exists(f"{workDir}/{name}.{yColName}.saved_state.pkl") and not args.forcelearning:
                        with open(f"{workDir}/{name}.{yColName}.saved_state.pkl", "rb") as f:
                            st = pickle.load(f)
                            feature_df_Xm_noise = st.pop(0)
                            feature_df_X = st.pop(0)
                            features = st.pop(0)
                            Y = st.pop(0)
                            Yeval = st.pop(0)
                            Ystr = st.pop(0)
                            Ycl = st.pop(0)
                            Yclm_noise = st.pop(0)
                    else:
                        df_X, df_Xm, df_Xm_noise, Ystr, Y, Yeval, Ycl, Yclm_noise = data_prepare(df, yColName, args.balance, seed)

                        feature_df_Xm_noise, feature_df_X, features = Select_features(
                            df_X, df_Xm, df_Xm_noise, Y, Ycl,
                            min_empty_steps=args.minemptystep)
                        with open(f"{workDir}/{name}.{yColName}.saved_state.pkl", "wb") as f:
                            st = pickle.dump([feature_df_Xm_noise, feature_df_X,
                                             features, Y, Yeval, Ystr, Ycl, Yclm_noise], f)

                gc.collect

                # features = features.loc[lambda x: x >= 1.0]
                # print(f"\n\n{features}")
                if features.size > 0:
                    try:
                        logging.info(f"\n\tClassification by field '{yColName}'")
                        logging.info(f"\n{features}\n")
                        features.to_excel(f"{workDir}/{name}.{yColName}.features.xlsx")
                        plot_scores(features, f"{workDir}/{name}.{yColName}.features")
                        # features = features.loc[lambda x: x > 0.0]
                        if not args.kanonly and features.size > 5:
                            features = get_important_features(features, args.opt_shift)
                            logging.info(f"\nOptimized set of features:\n{features}\n")

                        save_heatmap(f"{workDir}/{name}.{yColName}.heatmap", feature_df_X[features.index], Ystr, yColName)

                        metrics, rf_results = random_forest_classification(
                            feature_df_Xm_noise[features.index], Yclm_noise,
                            feature_df_X[features.index], Ycl
                        )
                        logging.info(f"\n\tRF metrics:\n{metrics_to_string(metrics)}\n")

                        metrics, c45_formular = c45_classification(
                            feature_df_Xm_noise[features.index], Yclm_noise,
                            feature_df_X[features.index], Ycl
                        )
                        logging.info(f"\n\tC4.5 metrics:\n{metrics_to_string(metrics)}\n")

                        best_acc, best_formular = KAN_learning(
                            torch.from_numpy(feature_df_Xm_noise[features.index].to_numpy()).float(),
                            torch.from_numpy(Y),
                            torch.from_numpy(feature_df_X[features.index].to_numpy()).float(),
                            torch.from_numpy(Yeval),
                            features.index.to_list(),
                            min_seed_steps=args.minseedstep,
                            seed_range=range(args.minrange, args.maxrange),
                            min_best_result_repeated=args.minbestresrepeate,
                            use_formular_accuracy=args.use_formular_accuracy,
                            force_formular_prune=args.force_formular_prune
                        )
                        gc.collect

                        logging.info(f"\n\tBest accuracy: {best_acc}")
                        logging.info("\n\tBest formulars:\n")
                        k = 1
                        for f in best_formular:
                            logging.info(f"\n\t\tF{k} = {f}\n")
                            k += 1

                        save_result_table(feature_df_X, yColName, Ystr, rf_results, c45_formular, best_formular, writer)
                        sheetCount += 1

                        # sympy.pprint(best_formular[0], use_unicode=True)
                        # sympy.pprint(best_formular[1], use_unicode=True)
                        # print("\n\n")
                        # if best_prune_acc > 0:
                        #    sympy.pprint(best_prune_formular[0], use_unicode=True)
                        #    sympy.pprint(best_prune_formular[1], use_unicode=True)

                        # sympy.preview(best_formular[0], output='pdf', viewer='okular')
                        # sympy.preview(best_formular[1], output='pdf', viewer='okular')
                    except Exception as e:
                        logging.error(f"Can not learn KAN model\nException: {type(e)}\n\t{str(e)}")
                        if log_level == logging.DEBUG:
                            logging.exception(e)
                else:
                    logging.error("Empty feature list")
            if sheetCount == 0:
                pd.DataFrame().to_excel(writer, index=False, sheet_name="empty")
    sys.exit()


if __name__ == "__main__":
    main()
