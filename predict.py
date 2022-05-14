#!/usr/bin/env python
# coding: utf-8

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle
import sys
import os


SEPSIS_LABEL = 'SepsisLabel'
REMOVED_FEATURES = ['Bilirubin_direct', 'TroponinI', 'Fibrinogen', 'Hgb']


def crop_df(patient_df):
    """
    Crop the input df to contain only rows upto (and including) the first tow with SepsisLabel=1.
    If no such row, return the patient_df as is.
    """
    if 1 in patient_df[SEPSIS_LABEL].unique():
        first_row_sepsis = patient_df[patient_df[SEPSIS_LABEL] == 1].iloc[0].name
        return patient_df.iloc[:first_row_sepsis + 1]
    return patient_df


def feature_agg(patient_df):
    """
    Aggregate the input patient_df as described.
    For each column (besides the label column), calculate the Mean, Std, Min and Max
    Return Series containing the new values.
    """
    patient_df = patient_df.drop([SEPSIS_LABEL], axis=1)
    mean = patient_df.mean().to_numpy()
    std = patient_df.std(ddof=0).to_numpy()
    mins = patient_df.min().to_numpy()
    maxs = patient_df.max().to_numpy()

    patient_vector = np.concatenate((mean, std, mins, maxs), axis=0)
    patient_series = pd.Series(patient_vector).fillna(-1)

    return patient_series


def aggregate_dataset(data_path, data_set, agg_method):
    """
    Aggregate the given data-set using the given aggregation method.
    For each patient in the data-set, create an aggregation vector, and construct a df out of all the patients.
    Save the new df to a psv file.
    """

    agg_df = pd.DataFrame()

    for patient_file in os.listdir(data_path):
        patient_df = pd.read_csv(os.path.join(data_path, patient_file), sep='|')
        patient_df = patient_df.drop(REMOVED_FEATURES, axis=1)
        patient_df = crop_df(patient_df)
        patient_series = agg_method(patient_df)
        is_sick = 1 if 1 in patient_df[SEPSIS_LABEL].unique() else 0
        patient_series['PatientLabel'] = is_sick

        patient_series['PatientId'] = int(patient_file.split('.')[0].split('_')[1])

        agg_df = agg_df.append(patient_series, ignore_index=True)

    return agg_df


def print_stats(values,preds,probas):
    print(f"F1 score {f1_score(values, preds, average='binary')}")
    print(f"Precision score {precision_score(values, preds, average='binary')}")
    print(f"Recall score {recall_score(values, preds, average='binary')}")
    print(f"Accuracy score {accuracy_score(values, preds)}")
    print(f"ROC-AUC score {roc_auc_score(values, probas)}")


def main():
    # load the pretrained model from the training phase
    rfc = pickle.load(open('RF_Classifier.pkl', 'rb'))
    test_path = str(sys.argv[1])

    aggregated_test = aggregate_dataset(test_path, "test_set_predict", feature_agg)
    patient_id = aggregated_test['PatientId']
    aggregated_test = aggregated_test.drop(['PatientId'], axis=1)
    X_test, Y_test = aggregated_test.loc[:, aggregated_test.columns != 'PatientLabel'], aggregated_test['PatientLabel']

    preds_rfc = rfc.predict(X_test)
    predict_df = pd.DataFrame(zip(patient_id, preds_rfc), columns=['Id', 'SepsisLabel'])
    predict_df.to_csv('prediction.csv', index=False, header=False)
    print_stats(Y_test, predict_df['SepsisLabel'].to_numpy(), rfc.predict_proba(X_test)[:, 1])


if __name__ == '__main__':
    main()
