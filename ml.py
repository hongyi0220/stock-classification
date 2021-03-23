import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import seaborn as sns
import numpy as np
import io

def do_ml():
    stocks_data1 = pd.read_csv('data/2014_Financial_Data.csv')
    stocks_data2 = pd.read_csv('data/2015_Financial_Data.csv')
    stocks_data3 = pd.read_csv('data/2016_Financial_Data.csv')
    stocks_data4 = pd.read_csv('data/2017_Financial_Data.csv')
    stocks_data_list = [stocks_data1, stocks_data2, stocks_data3, stocks_data4]
    for sd in stocks_data_list:
        sd.rename({sd.columns[sd.columns.str.contains('price var', case=False)][0]: 'price var'}, axis=1, inplace=True)
    stocks_data = pd.concat(stocks_data_list)
    stocks_data.reset_index(drop=True, inplace=True)
    # print(stocks_data.shape)
    stocks_data.to_csv('data/imputed_data.csv')

    stocks_data_test = pd.read_csv('data/2018_Financial_Data.csv')

    # Clean up data
    # print(stocks_data)
    stocks_data.drop(columns=['Sector'], axis=1, inplace=True)
    stocks_data_test.drop(columns=['Sector'], axis=1, inplace=True)
    stocks_data.drop(stocks_data.columns[stocks_data.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    stock_names = stocks_data_test.pop(
        stocks_data_test.columns[stocks_data_test.columns.str.contains('unnamed', case=False)][0])
    stocks_data_test.rename(
        {stocks_data_test.columns[stocks_data_test.columns.str.contains('price var', case=False)][0]: 'price var'},
        axis=1, inplace=True)
    # print(stocks_data.columns)
    price_var_ = stocks_data.pop('price var')
    price_var = stocks_data_test.pop('price var')

    # print('stock after dropping sector, unnamed, 2015 price var cols\n', stocks_data)
    labels = stocks_data.pop('Class')
    labels_test_pred_meth = stocks_data_test.pop('Class')
    # print(labels)
    # print(stocks_data)

    # Do feature elimination
    threshold_percent = 8
    threshold = (threshold_percent / 100) * stocks_data.shape[0]
    stocks_data = stocks_data.loc[:, stocks_data.isin([0]).sum() <= threshold]
    stocks_data = stocks_data.loc[:, stocks_data.isna().sum() <= threshold]
    # print(stocks_data.shape)
    features = stocks_data

    features_test = stocks_data_test.loc[:, stocks_data.columns]
    print('stock_data_test shape:', features_test.shape)

    # Fill in missing values
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputed_features = imp_mean.fit_transform(features)
    imputed_features_test = imp_mean.fit_transform(features_test)
    imputed_features_test = pd.DataFrame(imputed_features_test)

    # Add headers back
    imputed_features = pd.DataFrame(imputed_features, columns=features.columns)

    # Remove columns with same data
    nunique = imputed_features.apply(pd.Series.nunique)
    nunique_test = imputed_features_test.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    cols_to_drop_test = nunique_test[nunique_test == 1].index
    imputed_features.drop(cols_to_drop, axis=1, inplace=True)
    imputed_features_test.drop(cols_to_drop_test, axis=1, inplace=True)
    # print('after:', imputed_features.shape)

    # Export data to csv file
    data_print = pd.DataFrame(imputed_features, columns=imputed_features.columns).join(labels).to_csv(
        path_or_buf='data/Financial_Data_Imputed.csv', index=False)
    # print('imputed features:', imputed_features)

    # Divide dataset into training and test data
    features_train = imputed_features
    labels_train = labels

    # Standardize data
    scaler = StandardScaler()
    features_train_scaled = scaler.fit(features_train)
    features_train_scaled = scaler.transform(features_train)
    features_test_pred_meth = scaler.fit_transform(imputed_features_test)
    # print('scaled data mean:', features_train_scaled.mean(axis=0))
    # print('scaled data std:', features_train_scaled.std(axis=0))

    features_train_scaled = pd.DataFrame(features_train_scaled)
    features_test_pred_meth = pd.DataFrame(features_test_pred_meth)
    # print('FEATURE_TEST INDEX AFTER SCALING AND DATAFRAMING:', features_test_scaled.index)
    # print(features_train_transformed, 'shape:', features_train_transformed.shape)

    # Make corr heatmap b/w features
    corr = features_train_scaled.corr()
    print(corr)
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap)

    #bytes_image = io.BytesIO()
    #plt.savefig(bytes_image, format='png')
    plt.savefig('./plots/corr_heatmap.png')
    #bytes_image.seek(0)

    # Train model with algorithm
    # clf = RandomForestClassifier(random_state=0, criterion='entropy', max_depth=6, max_features='auto', n_estimators=128)

    #    clf.fit(features_train_scaled, labels_train)
    #    y_pred = clf.predict(features_test_scaled)
    #    print('accuracy w/ all features:', accuracy_score(labels_test, y_pred))
    #    tuned_parameters = {'n_estimators': [20, 32, 256, 512, 1024],
    #                        'max_depth': [2, 4, 5, 6, 7, 8, 10],
    #                        'max_features': ['auto', 'sqrt'],
    #                        'criterion': ['gini', 'entropy']}
    #    clf = GridSearchCV(RandomForestClassifier(random_state=1),
    #                        tuned_parameters,
    #                        n_jobs=6,
    #                        scoring='average_precision',
    #                        cv=5)
    # clf.fit(features_train_scaled, labels_train)

    # Store trained model for reuse
    # with open('models/rf.plk', 'wb') as file:
    #    joblib.dump(clf, file)

    #    print('Best score and parameters found on development set:')
    #    print('%0.3f for %r' % (clf.best_score_, clf.best_params_))

    # features_test_transformed = sel.transform(features_test_scaled)
    # features_test_scaled = sel2.transform(features_test_scaled)

    # Get 2019 price variations ONLY for the stocks in testing split
    init_invest_amount = 1000
    # print(features_test_scaled.index.values)
    # price_var_test = price_var.loc[features_test.index.values]
    price_var_test = price_var.loc[imputed_features_test.index.values]
    price_var_test = pd.DataFrame(price_var_test)
    # print(price_var_test.shape)
    # print('LABELS_TEST:', labels_test)
    #    pl_df = pd.DataFrame(np.array(labels_test), index=features_test.index.values,
    #                       columns=['class'])  # first column is the true class (buy, 1/ skip, 0)
    pl_df = pd.DataFrame(np.array(labels_test_pred_meth), index=imputed_features_test.index.values,
                         columns=['class'])  # first column is the true class (buy, 1/ skip, 0)
    # y_pred = clf.predict(features_test_scaled)

    # Reuse trained model
    with open('rf.plk', 'rb') as file:
        clf = joblib.load(file)

    y_pred = clf.predict(features_test_pred_meth)
    print('ypred:', y_pred)
    print(y_pred.shape)
    print(stock_names)
    print(stock_names.shape)

    # Show names of stocks picked
    for stock_name, pred in zip(stock_names, y_pred):
        if pred == 1:
            stock_name
    # print('stocks picked:', stock_names.iloc(axis=0)[[index for index in y_pred if index == 1], :])
    print('buy_pred/y_pred:', len(list(y for y in y_pred if y == 1)), '/', len(y_pred))
    # print(len(list(pred in y_pred for pred == 1)))
    buy_amount = init_invest_amount / len(list(y for y in y_pred if y == 1))
    pl_df['pred'] = y_pred

    pl_df['init invest amount'] = pl_df['pred'] * buy_amount
    pl_df['price var %'] = price_var_test['price var']
    pl_df['price var $'] = (price_var_test['price var'].values / 100) * pl_df['init invest amount']
    pl_df['final val'] = pl_df['init invest amount'] + pl_df['price var $']
    # print(pl_df)
    # y_pred = rgr.predict(features_test_scaled)

    total_init_value_rf = pl_df['init invest amount'].sum()
    total_final_value_rf = pl_df['final val'].sum()
    net_gain_rf = total_final_value_rf - total_init_value_rf
    percent_gain_rf = (net_gain_rf / total_init_value_rf) * 100
    sum_df = pd.DataFrame([total_init_value_rf, total_final_value_rf, percent_gain_rf],
                          index=['init val', 'final val', 'roi'], columns=['rf'])
    print(sum_df)

    print('accuracy after feat sel:', accuracy_score(labels_test_pred_meth, y_pred))

    # Make confusion matrix to track evaluation metrics
    cf_matrix = confusion_matrix(labels_test_pred_meth, y_pred)
    print(cf_matrix)
    cell_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    cell_counts = ["{0:0f}".format(v) for v in cf_matrix.flatten()]
    cell_percents = ["{0:.2%}".format(v) for v in cf_matrix.flatten() / np.sum(cf_matrix)]
    cell_labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(cell_names, cell_counts, cell_percents)]
    cell_labels = np.asarray(cell_labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=cell_labels, fmt='', cmap='Blues')

    # Make summary stats for the cf matrix
    accuracy = cf_matrix.trace() / float(np.sum(cf_matrix))
    precision = cf_matrix[1, 1] / sum(cf_matrix[:, 1])
    recall = cf_matrix[1, 1] / sum(cf_matrix[1, :])
    f1score = 2 * precision * recall / (precision + recall)
    sum_stats = "\nAccuracy:{:0.2f}\nRecall={:0.2f}\nPrecision={:0.2f}\nF1 Score={:0.2f}".format(accuracy, recall,
                                                                                                 precision, f1score)
    plt.xlabel(sum_stats)
    plt.savefig('./plots/confusion_matrix.png')

    # Classification report
    print(classification_report(labels_test_pred_meth, clf.predict(features_test_pred_meth),
                                target_names=['Skip', 'Buy']))

    # plotting feature importance
    importances = clf.feature_importances_
    # print('importances:', importances)
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.title("Feature Importance")
    plt.barh(range(features_train_scaled.shape[1]), importances[indices], color="b", yerr=std[indices], align="center")
    plt.yticks(range(features_train_scaled.shape[1]), features_train_scaled.columns)
    # plt.xlim([-1, features_train_transformed.shape[1]])
    plt.xlabel('Relative Importance')
    plt.savefig('./plots/feature_importance.png')
    plt.show()

    # Print the feature ranking
    feature_importance_df = pd.DataFrame(
        {'feature': features_train.columns, 'importance': clf.feature_importances_}).sort_values('importance',
                                                                                                 ascending=False)
    # print(feature_importance_df)

    #TODO:
    #   []line 161: zip stock names and y_pred, send it over upon get request
    #   []Clean up code
