import pandas as pd
import matplotlib.pyplot as plt
from pandas.tests.frame.methods.test_sort_values import ascending
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
import seaborn as sns
import numpy as np

if __name__ == '__main__':
    stocks_data = pd.read_csv('data/2018_Financial_Data.csv')


    # Clean up data
#    for i, row in stocks_data.iterrows():
#        price_var = row[stocks_data.columns.str.contains('price var', case=False)]
#        #print('price var:', price_var)
#        if price_var.item() < 10:
#            #print('class value changed')
#            stocks_data.at[i, 'Class'] = 0


    #print(stocks_data)
    stocks_data.drop(columns=['Sector'], axis=1, inplace=True)
    stocks_data.drop(stocks_data.columns[stocks_data.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    stocks_data.rename({stocks_data.columns[stocks_data.columns.str.contains('price var', case=False)][0]: 'price var'}, axis=1, inplace=True)
    #print(stocks_data.columns)
    price_var = stocks_data.pop('price var')


    #print('stock after dropping sector, unnamed, 2015 price var cols\n', stocks_data)
    labels = stocks_data.pop('Class')
    #labels = stocks_data.pop(stocks_data.columns[stocks_data.columns.str.contains('2015 price var', case=False)][0])
    #print(labels)
    #print(stocks_data)
    threshold_percent = 11
    threshold = (threshold_percent / 100) * 3808
    stocks_data = stocks_data.loc[:, stocks_data.isin([0]).sum() <= threshold]
    stocks_data = stocks_data.loc[:, stocks_data.isna().sum() <= threshold]
    #print(stocks_data.shape)
    features = stocks_data

    # Fill in missing values
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputed_features = imp_mean.fit_transform(features)

    # Add headers back
    imputed_features = pd.DataFrame(imputed_features, columns=features.columns)

    # Remove columns with same data
    nunique = imputed_features.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    imputed_features.drop(cols_to_drop, axis=1, inplace=True)
    #print('after:', imputed_features.shape)

    # Export data to csv file
    data_print = pd.DataFrame(imputed_features, columns=imputed_features.columns).join(labels).to_csv(path_or_buf='data/Financial_Data_Imputed.csv', index=False)
    #print('imputed features:', imputed_features)


    # Divide dataset into training and test data
#    kf = KFold(shuffle=True)
#    for train_indices, test_indices in kf.split(imputed_features, labels):
#        features_train = [imputed_features.iloc[i] for i in train_indices]
#        features_test = [imputed_features.iloc[i] for i in test_indices]
#        labels_train = [labels.iloc[i] for i in train_indices]
#        labels_test = [labels.iloc[i] for i in test_indices]
        #print(train_indices)
        #print(labels_train)
        #print(labels_test)
    features_train, features_test, labels_train, labels_test = train_test_split(imputed_features, labels, test_size=0.2, random_state=0)
    #features_train = features_train.iloc[:, :]
    #features_test = features_test.iloc[:, :]
    #features_train = pd.DataFrame(features_train)
    #features_test = pd.DataFrame(features_test)
    print('FEATURE_TEST INDEX:', features_test.index)
    #print(features_train)

    # Standardize data
    scaler = StandardScaler()
    features_train_scaled = scaler.fit(features_train)
    features_train_scaled = scaler.transform(features_train)
    features_test_scaled = scaler.transform(features_test)
    #print('scaled data mean:', features_train_scaled.mean(axis=0))
    #print('scaled data std:', features_train_scaled.std(axis=0))

    # Do feature elimination
    #sel = SelectKBest(f_classif, k=10).fit(features_train_scaled, labels_train)
    #print('feat sel score:', sel.scores_)
    #features_train_transformed = sel.transform(features_train_scaled)
    features_train_scaled = pd.DataFrame(features_train_scaled)
    features_test_scaled = pd.DataFrame(features_test_scaled)
    print('FEATURE_TEST INDEX AFTER SCALING AND DATAFRAMING:', features_test_scaled.index)
    #print(features_train_transformed, 'shape:', features_train_transformed.shape)


    # Make corr heatmap b/w features
    #corr = features_train_scaled.corr()
    #print(corr)
    #mask = np.triu(np.ones_like(corr, dtype=np.bool))
    #cmap = sns.diverging_palette(230, 20, as_cmap=True)
    #sns.heatmap(corr, mask=mask, cmap=cmap)


    # Train model with algorithm
    clf = RandomForestClassifier(random_state=0, criterion='entropy', max_depth=4, max_features='auto', n_estimators=20)
    #rgr = linear_model.LinearRegression()
    #rgr.fit(features_train_scaled, labels_train)

    # RFE
    #sel2 = RFE(rgr, n_features_to_select=10)
    #features_train_scaled = sel2.fit_transform(features_train_scaled, labels_train)
    #print(sel2.support_)
    #print(sel2.ranking_)
    #rgr.fit(features_train_scaled, labels_train)

    clf.fit(features_train_scaled, labels_train)
#    y_pred = clf.predict(features_test_scaled)
#    print('accuracy w/ all features:', accuracy_score(labels_test, y_pred))
    tuned_parameters = {'n_estimators': [32, 256, 512, 1024]}
                        #'max_depth': [4, 5, 6, 7, 8]}
                        #'max_features': ['auto', 'sqrt'],
                        #'criterion': ['gini', 'entropy']}
    #clf = GridSearchCV(RandomForestClassifier(random_state=1, criterion='entropy', max_features='auto', max_depth=4),
#                        tuned_parameters,
#                        n_jobs=6,
#                        scoring='average_precision',
#                        cv=5)
    #clf.fit(features_train_scaled, labels_train)
    #print('Best score and parameters found on development set:')
    #print('%0.3f for %r' % (clf.best_score_, clf.best_params_))
    #Best score and parameters found on development set: 0.321 for {'criterion': 'entropy', 'max_depth': 4, 'max_features': 'auto', 'n_estimators': 32}

    #features_test_transformed = sel.transform(features_test_scaled)
    #features_test_scaled = sel2.transform(features_test_scaled)

    # Get 2019 price variations ONLY for the stocks in testing split
    init_invest_amount = 1000
    #print(features_test_scaled.index.values)
    price_var_test = price_var.loc[features_test.index.values]
    price_var_test = pd.DataFrame(price_var_test)
    #print(price_var_test.shape)
    #print('LABELS_TEST:', labels_test)
    pl_df = pd.DataFrame(np.array(labels_test), index=features_test.index.values,
                       columns=['class'])  # first column is the true class (buy, 1/ skip, 0)
    y_pred = clf.predict(features_test_scaled)
    #print('ypred:', y_pred)
    print('buy_pred/y_pred:', len(list(y for y in y_pred if y == 1)), '/', len(y_pred))
    #print(len(list(pred in y_pred for pred == 1)))
    buy_amount = init_invest_amount / len(list(y for y in y_pred if y == 1))
    pl_df['pred'] = y_pred

    pl_df['init invest amount'] = pl_df['pred'] * buy_amount
    pl_df['price var %'] = price_var_test['price var']
    pl_df['price var $'] = (price_var_test['price var'].values / 100) * pl_df['init invest amount']
    pl_df['final val'] = pl_df['init invest amount'] + pl_df['price var $']
    print(pl_df)
    #y_pred = rgr.predict(features_test_scaled)

    total_init_value_rf = pl_df['init invest amount'].sum()
    total_final_value_rf = pl_df['final val'].sum()
    net_gain_rf = total_final_value_rf - total_init_value_rf
    percent_gain_rf = (net_gain_rf / total_init_value_rf) * 100
    sum_df = pd.DataFrame([total_init_value_rf, total_final_value_rf, percent_gain_rf], index=['init val', 'final val', 'roi'], columns=['rf'])
    print(sum_df)


    print('accuracy after feat sel:', accuracy_score(labels_test, y_pred))

#    # The coefficients
#    print('Coefficients: \n', rgr.coef_)
#    # The mean squared error
#    print('Mean squared error: %.2f'
#          % mean_squared_error(labels_test, y_pred))
#    # The coefficient of determination: 1 is perfect prediction
#    print('Coefficient of determination: %.2f'
#          % r2_score(labels_test, y_pred))

    # Plot outputs
#    plt.scatter(features_test_scaled[:, 0], labels_test, color='black')
#    plt.plot(features_test_scaled, y_pred, color='blue', linewidth=3)
#
#    plt.xticks(())
#    plt.yticks(())
#
#    plt.show()

    # Make confusion matrix to track evaluation metrics
    cf_matrix = confusion_matrix(labels_test, y_pred)
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
    f1score = 2*precision*recall / (precision + recall)
    sum_stats = "\nAccuracy:{:0.2f}\nRecall={:0.2f}\nPrecision={:0.2f}\nF1 Score={:0.2f}".format(accuracy, recall, precision, f1score)
    plt.xlabel(sum_stats)

    # Classification report
    print(classification_report(labels_test, clf.predict(features_test_scaled), target_names=['Skip', 'Buy']))

   # plotting feature importance
    importances = clf.feature_importances_
    #print('importances:', importances)
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.title("Feature Importance")
    plt.barh(range(features_train_scaled.shape[1]), importances[indices], color="b", yerr=std[indices], align="center")
    plt.yticks(range(features_train_scaled.shape[1]), features_train_scaled.columns)
    #plt.xlim([-1, features_train_transformed.shape[1]])
    plt.xlabel('Relative Importance')
    plt.show()

    # Print the feature ranking
    feature_importance_df = pd.DataFrame({'feature': features_train.columns, 'importance': clf.feature_importances_}).sort_values('importance', ascending=False)
    print(feature_importance_df)



#TODO:
# [x]Make a correlation graph between selected features
# [x]Explore RFE
# [x]Review & implement cross validation
# [x]Add column names in importance graph
# [x]Learn the basics of plt
# [x]Review and implement confusion matrix (evaluation metrics)
# [x]Create summary statistics of accuracy, precision, recall, f-score
# [x]Decide whether to use the dataset
# [x]Calculate price var based on a fixed amount investment divided equally into winning picks
# [x]Investigate class, price var % mismatch in pl_df (index f-d up during transformation, dataframing?)
# [x]Fix buy/skip column of pl_df
# []Try using aggregated dataset from year 2014-2018 and see if model improves
# []Fine tune algorithm(tune parameters, missing data threshold) to boost ROI and beat S&P500(y2019 ~28%)
# []Get Names of the true positive stocks
# []Draft the capstone topic approval form

