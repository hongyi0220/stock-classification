import pandas as pd
import matplotlib.pyplot as plt
from pandas.tests.frame.methods.test_sort_values import ascending
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
import seaborn as sns
import numpy as np

if __name__ == '__main__':
    stocks = pd.read_csv('data/2014_Financial_Data.csv')


    # Clean up data
    for i, row in stocks.iterrows():
        price_var = row[stocks.columns.str.contains('price var', case=False)]
        #print('price var:', price_var)
        if price_var.item() < 10:
            #print('class value changed')
            stocks.at[i, 'Class'] = 0

    #print(stocks)
    stocks.drop(columns=['Sector'], axis=1, inplace=True)
    stocks.drop(stocks.columns[stocks.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    stocks.drop(stocks.columns[stocks.columns.str.contains('2015 price var', case=False)], axis=1, inplace=True)
    #print('stock after dropping sector, unnamed, 2015 price var cols\n', stocks)
    labels = stocks.pop('Class')
    features = stocks

    # Fill in missing values
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    imputed_features = imp_mean.fit_transform(features)

    # Add headers back
    imputed_features = pd.DataFrame(imputed_features, columns=features.columns)

    # Remove columns with same data
    nunique = imputed_features.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    imputed_features.drop(cols_to_drop, axis=1, inplace=True)
    #print('after:', imputed_features.shape)

    # Export data to csv file
    pd.DataFrame(imputed_features, columns=imputed_features.columns).to_csv(path_or_buf='data/2014_Financial_Data_Imputed.csv', index=False)
    #print('imputed features:', imputed_features)


    # Divide dataset into training and test data
    kf = KFold(shuffle=True)
    for train_indices, test_indices in kf.split(imputed_features, labels):
        features_train = [imputed_features.iloc[i] for i in train_indices]
        features_test = [imputed_features.iloc[i] for i in test_indices]
        labels_train = [labels.iloc[i] for i in train_indices]
        labels_test = [labels.iloc[i] for i in test_indices]
        #print(train_indices)
        #print(labels_train)
        #print(labels_test)
    #features_train, features_test, labels_train, labels_test = train_test_split(imputed_features, labels, test_size=0.2, random_state=0)
    features_train = pd.DataFrame(features_train)
    features_test = pd.DataFrame(features_test)
    #print(features_train.shape)
    #print(features_train)

    # Standardize data
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_test_scaled = scaler.fit_transform(features_test)
    #print('scaled data mean:', features_train_scaled.mean(axis=0))
    #print('scaled data std:', features_train_scaled.std(axis=0))

    # Do feature elimination
    sel = SelectKBest(f_classif, k=10).fit(features_train_scaled, labels_train)
    #print('feat sel score:', sel.scores_)
    features_train_transformed = sel.transform(features_train_scaled)
    features_train_transformed = pd.DataFrame(features_train_transformed, columns=features_train.columns[sel.get_support()])
    #print(features_train_transformed, 'shape:', features_train_transformed.shape)

    # Make corr heatmap b/w features
    corr = features_train_transformed.corr()
    #print(corr)
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap)


    # Train model with algorithm
    clf = RandomForestClassifier(random_state=0)
#    clf.fit(features_train_scaled, labels_train)
#    y_pred = clf.predict(features_test_scaled)
#    print('accuracy w/ all features:', accuracy_score(labels_test, y_pred))

    # RFE
    #sel2 = RFE(clf, n_features_to_select=10)
    #features_train_transformed = sel2.fit_transform(features_train_scaled, labels_train)

    clf.fit(features_train_transformed, labels_train)
    features_test_transformed = sel.transform(features_test_scaled)
    y_pred2 = clf.predict(features_test_transformed)
    print('accuracy after feat sel:', accuracy_score(labels_test, y_pred2))
    #accuracy after feat sel: 0.541994750656168 k=20
    #0.6417322834645669 k=21
    #0.5997375328083989 k=22
    #0.5721784776902887 k=23
    #0.6614173228346457 k=24
    #0.6876640419947506 k=25
    #0.6968503937007874 k=30 0.699475065616798, 0.4921259842519685, 0.7125984251968503, 0.6968503937007874, 0.7152230971128609, 0.7007874015748031, 0.6469816272965879, 0.6837270341207349, 0.60498687664042, 0.5341207349081365, 0.6561679790026247
    #0.6889763779527559 k=31
    #0.55249343832021 k=32

    # Make confusion matrix to track evaluation metrics
    cf_matrix = confusion_matrix(labels_test, y_pred2)
    #print(cf_matrix)
    cell_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    cell_counts = ["{0:0f}".format(v) for v in cf_matrix.flatten()]
    cell_percents = ["{0:.2%}".format(v) for v in cf_matrix.flatten() / np.sum(cf_matrix)]
    cell_labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(cell_names, cell_counts, cell_percents)]
    cell_labels = np.asarray(cell_labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=cell_labels, fmt='', cmap='Blues')

    # plotting feature importance
    importances = clf.feature_importances_
    #print('importances:', importances)
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.title("Feature Importance")
    plt.barh(range(features_train_transformed.shape[1]), importances[indices], color="b", yerr=std[indices], align="center")
    plt.yticks(range(features_train_transformed.shape[1]), features_train_transformed.columns)
    #plt.xlim([-1, features_train_transformed.shape[1]])
    plt.xlabel('Relative Importance')
    plt.show()

    # Print the feature ranking
    feature_importance_df = pd.DataFrame({'feature': features_train.columns[sel.get_support()], 'importance': clf.feature_importances_}).sort_values('importance', ascending=False)
    print(feature_importance_df)



#TODO:
# [x]Make a correlation graph between selected features
# [x]Explore RFE
# [x]Review & implement cross validation
# [x]Add column names in importance graph
# [x]Learn the basics of plt
# [x]Review and implement confusion matrix (evaluation metrics)
# []Create summary statistics of accuracy, precision, recall, f-score
# [x]Decide whether to use the dataset
# []Draft the capstone topic approval form

