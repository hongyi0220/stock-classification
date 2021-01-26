import pandas as pd
import matplotlib.pyplot as plt
from pandas.tests.frame.methods.test_sort_values import ascending
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
import seaborn as sns
import numpy as np

if __name__ == '__main__':
    stocks = pd.read_csv('data/2014_Financial_Data.csv')

    corr = stocks.corr()
    #print(corr)
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap)

    stocks.drop(columns=['Sector'], axis=1, inplace=True)
    stocks.drop(stocks.columns[stocks.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    print('shape before drop', stocks.shape)
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
    features_train, features_test, labels_train, labels_test = train_test_split(imputed_features, labels, test_size=0.2)
    #print(features_train.shape)

    # Standardize data
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_test_scaled = scaler.fit_transform(features_test)
    #print('scaled data mean:', features_train_scaled.mean(axis=0))
    #print('scaled data std:', features_train_scaled.std(axis=0))

    # Do feature elimination
    sel = SelectKBest(f_classif, k=20).fit(features_train_scaled, labels_train)
    print('feat sel score:', sel.scores_)
    features_train_transformed = sel.transform(features_train_scaled)
    #print(features_train_transformed.shape)

    # Train model with algorithm
    clf = RandomForestClassifier()
    clf.fit(features_train_scaled, labels_train)
    y_pred = clf.predict(features_test_scaled)
    print('accuracy w/ all features:', accuracy_score(labels_test, y_pred))

#    clf.fit(features_train_transformed, labels_train)
#    features_test_transformed = sel.transform(features_test_scaled)
#    y_pred2 = clf.predict(features_test_transformed)
#    print('accuracy after feat sel:', accuracy_score(labels_test, y_pred2))
    #accuracy after feat sel: 0.5984251968503937

    # Plot feature importance
    #feature_importance_df = pd.DataFrame({'feature': features_train_transformed.columns, 'importance': clf.feature_importances_}).sort_values('importance', ascending=False)

    #print(feature_importance_df)

    #print(features, labels)
