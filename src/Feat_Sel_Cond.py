

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b8787480-9f61-4056-9876-795de25fc3f9"),
    Condition_Features=Input(rid="ri.foundry.main.dataset.60f2eb0d-7b8c-466f-9ae8-66105e882875")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def Feat_Sel_Cond(Condition_Features):
    #parameters
    maxCategories=4
    seed=42
    label="pasc_code_after_four_weeks"

    df = Condition_Features
    y = df.select(label).toPandas().values.ravel()
    X = df.drop(label, "features").toPandas()

    # stratified split
    class0 = df.filter(df["pasc_code_after_four_weeks"]==0)
    class1 = df.filter(df["pasc_code_after_four_weeks"]==1)
    print("Class 0 (pasc= 0): ", class0.count())
    print("Class 1 (pasc= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=seed)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=seed)

    train = train0.union(train1)
    test = test0.union(test1)

    y_train = train.select(label).toPandas().values.ravel()
    X_train = train.drop(label, "features").toPandas()
    y_test = test.select(label).toPandas().values.ravel()
    X_test = test.drop(label, "features").toPandas()
    columns = X_train.columns
    print("\n[Dataframe shape]")
    print("Train set: {}, Pandas DF Shape: {}\n{}".format(train.count(), X_train.shape, train.groupBy("pasc_code_after_four_weeks").count().toPandas()))
    print("Test set: {}, Pandas DF Shape: {}\n{}".format(test.count(), X_test.shape, test.groupBy("pasc_code_after_four_weeks").count().toPandas()))

    results = pd.DataFrame()

    def variance_threshold(X):
        from sklearn.feature_selection import VarianceThreshold

        # dropping columns where 1-threshold of the values are similar
        # a feature contains only 0s 80% of the time or only 1s 80% of the time
        sel = VarianceThreshold(threshold=.8*(1-.8))

        sel.fit_transform(X)
        selected_var = columns[sel.get_support()]

        result = pd.DataFrame({'feature': selected_var.to_list(),
                            'methods': 'Variance Threshold',
                            'model': 'n/a',
                            'importance': 1})
        print("\nMethod: Variance Threshold")
        print('n_features_selected:',selected_var.shape[0])
        # print('Features Selected: ', selected_var)
        return result

    def lasso(X, y):
        lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=seed).fit(X, y)

        print("\nMethod: Lasso regularization")
        print('Train Accuracy: {:0.2f}'.format(lasso.score(X, y)))

        df_lasso = pd.DataFrame()

        for c, cla in zip(lasso.coef_, range(-2,3,1)):
            temp = pd.DataFrame({'feature': columns, 'coef': c, 'class': cla})
            df_lasso = pd.concat([df_lasso, temp], ignore_index=True)

        df_lasso2 = df_lasso.groupby(['feature'], as_index=False).agg({'coef': 'sum'})
        df_lasso2['Model'] = 'Lasso'

        df_lasso3 = df_lasso2[df_lasso2['coef']!=0].copy()
        df_lasso3.loc[:,'importances'] = 1
        result = pd.DataFrame({'feature': df_lasso3['feature'],
                                'methods': 'Regularization',
                                'model': 'Lasso',
                                'importance': df_lasso3['importances']})
        return result

    def RF_Imp(X_train, y_train, X_test, y_test):
        estimator = RandomForestClassifier(random_state=seed, n_jobs=-1)
        estimator.fit(X_train, y_train)
        print("\nMethod: Random Forest Feature Importance")
        print("Train Score: {:0.2f}".format(estimator.score(X_train, y_train)))
        print("Test Score: {:0.2f}".format(estimator.score(X_test, y_test)))

        result = pd.DataFrame({'feature': columns,
                                'methods': 'Feature Importance',
                                'model': 'Random Forest',
                                'importance': estimator.feature_importances_})
        return result

    def PMI(X_train, y_train, X_test, y_test):
        from sklearn.inspection import permutation_importance

        result = pd.DataFrame()
        rf = RandomForestClassifier(random_state=seed, n_jobs=-1)
        ridge = LogisticRegression(penalty='l2', solver='liblinear', C=0.1, random_state=seed)

        def running(estimator, model):
            estimator.fit(X_train, y_train)
            print("\nMethod: Permutation Importance with",model)
            print("Train Score: {:0.2f}".format(estimator.score(X_train, y_train)))
            print("Test Score: {:0.2f}".format(estimator.score(X_test, y_test)))
            pmi = permutation_importance(estimator, X=X_test, y=y_test, scoring='accuracy', n_jobs=-1, random_state=seed)
            temp = pd.DataFrame({'feature': columns,
                                'methods': 'Permutation Importance',
                                'model': model,
                                'importance': pmi['importances_mean']})
            return temp

        result = pd.concat([result, running(rf, "Random Forest")], ignore_index=True)
        result = pd.concat([result, running(ridge, "Ridge")], ignore_index=True)

        return result

    def RFE(X_train, y_train, X_test, y_test):
        from sklearn.feature_selection import RFECV

        result = pd.DataFrame()

        rf = RandomForestClassifier(random_state=seed, n_jobs=-1)
        ridge = LogisticRegression(penalty='l2', solver='liblinear', C=0.1, random_state=seed)

        def running(estimator, model):
            estimator.fit(X_train, y_train)
            print("\nMethod: Recursive Feature Elimination with",model)
            print("Train Score: {:0.2f}".format(estimator.score(X_train, y_train)))
            print("Test Score: {:0.2f}".format(estimator.score(X_test, y_test)))
            rfe = RFECV(estimator, step=1, cv=5, n_jobs=-1)
            sel = rfe.fit(X_train, y_train)
            support = sel.support_
            print('n Selected:', sel.n_features_)
            temp = pd.DataFrame({'feature': columns,
                                  'methods': 'RFE',
                                  'model': model,
                                  'importance': [1 if ft in columns[support] else 0 for ft in columns]})
            return temp

        result = pd.concat([result, running(rf, "Random Forest")], ignore_index=True)
        result = pd.concat([result, running(ridge, "Ridge")], ignore_index=True)

        return result

    def SFS (X_train, y_train, X_test, y_test):
        from sklearn.feature_selection import SequentialFeatureSelector

        result = pd.DataFrame()

        # KNN = KNeighborsClassifier(n_neighbors=3)
        rf = RandomForestClassifier(random_state=seed)
        ridge = LogisticRegression(penalty='l2', solver='liblinear', C=0.1, random_state=seed)

        def running(estimator, model):
            estimator.fit(X_train, y_train)
            print("\nMethod: Sequential Feature Selecttion with", model)
            print("Train Score: {:0.2f}".format(estimator.score(X_train, y_train)))
            print("Test Score: {:0.2f}".format(estimator.score(X_test, y_test)))

            sfs = SequentialFeatureSelector(estimator, n_jobs=-1, n_features_to_select=0.5)
            sfs.fit(X_train, y_train)
            support = sfs.get_support()
            print('n Selected:', sfs.n_features_to_select_)

            temp = pd.DataFrame({'feature': columns,
                                'methods': 'SFS',
                                'model': model,
                                'importance': [1 if ft in columns[support] else 0 for ft in columns]})
            return temp

        # result = pd.concat([result, running(KNN, "KNN")], ignore_index=True)
        result = pd.concat([result, running(rf, "Random Forest")], ignore_index=True)
        result = pd.concat([result, running(ridge, "Ridge")], ignore_index=True)

        return result

    results = pd.concat([results, variance_threshold(X)], ignore_index=True)
    results = pd.concat([results, lasso(X, y)], ignore_index=True)
    results = pd.concat([results, RF_Imp(X_train, y_train, X_test, y_test)], ignore_index=True)
    results = pd.concat([results, PMI(X_train, y_train, X_test, y_test)], ignore_index=True)
    results = pd.concat([results, RFE(X_train, y_train, X_test, y_test)], ignore_index=True)
    results = pd.concat([results, SFS(X_train, y_train, X_test, y_test)], ignore_index=True)

    results['selected'] = results['importance'].apply(lambda x: 1 if x > 0 else 0)

    th_quantile=0.5
    th_rf = results[results['methods'] == 'Feature Importance']['importance'].quantile(th_quantile)
    print('\nThreshold for Random Forest Feature Importance is {:0.4f} at {:0.0f}th percentile'.format(th_rf, th_quantile*100))
    results['selected'] = results.apply(lambda x: 0 if ((x.methods == 'Feature Importance') & (x.importance < th_rf))
                                                    else x.selected, axis=1)

    results['Method'] = results.apply(lambda x: 'Variance Threshold' if x.methods == 'Variance Threshold'
                                                                    else str(x.methods) + " - " + str(x.model), axis=1)

    results['Count']=results.groupby('feature').transform(lambda x: x.sum())['selected']
    results = results.sort_values(by=['Count', 'feature'], ascending=True)

    return results
