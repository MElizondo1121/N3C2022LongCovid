
@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e182d719-d3e8-4168-857d-3d45f2ae76fd"),
    observation_Features_1_0=Input(rid="ri.foundry.main.dataset.0633476b-c599-44a7-8575-d2f44a9dedcf")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def Feat_Sel_Obs_1_0(observation_Features_1_0):
    #parameters
    maxCategories=4
    seed=42
    label="pasc_code_after_four_weeks"

    df = observation_Features_1_0
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

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.155ea8bc-546c-4772-92c9-2b51e08869b1"),
    DemORDia=Input(rid="ri.foundry.main.dataset.f27bd63a-3e37-4248-8978-9c9229c1a2ac"),
    DemORDiaTest=Input(rid="ri.foundry.main.dataset.ab2edfae-9f48-42ba-adb9-7429ad45b0fc")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
from pyspark.ml.feature import VectorIndexer
import numpy as np
from sklearn.metrics import confusion_matrix

def Final_Model(DemORDia, DemORDiaTest):

    #parameters
    maxCategories=4
    seed=42
    label="pasc_code_after_four_weeks"
    params={'maxDepth': 20,
            'impurity': 'gini',
            'minInstancesPerNode': 1,
            'numTrees': 50}
    print("Parameters: ",params)

    test = DemORDiaTest
    train = DemORDia.select(test.columns+[label])
    print("Train columns not in Test:", [c for c in train.columns if c not in test.columns])
    print("Test columns not in Train:", [c for c in test.columns if c not in train.columns])

    print(train.columns)
    print(test.columns)
    train.show(5)
    test.show(5)

    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=maxCategories).fit(train)

    print("\n[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy(label).count().toPandas().sort_values(by=label)))
    print("Test set: {}\n".format(test.count()))



    estimator = RandomForestClassifier(featuresCol='indexed_features', labelCol=label, seed=seed)
    estimator.setParams(**params)
    pipeline = Pipeline(stages=[featureIndexer, estimator])
    model = pipeline.fit(train)
    results_train = model.transform(train)
    results_test = model.transform(test)

    my_eval = MulticlassClassificationEvaluator(labelCol = label, predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol =label, rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select(label).collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("\n[Performance on Train set]")
    printPerformance(results_train)

    return results_test.select("person_id", "prediction")

@transform_pandas(
    Output(rid="ri.vector.main.execute.c008d8f7-c70f-46b8-a295-c610fa1e591b"),
    Demographics_Features=Input(rid="ri.foundry.main.dataset.14be43ee-dbda-4b0d-b622-9c49b2c3bbca")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def GradientBoost(Demographics_Features):
    #parameters
    maxCategories=4
    seed=42

    df = Demographics_Features
    featureIndexer =VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=maxCategories).fit(df)

    # stratified split
    class0 = df.filter(df["pasc_code_after_four_weeks"]==0)
    class1 = df.filter(df["pasc_code_after_four_weeks"]==1)
    print("Class 0 (pasc= 0): ", class0.count())
    print("Class 1 (pasc= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=seed)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=seed)

    train = train0.union(train1)
    test = test0.union(test1)
    print("[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas()))
    print("Test set: {}\n {}".format(test.count(), test.groupBy("pasc_code_after_four_weeks").count().toPandas()))

    estimator = GBTClassifier(featuresCol='indexed_features', labelCol="pasc_code_after_four_weeks", seed=seed,
                       maxIter=10)
    pipeline = Pipeline(stages=[featureIndexer, estimator])
    evaluator=MulticlassClassificationEvaluator(labelCol='pasc_code_after_four_weeks', predictionCol='prediction', metricName='accuracy')

    model = pipeline.fit(train)
    print('Default parameters accuracy on test set: {:f}'.format(evaluator.evaluate(model.transform(test))))

    ## full list of hyperparameters
    param_list = ['maxDepth', 'minInstancesPerNode', 'minInfoGain', 'subsamplingRate', 'minWeightFractionPerNode', 'maxIter']
    paramGrid = ParamGridBuilder()\
            .addGrid(estimator.maxDepth, [10, 20, 30])\
            .addGrid(estimator.minInstancesPerNode, [1, 5, 10])\
            .addGrid(estimator.minInfoGain, [0.0, 0.1, 0.5])\
            .addGrid(estimator.subsamplingRate, [0.1, 0.3, 0.5])\
            .addGrid(estimator.minWeightFractionPerNode, [0.0, 0.1, 0.3])\
            .addGrid(estimator.maxIter, [20, 30, 50])\
            .build()

     ## short list of hyperparameters
    # param_list = ['maxDepth', 'maxIter']
    # paramGrid = ParamGridBuilder()\
    #         .addGrid(estimator.maxDepth, [10, 30])\
    #         .addGrid(estimator.maxIter, [20, 50])\
    #         .build()

    tvs = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8,
                           seed=seed,
                           parallelism=10)

    model = tvs.fit(train)
    results = model.transform(test)
    print('Best model accuracy on test set: {:f}'.format(evaluator.evaluate(results)))

    best_params={param.name: value for param, value in zip(model.bestModel.stages[1].extractParamMap().keys(), model.bestModel.stages[1].extractParamMap().values()) if param.name in param_list}
    print("\n[Best hyperparameters]")
    for p in param_list: print('Best {}: {}'.format(p, best_params[p]))
    print(best_params,"\n")
    df_best_params = pd.DataFrame(data=best_params, index=[0])

    my_eval = MulticlassClassificationEvaluator(labelCol = 'pasc_code_after_four_weeks', predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    print("[Performance on test set]")
    for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))

    my_eval_bin = BinaryClassificationEvaluator(labelCol ='pasc_code_after_four_weeks', rawPredictionCol = 'prediction')
    metrics=['areaUnderROC', 'areaUnderPR']
    for m in metrics: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))

    y_test = np.array(results.select("pasc_code_after_four_weeks").collect()).flatten()
    y_pred = np.array(results.select("prediction").collect()).flatten()
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    results = model.transform(df)
    y_test = np.array(results.select("pasc_code_after_four_weeks").collect()).flatten()
    y_pred = np.array(results.select("prediction").collect()).flatten()
    print("\nonfusion Matrix on Whole DataFrame:\n", confusion_matrix(y_test, y_pred))

    return df_best_params

@transform_pandas(
    Output(rid="ri.vector.main.execute.36eac74e-bc80-403b-8358-58712ade737b"),
    Condition_Features=Input(rid="ri.foundry.main.dataset.60f2eb0d-7b8c-466f-9ae8-66105e882875")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def GradientBoostCond(Condition_Features):
    #parameters
    maxCategories=4
    seed=42

    df = Condition_Features
    featureIndexer =VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=maxCategories).fit(df)

    # stratified split
    class0 = df.filter(df["pasc_code_after_four_weeks"]==0)
    class1 = df.filter(df["pasc_code_after_four_weeks"]==1)
    print("Class 0 (pasc= 0): ", class0.count())
    print("Class 1 (pasc= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=seed)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=seed)

    train = train0.union(train1)
    test = test0.union(test1)
    print("[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas()))
    print("Test set: {}\n {}".format(test.count(), test.groupBy("pasc_code_after_four_weeks").count().toPandas()))

    estimator = GBTClassifier(featuresCol='indexed_features', labelCol="pasc_code_after_four_weeks", seed=seed,
                       maxIter=10)
    pipeline = Pipeline(stages=[featureIndexer, estimator])
    evaluator=MulticlassClassificationEvaluator(labelCol='pasc_code_after_four_weeks', predictionCol='prediction', metricName='accuracy')

    model = pipeline.fit(train)
    print('Default parameters accuracy on test set: {:f}'.format(evaluator.evaluate(model.transform(test))))

    ## full list of hyperparameters
    param_list = ['maxDepth', 'minInstancesPerNode', 'minInfoGain', 'subsamplingRate', 'minWeightFractionPerNode', 'maxIter']
    paramGrid = ParamGridBuilder()\
            .addGrid(estimator.maxDepth, [10, 20, 30])\
            .addGrid(estimator.minInstancesPerNode, [1, 5, 10])\
            .addGrid(estimator.minInfoGain, [0.0, 0.1, 0.5])\
            .addGrid(estimator.subsamplingRate, [0.1, 0.3, 0.5])\
            .addGrid(estimator.minWeightFractionPerNode, [0.0, 0.1, 0.3])\
            .addGrid(estimator.maxIter, [20, 30, 50])\
            .build()

     ## short list of hyperparameters
    # param_list = ['maxDepth', 'maxIter']
    # paramGrid = ParamGridBuilder()\
    #         .addGrid(estimator.maxDepth, [10, 30])\
    #         .addGrid(estimator.maxIter, [20, 50])\
    #         .build()

    tvs = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8,
                           seed=seed,
                           parallelism=10)

    model = tvs.fit(train)
    results = model.transform(test)
    print('Best model accuracy on test set: {:f}'.format(evaluator.evaluate(results)))

    best_params={param.name: value for param, value in zip(model.bestModel.stages[1].extractParamMap().keys(), model.bestModel.stages[1].extractParamMap().values())}
    print("\n[Best hyperparameters]")
    for p in param_list: print('Best {}: {}'.format(p, best_params[p]))
    print("\n")

    my_eval = MulticlassClassificationEvaluator(labelCol = 'pasc_code_after_four_weeks', predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    print("[Performance on test set]")
    for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))

    my_eval_bin = BinaryClassificationEvaluator(labelCol ='pasc_code_after_four_weeks', rawPredictionCol = 'prediction')
    metrics=['areaUnderROC', 'areaUnderPR']
    for m in metrics: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))

    y_test = np.array(results.select("pasc_code_after_four_weeks").collect()).flatten()
    y_pred = np.array(results.select("prediction").collect()).flatten()
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    results = model.transform(df)
    y_test = np.array(results.select("pasc_code_after_four_weeks").collect()).flatten()
    y_pred = np.array(results.select("prediction").collect()).flatten()
    print("\nonfusion Matrix on Whole DataFrame:\n", confusion_matrix(y_test, y_pred))

    return results

@transform_pandas(
    Output(rid="ri.vector.main.execute.5cc66224-1f8a-4e79-843c-0fdddfa53ac6"),
    Diagnosis_Features=Input(rid="ri.vector.main.execute.49f1d284-20c7-4aca-a3b3-6a1d25610f25")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def GradientBoostDiag(Diagnosis_Features):
    #parameters
    maxCategories=4
    seed=42

    df = Diagnosis_Features
    featureIndexer =VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=maxCategories).fit(df)

    # stratified split
    class0 = df.filter(df["pasc_code_after_four_weeks"]==0)
    class1 = df.filter(df["pasc_code_after_four_weeks"]==1)
    print("Class 0 (pasc= 0): ", class0.count())
    print("Class 1 (pasc= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=seed)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=seed)

    train = train0.union(train1)
    test = test0.union(test1)
    print("[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas()))
    print("Test set: {}\n {}".format(test.count(), test.groupBy("pasc_code_after_four_weeks").count().toPandas()))

    estimator = GBTClassifier(featuresCol='indexed_features', labelCol="pasc_code_after_four_weeks", seed=seed,
                       maxIter=10)
    pipeline = Pipeline(stages=[featureIndexer, estimator])
    evaluator=MulticlassClassificationEvaluator(labelCol='pasc_code_after_four_weeks', predictionCol='prediction', metricName='accuracy')

    model = pipeline.fit(train)
    print('Default parameters accuracy on test set: {:f}'.format(evaluator.evaluate(model.transform(test))))

    ## full list of hyperparameters
    param_list = ['maxDepth', 'minInstancesPerNode', 'minInfoGain', 'subsamplingRate', 'minWeightFractionPerNode', 'maxIter']
    paramGrid = ParamGridBuilder()\
            .addGrid(estimator.maxDepth, [10, 20, 30])\
            .addGrid(estimator.minInstancesPerNode, [1, 5, 10])\
            .addGrid(estimator.minInfoGain, [0.0, 0.1, 0.5])\
            .addGrid(estimator.subsamplingRate, [0.1, 0.3, 0.5])\
            .addGrid(estimator.minWeightFractionPerNode, [0.0, 0.1, 0.3])\
            .addGrid(estimator.maxIter, [20, 30, 50])\
            .build()

     ## short list of hyperparameters
    # param_list = ['maxDepth', 'maxIter']
    # paramGrid = ParamGridBuilder()\
    #         .addGrid(estimator.maxDepth, [10, 30])\
    #         .addGrid(estimator.maxIter, [20, 50])\
    #         .build()

    tvs = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8,
                           seed=seed,
                           parallelism=10)

    model = tvs.fit(train)
    results = model.transform(test)
    print('Best model accuracy on test set: {:f}'.format(evaluator.evaluate(results)))

    best_params={param.name: value for param, value in zip(model.bestModel.stages[1].extractParamMap().keys(), model.bestModel.stages[1].extractParamMap().values())}
    print("\n[Best hyperparameters]")
    for p in param_list: print('Best {}: {}'.format(p, best_params[p]))
    print("\n")

    my_eval = MulticlassClassificationEvaluator(labelCol = 'pasc_code_after_four_weeks', predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol ='pasc_code_after_four_weeks', rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select("pasc_code_after_four_weeks").collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("[Performance on Test set]")
    printPerformance(results)

    results = model.transform(df)
    print("\n[Performance on Whole set]")
    printPerformance(results)

    return results

@transform_pandas(
    Output(rid="ri.vector.main.execute.95acfb2e-48b0-4bd6-9827-be6251f2fd31"),
    Diagnosis_1_0=Input(rid="ri.foundry.main.dataset.b7b9f659-64e6-4e0a-b8c8-c3a97ece514e")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def GradientBoostDiag_1_0(Diagnosis_1_0):
    #parameters
    maxCategories=4
    seed=42

    df = Diagnosis_1_0
    featureIndexer =VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=maxCategories).fit(df)

    # stratified split
    class0 = df.filter(df["pasc_code_after_four_weeks"]==0)
    class1 = df.filter(df["pasc_code_after_four_weeks"]==1)
    print("Class 0 (pasc= 0): ", class0.count())
    print("Class 1 (pasc= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=seed)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=seed)

    train = train0.union(train1)
    test = test0.union(test1)
    print("[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas()))
    print("Test set: {}\n {}".format(test.count(), test.groupBy("pasc_code_after_four_weeks").count().toPandas()))

    estimator = GBTClassifier(featuresCol='indexed_features', labelCol="pasc_code_after_four_weeks", seed=seed,
                       maxIter=10)
    pipeline = Pipeline(stages=[featureIndexer, estimator])
    evaluator=MulticlassClassificationEvaluator(labelCol='pasc_code_after_four_weeks', predictionCol='prediction', metricName='accuracy')

    model = pipeline.fit(train)
    print('Default parameters accuracy on test set: {:f}'.format(evaluator.evaluate(model.transform(test))))

    ## full list of hyperparameters
    param_list = ['maxDepth', 'minInstancesPerNode', 'minInfoGain', 'subsamplingRate', 'minWeightFractionPerNode', 'maxIter']
    paramGrid = ParamGridBuilder()\
            .addGrid(estimator.maxDepth, [10, 20, 30])\
            .addGrid(estimator.minInstancesPerNode, [1, 5, 10])\
            .addGrid(estimator.minInfoGain, [0.0, 0.1, 0.5])\
            .addGrid(estimator.subsamplingRate, [0.1, 0.3, 0.5])\
            .addGrid(estimator.minWeightFractionPerNode, [0.0, 0.1, 0.3])\
            .addGrid(estimator.maxIter, [20, 30, 50])\
            .build()

     ## short list of hyperparameters
    # param_list = ['maxDepth', 'maxIter']
    # paramGrid = ParamGridBuilder()\
    #         .addGrid(estimator.maxDepth, [10, 30])\
    #         .addGrid(estimator.maxIter, [20, 50])\
    #         .build()

    tvs = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8,
                           seed=seed,
                           parallelism=10)

    model = tvs.fit(train)
    results = model.transform(test)
    print('Best model accuracy on test set: {:f}'.format(evaluator.evaluate(results)))

    best_params={param.name: value for param, value in zip(model.bestModel.stages[1].extractParamMap().keys(), model.bestModel.stages[1].extractParamMap().values())}
    print("\n[Best hyperparameters]")
    for p in param_list: print('Best {}: {}'.format(p, best_params[p]))
    print("\n")

    my_eval = MulticlassClassificationEvaluator(labelCol = 'pasc_code_after_four_weeks', predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol ='pasc_code_after_four_weeks', rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select("pasc_code_after_four_weeks").collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("[Performance on Test set]")
    printPerformance(results)

    results = model.transform(df)
    print("\n[Performance on Whole set]")
    printPerformance(results)

    return results

@transform_pandas(
    Output(rid="ri.vector.main.execute.d2b8dbff-67b2-44f9-97bd-ceef34b638b8"),
    Condition_Features=Input(rid="ri.foundry.main.dataset.60f2eb0d-7b8c-466f-9ae8-66105e882875")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def LogisticRegCond(Condition_Features):
    #parameters
    seed=42

    df = Condition_Features
    ## stratified split
    class0 = df.filter(df["pasc_code_after_four_weeks"]==0)
    class1 = df.filter(df["pasc_code_after_four_weeks"]==1)
    print("Class 0 (pasc= 0): ", class0.count())
    print("Class 1 (pasc= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=seed)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=seed)

    train = train0.union(train1)
    test = test0.union(test1)
    print("[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas()))
    print("Test set: {}\n {}".format(test.count(), test.groupBy("pasc_code_after_four_weeks").count().toPandas()))

    ## if regParam=0 -> Ridge regression, elasticNetParam=1 -> Lasso
    estimator = LogisticRegression(featuresCol='features', labelCol='pasc_code_after_four_weeks')
    evaluator=MulticlassClassificationEvaluator(labelCol='pasc_code_after_four_weeks', predictionCol='prediction', metricName='accuracy')

    model = estimator.fit(train)
    print('Default parameters accuracy on test set: {:f}'.format(evaluator.evaluate(model.transform(test))))

    param_list= ['maxIter', 'regParam', 'elasticNetParam']
    paramGrid = ParamGridBuilder()\
            .addGrid(estimator.maxIter, [10, 50, 100, 500, 1000])\
            .addGrid(estimator.regParam, [0.0, 0.001, 0.1, 0.5, 1])\
            .addGrid(estimator.elasticNetParam, [0.0, 0.001, 0.1, 0.5, 1])\
            .build()

    tvs = TrainValidationSplit(estimator=estimator,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8,
                           seed=seed,
                           parallelism=10)

    model = tvs.fit(train)
    results = model.transform(test)
    print('Best model accuracy on test set: {:f}'.format(evaluator.evaluate(results)))

    best_params={param.name: value for param, value in zip(model.bestModel.extractParamMap().keys(), model.bestModel.extractParamMap().values()) if param.name in param_list}
    print("\n[Best hyperparameters]")
    for p in param_list: print('Best {}: {}'.format(p, best_params[p]))
    print("\n")
    df_best_params = pd.DataFrame(data=best_params, index=[0])

    my_eval = MulticlassClassificationEvaluator(labelCol = 'pasc_code_after_four_weeks', predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol ='pasc_code_after_four_weeks', rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select("pasc_code_after_four_weeks").collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("[Performance on Test set]")
    printPerformance(results)

    results = model.transform(df)
    print("\n[Performance on Whole set]")
    printPerformance(results)

    estimator.setParams(**best_params)
    model = estimator.fit(df)
    results = model.transform(df)
    print("\n[Performance on Whole set with Fitting]")
    printPerformance(results)

    return df_best_params

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.efa2e793-596c-4a5e-9cd7-fa7aad2bd462"),
    Demographics_Features=Input(rid="ri.foundry.main.dataset.14be43ee-dbda-4b0d-b622-9c49b2c3bbca")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def LogisticRegDem(Demographics_Features):
    #parameters
    seed=42

    df = Demographics_Features
    ## stratified split
    class0 = df.filter(df["pasc_code_after_four_weeks"]==0)
    class1 = df.filter(df["pasc_code_after_four_weeks"]==1)
    print("Class 0 (pasc= 0): ", class0.count())
    print("Class 1 (pasc= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=seed)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=seed)

    train = train0.union(train1)
    test = test0.union(test1)
    print("[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas()))
    print("Test set: {}\n {}".format(test.count(), test.groupBy("pasc_code_after_four_weeks").count().toPandas()))

    ## if regParam=0 -> Ridge regression, elasticNetParam=1 -> Lasso
    estimator = LogisticRegression(featuresCol='features', labelCol='pasc_code_after_four_weeks')
    evaluator=MulticlassClassificationEvaluator(labelCol='pasc_code_after_four_weeks', predictionCol='prediction', metricName='accuracy')

    model = estimator.fit(train)
    print('Default parameters accuracy on test set: {:f}'.format(evaluator.evaluate(model.transform(test))))

    param_list= ['maxIter', 'regParam', 'elasticNetParam']
    paramGrid = ParamGridBuilder()\
            .addGrid(estimator.maxIter, [10, 50, 100, 500, 1000])\
            .addGrid(estimator.regParam, [0.0, 0.001, 0.1, 0.5, 1])\
            .addGrid(estimator.elasticNetParam, [0.0, 0.001, 0.1, 0.5, 1])\
            .build()

    tvs = TrainValidationSplit(estimator=estimator,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8,
                           seed=seed,
                           parallelism=10)

    model = tvs.fit(train)
    results = model.transform(test)
    print('Best model accuracy on test set: {:f}'.format(evaluator.evaluate(results)))

    best_params={param.name: value for param, value in zip(model.bestModel.extractParamMap().keys(), model.bestModel.extractParamMap().values()) if param.name in param_list}
    print("\n[Best hyperparameters]")
    for p in param_list: print('Best {}: {}'.format(p, best_params[p]))
    print("\n")
    df_best_params = pd.DataFrame(data=best_params, index=[0])

    my_eval = MulticlassClassificationEvaluator(labelCol = 'pasc_code_after_four_weeks', predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol ='pasc_code_after_four_weeks', rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select("pasc_code_after_four_weeks").collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("[Performance on Test set]")
    printPerformance(results)

    results = model.transform(df)
    print("\n[Performance on Whole set]")
    printPerformance(results)

    estimator.setParams(**best_params)
    model = estimator.fit(df)
    results = model.transform(df)
    print("\n[Performance on Whole set with Fitting]")
    printPerformance(results)

    return df_best_params

@transform_pandas(
    Output(rid="ri.vector.main.execute.b5b0dbf2-d984-4e07-ba38-e2ee17d279ed"),
    Diagnosis_Features=Input(rid="ri.vector.main.execute.49f1d284-20c7-4aca-a3b3-6a1d25610f25")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def LogisticRegDiag(Diagnosis_Features):
    #parameters
    seed=42

    df = Diagnosis_Features
    ## stratified split
    class0 = df.filter(df["pasc_code_after_four_weeks"]==0)
    class1 = df.filter(df["pasc_code_after_four_weeks"]==1)
    print("Class 0 (pasc= 0): ", class0.count())
    print("Class 1 (pasc= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=seed)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=seed)

    train = train0.union(train1)
    test = test0.union(test1)
    print("[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas()))
    print("Test set: {}\n {}".format(test.count(), test.groupBy("pasc_code_after_four_weeks").count().toPandas()))

    ## if regParam=0 -> Ridge regression, elasticNetParam=1 -> Lasso
    estimator = LogisticRegression(featuresCol='features', labelCol='pasc_code_after_four_weeks')
    evaluator=MulticlassClassificationEvaluator(labelCol='pasc_code_after_four_weeks', predictionCol='prediction', metricName='accuracy')

    model = estimator.fit(train)
    print('Default parameters accuracy on test set: {:f}'.format(evaluator.evaluate(model.transform(test))))

    param_list= ['maxIter', 'regParam', 'elasticNetParam']
    paramGrid = ParamGridBuilder()\
            .addGrid(estimator.maxIter, [10, 50, 100, 500, 1000])\
            .addGrid(estimator.regParam, [0.0, 0.001, 0.1, 0.5, 1])\
            .addGrid(estimator.elasticNetParam, [0.0, 0.001, 0.1, 0.5, 1])\
            .build()

    tvs = TrainValidationSplit(estimator=estimator,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8,
                           seed=seed,
                           parallelism=10)

    model = tvs.fit(train)
    results = model.transform(test)
    print('Best model accuracy on test set: {:f}'.format(evaluator.evaluate(results)))

    best_params={param.name: value for param, value in zip(model.bestModel.extractParamMap().keys(), model.bestModel.extractParamMap().values()) if param.name in param_list}
    print("\n[Best hyperparameters]")
    for p in param_list: print('Best {}: {}'.format(p, best_params[p]))
    print("\n")
    df_best_params = pd.DataFrame(data=best_params, index=[0])

    my_eval = MulticlassClassificationEvaluator(labelCol = 'pasc_code_after_four_weeks', predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol ='pasc_code_after_four_weeks', rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select("pasc_code_after_four_weeks").collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("[Performance on Test set]")
    printPerformance(results)

    results = model.transform(df)
    print("\n[Performance on Whole set]")
    printPerformance(results)

    estimator.setParams(**best_params)
    model = estimator.fit(df)
    results = model.transform(df)
    print("\n[Performance on Whole set with Fitting]")
    printPerformance(results)

    return df_best_params

@transform_pandas(
    Output(rid="ri.vector.main.execute.76f2d1ed-9583-4a68-bf8f-737952c55116"),
    Diagnosis_1_0=Input(rid="ri.foundry.main.dataset.b7b9f659-64e6-4e0a-b8c8-c3a97ece514e")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def LogisticRegDiag_1_0(Diagnosis_1_0):
    #parameters
    seed=42

    df = Diagnosis_1_0
    ## stratified split
    class0 = df.filter(df["pasc_code_after_four_weeks"]==0)
    class1 = df.filter(df["pasc_code_after_four_weeks"]==1)
    print("Class 0 (pasc= 0): ", class0.count())
    print("Class 1 (pasc= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=seed)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=seed)

    train = train0.union(train1)
    test = test0.union(test1)
    print("[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas()))
    print("Test set: {}\n {}".format(test.count(), test.groupBy("pasc_code_after_four_weeks").count().toPandas()))

    ## if regParam=0 -> Ridge regression, elasticNetParam=1 -> Lasso
    estimator = LogisticRegression(featuresCol='features', labelCol='pasc_code_after_four_weeks')
    evaluator=MulticlassClassificationEvaluator(labelCol='pasc_code_after_four_weeks', predictionCol='prediction', metricName='accuracy')

    model = estimator.fit(train)
    print('Default parameters accuracy on test set: {:f}'.format(evaluator.evaluate(model.transform(test))))

    param_list= ['maxIter', 'regParam', 'elasticNetParam']
    paramGrid = ParamGridBuilder()\
            .addGrid(estimator.maxIter, [10, 50, 100, 500, 1000])\
            .addGrid(estimator.regParam, [0.0, 0.001, 0.1, 0.5, 1])\
            .addGrid(estimator.elasticNetParam, [0.0, 0.001, 0.1, 0.5, 1])\
            .build()

    tvs = TrainValidationSplit(estimator=estimator,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8,
                           seed=seed,
                           parallelism=10)

    model = tvs.fit(train)
    results = model.transform(test)
    print('Best model accuracy on test set: {:f}'.format(evaluator.evaluate(results)))

    best_params={param.name: value for param, value in zip(model.bestModel.extractParamMap().keys(), model.bestModel.extractParamMap().values()) if param.name in param_list}
    print("\n[Best hyperparameters]")
    for p in param_list: print('Best {}: {}'.format(p, best_params[p]))
    print("\n")
    df_best_params = pd.DataFrame(data=best_params, index=[0])

    my_eval = MulticlassClassificationEvaluator(labelCol = 'pasc_code_after_four_weeks', predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol ='pasc_code_after_four_weeks', rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select("pasc_code_after_four_weeks").collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("[Performance on Test set]")
    printPerformance(results)

    results = model.transform(df)
    print("\n[Performance on Whole set]")
    printPerformance(results)

    estimator.setParams(**best_params)
    model = estimator.fit(df)
    results = model.transform(df)
    print("\n[Performance on Whole set with Fitting]")
    printPerformance(results)

    return df_best_params

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.83ef5d36-3f80-4bc2-b813-841e2da1b1bc"),
    Condition_Features=Input(rid="ri.foundry.main.dataset.60f2eb0d-7b8c-466f-9ae8-66105e882875")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def RandomForestCond(Condition_Features):
    #parameters
    maxCategories=4
    seed=42

    df = Condition_Features
    featureIndexer =VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=maxCategories).fit(df)

    # stratified split
    class0 = df.filter(df["pasc_code_after_four_weeks"]==0)
    class1 = df.filter(df["pasc_code_after_four_weeks"]==1)
    print("Class 0 (pasc= 0): ", class0.count())
    print("Class 1 (pasc= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=seed)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=seed)

    train = train0.union(train1)
    test = test0.union(test1)
    print("[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas()))
    print("Test set: {}\n {}".format(test.count(), test.groupBy("pasc_code_after_four_weeks").count().toPandas()))

    estimator = RandomForestClassifier(featuresCol='indexed_features', labelCol="pasc_code_after_four_weeks", seed=seed,
                                numTrees=10)
    pipeline = Pipeline(stages=[featureIndexer, estimator])
    evaluator=MulticlassClassificationEvaluator(labelCol='pasc_code_after_four_weeks', predictionCol='prediction', metricName='accuracy')

    model = pipeline.fit(train)
    print('Default parameters accuracy on test set: {:f}'.format(evaluator.evaluate(model.transform(test))))

    ## full list of hyperparameters
    param_list = ['maxDepth', 'impurity', 'minInstancesPerNode', 'numTrees']
    paramGrid = ParamGridBuilder()\
            .addGrid(estimator.maxDepth, [10, 20, 30])\
            .addGrid(estimator.impurity, ['entropy', 'gini'])\
            .addGrid(estimator.minInstancesPerNode, [1, 5, 10])\
            .addGrid(estimator.numTrees, [20, 50, 100])\
            .build()

    ## short list of hyperparameters
    # param_list = ['maxDepth', 'impurity', 'numTrees']
    # paramGrid = ParamGridBuilder()\
    #         .addGrid(estimator.maxDepth, [10, 30])\
    #         .addGrid(estimator.impurity, ['entropy', 'gini'])\
    #         .addGrid(estimator.numTrees, [20, 100])\
    #         .build()

    tvs = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8,
                           seed=seed,
                           parallelism=10)
    model = tvs.fit(train)
    results = model.transform(test)
    print('Best model accuracy on test set: {:f}'.format(evaluator.evaluate(results)))

    best_params={param.name: value for param, value in zip(model.bestModel.stages[1].extractParamMap().keys(), model.bestModel.stages[1].extractParamMap().values()) if param.name in param_list}
    print("\n[Best hyperparameters]")
    for p in param_list: print('Best {}: {}'.format(p, best_params[p]))
    print("\n")
    df_best_params = pd.DataFrame(data=best_params, index=[0])

    my_eval = MulticlassClassificationEvaluator(labelCol = 'pasc_code_after_four_weeks', predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol ='pasc_code_after_four_weeks', rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select("pasc_code_after_four_weeks").collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("[Performance on Test set]")
    printPerformance(results)

    results = model.transform(df)
    print("\n[Performance on Whole set]")
    printPerformance(results)

    estimator.setParams(**best_params)
    model = pipeline.fit(df)
    results = model.transform(df)
    print("\n[Performance on Whole set with Fitting]")
    printPerformance(results)

    return df_best_params

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.819a3389-a43d-47ca-a87b-1e274c8c0087"),
    Condition_Features_1_0=Input(rid="ri.foundry.main.dataset.09f400a0-17c7-4e67-8ec4-97a1233caebd")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def RandomForestCond_1_0(Condition_Features_1_0):
    #parameters
    maxCategories=4
    seed=42

    df = Condition_Features_1_0
    featureIndexer =VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=maxCategories).fit(df)

    # stratified split
    class0 = df.filter(df["pasc_code_after_four_weeks"]==0)
    class1 = df.filter(df["pasc_code_after_four_weeks"]==1)
    print("Class 0 (pasc= 0): ", class0.count())
    print("Class 1 (pasc= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=seed)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=seed)

    train = train0.union(train1)
    test = test0.union(test1)
    print("[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas()))
    print("Test set: {}\n {}".format(test.count(), test.groupBy("pasc_code_after_four_weeks").count().toPandas()))

    estimator = RandomForestClassifier(featuresCol='indexed_features', labelCol="pasc_code_after_four_weeks", seed=seed,
                                numTrees=10)
    pipeline = Pipeline(stages=[featureIndexer, estimator])
    evaluator=MulticlassClassificationEvaluator(labelCol='pasc_code_after_four_weeks', predictionCol='prediction', metricName='accuracy')

    model = pipeline.fit(train)
    print('Default parameters accuracy on test set: {:f}'.format(evaluator.evaluate(model.transform(test))))

    ## full list of hyperparameters
    param_list = ['maxDepth', 'impurity', 'minInstancesPerNode', 'numTrees']
    paramGrid = ParamGridBuilder()\
            .addGrid(estimator.maxDepth, [10, 20, 30])\
            .addGrid(estimator.impurity, ['entropy', 'gini'])\
            .addGrid(estimator.minInstancesPerNode, [1, 5, 10])\
            .addGrid(estimator.numTrees, [20, 50, 100])\
            .build()

    ## short list of hyperparameters
    # param_list = ['maxDepth', 'impurity', 'numTrees']
    # paramGrid = ParamGridBuilder()\
    #         .addGrid(estimator.maxDepth, [10, 30])\
    #         .addGrid(estimator.impurity, ['entropy', 'gini'])\
    #         .addGrid(estimator.numTrees, [20, 100])\
    #         .build()

    tvs = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8,
                           seed=seed,
                           parallelism=10)
    model = tvs.fit(train)
    results = model.transform(test)
    print('Best model accuracy on test set: {:f}'.format(evaluator.evaluate(results)))

    best_params={param.name: value for param, value in zip(model.bestModel.stages[1].extractParamMap().keys(), model.bestModel.stages[1].extractParamMap().values()) if param.name in param_list}
    print("\n[Best hyperparameters]")
    for p in param_list: print('Best {}: {}'.format(p, best_params[p]))
    print("\n")
    df_best_params = pd.DataFrame(data=best_params, index=[0])

    my_eval = MulticlassClassificationEvaluator(labelCol = 'pasc_code_after_four_weeks', predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol ='pasc_code_after_four_weeks', rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select("pasc_code_after_four_weeks").collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("[Performance on Test set]")
    printPerformance(results)

    results = model.transform(df)
    print("\n[Performance on Whole set]")
    printPerformance(results)

    estimator.setParams(**best_params)
    model = pipeline.fit(df)
    results = model.transform(df)
    print("\n[Performance on Whole set with Fitting]")
    printPerformance(results)

    return df_best_params

@transform_pandas(
    Output(rid="ri.vector.main.execute.e5f1f959-2bf2-416d-881a-a02155c90d2f"),
    Cond_Features_1_0=Input(rid="ri.foundry.main.dataset.0a8462a4-1d09-4a31-a0c7-2a36dc7ab283"),
    Condition_Features_1_0=Input(rid="ri.foundry.main.dataset.09f400a0-17c7-4e67-8ec4-97a1233caebd"),
    RandomForestCond_1_0=Input(rid="ri.foundry.main.dataset.819a3389-a43d-47ca-a87b-1e274c8c0087")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
from pyspark.ml.feature import VectorIndexer
import numpy as np
from sklearn.metrics import confusion_matrix

def RandomForestCond_1_0_Test(Condition_Features_1_0, Cond_Features_1_0, RandomForestCond_1_0):
    best_params=RandomForestCond_1_0.to_dict('records')
    print("Best parameters: ",best_params[0])
    #parameters
    maxCategories=4
    seed=42
    label="pasc_code_after_four_weeks"

    train = Condition_Features_1_0
    test = Cond_Features_1_0
    print("Train columns not in Test:", [c for c in train.columns if c not in test.columns])
    print("Test columns not in Train:", [c for c in test.columns if c not in train.columns])
    print(train.columns)
    print(test.columns)

    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=maxCategories).fit(train)

    print("\n[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas().sort_values(by=label)))

    estimator = RandomForestClassifier(featuresCol='indexed_features', labelCol=label, seed=seed)
    estimator.setParams(**best_params[0])
    pipeline = Pipeline(stages=[featureIndexer, estimator])
    model = pipeline.fit(train)
    results_train = model.transform(train)
    results_test = model.transform(test)

    my_eval = MulticlassClassificationEvaluator(labelCol = label, predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol =label, rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select(label).collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("\n[Performance on Train set]")
    printPerformance(results_train)

    return results_test

@transform_pandas(
    Output(rid="ri.vector.main.execute.3139fc1c-25ad-4ea2-bb24-61dcfd7b2e35"),
    Cond_Features=Input(rid="ri.foundry.main.dataset.a15e6e2a-1451-433e-adde-d102587615e3"),
    Condition_Features=Input(rid="ri.foundry.main.dataset.60f2eb0d-7b8c-466f-9ae8-66105e882875"),
    RandomForestCond=Input(rid="ri.foundry.main.dataset.83ef5d36-3f80-4bc2-b813-841e2da1b1bc")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
from pyspark.ml.feature import VectorIndexer
import numpy as np
from sklearn.metrics import confusion_matrix

def RandomForestCond_Test(Condition_Features, Cond_Features, RandomForestCond):
    best_params=RandomForestCond.to_dict('records')
    print("Best parameters: ",best_params[0])
    #parameters
    maxCategories=4
    seed=42
    label="pasc_code_after_four_weeks"

    train = Condition_Features
    test = Cond_Features
    print("Train columns not in Test:", [c for c in train.columns if c not in test.columns])
    print("Test columns not in Train:", [c for c in test.columns if c not in train.columns])
    # df = train.union(test)

    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=maxCategories).fit(train)

    print("\n[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas().sort_values(by=label)))

    estimator = RandomForestClassifier(featuresCol='indexed_features', labelCol=label, seed=seed)
    estimator.setParams(**best_params[0])
    pipeline = Pipeline(stages=[featureIndexer, estimator])
    model = pipeline.fit(train)
    results_train = model.transform(train)
    results_test = model.transform(test)

    my_eval = MulticlassClassificationEvaluator(labelCol = label, predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol =label, rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select(label).collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("\n[Performance on Train set]")
    printPerformance(results_train)

    return results_test

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c9643448-a561-4557-8d94-d094b8f166c8"),
    Demographics_Features=Input(rid="ri.foundry.main.dataset.14be43ee-dbda-4b0d-b622-9c49b2c3bbca")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def RandomForestDem(Demographics_Features):
    #parameters
    maxCategories=4
    seed=42

    df = Demographics_Features
    featureIndexer =VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=maxCategories).fit(df)

    # stratified split
    class0 = df.filter(df["pasc_code_after_four_weeks"]==0)
    class1 = df.filter(df["pasc_code_after_four_weeks"]==1)
    print("Class 0 (pasc= 0): ", class0.count())
    print("Class 1 (pasc= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=seed)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=seed)

    train = train0.union(train1)
    test = test0.union(test1)
    print("[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas()))
    print("Test set: {}\n {}".format(test.count(), test.groupBy("pasc_code_after_four_weeks").count().toPandas()))

    estimator = RandomForestClassifier(featuresCol='indexed_features', labelCol="pasc_code_after_four_weeks", seed=seed,
                                numTrees=10)
    pipeline = Pipeline(stages=[featureIndexer, estimator])
    evaluator=MulticlassClassificationEvaluator(labelCol='pasc_code_after_four_weeks', predictionCol='prediction', metricName='accuracy')

    model = pipeline.fit(train)
    print('Default parameters accuracy on test set: {:f}'.format(evaluator.evaluate(model.transform(test))))

    ## full list of hyperparameters
    param_list = ['maxDepth', 'impurity', 'minInstancesPerNode', 'numTrees']
    paramGrid = ParamGridBuilder()\
            .addGrid(estimator.maxDepth, [10, 20, 30])\
            .addGrid(estimator.impurity, ['entropy', 'gini'])\
            .addGrid(estimator.minInstancesPerNode, [1, 5, 10])\
            .addGrid(estimator.numTrees, [20, 50, 100])\
            .build()

    ## short list of hyperparameters
    # param_list = ['maxDepth', 'impurity', 'numTrees']
    # paramGrid = ParamGridBuilder()\
    #         .addGrid(estimator.maxDepth, [10, 30])\
    #         .addGrid(estimator.impurity, ['entropy', 'gini'])\
    #         .addGrid(estimator.numTrees, [20, 100])\
    #         .build()

    tvs = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8,
                           seed=seed,
                           parallelism=10)
    model = tvs.fit(train)
    results = model.transform(test)
    print('Best model accuracy on test set: {:f}'.format(evaluator.evaluate(results)))

    best_params={param.name: value for param, value in zip(model.bestModel.stages[1].extractParamMap().keys(), model.bestModel.stages[1].extractParamMap().values()) if param.name in param_list}
    print("\n[Best hyperparameters]")
    for p in param_list: print('Best {}: {}'.format(p, best_params[p]))
    print("\n")
    df_best_params = pd.DataFrame(data=best_params, index=[0])

    my_eval = MulticlassClassificationEvaluator(labelCol = 'pasc_code_after_four_weeks', predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol ='pasc_code_after_four_weeks', rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select("pasc_code_after_four_weeks").collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("[Performance on Test set]")
    printPerformance(results)

    results = model.transform(df)
    print("\n[Performance on Whole set]")
    printPerformance(results)

    estimator.setParams(**best_params)
    model = pipeline.fit(df)
    results = model.transform(df)
    print("\n[Performance on Whole set with Fitting]")
    printPerformance(results)

    return df_best_params

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.963fe618-d85c-4d76-968a-a3c759f6b409"),
    Demographic_test_Features=Input(rid="ri.foundry.main.dataset.2bfd07f4-4dd7-42b9-b6c9-59bd790d8abb"),
    Demographics_Features=Input(rid="ri.foundry.main.dataset.14be43ee-dbda-4b0d-b622-9c49b2c3bbca"),
    RandomForestDem=Input(rid="ri.foundry.main.dataset.c9643448-a561-4557-8d94-d094b8f166c8")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
from pyspark.ml.feature import VectorIndexer
import numpy as np
from sklearn.metrics import confusion_matrix

def RandomForestDem_Test(Demographics_Features, Demographic_test_Features, RandomForestDem):
    best_params=RandomForestDem.to_dict('records')
    print("Best parameters: ",best_params[0])
    #parameters
    maxCategories=4
    seed=42
    label="pasc_code_after_four_weeks"


    test = Demographic_test_Features
    train = Demographics_Features.select(test.columns+[label])
    print("Train columns not in Test:", [c for c in train.columns if c not in test.columns])
    print("Test columns not in Train:", [c for c in test.columns if c not in train.columns])
    print(train.columns)
    print(test.columns)

    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=maxCategories).fit(train)

    print("\n[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas().sort_values(by=label)))
    # print("Train set: {}\n".format(test.count())

    estimator = RandomForestClassifier(featuresCol='indexed_features', labelCol=label, seed=seed)
    estimator.setParams(**best_params[0])
    pipeline = Pipeline(stages=[featureIndexer, estimator])
    model = pipeline.fit(train)
    results_train = model.transform(train)
    results_test = model.transform(test)

    my_eval = MulticlassClassificationEvaluator(labelCol = label, predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol =label, rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select(label).collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("\n[Performance on Train set]")
    printPerformance(results_train)

    return results_test

@transform_pandas(
    Output(rid="ri.vector.main.execute.94609a36-719d-44e0-a16d-9451115bf091"),
    Diagnosis_1_0=Input(rid="ri.foundry.main.dataset.b7b9f659-64e6-4e0a-b8c8-c3a97ece514e"),
    Diagnosis_Feature_1_0=Input(rid="ri.foundry.main.dataset.1948d0b2-7832-42de-9d07-323558202543"),
    RandomForestDiag_1_0=Input(rid="ri.foundry.main.dataset.739b46ba-6c3e-4162-a6b8-2bc7f488ff9a")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
from pyspark.ml.feature import VectorIndexer
import numpy as np
from sklearn.metrics import confusion_matrix

def RandomForestDia_1_0_Test(Diagnosis_1_0, Diagnosis_Feature_1_0, RandomForestDiag_1_0):
    best_params=RandomForestDiag_1_0.to_dict('records')
    print("Best parameters: ",best_params[0])
    #parameters
    maxCategories=4
    seed=42
    label="pasc_code_after_four_weeks"

    test = Diagnosis_Feature_1_0
    train = Diagnosis_1_0.select(test.columns+[label])
    print("Train columns not in Test:", [c for c in train.columns if c not in test.columns])
    print("Test columns not in Train:", [c for c in test.columns if c not in train.columns])
    print(train.columns)
    print(test.columns)
    train.show(5)
    test.show(5)

    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=maxCategories).fit(train)

    print("\n[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas().sort_values(by=label)))

    estimator = RandomForestClassifier(featuresCol='indexed_features', labelCol=label, seed=seed)
    estimator.setParams(**best_params[0])
    pipeline = Pipeline(stages=[featureIndexer, estimator])
    model = pipeline.fit(train)
    results_train = model.transform(train)
    results_test = model.transform(test)

    my_eval = MulticlassClassificationEvaluator(labelCol = label, predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol =label, rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select(label).collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("\n[Performance on Train set]")
    printPerformance(results_train)

    return results_test

@transform_pandas(
    Output(rid="ri.vector.main.execute.66cf735d-61f9-4cec-91ce-b125fac6376a"),
    Diagnosis_Feature=Input(rid="ri.foundry.main.dataset.445fc6a7-da36-4ac0-b613-d8c4b20db14c"),
    Diagnosis_Features=Input(rid="ri.vector.main.execute.49f1d284-20c7-4aca-a3b3-6a1d25610f25"),
    RandomForestDiag=Input(rid="ri.foundry.main.dataset.92975356-0735-4fe5-8071-510688a853f7")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
from pyspark.ml.feature import VectorIndexer
import numpy as np
from sklearn.metrics import confusion_matrix

def RandomForestDia_Test(Diagnosis_Features, Diagnosis_Feature, RandomForestDiag):
    best_params=RandomForestDiag.to_dict('records')
    print("Best parameters: ",best_params[0])
    #parameters
    maxCategories=4
    seed=42
    label="pasc_code_after_four_weeks"

    train = Diagnosis_Features
    test = Diagnosis_Feature
    print("Train columns not in Test:", [c for c in train.columns if c not in test.columns])
    print("Test columns not in Train:", [c for c in test.columns if c not in train.columns])
    print(train.columns)
    print(test.columns)

    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=maxCategories).fit(train)

    print("\n[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas().sort_values(by=label)))

    estimator = RandomForestClassifier(featuresCol='indexed_features', labelCol=label, seed=seed)
    estimator.setParams(**best_params[0])
    pipeline = Pipeline(stages=[featureIndexer, estimator])
    model = pipeline.fit(train)
    results_train = model.transform(train)
    results_test = model.transform(test)

    my_eval = MulticlassClassificationEvaluator(labelCol = label, predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol =label, rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select(label).collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("\n[Performance on Train set]")
    printPerformance(results_train)

    return results_test

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.92975356-0735-4fe5-8071-510688a853f7"),
    Diagnosis_Features=Input(rid="ri.vector.main.execute.49f1d284-20c7-4aca-a3b3-6a1d25610f25")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def RandomForestDiag(Diagnosis_Features):
    #parameters
    maxCategories=4
    seed=42

    df = Diagnosis_Features
    featureIndexer =VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=maxCategories).fit(df)

    # stratified split
    class0 = df.filter(df["pasc_code_after_four_weeks"]==0)
    class1 = df.filter(df["pasc_code_after_four_weeks"]==1)
    print("Class 0 (pasc= 0): ", class0.count())
    print("Class 1 (pasc= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=seed)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=seed)

    train = train0.union(train1)
    test = test0.union(test1)
    print("[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas()))
    print("Test set: {}\n {}".format(test.count(), test.groupBy("pasc_code_after_four_weeks").count().toPandas()))

    estimator = RandomForestClassifier(featuresCol='indexed_features', labelCol="pasc_code_after_four_weeks", seed=seed,
                                numTrees=10)
    pipeline = Pipeline(stages=[featureIndexer, estimator])
    evaluator=MulticlassClassificationEvaluator(labelCol='pasc_code_after_four_weeks', predictionCol='prediction', metricName='accuracy')

    model = pipeline.fit(train)
    print('Default parameters accuracy on test set: {:f}'.format(evaluator.evaluate(model.transform(test))))

    ## full list of hyperparameters
    param_list = ['maxDepth', 'impurity', 'minInstancesPerNode', 'numTrees']
    paramGrid = ParamGridBuilder()\
            .addGrid(estimator.maxDepth, [10, 20, 30])\
            .addGrid(estimator.impurity, ['entropy', 'gini'])\
            .addGrid(estimator.minInstancesPerNode, [1, 5, 10])\
            .addGrid(estimator.numTrees, [20, 50, 100])\
            .build()

    ## short list of hyperparameters
    # param_list = ['maxDepth', 'impurity', 'numTrees']
    # paramGrid = ParamGridBuilder()\
    #         .addGrid(estimator.maxDepth, [10, 30])\
    #         .addGrid(estimator.impurity, ['entropy', 'gini'])\
    #         .addGrid(estimator.numTrees, [20, 100])\
    #         .build()

    tvs = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8,
                           seed=seed,
                           parallelism=10)
    model = tvs.fit(train)
    results = model.transform(test)
    print('Best model accuracy on test set: {:f}'.format(evaluator.evaluate(results)))

    best_params={param.name: value for param, value in zip(model.bestModel.stages[1].extractParamMap().keys(), model.bestModel.stages[1].extractParamMap().values()) if param.name in param_list}
    print("\n[Best hyperparameters]")
    for p in param_list: print('Best {}: {}'.format(p, best_params[p]))
    print("\n")
    df_best_params = pd.DataFrame(data=best_params, index=[0])

    my_eval = MulticlassClassificationEvaluator(labelCol = 'pasc_code_after_four_weeks', predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol ='pasc_code_after_four_weeks', rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select("pasc_code_after_four_weeks").collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("[Performance on Test set]")
    printPerformance(results)

    results = model.transform(df)
    print("\n[Performance on Whole set]")
    printPerformance(results)

    estimator.setParams(**best_params)
    model = pipeline.fit(df)
    results = model.transform(df)
    print("\n[Performance on Whole set with Fitting]")
    printPerformance(results)

    return df_best_params

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.739b46ba-6c3e-4162-a6b8-2bc7f488ff9a"),
    Diagnosis_1_0=Input(rid="ri.foundry.main.dataset.b7b9f659-64e6-4e0a-b8c8-c3a97ece514e")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def RandomForestDiag_1_0(Diagnosis_1_0):
    #parameters
    maxCategories=4
    seed=42

    df = Diagnosis_1_0
    featureIndexer =VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=maxCategories).fit(df)

    # stratified split
    class0 = df.filter(df["pasc_code_after_four_weeks"]==0)
    class1 = df.filter(df["pasc_code_after_four_weeks"]==1)
    print("Class 0 (pasc= 0): ", class0.count())
    print("Class 1 (pasc= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=seed)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=seed)

    train = train0.union(train1)
    test = test0.union(test1)
    print("[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas()))
    print("Test set: {}\n {}".format(test.count(), test.groupBy("pasc_code_after_four_weeks").count().toPandas()))

    estimator = RandomForestClassifier(featuresCol='indexed_features', labelCol="pasc_code_after_four_weeks", seed=seed,
                                numTrees=10)
    pipeline = Pipeline(stages=[featureIndexer, estimator])
    evaluator=MulticlassClassificationEvaluator(labelCol='pasc_code_after_four_weeks', predictionCol='prediction', metricName='accuracy')

    model = pipeline.fit(train)
    print('Default parameters accuracy on test set: {:f}'.format(evaluator.evaluate(model.transform(test))))

    ## full list of hyperparameters
    param_list = ['maxDepth', 'impurity', 'minInstancesPerNode', 'numTrees']
    paramGrid = ParamGridBuilder()\
            .addGrid(estimator.maxDepth, [10, 20, 30])\
            .addGrid(estimator.impurity, ['entropy', 'gini'])\
            .addGrid(estimator.minInstancesPerNode, [1, 5, 10])\
            .addGrid(estimator.numTrees, [20, 50, 100])\
            .build()

    ## short list of hyperparameters
    # param_list = ['maxDepth', 'impurity', 'numTrees']
    # paramGrid = ParamGridBuilder()\
    #         .addGrid(estimator.maxDepth, [10, 30])\
    #         .addGrid(estimator.impurity, ['entropy', 'gini'])\
    #         .addGrid(estimator.numTrees, [20, 100])\
    #         .build()

    tvs = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8,
                           seed=seed,
                           parallelism=10)
    model = tvs.fit(train)
    results = model.transform(test)
    print('Best model accuracy on test set: {:f}'.format(evaluator.evaluate(results)))

    best_params={param.name: value for param, value in zip(model.bestModel.stages[1].extractParamMap().keys(), model.bestModel.stages[1].extractParamMap().values()) if param.name in param_list}
    print("\n[Best hyperparameters]")
    for p in param_list: print('Best {}: {}'.format(p, best_params[p]))
    print("\n")
    df_best_params = pd.DataFrame(data=best_params, index=[0])

    my_eval = MulticlassClassificationEvaluator(labelCol = 'pasc_code_after_four_weeks', predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol ='pasc_code_after_four_weeks', rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select("pasc_code_after_four_weeks").collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("[Performance on Test set]")
    printPerformance(results)

    results = model.transform(df)
    print("\n[Performance on Whole set]")
    printPerformance(results)

    estimator.setParams(**best_params)
    model = pipeline.fit(df)
    results = model.transform(df)
    print("\n[Performance on Whole set with Fitting]")
    printPerformance(results)

    return df_best_params

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.531c3a1c-ae0c-41ae-a7cd-5a68d9b2bfdf"),
    observation_Features=Input(rid="ri.foundry.main.dataset.1795a3cf-d1e3-4886-ba73-f13e71d175ab")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def RandomForestObs(observation_Features):
    #parameters
    maxCategories=4
    seed=42
    label = "pasc_code_after_four_weeks"
    df = observation_Features
    featureIndexer =VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=maxCategories).fit(df)

    # stratified split
    class0 = df.filter(df["pasc_code_after_four_weeks"]==0)
    class1 = df.filter(df["pasc_code_after_four_weeks"]==1)
    print("Class 0 (pasc= 0): ", class0.count())
    print("Class 1 (pasc= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=seed)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=seed)

    train = train0.union(train1)
    test = test0.union(test1)
    print("[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas()))
    print("Test set: {}\n {}".format(test.count(), test.groupBy("pasc_code_after_four_weeks").count().toPandas()))

    estimator = RandomForestClassifier(featuresCol='indexed_features', labelCol="pasc_code_after_four_weeks", seed=seed,
                                numTrees=10)
    pipeline = Pipeline(stages=[featureIndexer, estimator])
    evaluator=MulticlassClassificationEvaluator(labelCol='pasc_code_after_four_weeks', predictionCol='prediction', metricName='accuracy')

    model = pipeline.fit(train)
    print('Default parameters accuracy on test set: {:f}'.format(evaluator.evaluate(model.transform(test))))

    ## full list of hyperparameters
    param_list = ['maxDepth', 'impurity', 'minInstancesPerNode', 'numTrees']
    paramGrid = ParamGridBuilder()\
            .addGrid(estimator.maxDepth, [10, 20, 30])\
            .addGrid(estimator.impurity, ['entropy', 'gini'])\
            .addGrid(estimator.minInstancesPerNode, [1, 5, 10])\
            .addGrid(estimator.numTrees, [20, 50, 100])\
            .build()

    ## short list of hyperparameters
    # param_list = ['maxDepth', 'impurity', 'numTrees']
    # paramGrid = ParamGridBuilder()\
    #         .addGrid(estimator.maxDepth, [10, 30])\
    #         .addGrid(estimator.impurity, ['entropy', 'gini'])\
    #         .addGrid(estimator.numTrees, [20, 100])\
    #         .build()

    tvs = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8,
                           seed=seed,
                           parallelism=10)
    model = tvs.fit(train)
    results = model.transform(test)
    print('Best model accuracy on test set: {:f}'.format(evaluator.evaluate(results)))

    best_params={param.name: value for param, value in zip(model.bestModel.stages[1].extractParamMap().keys(), model.bestModel.stages[1].extractParamMap().values()) if param.name in param_list}
    print("\n[Best hyperparameters]")
    for p in param_list: print('Best {}: {}'.format(p, best_params[p]))
    print("\n")
    df_best_params = pd.DataFrame(data=best_params, index=[0])

    my_eval = MulticlassClassificationEvaluator(labelCol = 'pasc_code_after_four_weeks', predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol ='pasc_code_after_four_weeks', rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select("pasc_code_after_four_weeks").collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("[Performance on Test set]")
    printPerformance(results)

    results = model.transform(df)
    print("\n[Performance on Whole set]")
    printPerformance(results)

    estimator.setParams(**best_params)
    model = pipeline.fit(df)
    results = model.transform(df)
    print("\n[Performance on Whole set with Fitting]")
    printPerformance(results)

    return df_best_params

@transform_pandas(
    Output(rid="ri.vector.main.execute.f44863fc-cd8a-4326-b439-dac041cd4018"),
    RandomForestObs_1_0=Input(rid="ri.foundry.main.dataset.8095aa62-be63-4c68-bcdb-a8135f9bb57e"),
    obs_features_1_0=Input(rid="ri.foundry.main.dataset.0daaade6-46a5-47ad-a941-00e5201adf7a"),
    observation_Features_1_0=Input(rid="ri.foundry.main.dataset.0633476b-c599-44a7-8575-d2f44a9dedcf")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
from pyspark.ml.feature import VectorIndexer
import numpy as np
from sklearn.metrics import confusion_matrix

def RandomForestObs1_0_Test(observation_Features_1_0, obs_features_1_0, RandomForestObs_1_0):
    best_params=RandomForestObs_1_0.to_dict('records')
    print("Best parameters: ",best_params[0])
    #parameters
    maxCategories=4
    seed=42
    label="pasc_code_after_four_weeks"

    train = observation_Features_1_0
    test = obs_features_1_0
    print("Train columns not in Test:", [c for c in train.columns if c not in test.columns])
    print("Test columns not in Train:", [c for c in test.columns if c not in train.columns])
    print(train.columns)
    print(test.columns)

    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=maxCategories).fit(train)

    print("\n[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas().sort_values(by=label)))

    estimator = RandomForestClassifier(featuresCol='indexed_features', labelCol=label, seed=seed)
    estimator.setParams(**best_params[0])
    pipeline = Pipeline(stages=[featureIndexer, estimator])
    model = pipeline.fit(train)
    results_train = model.transform(train)
    results_test = model.transform(test)

    my_eval = MulticlassClassificationEvaluator(labelCol = label, predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol =label, rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select(label).collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("\n[Performance on Train set]")
    printPerformance(results_train)

    return results_test

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.8095aa62-be63-4c68-bcdb-a8135f9bb57e"),
    observation_Features_1_0=Input(rid="ri.foundry.main.dataset.0633476b-c599-44a7-8575-d2f44a9dedcf")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def RandomForestObs_1_0(observation_Features_1_0):
    #parameters
    maxCategories=4
    seed=42
    label = "pasc_code_after_four_weeks"
    df = observation_Features_1_0
    featureIndexer =VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=maxCategories).fit(df)

    # stratified split
    class0 = df.filter(df["pasc_code_after_four_weeks"]==0)
    class1 = df.filter(df["pasc_code_after_four_weeks"]==1)
    print("Class 0 (pasc= 0): ", class0.count())
    print("Class 1 (pasc= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=seed)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=seed)

    train = train0.union(train1)
    test = test0.union(test1)
    print("[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas()))
    print("Test set: {}\n {}".format(test.count(), test.groupBy("pasc_code_after_four_weeks").count().toPandas()))

    estimator = RandomForestClassifier(featuresCol='indexed_features', labelCol="pasc_code_after_four_weeks", seed=seed,
                                numTrees=10)
    pipeline = Pipeline(stages=[featureIndexer, estimator])
    evaluator=MulticlassClassificationEvaluator(labelCol='pasc_code_after_four_weeks', predictionCol='prediction', metricName='accuracy')

    model = pipeline.fit(train)
    print('Default parameters accuracy on test set: {:f}'.format(evaluator.evaluate(model.transform(test))))

    ## full list of hyperparameters
    param_list = ['maxDepth', 'impurity', 'minInstancesPerNode', 'numTrees']
    paramGrid = ParamGridBuilder()\
            .addGrid(estimator.maxDepth, [10, 20, 30])\
            .addGrid(estimator.impurity, ['entropy', 'gini'])\
            .addGrid(estimator.minInstancesPerNode, [1, 5, 10])\
            .addGrid(estimator.numTrees, [20, 50, 100])\
            .build()

    ## short list of hyperparameters
    # param_list = ['maxDepth', 'impurity', 'numTrees']
    # paramGrid = ParamGridBuilder()\
    #         .addGrid(estimator.maxDepth, [10, 30])\
    #         .addGrid(estimator.impurity, ['entropy', 'gini'])\
    #         .addGrid(estimator.numTrees, [20, 100])\
    #         .build()

    tvs = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8,
                           seed=seed,
                           parallelism=10)
    model = tvs.fit(train)
    results = model.transform(test)
    print('Best model accuracy on test set: {:f}'.format(evaluator.evaluate(results)))

    best_params={param.name: value for param, value in zip(model.bestModel.stages[1].extractParamMap().keys(), model.bestModel.stages[1].extractParamMap().values()) if param.name in param_list}
    print("\n[Best hyperparameters]")
    for p in param_list: print('Best {}: {}'.format(p, best_params[p]))
    print("\n")
    df_best_params = pd.DataFrame(data=best_params, index=[0])

    my_eval = MulticlassClassificationEvaluator(labelCol = 'pasc_code_after_four_weeks', predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol ='pasc_code_after_four_weeks', rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select("pasc_code_after_four_weeks").collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("[Performance on Test set]")
    printPerformance(results)

    results = model.transform(df)
    print("\n[Performance on Whole set]")
    printPerformance(results)

    estimator.setParams(**best_params)
    model = pipeline.fit(df)
    results = model.transform(df)
    print("\n[Performance on Whole set with Fitting]")
    printPerformance(results)

    return df_best_params

@transform_pandas(
    Output(rid="ri.vector.main.execute.1d23fcec-e828-448e-a1b4-a6f6e015cad8"),
    RandomForestObs=Input(rid="ri.foundry.main.dataset.531c3a1c-ae0c-41ae-a7cd-5a68d9b2bfdf"),
    obs_features=Input(rid="ri.foundry.main.dataset.16de7401-d0bc-4e71-ad07-c58bcb94e8db"),
    observation_Features=Input(rid="ri.foundry.main.dataset.1795a3cf-d1e3-4886-ba73-f13e71d175ab")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
from pyspark.ml.feature import VectorIndexer
import numpy as np
from sklearn.metrics import confusion_matrix

def RandomForestObs_Test(observation_Features, obs_features, RandomForestObs):
    best_params=RandomForestObs.to_dict('records')
    print("Best parameters: ",best_params[0])
    #parameters
    maxCategories=4
    seed=42
    label="pasc_code_after_four_weeks"

    train = observation_Features
    test = obs_features
    print("Train columns not in Test:", [c for c in train.columns if c not in test.columns])
    print("Test columns not in Train:", [c for c in test.columns if c not in train.columns])
    print(train.columns)
    print(test.columns)
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=maxCategories).fit(train)

    print("\n[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas().sort_values(by=label)))

    estimator = RandomForestClassifier(featuresCol='indexed_features', labelCol=label, seed=seed)
    estimator.setParams(**best_params[0])
    pipeline = Pipeline(stages=[featureIndexer, estimator])
    model = pipeline.fit(train)
    results_train = model.transform(train)
    results_test = model.transform(test)

    my_eval = MulticlassClassificationEvaluator(labelCol = label, predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol =label, rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select(label).collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("\n[Performance on Train set]")
    printPerformance(results_train)

    return results_test

@transform_pandas(
    Output(rid="ri.vector.main.execute.abb9d7f6-5062-4d9b-bacb-2a62e03a71d3"),
    Condition_Features_1_0=Input(rid="ri.foundry.main.dataset.09f400a0-17c7-4e67-8ec4-97a1233caebd")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def decTree(Condition_Features_1_0):
    #parameters
    maxCategories=4
    seed=42

    df = Condition_Features_1_0
    featureIndexer =VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=maxCategories).fit(df)

    # stratified split
    class0 = df.filter(df["pasc_code_after_four_weeks"]==0)
    class1 = df.filter(df["pasc_code_after_four_weeks"]==1)
    print("Class 0 (pasc= 0): ", class0.count())
    print("Class 1 (pasc= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=seed)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=seed)

    train = train0.union(train1)
    test = test0.union(test1)
    print("[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas()))
    print("Test set: {}\n {}".format(test.count(), test.groupBy("pasc_code_after_four_weeks").count().toPandas()))

    estimator = DecisionTreeClassifier(featuresCol='indexed_features', labelCol="pasc_code_after_four_weeks", seed=seed)
    pipeline = Pipeline(stages=[featureIndexer, estimator])
    evaluator=MulticlassClassificationEvaluator(labelCol='pasc_code_after_four_weeks', predictionCol='prediction', metricName='accuracy')

    model = pipeline.fit(train)
    print('Default parameters accuracy on test set: {:f}'.format(evaluator.evaluate(model.transform(test))))

    param_list = ['maxDepth', 'impurity', 'minInstancesPerNode', 'minInfoGain', 'minWeightFractionPerNode']
    paramGrid = ParamGridBuilder()\
            .addGrid(estimator.maxDepth, [10, 20, 30])\
            .addGrid(estimator.impurity, ['entropy', 'gini'])\
            .addGrid(estimator.minInstancesPerNode, [1, 5, 10])\
            .addGrid(estimator.minInfoGain, [0.0, 0.1, 0.5])\
            .addGrid(estimator.minWeightFractionPerNode, [0.0, 0.1, 0.3])\
            .build()

    tvs = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8,
                           seed=seed,
                           parallelism=10)

    model = tvs.fit(train)
    results = model.transform(test)
    print('Best model accuracy on test set: {:f}'.format(evaluator.evaluate(results)))

    best_params={param.name: value for param, value in zip(model.bestModel.stages[1].extractParamMap().keys(), model.bestModel.stages[1].extractParamMap().values())}
    print("\n[Best hyperparameters]")
    for p in param_list: print('Best {}: {}'.format(p, best_params[p]))
    print("\n")

    my_eval = MulticlassClassificationEvaluator(labelCol = 'pasc_code_after_four_weeks', predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    print("[Performance on test set]")
    for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))

    my_eval_bin = BinaryClassificationEvaluator(labelCol ='pasc_code_after_four_weeks', rawPredictionCol = 'prediction')
    metrics=['areaUnderROC', 'areaUnderPR']
    for m in metrics: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))

    y_test = np.array(results.select("pasc_code_after_four_weeks").collect()).flatten()
    y_pred = np.array(results.select("prediction").collect()).flatten()
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    results = model.transform(df)
    y_test = np.array(results.select("pasc_code_after_four_weeks").collect()).flatten()
    y_pred = np.array(results.select("prediction").collect()).flatten()
    print("\nonfusion Matrix on Whole DataFrame:\n", confusion_matrix(y_test, y_pred))

    return results

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0e297085-73be-4c30-b1fe-14a3ad5a6731"),
    Final_Model=Input(rid="ri.foundry.main.dataset.155ea8bc-546c-4772-92c9-2b51e08869b1")
)
def drop(Final_Model):
    df = Final_Model
    df = df.dropna()

    return df

@transform_pandas(
    Output(rid="ri.vector.main.execute.3f3a4e26-2d2a-47df-b2ee-e218bb999d66"),
    Condition_Features_1_0=Input(rid="ri.foundry.main.dataset.09f400a0-17c7-4e67-8ec4-97a1233caebd")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def gradientBoost(Condition_Features_1_0):
    #parameters
    maxCategories=4
    seed=42

    df = Condition_Features_1_0
    featureIndexer =VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=maxCategories).fit(df)

    # stratified split
    class0 = df.filter(df["pasc_code_after_four_weeks"]==0)
    class1 = df.filter(df["pasc_code_after_four_weeks"]==1)
    print("Class 0 (pasc= 0): ", class0.count())
    print("Class 1 (pasc= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=seed)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=seed)

    train = train0.union(train1)
    test = test0.union(test1)
    print("[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas()))
    print("Test set: {}\n {}".format(test.count(), test.groupBy("pasc_code_after_four_weeks").count().toPandas()))

    estimator = GBTClassifier(featuresCol='indexed_features', labelCol="pasc_code_after_four_weeks", seed=seed,
                       maxIter=10)
    pipeline = Pipeline(stages=[featureIndexer, estimator])
    evaluator=MulticlassClassificationEvaluator(labelCol='pasc_code_after_four_weeks', predictionCol='prediction', metricName='accuracy')

    model = pipeline.fit(train)
    print('Default parameters accuracy on test set: {:f}'.format(evaluator.evaluate(model.transform(test))))

    ## full list of hyperparameters
    param_list = ['maxDepth', 'minInstancesPerNode', 'minInfoGain', 'subsamplingRate', 'minWeightFractionPerNode', 'maxIter']
    paramGrid = ParamGridBuilder()\
            .addGrid(estimator.maxDepth, [10, 20, 30])\
            .addGrid(estimator.minInstancesPerNode, [1, 5, 10])\
            .addGrid(estimator.minInfoGain, [0.0, 0.1, 0.5])\
            .addGrid(estimator.subsamplingRate, [0.1, 0.3, 0.5])\
            .addGrid(estimator.minWeightFractionPerNode, [0.0, 0.1, 0.3])\
            .addGrid(estimator.maxIter, [20, 30, 50])\
            .build()

     ## short list of hyperparameters
    # param_list = ['maxDepth', 'maxIter']
    # paramGrid = ParamGridBuilder()\
    #         .addGrid(estimator.maxDepth, [10, 30])\
    #         .addGrid(estimator.maxIter, [20, 50])\
    #         .build()

    tvs = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8,
                           seed=seed,
                           parallelism=10)

    model = tvs.fit(train)
    results = model.transform(test)
    print('Best model accuracy on test set: {:f}'.format(evaluator.evaluate(results)))

    best_params={param.name: value for param, value in zip(model.bestModel.stages[1].extractParamMap().keys(), model.bestModel.stages[1].extractParamMap().values())}
    print("\n[Best hyperparameters]")
    for p in param_list: print('Best {}: {}'.format(p, best_params[p]))
    print("\n")

    my_eval = MulticlassClassificationEvaluator(labelCol = 'pasc_code_after_four_weeks', predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    print("[Performance on test set]")
    for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))

    my_eval_bin = BinaryClassificationEvaluator(labelCol ='pasc_code_after_four_weeks', rawPredictionCol = 'prediction')
    metrics=['areaUnderROC', 'areaUnderPR']
    for m in metrics: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))

    y_test = np.array(results.select("pasc_code_after_four_weeks").collect()).flatten()
    y_pred = np.array(results.select("prediction").collect()).flatten()
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    results = model.transform(df)
    y_test = np.array(results.select("pasc_code_after_four_weeks").collect()).flatten()
    y_pred = np.array(results.select("prediction").collect()).flatten()
    print("\nonfusion Matrix on Whole DataFrame:\n", confusion_matrix(y_test, y_pred))

    return results

@transform_pandas(
    Output(rid="ri.vector.main.execute.2b96d4af-fa47-4331-b6b3-7195f144cd4e"),
    Condition_Features_1_0=Input(rid="ri.foundry.main.dataset.09f400a0-17c7-4e67-8ec4-97a1233caebd")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def logReg(Condition_Features_1_0):
    #parameters
    seed=42

    df = Condition_Features_1_0
    ## stratified split
    class0 = df.filter(df["pasc_code_after_four_weeks"]==0)
    class1 = df.filter(df["pasc_code_after_four_weeks"]==1)
    print("Class 0 (pasc= 0): ", class0.count())
    print("Class 1 (pasc= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=seed)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=seed)

    train = train0.union(train1)
    test = test0.union(test1)
    print("[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("pasc_code_after_four_weeks").count().toPandas()))
    print("Test set: {}\n {}".format(test.count(), test.groupBy("pasc_code_after_four_weeks").count().toPandas()))

    ## if regParam=0 -> Ridge regression, elasticNetParam=1 -> Lasso
    estimator = LogisticRegression(featuresCol='features', labelCol='pasc_code_after_four_weeks')
    evaluator=MulticlassClassificationEvaluator(labelCol='pasc_code_after_four_weeks', predictionCol='prediction', metricName='accuracy')

    model = estimator.fit(train)
    print('Default parameters accuracy on test set: {:f}'.format(evaluator.evaluate(model.transform(test))))

    param_list= ['maxIter', 'regParam', 'elasticNetParam']
    paramGrid = ParamGridBuilder()\
            .addGrid(estimator.maxIter, [10, 50, 100, 500, 1000])\
            .addGrid(estimator.regParam, [0.0, 0.001, 0.1, 0.5, 1])\
            .addGrid(estimator.elasticNetParam, [0.0, 0.001, 0.1, 0.5, 1])\
            .build()

    tvs = TrainValidationSplit(estimator=estimator,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8,
                           seed=seed,
                           parallelism=10)

    model = tvs.fit(train)
    results = model.transform(test)
    print('Best model accuracy on test set: {:f}'.format(evaluator.evaluate(results)))

    best_params={param.name: value for param, value in zip(model.bestModel.extractParamMap().keys(), model.bestModel.extractParamMap().values()) if param.name in param_list}
    print("\n[Best hyperparameters]")
    for p in param_list: print('Best {}: {}'.format(p, best_params[p]))
    print("\n")
    df_best_params = pd.DataFrame(data=best_params, index=[0])

    my_eval = MulticlassClassificationEvaluator(labelCol = 'pasc_code_after_four_weeks', predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol ='pasc_code_after_four_weeks', rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select("pasc_code_after_four_weeks").collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("[Performance on Test set]")
    printPerformance(results)

    results = model.transform(df)
    print("\n[Performance on Whole set]")
    printPerformance(results)

    estimator.setParams(**best_params)
    model = estimator.fit(df)
    results = model.transform(df)
    print("\n[Performance on Whole set with Fitting]")
    printPerformance(results)

    return df_best_params

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e90f57b1-7d56-40d3-98bd-92ad05217424"),
    Demographics_=Input(rid="ri.foundry.main.dataset.3e49bf18-4379-49db-84c1-a86770214a78"),
    age=Input(rid="ri.foundry.main.dataset.8a1f50d0-81e8-4670-9a3a-099594870b0b")
)
def mergedDemo(Demographics_, age):
    df = Demographics_
    df_age = age
    df_ =df.join(df_age, on=['person_id'])
    return df_

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c9223d90-a29f-442b-ad11-10bd70ecd355"),
    merged=Input(rid="ri.foundry.main.dataset.43b894f3-c1bf-4fdf-8305-c88cca2fa7aa")
)
def merged_cleaned(merged):
    df = merged
    df=df.withColumn('age', df.age_.cast('int'))
    df=df.withColumn('year_of_birth', df.year_of_birth.cast('int'))


    df_ = df.withColumn('pasc',when((col('pasc_code_after_four_weeks') == 1)| (col('pasc_code_prior_four_weeks')==1), 1).otherwise(0))

    df_ = df_.withColumn('ageGroup_infant', when(col('age') < 2, 1)).withColumn('ageGroup_toddler', when(((col('age') >= 2) & (col('age') < 4)), 1)).withColumn('ageGroup_adolescent', when(((col('age') >= 4) & (col('age') < 14)), 1)).withColumn('ageGroup_youngAd', when(((col('age') >= 14) & (col('age') < 30)), 1)).withColumn('ageGroup_adult', when(((col('age') >= 30) & (col('age') < 50)), 1)).withColumn('ageGroup_olderAd', when(((col('age') >= 50) & (col('age') < 90)), 1)).withColumn('ageGroup_elderly', when((col('is_age_90_or_older') == 'true'), 1)).fillna(0)

    gender = df_.select('gender_concept_name').distinct().collect()
    race = df_.select('race_concept_name').distinct().collect()
    ethnicity = df_.select('ethnicity_concept_name').distinct().collect()

    df_ = df_.withColumn('gender_fem', when(col('gender_concept_name') == 'FEMALE', 1)).withColumn('gender_mal', when(col('gender_concept_name') == 'MALE', 1)).withColumn('gender_unk', when(col('gender_concept_name') == 'UNKN', 1)).fillna(0).drop('gender_concept_name', 'gender_concept_id')

    df_ = df_.withColumn('race_none', when(col('race_concept_name') == 'None', 1)).withColumn('race_mult', when(col('race_concept_name') == 'Multiple race', 1)).withColumn('race_unk', when(col('race_concept_name') == 'Unknown', 1)).withColumn('race_whi', when(col('race_concept_name') == 'White', 1)).withColumn('race_his', when(col('race_concept_name') == 'Hispanic', 1)).withColumn('race_asi', when(col('race_concept_name') == 'Asian', 1)).withColumn('race_bla', when(col('race_concept_name') == 'Black or African American', 1)).withColumn('race_bla', when(col('race_concept_name') == 'Black or African American', 1)).withColumn('race_nat', when(col('race_concept_name') == 'Native Hawaiian or Other Pacific Islander', 1)).withColumn('race_ind', when(col('race_concept_name') == 'American Indian or Alaska Native', 1)).drop('race_concept_name','race_concept_id')

    df_ = df_.withColumn('ethnicity_unk', when(col('ethnicity_concept_name') == 'Unknown', 1)).withColumn('ethnicity_his', when(col('ethnicity_concept_name') == 'Hispanic or Latino', 1)).withColumn('ethnicity_notHis', when(col('ethnicity_concept_name') == 'Not Hispanic or Latino', 1)).fillna(0).drop('ethnicity_concept_name', 'pasc', 'pasc_code_prior_four_weeks').fillna(0).drop('ethnicity_concept_name','ethnicity_concept_id',  'location_id', 'year_of_birth','year_of_birth_', 'is_age_90_or_older', 'time_to_pasc','age_' , 'covid_index')

    return df_


@transform_pandas(
    Output(rid="ri.foundry.main.dataset.16de7401-d0bc-4e71-ad07-c58bcb94e8db"),
    Observations_test=Input(rid="ri.foundry.main.dataset.6584d859-4f77-4a5f-9f14-0cc8759f1905")
)
def obs_features(Observations_test):
    #Model information includes names and what the observations are labeled as
    require_vaccine = ['44784283','37109774','45766222']
    prior_procedure =['4215685','4203722']
    overweight = ['4060705','4060985']
    never_used_tobacco =['4046550','2101934']
    never_smoked =['45765920','45879404']
    malnutrition =['4174707','434750','42872393']
    malignant_disease =['44784264','4326679','4179242']
    long_term =['46272451','46273937','46272734','46272453','46272452','46272450','36717001']
    history_obs = ['4214956','4143274']
    former_smoker =['45765917','45883458']
    family_history =['4179963','4171594','4167217','4144264','4051568','4051114','4023153']
    fall =['441749','436583']
    drug_indicated =['4202797','4047982']
    current_smoker =['45884037','45881517','40766945','4276526','3004518']
    contraceptive =['45773394','45765650']
    allergy =['4241527','439224']
    abnormal = ['439141', '437643','435928']
    colsss = ['person_id', 'observation_id', 'visit_occurrence_id', 'observation_concept_id', 'observation_period_id', 'observation_period_duration']
    df = Observations_test.drop('visit_occurrence_id')
    df = df.withColumn('tobacco_product', when(col('observation_concept_id') == '40766721', col('observation_period_duration')))
    df = df.withColumn('symptoms_aggravating', when(col('observation_concept_id')== '4037487', col('observation_period_duration')))
    df = df.withColumn('respiration_rate', when(col('observation_concept_id')== '4256640', col('observation_period_duration')))
    df = df.withColumn('severely_obese', when(col('observation_concept_id')== '3005879', col('observation_period_duration')))
    df = df.withColumn('require_vaccine', when(col('observation_concept_id').isin(require_vaccine), col('observation_period_duration')))
    df = df.withColumn('prior_procedure', when(col('observation_concept_id').isin(prior_procedure), col('observation_period_duration')))
    df = df.withColumn('post_op_care', when(col('observation_concept_id')== '4046550', col('observation_period_duration')))
    df = df.withColumn('overweight', when(col('observation_concept_id').isin(overweight), col('observation_period_duration')))
    df = df.withColumn('overexertion', when(col('observation_concept_id')== '4166272', col('observation_period_duration')))
    df = df.withColumn('never_used_tobacco', when(col('observation_concept_id').isin(never_used_tobacco), col('observation_period_duration')))
    df = df.withColumn('never_smoked', when(col('observation_concept_id').isin(never_smoked), col('observation_period_duration')))
    df = df.withColumn('malnutrition', when(col('observation_concept_id').isin(malnutrition), col('observation_period_duration')))
    df = df.withColumn('malignant_disease', when(col('observation_concept_id').isin(malignant_disease), col('observation_period_duration')))
    df = df.withColumn('long_term', when(col('observation_concept_id').isin(long_term), col('observation_period_duration')))
    df = df.withColumn('history_obs', when(col('observation_concept_id').isin(history_obs), col('observation_period_duration')))
    df = df.withColumn('high_risk_pregnancy', when(col('observation_concept_id')== '45765728', col('observation_period_duration')))
    df = df.withColumn('health_status', when(col('observation_concept_id')== '4247398', col('observation_period_duration')))
    df = df.withColumn('former_smoker', when(col('observation_concept_id').isin(former_smoker), col('observation_period_duration')))
    df = df.withColumn('fetal_disorder', when(col('observation_concept_id')== '43530881', col('observation_period_duration')))
    df = df.withColumn('family_history', when(col('observation_concept_id').isin(family_history), col('observation_period_duration')))
    df = df.withColumn('fall', when(col('observation_concept_id').isin(fall), col('observation_period_duration')))
    df = df.withColumn('drug_indicated', when(col('observation_concept_id').isin(drug_indicated), col('observation_period_duration')))
    df = df.withColumn('dialysis', when(col('observation_concept_id')== '4019967', col('observation_period_duration')))
    df = df.withColumn('current_smoker', when(col('observation_concept_id').isin(current_smoker), col('observation_period_duration')))
    df = df.withColumn('contraceptive', when(col('observation_concept_id').isin(contraceptive), col('observation_period_duration')))
    df = df.withColumn('congregate_care_setting', when(col('observation_concept_id')== '36661383', col('observation_period_duration')))
    df = df.withColumn('antenatal_care', when(col('observation_concept_id')== '4047564', col('observation_period_duration')))
    df = df.withColumn('allergy', when(col('observation_concept_id').isin(allergy), col('observation_period_duration')))
    df = df.withColumn('alcohol', when(col('observation_concept_id')== '45766930', col('observation_period_duration')))
    df = df.withColumn('abnormal', when(col('observation_concept_id').isin(abnormal), col('observation_period_duration')))
    df = df.withColumn('accident', when(col('observation_concept_id')== '440279', col('observation_period_duration')))

    ids = ['accident', 'abnormal', 'alcohol', 'allergy', 'antenatal_care', 'congregate_care_setting', 'contraceptive', 'current_smoker', 'dialysis', 'drug_indicated', 'fall', 'family_history', 'fetal_disorder', 'former_smoker', 'health_status', 'high_risk_pregnancy', 'history_obs', 'long_term', 'malignant_disease', 'malnutrition', 'never_smoked', 'never_used_tobacco', 'overexertion', 'overweight', 'post_op_care', 'prior_procedure', 'require_vaccine', 'respiration_rate', 'severely_obese', 'symptoms_aggravating', 'tobacco_product']
    df_ = df.groupby('person_id').agg(first('tobacco_product', ignorenulls=True).alias('tobacco_product_'),first('symptoms_aggravating', ignorenulls=True).alias('symptoms_aggravating_'),first('severely_obese', ignorenulls=True).alias('severely_obese_'),first('respiration_rate' , ignorenulls=True).alias('respiration_rate_'),first('require_vaccine' , ignorenulls=True).alias('require_vaccine_'),first('prior_procedure' , ignorenulls=True).alias('prior_procedure_'),first('post_op_care' , ignorenulls=True).alias('post_op_care_'),first('overweight', ignorenulls=True).alias('overweight_'),first('overexertion', ignorenulls=True).alias('overexertion_'),first('never_used_tobacco' , ignorenulls=True).alias('never_used_tobacco_'),first('never_smoked' , ignorenulls=True).alias('never_smoked_'),first('malignant_disease' , ignorenulls=True).alias('malignant_disease_'),first('malnutrition', ignorenulls=True).alias('malnutrition_'),first('long_term', ignorenulls=True).alias('long_term_'),first('history_obs', ignorenulls=True).alias('history_obs_'),first('high_risk_pregnancy', ignorenulls=True).alias('high_risk_pregnancy_'),first('health_status', ignorenulls=True).alias('health_status_'),first('former_smoker', ignorenulls=True).alias('former_smoker_'),first('fetal_disorder', ignorenulls=True).alias('fetal_disorder_'),first('family_history', ignorenulls=True).alias('family_history_'),first('fall', ignorenulls=True).alias('fall_'),first('drug_indicated', ignorenulls=True).alias('drug_indicated_'),first('dialysis', ignorenulls=True).alias('dialysis_'),first('current_smoker', ignorenulls=True).alias('current_smoker_'),first('contraceptive', ignorenulls=True).alias('contraceptive_'),first('congregate_care_setting', ignorenulls=True).alias('congregate_care_setting_'),first('antenatal_care', ignorenulls=True).alias('antenatal_care_'),first('allergy', ignorenulls=True).alias('allergy_'),first('alcohol', ignorenulls=True).alias('alcohol_'),first('abnormal', ignorenulls=True).alias('abnormal_'),first('accident', ignorenulls=True).alias('accident_')).fillna(0)
    #obs = [accident_, abnormal_, alcohol_, allergy_, antenatal_care_, congregate_care_setting, contraceptive, current_smoker_, dialysis_, drug_indicated_, fall_, family_history_, fetal_disorder_, former_smoker_, health_status_, high_risk_pregnancy_, history_obs_, long_term, malignant_disease, malnutrition_, never_smoked_, never_used_tobacco_, overexertion_, overweight_, post_op_care_, prior_procedure_, require_vaccine_, respiration_rate_, severely_obese, symptoms_aggravating, tobacco_product_]

    df = df.join(df_, on=['person_id'], how ='fullouter').drop('accident', 'abnormal', 'alcohol', 'allergy', 'antenatal_care', 'congregate_care_setting', 'contraceptive', 'current_smoker', 'dialysis', 'drug_indicated', 'fall', 'family_history', 'fetal_disorder', 'former_smoker', 'health_status', 'high_risk_pregnancy', 'history_obs', 'long_term', 'malignant_disease', 'malnutrition', 'never_smoked', 'never_used_tobacco', 'overexertion', 'overweight', 'post_op_care', 'prior_procedure', 'require_vaccine', 'respiration_rate', 'severely_obese', 'symptoms_aggravating', 'tobacco_product').fillna(0)

    personIndex = StringIndexer(inputCol = 'person_id', outputCol= 'personIndex')
    #observationIndex= StringIndexer(inputCol = 'observation_concept_id', outputCol= 'observationIndex').setHandleInvalid("skip")
    #'dataPartnerIndex' is removed from assembler, result
    encoded_df = Pipeline(stages=[personIndex]).fit(df).transform(df)

    assembler = VectorAssembler().setInputCols(['personIndex', 'accident_', 'abnormal_', 'alcohol_', 'allergy_', 'antenatal_care_', 'congregate_care_setting_', 'contraceptive_', 'current_smoker_', 'dialysis_', 'drug_indicated_', 'fall_', 'family_history_', 'fetal_disorder_', 'former_smoker_', 'health_status_', 'high_risk_pregnancy_', 'history_obs_', 'long_term_', 'malignant_disease_', 'malnutrition_', 'never_smoked_', 'never_used_tobacco_', 'overexertion_', 'overweight_', 'post_op_care_', 'prior_procedure_', 'require_vaccine_', 'respiration_rate_', 'severely_obese_', 'symptoms_aggravating_', 'tobacco_product_']).setOutputCol('features')

    result = assembler.transform(encoded_df)
    #result = result.select('person_id','personIndex','personIndex', 'accident_', 'abnormal_', 'alcohol_', 'allergy_', 'antenatal_care_', 'congregate_care_setting_', 'contraceptive_', 'current_smoker_', 'dialysis_', 'drug_indicated_', 'fall_', 'family_history_', 'fetal_disorder_', 'former_smoker_', 'health_status_', 'high_risk_pregnancy_', 'history_obs_', 'long_term_', 'malignant_disease_', 'malnutrition_', 'never_smoked_', 'never_used_tobacco_', 'overexertion_', 'overweight_', 'post_op_care_', 'prior_procedure_', 'require_vaccine_', 'respiration_rate_', 'severely_obese_', 'symptoms_aggravating_', 'tobacco_product_','features')

    result = result.dropDuplicates(['features']).drop('observation_concept_id', 'observation_period_id', 'observation_id', 'observation_date', 'data_partner_id', 'visit_concept_id')
    result = result.na.fill(0)
    return result

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0daaade6-46a5-47ad-a941-00e5201adf7a"),
    Observations_test=Input(rid="ri.foundry.main.dataset.6584d859-4f77-4a5f-9f14-0cc8759f1905")
)
def obs_features_1_0(Observations_test):
    #Model information includes names and what the observations are labeled as
    require_vaccine = ['44784283','37109774','45766222']
    prior_procedure =['4215685','4203722']
    overweight = ['4060705','4060985']
    never_used_tobacco =['4046550','2101934']
    never_smoked =['45765920','45879404']
    malnutrition =['4174707','434750','42872393']
    malignant_disease =['44784264','4326679','4179242']
    long_term =['46272451','46273937','46272734','46272453','46272452','46272450','36717001']
    history_obs = ['4214956','4143274']
    former_smoker =['45765917','45883458']
    family_history =['4179963','4171594','4167217','4144264','4051568','4051114','4023153']
    fall =['441749','436583']
    drug_indicated =['4202797','4047982']
    current_smoker =['45884037','45881517','40766945','4276526','3004518']
    contraceptive =['45773394','45765650']
    allergy =['4241527','439224']
    abnormal = ['439141', '437643','435928']
    colsss = ['person_id', 'observation_id', 'visit_occurrence_id', 'observation_concept_id', 'observation_period_id', 'observation_period_duration']
    df = Observations_test.drop('visit_occurrence_id')
    df = df.withColumn('tobacco_product', when(col('observation_concept_id') == '40766721', 1))
    df = df.withColumn('symptoms_aggravating', when(col('observation_concept_id')== '4037487', 1))
    df = df.withColumn('respiration_rate', when(col('observation_concept_id')== '4256640', 1))
    df = df.withColumn('severely_obese', when(col('observation_concept_id')== '3005879', 1))
    df = df.withColumn('require_vaccine', when(col('observation_concept_id').isin(require_vaccine), 1))
    df = df.withColumn('prior_procedure', when(col('observation_concept_id').isin(prior_procedure), 1))
    df = df.withColumn('post_op_care', when(col('observation_concept_id')== '4046550', 1))
    df = df.withColumn('overweight', when(col('observation_concept_id').isin(overweight), 1))
    df = df.withColumn('overexertion', when(col('observation_concept_id')== '4166272', 1))
    df = df.withColumn('never_used_tobacco', when(col('observation_concept_id').isin(never_used_tobacco), 1))
    df = df.withColumn('never_smoked', when(col('observation_concept_id').isin(never_smoked), 1))
    df = df.withColumn('malnutrition', when(col('observation_concept_id').isin(malnutrition), 1))
    df = df.withColumn('malignant_disease', when(col('observation_concept_id').isin(malignant_disease), 1))
    df = df.withColumn('long_term', when(col('observation_concept_id').isin(long_term), 1))
    df = df.withColumn('history_obs', when(col('observation_concept_id').isin(history_obs), 1))
    df = df.withColumn('high_risk_pregnancy', when(col('observation_concept_id')== '45765728', 1))
    df = df.withColumn('health_status', when(col('observation_concept_id')== '4247398', 1))
    df = df.withColumn('former_smoker', when(col('observation_concept_id').isin(former_smoker), 1))
    df = df.withColumn('fetal_disorder', when(col('observation_concept_id')== '43530881', 1))
    df = df.withColumn('family_history', when(col('observation_concept_id').isin(family_history), 1))
    df = df.withColumn('fall', when(col('observation_concept_id').isin(fall), 1))
    df = df.withColumn('drug_indicated', when(col('observation_concept_id').isin(drug_indicated), 1))
    df = df.withColumn('dialysis', when(col('observation_concept_id')== '4019967', 1))
    df = df.withColumn('current_smoker', when(col('observation_concept_id').isin(current_smoker), 1))
    df = df.withColumn('contraceptive', when(col('observation_concept_id').isin(contraceptive), 1))
    df = df.withColumn('congregate_care_setting', when(col('observation_concept_id')== '36661383', 1))
    df = df.withColumn('antenatal_care', when(col('observation_concept_id')== '4047564', 1))
    df = df.withColumn('allergy', when(col('observation_concept_id').isin(allergy), 1))
    df = df.withColumn('alcohol', when(col('observation_concept_id')== '45766930', 1))
    df = df.withColumn('abnormal', when(col('observation_concept_id').isin(abnormal), 1))
    df = df.withColumn('accident', when(col('observation_concept_id')== '440279', 1))

    ids = ['accident', 'abnormal', 'alcohol', 'allergy', 'antenatal_care', 'congregate_care_setting', 'contraceptive', 'current_smoker', 'dialysis', 'drug_indicated', 'fall', 'family_history', 'fetal_disorder', 'former_smoker', 'health_status', 'high_risk_pregnancy', 'history_obs', 'long_term', 'malignant_disease', 'malnutrition', 'never_smoked', 'never_used_tobacco', 'overexertion', 'overweight', 'post_op_care', 'prior_procedure', 'require_vaccine', 'respiration_rate', 'severely_obese', 'symptoms_aggravating', 'tobacco_product']
    df_ = df.groupby('person_id').agg(first('tobacco_product', ignorenulls=True).alias('tobacco_product_'),first('symptoms_aggravating', ignorenulls=True).alias('symptoms_aggravating_'),first('severely_obese', ignorenulls=True).alias('severely_obese_'),first('respiration_rate' , ignorenulls=True).alias('respiration_rate_'),first('require_vaccine' , ignorenulls=True).alias('require_vaccine_'),first('prior_procedure' , ignorenulls=True).alias('prior_procedure_'),first('post_op_care' , ignorenulls=True).alias('post_op_care_'),first('overweight', ignorenulls=True).alias('overweight_'),first('overexertion', ignorenulls=True).alias('overexertion_'),first('never_used_tobacco' , ignorenulls=True).alias('never_used_tobacco_'),first('never_smoked' , ignorenulls=True).alias('never_smoked_'),first('malignant_disease' , ignorenulls=True).alias('malignant_disease_'),first('malnutrition', ignorenulls=True).alias('malnutrition_'),first('long_term', ignorenulls=True).alias('long_term_'),first('history_obs', ignorenulls=True).alias('history_obs_'),first('high_risk_pregnancy', ignorenulls=True).alias('high_risk_pregnancy_'),first('health_status', ignorenulls=True).alias('health_status_'),first('former_smoker', ignorenulls=True).alias('former_smoker_'),first('fetal_disorder', ignorenulls=True).alias('fetal_disorder_'),first('family_history', ignorenulls=True).alias('family_history_'),first('fall', ignorenulls=True).alias('fall_'),first('drug_indicated', ignorenulls=True).alias('drug_indicated_'),first('dialysis', ignorenulls=True).alias('dialysis_'),first('current_smoker', ignorenulls=True).alias('current_smoker_'),first('contraceptive', ignorenulls=True).alias('contraceptive_'),first('congregate_care_setting', ignorenulls=True).alias('congregate_care_setting_'),first('antenatal_care', ignorenulls=True).alias('antenatal_care_'),first('allergy', ignorenulls=True).alias('allergy_'),first('alcohol', ignorenulls=True).alias('alcohol_'),first('abnormal', ignorenulls=True).alias('abnormal_'),first('accident', ignorenulls=True).alias('accident_')).fillna(0)
    #obs = [accident_, abnormal_, alcohol_, allergy_, antenatal_care_, congregate_care_setting, contraceptive, current_smoker_, dialysis_, drug_indicated_, fall_, family_history_, fetal_disorder_, former_smoker_, health_status_, high_risk_pregnancy_, history_obs_, long_term, malignant_disease, malnutrition_, never_smoked_, never_used_tobacco_, overexertion_, overweight_, post_op_care_, prior_procedure_, require_vaccine_, respiration_rate_, severely_obese, symptoms_aggravating, tobacco_product_]

    df = df.join(df_, on=['person_id'], how ='fullouter').drop('accident', 'abnormal', 'alcohol', 'allergy', 'antenatal_care', 'congregate_care_setting', 'contraceptive', 'current_smoker', 'dialysis', 'drug_indicated', 'fall', 'family_history', 'fetal_disorder', 'former_smoker', 'health_status', 'high_risk_pregnancy', 'history_obs', 'long_term', 'malignant_disease', 'malnutrition', 'never_smoked', 'never_used_tobacco', 'overexertion', 'overweight', 'post_op_care', 'prior_procedure', 'require_vaccine', 'respiration_rate', 'severely_obese', 'symptoms_aggravating', 'tobacco_product').fillna(0)

    personIndex = StringIndexer(inputCol = 'person_id', outputCol= 'personIndex')
    #observationIndex= StringIndexer(inputCol = 'observation_concept_id', outputCol= 'observationIndex').setHandleInvalid("skip")
    #'dataPartnerIndex' is removed from assembler, result
    encoded_df = Pipeline(stages=[personIndex]).fit(df).transform(df)

    assembler = VectorAssembler().setInputCols(['personIndex', 'accident_', 'abnormal_', 'alcohol_', 'allergy_', 'antenatal_care_', 'congregate_care_setting_', 'contraceptive_', 'current_smoker_', 'dialysis_', 'drug_indicated_', 'fall_', 'family_history_', 'fetal_disorder_', 'former_smoker_', 'health_status_', 'high_risk_pregnancy_', 'history_obs_', 'long_term_', 'malignant_disease_', 'malnutrition_', 'never_smoked_', 'never_used_tobacco_', 'overexertion_', 'overweight_', 'post_op_care_', 'prior_procedure_', 'require_vaccine_', 'respiration_rate_', 'severely_obese_', 'symptoms_aggravating_', 'tobacco_product_']).setOutputCol('features')

    result = assembler.transform(encoded_df)
    #result = result.select('person_id','personIndex','personIndex', 'accident_', 'abnormal_', 'alcohol_', 'allergy_', 'antenatal_care_', 'congregate_care_setting_', 'contraceptive_', 'current_smoker_', 'dialysis_', 'drug_indicated_', 'fall_', 'family_history_', 'fetal_disorder_', 'former_smoker_', 'health_status_', 'high_risk_pregnancy_', 'history_obs_', 'long_term_', 'malignant_disease_', 'malnutrition_', 'never_smoked_', 'never_used_tobacco_', 'overexertion_', 'overweight_', 'post_op_care_', 'prior_procedure_', 'require_vaccine_', 'respiration_rate_', 'severely_obese_', 'symptoms_aggravating_', 'tobacco_product_','features')

    result = result.dropDuplicates(['features']).drop('observation_concept_id', 'observation_period_id', 'observation_id', 'observation_date', 'data_partner_id', 'visit_concept_id')
    result = result.na.fill(0)
    return result

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.1795a3cf-d1e3-4886-ba73-f13e71d175ab"),
    Observations_=Input(rid="ri.foundry.main.dataset.a5294430-8bfc-4c3e-a061-3c139b2f6e98"),
    person_pasc=Input(rid="ri.vector.main.execute.3267314b-ece5-40c6-9def-fc9995a8a416")
)
def observation_Features(Observations_, person_pasc):
    #Model information includes names and what the observations are labeled as
    require_vaccine = ['44784283','37109774','45766222']
    prior_procedure =['4215685','4203722']
    overweight = ['4060705','4060985']
    never_used_tobacco =['4046550','2101934']
    never_smoked =['45765920','45879404']
    malnutrition =['4174707','434750','42872393']
    malignant_disease =['44784264','4326679','4179242']
    long_term =['46272451','46273937','46272734','46272453','46272452','46272450','36717001']
    history_obs = ['4214956','4143274']
    former_smoker =['45765917','45883458']
    family_history =['4179963','4171594','4167217','4144264','4051568','4051114','4023153']
    fall =['441749','436583']
    drug_indicated =['4202797','4047982']
    current_smoker =['45884037','45881517','40766945','4276526','3004518']
    contraceptive =['45773394','45765650']
    allergy =['4241527','439224']
    abnormal = ['439141', '437643','435928']
    colsss = ['person_id', 'observation_id', 'visit_occurrence_id', 'observation_concept_id', 'observation_period_id', 'observation_period_duration']
    df = Observations_.drop('visit_occurrence_id')
    df_pasc = person_pasc
    df = df.withColumn('tobacco_product', when(col('observation_concept_id') == '40766721', col('observation_period_duration')))
    df = df.withColumn('symptoms_aggravating', when(col('observation_concept_id')== '4037487', col('observation_period_duration')))
    df = df.withColumn('respiration_rate', when(col('observation_concept_id')== '4256640', col('observation_period_duration')))
    df = df.withColumn('severely_obese', when(col('observation_concept_id')== '3005879', col('observation_period_duration')))
    df = df.withColumn('require_vaccine', when(col('observation_concept_id').isin(require_vaccine), col('observation_period_duration')))
    df = df.withColumn('prior_procedure', when(col('observation_concept_id').isin(prior_procedure), col('observation_period_duration')))
    df = df.withColumn('post_op_care', when(col('observation_concept_id')== '4046550', col('observation_period_duration')))
    df = df.withColumn('overweight', when(col('observation_concept_id').isin(overweight), col('observation_period_duration')))
    df = df.withColumn('overexertion', when(col('observation_concept_id')== '4166272', col('observation_period_duration')))
    df = df.withColumn('never_used_tobacco', when(col('observation_concept_id').isin(never_used_tobacco), col('observation_period_duration')))
    df = df.withColumn('never_smoked', when(col('observation_concept_id').isin(never_smoked), col('observation_period_duration')))
    df = df.withColumn('malnutrition', when(col('observation_concept_id').isin(malnutrition), col('observation_period_duration')))
    df = df.withColumn('malignant_disease', when(col('observation_concept_id').isin(malignant_disease), col('observation_period_duration')))
    df = df.withColumn('long_term', when(col('observation_concept_id').isin(long_term), col('observation_period_duration')))
    df = df.withColumn('history_obs', when(col('observation_concept_id').isin(history_obs), col('observation_period_duration')))
    df = df.withColumn('high_risk_pregnancy', when(col('observation_concept_id')== '45765728', col('observation_period_duration')))
    df = df.withColumn('health_status', when(col('observation_concept_id')== '4247398', col('observation_period_duration')))
    df = df.withColumn('former_smoker', when(col('observation_concept_id').isin(former_smoker), col('observation_period_duration')))
    df = df.withColumn('fetal_disorder', when(col('observation_concept_id')== '43530881', col('observation_period_duration')))
    df = df.withColumn('family_history', when(col('observation_concept_id').isin(family_history), col('observation_period_duration')))
    df = df.withColumn('fall', when(col('observation_concept_id').isin(fall), col('observation_period_duration')))
    df = df.withColumn('drug_indicated', when(col('observation_concept_id').isin(drug_indicated), col('observation_period_duration')))
    df = df.withColumn('dialysis', when(col('observation_concept_id')== '4019967', col('observation_period_duration')))
    df = df.withColumn('current_smoker', when(col('observation_concept_id').isin(current_smoker), col('observation_period_duration')))
    df = df.withColumn('contraceptive', when(col('observation_concept_id').isin(contraceptive), col('observation_period_duration')))
    df = df.withColumn('congregate_care_setting', when(col('observation_concept_id')== '36661383', col('observation_period_duration')))
    df = df.withColumn('antenatal_care', when(col('observation_concept_id')== '4047564', col('observation_period_duration')))
    df = df.withColumn('allergy', when(col('observation_concept_id').isin(allergy), col('observation_period_duration')))
    df = df.withColumn('alcohol', when(col('observation_concept_id')== '45766930', col('observation_period_duration')))
    df = df.withColumn('abnormal', when(col('observation_concept_id').isin(abnormal), col('observation_period_duration')))
    df = df.withColumn('accident', when(col('observation_concept_id')== '440279', col('observation_period_duration')))

    ids = ['accident', 'abnormal', 'alcohol', 'allergy', 'antenatal_care', 'congregate_care_setting', 'contraceptive', 'current_smoker', 'dialysis', 'drug_indicated', 'fall', 'family_history', 'fetal_disorder', 'former_smoker', 'health_status', 'high_risk_pregnancy', 'history_obs', 'long_term', 'malignant_disease', 'malnutrition', 'never_smoked', 'never_used_tobacco', 'overexertion', 'overweight', 'post_op_care', 'prior_procedure', 'require_vaccine', 'respiration_rate', 'severely_obese', 'symptoms_aggravating', 'tobacco_product']
    df_ = df.groupby('person_id').agg(first('tobacco_product', ignorenulls=True).alias('tobacco_product_'),first('symptoms_aggravating', ignorenulls=True).alias('symptoms_aggravating_'),first('severely_obese', ignorenulls=True).alias('severely_obese_'),first('respiration_rate' , ignorenulls=True).alias('respiration_rate_'),first('require_vaccine' , ignorenulls=True).alias('require_vaccine_'),first('prior_procedure' , ignorenulls=True).alias('prior_procedure_'),first('post_op_care' , ignorenulls=True).alias('post_op_care_'),first('overweight', ignorenulls=True).alias('overweight_'),first('overexertion', ignorenulls=True).alias('overexertion_'),first('never_used_tobacco' , ignorenulls=True).alias('never_used_tobacco_'),first('never_smoked' , ignorenulls=True).alias('never_smoked_'),first('malignant_disease' , ignorenulls=True).alias('malignant_disease_'),first('malnutrition', ignorenulls=True).alias('malnutrition_'),first('long_term', ignorenulls=True).alias('long_term_'),first('history_obs', ignorenulls=True).alias('history_obs_'),first('high_risk_pregnancy', ignorenulls=True).alias('high_risk_pregnancy_'),first('health_status', ignorenulls=True).alias('health_status_'),first('former_smoker', ignorenulls=True).alias('former_smoker_'),first('fetal_disorder', ignorenulls=True).alias('fetal_disorder_'),first('family_history', ignorenulls=True).alias('family_history_'),first('fall', ignorenulls=True).alias('fall_'),first('drug_indicated', ignorenulls=True).alias('drug_indicated_'),first('dialysis', ignorenulls=True).alias('dialysis_'),first('current_smoker', ignorenulls=True).alias('current_smoker_'),first('contraceptive', ignorenulls=True).alias('contraceptive_'),first('congregate_care_setting', ignorenulls=True).alias('congregate_care_setting_'),first('antenatal_care', ignorenulls=True).alias('antenatal_care_'),first('allergy', ignorenulls=True).alias('allergy_'),first('alcohol', ignorenulls=True).alias('alcohol_'),first('abnormal', ignorenulls=True).alias('abnormal_'),first('accident', ignorenulls=True).alias('accident_')).fillna(0)
    #obs = [accident_, abnormal_, alcohol_, allergy_, antenatal_care_, congregate_care_setting, contraceptive, current_smoker_, dialysis_, drug_indicated_, fall_, family_history_, fetal_disorder_, former_smoker_, health_status_, high_risk_pregnancy_, history_obs_, long_term, malignant_disease, malnutrition_, never_smoked_, never_used_tobacco_, overexertion_, overweight_, post_op_care_, prior_procedure_, require_vaccine_, respiration_rate_, severely_obese, symptoms_aggravating, tobacco_product_]

    df = df.join(df_, on=['person_id'], how ='fullouter').drop('accident', 'abnormal', 'alcohol', 'allergy', 'antenatal_care', 'congregate_care_setting', 'contraceptive', 'current_smoker', 'dialysis', 'drug_indicated', 'fall', 'family_history', 'fetal_disorder', 'former_smoker', 'health_status', 'high_risk_pregnancy', 'history_obs', 'long_term', 'malignant_disease', 'malnutrition', 'never_smoked', 'never_used_tobacco', 'overexertion', 'overweight', 'post_op_care', 'prior_procedure', 'require_vaccine', 'respiration_rate', 'severely_obese', 'symptoms_aggravating', 'tobacco_product').join(df_pasc ,on= ['person_id']).fillna(0)

    personIndex = StringIndexer(inputCol = 'person_id', outputCol= 'personIndex')
    #observationIndex= StringIndexer(inputCol = 'observation_concept_id', outputCol= 'observationIndex').setHandleInvalid("skip")
    #'dataPartnerIndex' is removed from assembler, result
    encoded_df = Pipeline(stages=[personIndex]).fit(df).transform(df)

    assembler = VectorAssembler().setInputCols(['personIndex', 'accident_', 'abnormal_', 'alcohol_', 'allergy_', 'antenatal_care_', 'congregate_care_setting_', 'contraceptive_', 'current_smoker_', 'dialysis_', 'drug_indicated_', 'fall_', 'family_history_', 'fetal_disorder_', 'former_smoker_', 'health_status_', 'high_risk_pregnancy_', 'history_obs_', 'long_term_', 'malignant_disease_', 'malnutrition_', 'never_smoked_', 'never_used_tobacco_', 'overexertion_', 'overweight_', 'post_op_care_', 'prior_procedure_', 'require_vaccine_', 'respiration_rate_', 'severely_obese_', 'symptoms_aggravating_', 'tobacco_product_']).setOutputCol('features')

    result = assembler.transform(encoded_df)
    #result = result.select('person_id','personIndex','personIndex', 'accident_', 'abnormal_', 'alcohol_', 'allergy_', 'antenatal_care_', 'congregate_care_setting_', 'contraceptive_', 'current_smoker_', 'dialysis_', 'drug_indicated_', 'fall_', 'family_history_', 'fetal_disorder_', 'former_smoker_', 'health_status_', 'high_risk_pregnancy_', 'history_obs_', 'long_term_', 'malignant_disease_', 'malnutrition_', 'never_smoked_', 'never_used_tobacco_', 'overexertion_', 'overweight_', 'post_op_care_', 'prior_procedure_', 'require_vaccine_', 'respiration_rate_', 'severely_obese_', 'symptoms_aggravating_', 'tobacco_product_','features')

    result = result.dropDuplicates(['features']).drop('observation_concept_id', 'observation_period_id', 'observation_id', 'observation_date', 'data_partner_id', 'visit_concept_id')
    result = result.na.fill(0)
    return result

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0633476b-c599-44a7-8575-d2f44a9dedcf"),
    Observations_=Input(rid="ri.foundry.main.dataset.a5294430-8bfc-4c3e-a061-3c139b2f6e98"),
    person_pasc=Input(rid="ri.vector.main.execute.3267314b-ece5-40c6-9def-fc9995a8a416")
)
def observation_Features_1_0(Observations_, person_pasc):
    #Model information includes names and what the observations are labeled as
    require_vaccine = ['44784283','37109774','45766222']
    prior_procedure =['4215685','4203722']
    overweight = ['4060705','4060985']
    never_used_tobacco =['4046550','2101934']
    never_smoked =['45765920','45879404']
    malnutrition =['4174707','434750','42872393']
    malignant_disease =['44784264','4326679','4179242']
    long_term =['46272451','46273937','46272734','46272453','46272452','46272450','36717001']
    history_obs = ['4214956','4143274']
    former_smoker =['45765917','45883458']
    family_history =['4179963','4171594','4167217','4144264','4051568','4051114','4023153']
    fall =['441749','436583']
    drug_indicated =['4202797','4047982']
    current_smoker =['45884037','45881517','40766945','4276526','3004518']
    contraceptive =['45773394','45765650']
    allergy =['4241527','439224']
    abnormal = ['439141', '437643','435928']
    colsss = ['person_id', 'observation_id', 'visit_occurrence_id', 'observation_concept_id', 'observation_period_id', 'observation_period_duration']
    df = Observations_.drop('visit_occurrence_id')
    df_pasc = person_pasc
    df = df.withColumn('tobacco_product', when(col('observation_concept_id') == '40766721', 1))
    df = df.withColumn('symptoms_aggravating', when(col('observation_concept_id')== '4037487', 1))
    df = df.withColumn('respiration_rate', when(col('observation_concept_id')== '4256640', 1))
    df = df.withColumn('severely_obese', when(col('observation_concept_id')== '3005879', 1))
    df = df.withColumn('require_vaccine', when(col('observation_concept_id').isin(require_vaccine), 1))
    df = df.withColumn('prior_procedure', when(col('observation_concept_id').isin(prior_procedure), 1))
    df = df.withColumn('post_op_care', when(col('observation_concept_id')== '4046550', 1))
    df = df.withColumn('overweight', when(col('observation_concept_id').isin(overweight), 1))
    df = df.withColumn('overexertion', when(col('observation_concept_id')== '4166272', 1))
    df = df.withColumn('never_used_tobacco', when(col('observation_concept_id').isin(never_used_tobacco), 1))
    df = df.withColumn('never_smoked', when(col('observation_concept_id').isin(never_smoked), 1))
    df = df.withColumn('malnutrition', when(col('observation_concept_id').isin(malnutrition), 1))
    df = df.withColumn('malignant_disease', when(col('observation_concept_id').isin(malignant_disease), 1))
    df = df.withColumn('long_term', when(col('observation_concept_id').isin(long_term), 1))
    df = df.withColumn('history_obs', when(col('observation_concept_id').isin(history_obs), 1))
    df = df.withColumn('high_risk_pregnancy', when(col('observation_concept_id')== '45765728', 1))
    df = df.withColumn('health_status', when(col('observation_concept_id')== '4247398', 1))
    df = df.withColumn('former_smoker', when(col('observation_concept_id').isin(former_smoker), 1))
    df = df.withColumn('fetal_disorder', when(col('observation_concept_id')== '43530881', 1))
    df = df.withColumn('family_history', when(col('observation_concept_id').isin(family_history), 1))
    df = df.withColumn('fall', when(col('observation_concept_id').isin(fall), 1))
    df = df.withColumn('drug_indicated', when(col('observation_concept_id').isin(drug_indicated), 1))
    df = df.withColumn('dialysis', when(col('observation_concept_id')== '4019967', 1))
    df = df.withColumn('current_smoker', when(col('observation_concept_id').isin(current_smoker), 1))
    df = df.withColumn('contraceptive', when(col('observation_concept_id').isin(contraceptive), 1))
    df = df.withColumn('congregate_care_setting', when(col('observation_concept_id')== '36661383', 1))
    df = df.withColumn('antenatal_care', when(col('observation_concept_id')== '4047564', 1))
    df = df.withColumn('allergy', when(col('observation_concept_id').isin(allergy), 1))
    df = df.withColumn('alcohol', when(col('observation_concept_id')== '45766930', 1))
    df = df.withColumn('abnormal', when(col('observation_concept_id').isin(abnormal), 1))
    df = df.withColumn('accident', when(col('observation_concept_id')== '440279', 1))

    ids = ['accident', 'abnormal', 'alcohol', 'allergy', 'antenatal_care', 'congregate_care_setting', 'contraceptive', 'current_smoker', 'dialysis', 'drug_indicated', 'fall', 'family_history', 'fetal_disorder', 'former_smoker', 'health_status', 'high_risk_pregnancy', 'history_obs', 'long_term', 'malignant_disease', 'malnutrition', 'never_smoked', 'never_used_tobacco', 'overexertion', 'overweight', 'post_op_care', 'prior_procedure', 'require_vaccine', 'respiration_rate', 'severely_obese', 'symptoms_aggravating', 'tobacco_product']
    df_ = df.groupby('person_id').agg(first('tobacco_product', ignorenulls=True).alias('tobacco_product_'),first('symptoms_aggravating', ignorenulls=True).alias('symptoms_aggravating_'),first('severely_obese', ignorenulls=True).alias('severely_obese_'),first('respiration_rate' , ignorenulls=True).alias('respiration_rate_'),first('require_vaccine' , ignorenulls=True).alias('require_vaccine_'),first('prior_procedure' , ignorenulls=True).alias('prior_procedure_'),first('post_op_care' , ignorenulls=True).alias('post_op_care_'),first('overweight', ignorenulls=True).alias('overweight_'),first('overexertion', ignorenulls=True).alias('overexertion_'),first('never_used_tobacco' , ignorenulls=True).alias('never_used_tobacco_'),first('never_smoked' , ignorenulls=True).alias('never_smoked_'),first('malignant_disease' , ignorenulls=True).alias('malignant_disease_'),first('malnutrition', ignorenulls=True).alias('malnutrition_'),first('long_term', ignorenulls=True).alias('long_term_'),first('history_obs', ignorenulls=True).alias('history_obs_'),first('high_risk_pregnancy', ignorenulls=True).alias('high_risk_pregnancy_'),first('health_status', ignorenulls=True).alias('health_status_'),first('former_smoker', ignorenulls=True).alias('former_smoker_'),first('fetal_disorder', ignorenulls=True).alias('fetal_disorder_'),first('family_history', ignorenulls=True).alias('family_history_'),first('fall', ignorenulls=True).alias('fall_'),first('drug_indicated', ignorenulls=True).alias('drug_indicated_'),first('dialysis', ignorenulls=True).alias('dialysis_'),first('current_smoker', ignorenulls=True).alias('current_smoker_'),first('contraceptive', ignorenulls=True).alias('contraceptive_'),first('congregate_care_setting', ignorenulls=True).alias('congregate_care_setting_'),first('antenatal_care', ignorenulls=True).alias('antenatal_care_'),first('allergy', ignorenulls=True).alias('allergy_'),first('alcohol', ignorenulls=True).alias('alcohol_'),first('abnormal', ignorenulls=True).alias('abnormal_'),first('accident', ignorenulls=True).alias('accident_')).fillna(0)
    #obs = [accident_, abnormal_, alcohol_, allergy_, antenatal_care_, congregate_care_setting, contraceptive, current_smoker_, dialysis_, drug_indicated_, fall_, family_history_, fetal_disorder_, former_smoker_, health_status_, high_risk_pregnancy_, history_obs_, long_term, malignant_disease, malnutrition_, never_smoked_, never_used_tobacco_, overexertion_, overweight_, post_op_care_, prior_procedure_, require_vaccine_, respiration_rate_, severely_obese, symptoms_aggravating, tobacco_product_]

    df = df.join(df_, on=['person_id'], how ='fullouter').drop('accident', 'abnormal', 'alcohol', 'allergy', 'antenatal_care', 'congregate_care_setting', 'contraceptive', 'current_smoker', 'dialysis', 'drug_indicated', 'fall', 'family_history', 'fetal_disorder', 'former_smoker', 'health_status', 'high_risk_pregnancy', 'history_obs', 'long_term', 'malignant_disease', 'malnutrition', 'never_smoked', 'never_used_tobacco', 'overexertion', 'overweight', 'post_op_care', 'prior_procedure', 'require_vaccine', 'respiration_rate', 'severely_obese', 'symptoms_aggravating', 'tobacco_product').join(df_pasc ,on= ['person_id']).fillna(0)

    personIndex = StringIndexer(inputCol = 'person_id', outputCol= 'personIndex')
    #observationIndex= StringIndexer(inputCol = 'observation_concept_id', outputCol= 'observationIndex').setHandleInvalid("skip")
    #'dataPartnerIndex' is removed from assembler, result
    encoded_df = Pipeline(stages=[personIndex]).fit(df).transform(df)

    assembler = VectorAssembler().setInputCols(['personIndex', 'accident_', 'abnormal_', 'alcohol_', 'allergy_', 'antenatal_care_', 'congregate_care_setting_', 'contraceptive_', 'current_smoker_', 'dialysis_', 'drug_indicated_', 'fall_', 'family_history_', 'fetal_disorder_', 'former_smoker_', 'health_status_', 'high_risk_pregnancy_', 'history_obs_', 'long_term_', 'malignant_disease_', 'malnutrition_', 'never_smoked_', 'never_used_tobacco_', 'overexertion_', 'overweight_', 'post_op_care_', 'prior_procedure_', 'require_vaccine_', 'respiration_rate_', 'severely_obese_', 'symptoms_aggravating_', 'tobacco_product_']).setOutputCol('features')

    result = assembler.transform(encoded_df)
    #result = result.select('person_id','personIndex','personIndex', 'accident_', 'abnormal_', 'alcohol_', 'allergy_', 'antenatal_care_', 'congregate_care_setting_', 'contraceptive_', 'current_smoker_', 'dialysis_', 'drug_indicated_', 'fall_', 'family_history_', 'fetal_disorder_', 'former_smoker_', 'health_status_', 'high_risk_pregnancy_', 'history_obs_', 'long_term_', 'malignant_disease_', 'malnutrition_', 'never_smoked_', 'never_used_tobacco_', 'overexertion_', 'overweight_', 'post_op_care_', 'prior_procedure_', 'require_vaccine_', 'respiration_rate_', 'severely_obese_', 'symptoms_aggravating_', 'tobacco_product_','features')

    result = result.dropDuplicates(['features']).drop('observation_concept_id', 'observation_period_id', 'observation_id', 'observation_date', 'data_partner_id', 'visit_concept_id')
    result = result.na.fill(0)
    return result

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.a361f40c-a709-4e5f-98cc-fd72eadb66be"),
    Demographics_Features=Input(rid="ri.foundry.main.dataset.14be43ee-dbda-4b0d-b622-9c49b2c3bbca"),
    Diagnosis_Features=Input(rid="ri.vector.main.execute.49f1d284-20c7-4aca-a3b3-6a1d25610f25")
)
def patientsMissingDiag(Diagnosis_Features, Demographics_Features):
    df = Diagnosis_Features
    df_= Demographics_Features
    dia_patients = [row[0] for row in df.select('person_id').distinct().collect()]
    patients = [row[0] for row in df_.select('person_id').distinct().collect()]

    s = set(dia_patients)
    temp = [x for x in patients if x not in s]
    print('patients missing diagnosis:' , len(temp))
    df_= df_[df_.person_id.isin(temp)]
    s = set(patients)

    return df_

@transform_pandas(
    Output(rid="ri.vector.main.execute.3267314b-ece5-40c6-9def-fc9995a8a416"),
    merged_cleaned=Input(rid="ri.foundry.main.dataset.c9223d90-a29f-442b-ad11-10bd70ecd355")
)
def person_pasc(merged_cleaned):
    df = merged_cleaned
    df = df.select('person_id', 'pasc_code_after_four_weeks')
    return df
