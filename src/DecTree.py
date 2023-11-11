
@transform_pandas(
    Output(rid="ri.vector.main.execute.baf86ffc-9983-48f5-83ef-6969f0644e73"),
    Demographics_Features=Input(rid="ri.foundry.main.dataset.14be43ee-dbda-4b0d-b622-9c49b2c3bbca")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def DecTree(Demographics_Features):
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
