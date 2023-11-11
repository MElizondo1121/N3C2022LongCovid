import pandas as pd
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StringIndexer,StandardScaler
from pyspark.ml import Pipeline

from pyspark.sql.functions import col, when, sum, first,  countDistinct, max
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.mllib.stat import Statistics
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.sql import Window
import pyspark.sql.functions as f



@transform_pandas(
    Output(rid="ri.foundry.main.dataset.a15e6e2a-1451-433e-adde-d102587615e3"),
    Conditions_test=Input(rid="ri.foundry.main.dataset.38503fb2-3f49-4ea6-a2fd-be05ece66960")
)
def Cond_Features(Conditions_test):
    df = Conditions_test
    fever = ['Fever','Postprocedural fever']
    cough = ['Chronic cough','Cough', 'Chronic fatigue syndrome']
    fatigue = ['Fatigue','Postviral fatigue syndrome']
    covid = ['COVID-19','Post-acute COVID-19']
    renal = ['Acquired renal cystic disease','Acute adrenal insufficiency','Acute renal failure syndrome','Adrenal cortical hypofunction','Benign neoplasm of left adrenal gland','Benign neoplasm of right adrenal gland','Disorder of adrenal gland','End-stage renal disease','Hepatorenal syndrome','Hyperparathyroidism due to renal insufficiency','Hypertensive heart and renal disease with (congestive) heart failure','Hypertensive renal failure','Renal disorder due to type 2 diabetes mellitus','Renal hypertension','Renal osteodystrophy']
    obesity = ['Maternal obesity syndrome', 'Morbid obesity','Obesity','Severe obesity', 'Simple obesity', 'Drug-induced obesity']
    brain_injury = ['Perinatal anoxic-ischemic brain injury',
'Traumatic brain injury with brief loss of consciousness','Diffuse brain injury','Traumatic brain injury','Traumatic brain injury with prolonged loss of consciousness','Traumatic brain injury with moderate loss of consciousness','Traumatic brain injury with loss of consciousness', 'Focal brain injury','Traumatic brain injury with no loss of consciousness']
    deformity_foot = ['Acquired deformity of foot', 'Congenital valgus deformity of foot', 'Congenital deformity of foot', 'Acquired equinus deformity of foot', 'Acquired cavus deformity of foot', 'Congenital varus deformity of foot', 'Acquired cavovarus deformity of foot']
    respiratory_failure = ['Acute hypoxemic respiratory failure', 'Chronic hypoxemic respiratory failure','Acute on chronic hypoxemic respiratory failure', 'Acute hypercapnic respiratory failure','Acute on chronic hypercapnic respiratory failure','Hypoxemic respiratory failure','Chronic hypercapnic respiratory failure','Acute respiratory failure','Chronic respiratory failure','Acute-on-chronic respiratory failure','Postprocedural respiratory failure','Neonatal respiratory failure','Hypercapnic respiratory failure']
    oltagia =['Referred otalgia','Otogenic otalgia', 'Otogenic otalgia of left ear', 'Otalgia', 'Otalgia of right ear', 'Otalgia of left ear']
    venticular = ['Supraventricular tachycardia', 'Ventricular tachycardia',
'Paroxysmal ventricular tachycardia','Paroxysmal supraventricular tachycardia','Recurrent ventricular tachycardia','Nonsustained ventricular tachycardia','Re-entrant atrioventricular tachycardia']
    elevation = ['Acute non-ST segment elevation myocardial infarction','Acute ST segment elevation myocardial infarction','Acute ST segment elevation myocardial infarction involving left anterior descending coronary artery','Acute ST segment elevation myocardial infarction due to left coronary artery occlusion','Acute ST segment elevation myocardial infarction due to occlusion of anterior descending branch of left coronary artery','ST segment elevation']
    bypass = ['Aortocoronary bypass graft present']
    trial_fib = ['Atrial fibrillation','Paroxysmal atrial fibrillation','Chronic atrial fibrillation','Atrial fibrillation with rapid ventricular response','Atrial fibrillation and flutter']
    disorders = ['Disorder of immune function','Disorder of salivary gland','Disorder of male genital organ']
    effusion_of_joint = ['Effusion of joint', 'Effusion of joint of shoulder region', 'Effusion of joint of hand', 'Effusion of joint of right ankle', 'Effusion of joint of left ankle', 'Effusion of joint of left hip','Effusion of joint of left knee','Effusion of joint of multiple sites','Effusion of joint of pelvic region','Effusion of joint of right elbow'
'Effusion of joint of left elbow']
    hernia = ['Hernia of abdominal cavity','Hernia of abdominal wall']
    nutricional_def = ['Nutritional deficiency disorder','Dilated cardiomyopathy due to nutritional deficiency']
    pain_in_limb = ['Pain in limb','Pain in limb - multiple']
    pain_in_hand = ['Pain in right hand','Joint pain in right hand']
    cyst = ['Pilonidal cyst']
    loss = ['Loss of taste']
    seasonal =['Seasonal allergic rhinitis']
    #df = df.drop('condition_occurrence_count').dropDuplicates()

    aggregated = df.groupby('condition_concept_name', 'person_id').agg(max('condition_duration').alias('total_condition_duration'))
    agg = df.groupby('condition_concept_name', 'person_id').agg(sum('condition_occurrence_count').alias('total_occurrence_count'))

    df = df.join(aggregated, on =['person_id', 'condition_concept_name'])
    df = df.join(agg, on =['person_id', 'condition_concept_name'])

    df = df.withColumn('LossOfTaste_Cond', when(col('condition_concept_name').isin(loss), col('total_condition_duration')))
    df = df.withColumn('Allergic_rhinitis_Cond', when(col('condition_concept_name').isin(seasonal), col('total_condition_duration')))
    df = df.withColumn('Bypass_graft_Cond', when(col('condition_concept_name').isin(bypass), col('total_condition_duration')))
    df = df.withColumn('Cough_Cond', when(col('condition_concept_name').isin(cough), col('total_condition_duration')))
    df = df.withColumn('Fever_Cond', when(col('condition_concept_name').isin(fever),  col('total_condition_duration')))
    df = df.withColumn('Fatigue_Cond', when(col('condition_concept_name').isin(fatigue), col('total_condition_duration')))
    df = df.withColumn('Renal_Cond', when(col('condition_concept_name').isin(renal), col('total_condition_duration')))
    df = df.withColumn('Obesity_Cond', when(col('condition_concept_name').isin(obesity), col('total_condition_duration')))
    df = df.withColumn('Covid_Cond', when(col('condition_concept_name').isin(covid), col('total_condition_duration')))
    df = df.withColumn('Brain_injury_Cond', when(col('condition_concept_name').isin(brain_injury), col('total_condition_duration')))
    df = df.withColumn('Deformity_foot_Cond', when(col('condition_concept_name').isin(deformity_foot), col('total_condition_duration')))
    df = df.withColumn('Respiratory_fail_Cond', when(col('condition_concept_name').isin(respiratory_failure), col('total_condition_duration')))
    df = df.withColumn('Oltagia_Cond', when(col('condition_concept_name').isin(oltagia), col('total_condition_duration')))
    df = df.withColumn('Venticular_Cond', when(col('condition_concept_name').isin(venticular), col('total_condition_duration')))
    df = df.withColumn('Elevation_Cond', when(col('condition_concept_name').isin(elevation), col('total_condition_duration')))
    df = df.withColumn('Trial_fib_Cond', when(col('condition_concept_name').isin(trial_fib), col('total_condition_duration')))
    df = df.withColumn('Disorders_Cond', when(col('condition_concept_name').isin(disorders), col('total_condition_duration')))
    df = df.withColumn('Effusion_Cond', when(col('condition_concept_name').isin(effusion_of_joint), col('total_condition_duration')))
    df = df.withColumn('Hernia_Cond', when(col('condition_concept_name').isin(hernia), col('total_condition_duration')))
    df = df.withColumn('Nutricional_def_Cond', when(col('condition_concept_name').isin(nutricional_def), col('total_condition_duration')))
    df = df.withColumn('Pain_limb_Cond', when(col('condition_concept_name').isin(pain_in_limb), col('total_condition_duration')))
    df = df.withColumn('Pain_hand_Cond', when(col('condition_concept_name').isin(pain_in_hand), col('total_condition_duration')))
    df = df.withColumn('Cyst_Cond', when(col('condition_concept_name').isin(cyst), col('total_condition_duration')))

    df = df.withColumn('Other_Cond', when((col('condition_concept_name').isin(cough) ==False) & (col('condition_concept_name').isin(fever) ==False)& (col('condition_concept_name').isin(fatigue) ==False)&  (col('condition_concept_name').isin(renal) ==False)& (col('condition_concept_name').isin(obesity) ==False)&(col('condition_concept_name').isin(covid) ==False)& (col('condition_concept_name').isin(brain_injury) ==False)&(col('condition_concept_name').isin(deformity_foot) == False)&(col('condition_concept_name').isin(respiratory_failure) == False)&(col('condition_concept_name').isin(oltagia) == False)&(col('condition_concept_name').isin(venticular) == False)&(col('condition_concept_name').isin(elevation) == False)&(col('condition_concept_name').isin(trial_fib) == False)&(col('condition_concept_name').isin(disorders) == False)&(col('condition_concept_name').isin(effusion_of_joint) == False)&(col('condition_concept_name').isin(hernia) == False)&(col('condition_concept_name').isin(nutricional_def) == False)&(col('condition_concept_name').isin(pain_in_limb) == False)&(col('condition_concept_name').isin(pain_in_hand) == False)&(col('condition_concept_name').isin(cyst) == False)&(col('condition_concept_name').isin(loss) == False)&(col('condition_concept_name').isin(seasonal) == False)&(col('condition_concept_name').isin(bypass) == False),col('total_condition_duration')))

    df_ = df.groupby('person_id').agg(first('LossOfTaste_Cond', ignorenulls=True).alias('LossOfTaste_Cond_'),first('Cough_Cond', ignorenulls=True).alias('Cough_Cond_'), first('Allergic_rhinitis_Cond', ignorenulls=True).alias('Allergic_rhinitis_Cond_'),first('Covid_Cond' , ignorenulls=True).alias('Covid_Cond_'),first('Renal_Cond' , ignorenulls=True).alias('Renal_Cond_'),first('Obesity_Cond' , ignorenulls=True).alias('Obesity_Cond_'),first('Fever_Cond' , ignorenulls=True).alias('Fever_Cond_'),first('Fatigue_Cond', ignorenulls=True).alias('Fatigue_Cond_'), first('Other_Cond', ignorenulls=True).alias('Other_Cond_'),first('Bypass_graft_Cond' , ignorenulls=True).alias('Bypass_graft_Cond_'),first('Deformity_foot_Cond' , ignorenulls=True).alias('Deformity_foot_Cond_'), first('Respiratory_fail_Cond' , ignorenulls=True).alias('Respiratory_fail_Cond_'),first('Brain_injury_Cond', ignorenulls=True).alias('Brain_injury_Cond_'),first('Oltagia_Cond', ignorenulls=True).alias('Oltagia_Cond_'),first('Venticular_Cond', ignorenulls=True).alias('Venticular_Cond_'),first('Elevation_Cond', ignorenulls=True).alias('Elevation_Cond_'),first('Trial_fib_Cond', ignorenulls=True).alias('Trial_fib_Cond_'),first('Disorders_Cond', ignorenulls=True).alias('Disorders_Cond_'),first('Effusion_Cond', ignorenulls=True).alias('Effusion_Cond_'),first('Hernia_Cond', ignorenulls=True).alias('Hernia_Cond_'),first('Nutricional_def_Cond', ignorenulls=True).alias('Nutricional_def_Cond_'),first('Pain_limb_Cond', ignorenulls=True).alias('Pain_limb_Cond_'),first('Pain_hand_Cond', ignorenulls=True).alias('Pain_hand_Cond_'),first('Cyst_Cond', ignorenulls=True).alias('Cyst_Cond_'))
    df = df.join(df_, on =['person_id'], how='full_outer').fillna(0)


    personIndex = StringIndexer(inputCol = 'person_id', outputCol= 'personIndex')
    conditionIndex= StringIndexer(inputCol = 'condition_concept_id', outputCol= 'conditionIndex').setHandleInvalid("skip")
    #'dataPartnerIndex' is removed from assembler, result
    encoded_df = Pipeline(stages=[personIndex, conditionIndex]).fit(df).transform(df)

    assembler = VectorAssembler().setInputCols(['personIndex','LossOfTaste_Cond_','Cough_Cond_','Allergic_rhinitis_Cond_','Covid_Cond_','Renal_Cond_','Obesity_Cond_','Fever_Cond_','Fatigue_Cond_','Other_Cond_','Bypass_graft_Cond_','Deformity_foot_Cond_','Respiratory_fail_Cond_','Brain_injury_Cond_','Oltagia_Cond_','Venticular_Cond_','Elevation_Cond_','Trial_fib_Cond_','Disorders_Cond_','Effusion_Cond_','Hernia_Cond_','Nutricional_def_Cond_','Pain_limb_Cond_','Pain_hand_Cond_','Cyst_Cond_']).setOutputCol('features')

    result = assembler.transform(encoded_df)
    result = result.select('person_id','personIndex', 'LossOfTaste_Cond_','Cough_Cond_','Allergic_rhinitis_Cond_','Covid_Cond_','Renal_Cond_','Obesity_Cond_','Fever_Cond_','Fatigue_Cond_','Other_Cond_','Bypass_graft_Cond_','Deformity_foot_Cond_','Respiratory_fail_Cond_','Brain_injury_Cond_','Oltagia_Cond_','Venticular_Cond_','Elevation_Cond_','Trial_fib_Cond_','Disorders_Cond_','Effusion_Cond_','Hernia_Cond_','Nutricional_def_Cond_','Pain_limb_Cond_','Pain_hand_Cond_','Cyst_Cond_','features')

    result = result.dropDuplicates(['features'])
    result = result.na.fill(0)
    return result
