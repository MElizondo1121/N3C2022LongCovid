

@transform_pandas(
    Output(rid="ri.vector.main.execute.9182bf31-1827-420b-a5b8-d19b1b9ef35c"),
    Drugs_test=Input(rid="ri.foundry.main.dataset.ea0dbace-450a-4011-8439-13bde05c3a06")
)
def Drugs_prep(Drugs_test):
    Drugs_1 = Drugs_test
    #Diagnosis includes Demographics, Conditions and Observations
    df = Drugs_1.drop('visit_occurrence_id')
    drug_ids = ['0.4 ML enoxaparin sodium 100 MG/ML Prefilled Syringe [Lovenox]','10 ML bupivacaine hydrochloride 2.5 MG/ML Injection','sodium chloride 9 MG/ML Injectable Solution','2 ML ondansetron 2 MG/ML Injection','Senna pod','atenolol 25 MG Oral Tablet','doxycycline hyclate 100 MG Oral Capsule [Pelodis]','enoxaparin Prefilled Syringe','fluorescein','metoprolol','midazolam 1 MG/ML Injectable Solution','naproxen 500 MG Oral Tablet','nicotine','ofloxacin','omeprazole','polyethylene glycol 3350 17000 MG Powder for Oral Solution','potassium chloride','vancomycin','zolpidem tartrate 5 MG Oral Tablet']

    df = df.withColumn('Enoxaparin_drug', when(col('drug_concept_name') == '0.4 ML enoxaparin sodium 100 MG/ML Prefilled Syringe [Lovenox]', col('drug_exposure_duration')))
    df = df.withColumn('Bupivacaine_drug', when(col('drug_concept_name')=='10 ML bupivacaine hydrochloride 2.5 MG/ML Injection' ,col('drug_exposure_duration')))
    df = df.withColumn('Sodium_chlo_drug', when(col('drug_concept_name')=='sodium chloride 9 MG/ML Injectable Solution' ,col('drug_exposure_duration')))
    df = df.withColumn('Ondansetron_drug', when(col('drug_concept_name') =='2 ML ondansetron 2 MG/ML Injection',col('drug_exposure_duration')))
    df = df.withColumn('Sennapod_drug', when(col('drug_concept_name') =='Senna pod',col('drug_exposure_duration')))
    df = df.withColumn('Atenolol_drug', when(col('drug_concept_name') =='atenolol 25 MG Oral Tablet',col('drug_exposure_duration')))
    df = df.withColumn('Doxy_drug', when(col('drug_concept_name') =='doxycycline hyclate 100 MG Oral Capsule [Pelodis]',col('drug_exposure_duration')))
    df = df.withColumn('Enoxaparin_drug', when(col('drug_concept_name') =='enoxaparin Prefilled Syringe',col('drug_exposure_duration')))
    df = df.withColumn('Fluorescein_drug', when(col('drug_concept_name') =='fluorescein',col('drug_exposure_duration')))
    df = df.withColumn('Metoprolol_drug', when(col('drug_concept_name') =='metoprolol',col('drug_exposure_duration')))
    df = df.withColumn('Midazolam_drug', when(col('drug_concept_name') =='midazolam 1 MG/ML Injectable Solution',col('drug_exposure_duration')))
    df = df.withColumn('Naproxen_drug', when(col('drug_concept_name') =='naproxen 500 MG Oral Tablet',col('drug_exposure_duration')))
    df = df.withColumn('Nicotine_drug', when(col('drug_concept_name') =='nicotine',col('drug_exposure_duration')))
    df = df.withColumn('Ofloxacin_drug', when(col('drug_concept_name') =='ofloxacin',col('drug_exposure_duration')))
    df = df.withColumn('Omeprazole_drug', when(col('drug_concept_name') =='omeprazole',col('drug_exposure_duration')))
    df = df.withColumn('Polyethykene_drug', when(col('drug_concept_name') =='polyethylene glycol 3350 17000 MG Powder for Oral Solution',col('drug_exposure_duration')))
    df = df.withColumn('Potassium_drug', when(col('drug_concept_name') =='potassium chloride',col('drug_exposure_duration')))
    df = df.withColumn('Vancomycin_drug', when(col('drug_concept_name') =='vancomycin',col('drug_exposure_duration')))
    df = df.withColumn('Zolpidem_drug', when(col('drug_concept_name') == 'zolpidem tartrate 5 MG Oral Tablet',col('drug_exposure_duration'))).withColumn('Other_drug', when((col('drug_concept_name').isin(drug_ids) ==False), col('drug_exposure_duration')))

    df_ = df.groupby('person_id').agg(first('Enoxaparin_drug', ignorenulls=True).alias('Enoxaparin_drug_'),first('Other_drug', ignorenulls=True).alias('Other_drug_'),first('Bupivacaine_drug', ignorenulls=True).alias('Bupivacaine_drug_'),first('Sodium_chlo_drug', ignorenulls=True).alias('Sodium_chlo_drug_'),first('Ondansetron_drug', ignorenulls=True).alias('Ondansetron_drug_'),first('Sennapod_drug', ignorenulls=True).alias('Sennapod_drug_'),first('Atenolol_drug', ignorenulls=True).alias('Atenolol_drug_'),first('Doxy_drug', ignorenulls=True).alias('Doxy_drug_'),first('Fluorescein_drug', ignorenulls=True).alias('Fluorescein_drug_'),first('Metoprolol_drug', ignorenulls=True).alias('Metoprolol_drug_'),first('Midazolam_drug', ignorenulls=True).alias('Midazolam_drug_'),first('Naproxen_drug', ignorenulls=True).alias('Naproxen_drug_'),first('Nicotine_drug', ignorenulls=True).alias('Nicotine_drug_'),first('Ofloxacin_drug', ignorenulls=True).alias('Ofloxacin_drug_'),first('Omeprazole_drug', ignorenulls=True).alias('Omeprazole_drug_'),first('Polyethykene_drug', ignorenulls=True).alias('Polyethykene_drug_'),first('Potassium_drug', ignorenulls=True).alias('Potassium_drug_'),first('Vancomycin_drug', ignorenulls=True).alias('Vancomycin_drug_'),first('Zolpidem_drug', ignorenulls=True).alias('Zolpidem_drug_'))
    df = df.join(df_, on =['person_id'], how='full_outer').fillna(0)

    cols = ['personIndex','Other_drug_', 'Enoxaparin_drug_', 'Bupivacaine_drug_', 'Sodium_chlo_drug_', 'Ondansetron_drug_', 'Sennapod_drug_', 'Atenolol_drug_', 'Doxy_drug_', 'Fluorescein_drug_', 'Metoprolol_drug_', 'Midazolam_drug_', 'Naproxen_drug_', 'Nicotine_drug_', 'Ofloxacin_drug_', 'Omeprazole_drug_', 'Polyethykene_drug_', 'Potassium_drug_', 'Vancomycin_drug_', 'Zolpidem_drug_']#, 'total_drug_exposure_duration', 'total_drug_era_duration', 'total_drug_exposure_count', 'total_gap_days']
    personIndex = StringIndexer(inputCol = 'person_id', outputCol= 'personIndex')
    drugIndex= StringIndexer(inputCol = 'drug_concept_id', outputCol= 'drugIndex').setHandleInvalid("skip")
    #'dataPartnerIndex' is removed from assembler, result
    encoded_df = Pipeline(stages=[personIndex, drugIndex]).fit(df).transform(df)

    assembler = VectorAssembler().setInputCols(cols).setOutputCol('features')

    result = assembler.transform(encoded_df)
    result = result.select(['person_id','personIndex','Other_drug_', 'Enoxaparin_drug_', 'Bupivacaine_drug_', 'Sodium_chlo_drug_', 'Ondansetron_drug_', 'Sennapod_drug_', 'Atenolol_drug_', 'Doxy_drug_', 'Fluorescein_drug_', 'Metoprolol_drug_', 'Midazolam_drug_', 'Naproxen_drug_', 'Nicotine_drug_', 'Ofloxacin_drug_', 'Omeprazole_drug_', 'Polyethykene_drug_', 'Potassium_drug_', 'Vancomycin_drug_', 'Zolpidem_drug_','features'])

    result = result.dropDuplicates(['features'])
    result = result.fillna(0)
    return result
