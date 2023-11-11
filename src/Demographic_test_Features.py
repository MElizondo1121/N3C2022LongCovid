
@transform_pandas(
    Output(rid="ri.foundry.main.dataset.2bfd07f4-4dd7-42b9-b6c9-59bd790d8abb"),
    Dem_Prep=Input(rid="ri.vector.main.execute.81a5a61b-f6f3-4e00-8b47-94a1569a98a9")
)
def Demographic_test_Features(Dem_Prep):
    df = Dem_Prep
    personIndex = StringIndexer(inputCol = 'person_id', outputCol= 'personIndex')

    #'dataPartnerIndex' is removed from assembler, result
    encoded_df = Pipeline(stages=[personIndex]).fit(df).transform(df)
    cols = ['personIndex', "age","ageGroup_infant","ageGroup_toddler","ageGroup_adolescent","ageGroup_youngAd","ageGroup_adult","ageGroup_olderAd","ageGroup_elderly","gender_fem","gender_mal",'gender_unk', "race_none","race_mult","race_unk","race_whi","race_his","race_asi","race_bla","race_nat","race_ind","ethnicity_unk","ethnicity_his","ethnicity_notHis"]

    assembler = VectorAssembler().setInputCols(cols).setOutputCol('features')
    result = assembler.transform(encoded_df)
    #result = result.dropDuplicates(['features'])
    result = result.drop('person_id')

    return result
