
@transform_pandas(
    Output(rid="ri.foundry.main.dataset.14be43ee-dbda-4b0d-b622-9c49b2c3bbca"),
    merged_cleaned=Input(rid="ri.foundry.main.dataset.c9223d90-a29f-442b-ad11-10bd70ecd355")
)
def Demographics_Features(merged_cleaned):
    df = merged_cleaned.fillna(0)
    personIndex = StringIndexer(inputCol = 'person_id', outputCol= 'personIndex')

    #'dataPartnerIndex' is removed from assembler, result
    encoded_df = Pipeline(stages=[personIndex]).fit(df).transform(df)
    cols = ['personIndex', "age","ageGroup_infant","ageGroup_toddler","ageGroup_adolescent","ageGroup_youngAd","ageGroup_adult","ageGroup_olderAd","ageGroup_elderly","gender_fem","gender_mal","gender_unk","race_none","race_mult","race_unk","race_whi","race_his","race_asi","race_bla","race_nat","race_ind","ethnicity_unk","ethnicity_his","ethnicity_notHis"]

    assembler = VectorAssembler().setInputCols(cols).setOutputCol('features')
    result = assembler.transform(encoded_df)
    #result = result.dropDuplicates(['features'])

    return result
