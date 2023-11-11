

@transform_pandas(
    Output(rid="ri.vector.main.execute.81a5a61b-f6f3-4e00-8b47-94a1569a98a9"),
    Person_test=Input(rid="ri.foundry.main.dataset.f4657f54-6a0e-4847-b1d1-01dd524b592b")
)
def Dem_Prep(Person_test):
    Person = Person_test
    df = Person.drop('location_id', 'care_site_id', 'provider_id', 'data_partner_id', 'month_of_birth', 'gender_concept_id', 'race_concept_id', 'ethnicity_concept_id')
    valueWhenTrue = 1932
    df = df.withColumn(
        'year_of_birth',
        when(
            col('is_age_90_or_older') == 'true',
            valueWhenTrue
            ).otherwise(col('year_of_birth')))
    gender = df.select('gender_concept_name').distinct().collect()
    race = df.select('race_concept_name').distinct().collect()
    ethnicity = df.select('ethnicity_concept_name').distinct().collect()
    df_ = df.withColumn('gender_fem', when(col('gender_concept_name') == 'FEMALE', 1)).withColumn('gender_mal', when(col('gender_concept_name') == 'MALE', 1)).fillna(0)

    df_ = df_.withColumn('gender_unk', when((col('gender_fem') == 0) & (col('gender_mal') == 0), 1))

    df_ = df_.withColumn('race_none', when(col('race_concept_name') == 'None', 1)).withColumn('race_mult', when(col('race_concept_name') == 'Multiple race', 1)).withColumn('race_unk', when(col('race_concept_name') == 'Unknown', 1)).withColumn('race_whi', when(col('race_concept_name') == 'White', 1)).withColumn('race_his', when(col('race_concept_name') == 'Hispanic', 1)).withColumn('race_asi', when(col('race_concept_name') == 'Asian', 1)).withColumn('race_bla', when(col('race_concept_name') == 'Black or African American', 1)).withColumn('race_bla', when(col('race_concept_name') == 'Black or African American', 1)).withColumn('race_nat', when(col('race_concept_name') == 'Native Hawaiian or Other Pacific Islander', 1)).withColumn('race_ind', when(col('race_concept_name') == 'American Indian or Alaska Native', 1))

    df_ = df_.withColumn('ethnicity_unk', when(col('ethnicity_concept_name') == 'No matching concept', 1)).withColumn('ethnicity_his', when(col('ethnicity_concept_name') == 'Hispanic or Latino', 1)).withColumn('ethnicity_notHis', when(col('ethnicity_concept_name') == 'Not Hispanic or Latino', 1)).drop('gender_concept_name', 'race_concept_name', 'ethnicity_concept_name', 'year_of_birth', 'is_age_90_or_older').fillna(0)
    df = df.join(df_, on='person_id').drop('gender_concept_name', 'race_concept_name', 'ethnicity_concept_name').withColumn('age', (2021 - col('year_of_birth')))

    df = df.withColumn('ageGroup_infant', when(col('age') < 2, 1)).withColumn('ageGroup_toddler', when(((col('age') >= 2) & (col('age') < 4)), 1)).withColumn('ageGroup_adolescent', when(((col('age') >= 4) & (col('age') < 14)), 1)).withColumn('ageGroup_youngAd', when(((col('age') >= 14) & (col('age') < 30)), 1)).withColumn('ageGroup_adult', when(((col('age') >= 30) & (col('age') < 50)), 1)).withColumn('ageGroup_olderAd', when(((col('age') >= 50) & (col('age') < 90)), 1)).withColumn('ageGroup_elderly', when((col('is_age_90_or_older') == 'true'), 1)).fillna(0).drop('is_age_90_or_older', 'year_of_birth').fillna(0)

    return df
