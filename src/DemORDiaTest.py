

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ab2edfae-9f48-42ba-adb9-7429ad45b0fc"),
    Demographic_test_Features=Input(rid="ri.foundry.main.dataset.2bfd07f4-4dd7-42b9-b6c9-59bd790d8abb"),
    Diagnosis_Feature_1_0=Input(rid="ri.foundry.main.dataset.1948d0b2-7832-42de-9d07-323558202543")
)
def DemORDiaTest(Demographic_test_Features, Diagnosis_Feature_1_0):
    df_dem = Demographic_test_Features.drop('features')
    df_dia = Diagnosis_Feature_1_0.drop('features')

    df_dia =df_dia.unionByName(df_dem, allowMissingColumns= True)
    df_dia = df_dia.drop('personIndex').fillna(0)
    cols = df_dia.columns
    assembler = VectorAssembler().setInputCols(['LossOfTaste_Cond_','Cough_Cond_','Allergic_rhinitis_Cond_','Covid_Cond_','Renal_Cond_','Obesity_Cond_','Fever_Cond_','Fatigue_Cond_','Other_Cond_','Bypass_graft_Cond_','Deformity_foot_Cond_','Respiratory_fail_Cond_','Brain_injury_Cond_','Oltagia_Cond_','Venticular_Cond_','Elevation_Cond_','Trial_fib_Cond_','Disorders_Cond_','Effusion_Cond_','Hernia_Cond_','Nutricional_def_Cond_','Pain_limb_Cond_','Pain_hand_Cond_','Cyst_Cond_','gender_fem','gender_mal','gender_unk','race_none','race_mult','race_unk','race_whi','race_his','race_asi','race_bla','race_nat','race_ind','ethnicity_unk','ethnicity_his','ethnicity_notHis','age','ageGroup_infant','ageGroup_toddler','ageGroup_adolescent','ageGroup_youngAd','ageGroup_adult','ageGroup_olderAd','ageGroup_elderly','Other_drug_','Enoxaparin_drug_','Bupivacaine_drug_','Sodium_chlo_drug_','Ondansetron_drug_','Sennapod_drug_','Atenolol_drug_','Doxy_drug_','Fluorescein_drug_','Metoprolol_drug_','Midazolam_drug_','Naproxen_drug_','Nicotine_drug_','Ofloxacin_drug_','Omeprazole_drug_','Polyethykene_drug_','Potassium_drug_','Vancomycin_drug_','Zolpidem_drug_']).setOutputCol('features')
    result = assembler.transform(df_dia)
    result = result.dropDuplicates(['features'])

    return result
