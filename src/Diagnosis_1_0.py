
@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b7b9f659-64e6-4e0a-b8c8-c3a97ece514e"),
    Condition_Features_1_0=Input(rid="ri.foundry.main.dataset.09f400a0-17c7-4e67-8ec4-97a1233caebd"),
    Drugs_prepare=Input(rid="ri.foundry.main.dataset.e7869727-7803-45a4-8285-3848d6c97e3e"),
    merged_cleaned=Input(rid="ri.foundry.main.dataset.c9223d90-a29f-442b-ad11-10bd70ecd355"),
    observation_Features_1_0=Input(rid="ri.foundry.main.dataset.0633476b-c599-44a7-8575-d2f44a9dedcf")
)
def Diagnosis_1_0(Condition_Features_1_0, merged_cleaned, Drugs_prepare,observation_Features_1_0):
    #Diagnosis includes Demographics, Conditions and Observations
    df = Condition_Features_1_0.drop('features')
    df_ = merged_cleaned
    df_drugs = Drugs_prepare.drop('features', 'personIndex')
    #df_obs = observation_Features_1_0.drop('features', 'personIndex')
    df = df.join(df_, on =['person_id', 'pasc_code_after_four_weeks'])
    df = df.join(df_drugs, on =['person_id'])#.join(df_obs, on =['person_id', 'pasc_code_after_four_weeks'])

    cols = ['personIndex', 'LossOfTaste_Cond_', 'Cough_Cond_', 'Allergic_rhinitis_Cond_', 'Covid_Cond_', 'Renal_Cond_', 'Obesity_Cond_', 'Fever_Cond_', 'Fatigue_Cond_', 'Other_Cond_', 'Bypass_graft_Cond_', 'Deformity_foot_Cond_', 'Respiratory_fail_Cond_', 'Brain_injury_Cond_', 'Oltagia_Cond_', 'Venticular_Cond_', 'Elevation_Cond_', 'Trial_fib_Cond_', 'Disorders_Cond_', 'Effusion_Cond_', 'Hernia_Cond_', 'Nutricional_def_Cond_', 'Pain_limb_Cond_', 'Pain_hand_Cond_', 'Cyst_Cond_', 'age', 'ageGroup_infant', 'ageGroup_toddler', 'ageGroup_adolescent', 'ageGroup_youngAd', 'ageGroup_adult', 'ageGroup_olderAd', 'ageGroup_elderly', 'gender_fem', 'gender_mal', 'gender_unk', 'race_none', 'race_mult', 'race_unk', 'race_whi', 'race_his', 'race_asi', 'race_bla', 'race_nat', 'race_ind', 'ethnicity_unk', 'ethnicity_his', 'ethnicity_notHis', 'Other_drug_', 'Enoxaparin_drug_', 'Bupivacaine_drug_', 'Sodium_chlo_drug_', 'Ondansetron_drug_', 'Sennapod_drug_', 'Atenolol_drug_', 'Doxy_drug_', 'Fluorescein_drug_', 'Metoprolol_drug_', 'Midazolam_drug_', 'Naproxen_drug_', 'Nicotine_drug_', 'Ofloxacin_drug_', 'Omeprazole_drug_', 'Polyethykene_drug_', 'Potassium_drug_', 'Vancomycin_drug_', 'Zolpidem_drug_']#,'accident_', 'abnormal_', 'alcohol_', 'allergy_', 'antenatal_care_', 'congregate_care_setting_', 'contraceptive_', 'current_smoker_', 'dialysis_', 'drug_indicated_', 'fall_', 'family_history_', 'fetal_disorder_', 'former_smoker_', 'health_status_', 'high_risk_pregnancy_', 'history_obs_', 'long_term_', 'malignant_disease_', 'malnutrition_', 'never_smoked_', 'never_used_tobacco_', 'overexertion_', 'overweight_', 'post_op_care_', 'prior_procedure_', 'require_vaccine_', 'respiration_rate_', 'severely_obese_', 'symptoms_aggravating_', 'tobacco_product_']
    print(df.columns)
    assembler = VectorAssembler().setInputCols(cols).setOutputCol('features')
    result = assembler.transform(df)
    result = result.dropDuplicates(['features'])

    return result
