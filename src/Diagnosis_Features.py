

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.445fc6a7-da36-4ac0-b613-d8c4b20db14c"),
    Cond_Features=Input(rid="ri.foundry.main.dataset.a15e6e2a-1451-433e-adde-d102587615e3"),
    Dem_Prep=Input(rid="ri.vector.main.execute.81a5a61b-f6f3-4e00-8b47-94a1569a98a9"),
    Drugs_prep=Input(rid="ri.vector.main.execute.9182bf31-1827-420b-a5b8-d19b1b9ef35c"),
    obs_features=Input(rid="ri.foundry.main.dataset.16de7401-d0bc-4e71-ad07-c58bcb94e8db")
)
def Diagnosis_Feature(Cond_Features, Drugs_prep, Dem_Prep, obs_features):
    #Diagnosis includes Demographics, Conditions and Observations
    df = Cond_Features.drop('features')
    df_ = Dem_Prep
    df_drugs = Drugs_prep.drop('features', 'personIndex')
    #df_obs = obs_features.drop('features', 'personIndex')
    df = df.join(df_, on =['person_id']).join(df_drugs, on =['person_id'])#.join(df_obs, on =['person_id'])

    cols = ['personIndex', 'LossOfTaste_Cond_', 'Cough_Cond_', 'Allergic_rhinitis_Cond_', 'Covid_Cond_', 'Renal_Cond_', 'Obesity_Cond_', 'Fever_Cond_', 'Fatigue_Cond_', 'Other_Cond_', 'Bypass_graft_Cond_', 'Deformity_foot_Cond_', 'Respiratory_fail_Cond_', 'Brain_injury_Cond_', 'Oltagia_Cond_', 'Venticular_Cond_', 'Elevation_Cond_', 'Trial_fib_Cond_', 'Disorders_Cond_', 'Effusion_Cond_', 'Hernia_Cond_', 'Nutricional_def_Cond_', 'Pain_limb_Cond_', 'Pain_hand_Cond_', 'Cyst_Cond_', 'age', 'ageGroup_infant', 'ageGroup_toddler', 'ageGroup_adolescent', 'ageGroup_youngAd', 'ageGroup_adult', 'ageGroup_olderAd', 'ageGroup_elderly', 'gender_fem', 'gender_mal', 'gender_unk','race_none', 'race_mult', 'race_unk', 'race_whi', 'race_his', 'race_asi', 'race_bla', 'race_nat', 'race_ind', 'ethnicity_unk', 'ethnicity_his', 'ethnicity_notHis', 'Other_drug_', 'Enoxaparin_drug_', 'Bupivacaine_drug_', 'Sodium_chlo_drug_', 'Ondansetron_drug_', 'Sennapod_drug_', 'Atenolol_drug_', 'Doxy_drug_', 'Fluorescein_drug_', 'Metoprolol_drug_', 'Midazolam_drug_', 'Naproxen_drug_', 'Nicotine_drug_', 'Ofloxacin_drug_', 'Omeprazole_drug_', 'Polyethykene_drug_', 'Potassium_drug_', 'Vancomycin_drug_', 'Zolpidem_drug_']#,'accident_', 'abnormal_', 'alcohol_', 'allergy_', 'antenatal_care_', 'congregate_care_setting_', 'contraceptive_', 'current_smoker_', 'dialysis_', 'drug_indicated_', 'fall_', 'family_history_', 'fetal_disorder_', 'former_smoker_', 'health_status_', 'high_risk_pregnancy_', 'history_obs_', 'long_term_', 'malignant_disease_', 'malnutrition_', 'never_smoked_', 'never_used_tobacco_', 'overexertion_', 'overweight_', 'post_op_care_', 'prior_procedure_', 'require_vaccine_', 'respiration_rate_', 'severely_obese_', 'symptoms_aggravating_', 'tobacco_product_']
    print(df.columns)
    assembler = VectorAssembler().setInputCols(cols).setOutputCol('features')
    result = assembler.transform(df)
    result = result.dropDuplicates(['features'])

    return result
