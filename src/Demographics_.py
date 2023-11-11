
@transform_pandas(
    Output(rid="ri.foundry.main.dataset.3e49bf18-4379-49db-84c1-a86770214a78"),
    Demographics=Input(rid="ri.foundry.main.dataset.9fdbe269-952e-4405-8bdc-22ba5e7a366f")
)
def Demographics_(Demographics):
    df = Demographics
    valueWhenTrue = 1932
    df_ = df.withColumn(
        'year_of_birth',
        when(
            col('is_age_90_or_older') == 'true',
            valueWhenTrue
            ).otherwise(col('year_of_birth')))

    return df_
