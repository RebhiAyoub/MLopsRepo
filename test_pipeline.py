from model_pipeline import load_data, prepare


def test_prepare_adds_required_columns():
    df = load_data()
    df_prep = prepare(df)

    assert "hour" in df_prep.columns
    assert "day_of_week" in df_prep.columns
    assert len(df_prep) > 0
