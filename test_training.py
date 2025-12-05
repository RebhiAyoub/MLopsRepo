from model_pipeline import (
    load_data,
    prepare,
    data_preparation,
    train_model,
    evaluate,
)


def test_data_preparation_shapes():
    df = load_data()
    df_prep = prepare(df)
    X_train, X_test, y_train, y_test = data_preparation(df_prep)

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]


def test_train_model_returns_fitted_model():
    df = load_data()
    df_prep = prepare(df)
    X_train, X_test, y_train, y_test = data_preparation(df_prep)

    model = train_model(X_train, y_train)
    preds = model.predict(X_test)

    assert len(preds) == len(y_test)


def test_evaluate_returns_metrics():
    df = load_data()
    df_prep = prepare(df)
    X_train, X_test, y_train, y_test = data_preparation(df_prep)

    model = train_model(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    assert "RMSE" in metrics
    assert "MAE" in metrics
    assert "R2" in metrics
    assert metrics["RMSE"] > 0
