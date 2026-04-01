"""
maegigal - 개인 프로젝트 메인 파일

이 파일을 시작점으로 코딩을 시작하세요!
"""


def main():
    print("Hello, World! 🚀")
    print("여기서부터 코딩을 시작하세요.")
    
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder, StandardScaler


    # =========================
    # 1. 데이터 불러오기
    # =========================

    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')


    # =========================
    # 2. 타깃 / 식별자 분리
    # =========================

    target_col = "returned"
    id_col = "order_id"

    X_train = train.drop(columns=[target_col]).copy()
    y_train = train[target_col].copy()
    X_test = test.copy()

    X_train.drop(columns=[id_col], inplace=True)
    X_test.drop(columns=[id_col], inplace=True)


    # =========================
    # 3. 수치형 / 범주형 변수 구분
    # =========================

    num_cols = X_train.select_dtypes(exclude=["object", "string"]).columns
    obj_cols = X_train.select_dtypes(include=["object", "string"]).columns


    # =========================
    # 4. 이상값 -> NaN 변환
    # =========================

    X_train.loc[X_train["product_price"] <= 0, "product_price"] = np.nan
    X_test.loc[X_test["product_price"] <= 0, "product_price"] = np.nan

    X_train.loc[X_train["discount_percent"] < 0, "discount_percent"] = np.nan
    X_test.loc[X_test["discount_percent"] < 0, "discount_percent"] = np.nan

    X_train.loc[
        (X_train["product_rating"] < 0) | (X_train["product_rating"] > 5),
        "product_rating"
    ] = np.nan
    X_test.loc[
        (X_test["product_rating"] < 0) | (X_test["product_rating"] > 5),
        "product_rating"
    ] = np.nan

    X_train.loc[
        (X_train["past_return_rate"] < 0) | (X_train["past_return_rate"] > 1),
        "past_return_rate"
    ] = np.nan
    X_test.loc[
        (X_test["past_return_rate"] < 0) | (X_test["past_return_rate"] > 1),
        "past_return_rate"
    ] = np.nan

    X_train.loc[X_train["session_length_minutes"] < 0, "session_length_minutes"] = np.nan
    X_test.loc[X_test["session_length_minutes"] < 0, "session_length_minutes"] = np.nan

    X_train.loc[X_train["num_product_views"] < 0, "num_product_views"] = np.nan
    X_test.loc[X_test["num_product_views"] < 0, "num_product_views"] = np.nan


    # =========================
    # 5. 이상값 대체
    # =========================

    # product_price -> product_category별 중앙값
    price_median_by_cat = X_train.groupby("product_category")["product_price"].median()

    X_train["product_price"] = X_train["product_price"].fillna(
        X_train["product_category"].map(price_median_by_cat)
    )
    X_test["product_price"] = X_test["product_price"].fillna(
        X_test["product_category"].map(price_median_by_cat)
    )

    X_train["product_price"] = X_train["product_price"].fillna(X_train["product_price"].median())
    X_test["product_price"] = X_test["product_price"].fillna(X_train["product_price"].median())

    # discount_percent -> 전체 중앙값
    X_train["discount_percent"] = X_train["discount_percent"].fillna(X_train["discount_percent"].median())
    X_test["discount_percent"] = X_test["discount_percent"].fillna(X_train["discount_percent"].median())

    # product_rating -> product_category별 중앙값
    rating_median_by_cat = X_train.groupby("product_category")["product_rating"].median()

    X_train["product_rating"] = X_train["product_rating"].fillna(
        X_train["product_category"].map(rating_median_by_cat)
    )
    X_test["product_rating"] = X_test["product_rating"].fillna(
        X_test["product_category"].map(rating_median_by_cat)
    )

    X_train["product_rating"] = X_train["product_rating"].fillna(X_train["product_rating"].median())
    X_test["product_rating"] = X_test["product_rating"].fillna(X_train["product_rating"].median())

    # 나머지 이상값 컬럼 -> 전체 중앙값
    for col in ["past_return_rate", "session_length_minutes", "num_product_views"]:
        X_train[col] = X_train[col].fillna(X_train[col].median())
        X_test[col] = X_test[col].fillna(X_train[col].median())


    # =========================
    # 6. 범주형 변수 원-핫 인코딩
    # =========================

    for col in obj_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    X_train_enc = pd.DataFrame(
        encoder.fit_transform(X_train[obj_cols]),
        columns=encoder.get_feature_names_out(obj_cols),
        index=X_train.index
    )

    X_test_enc = pd.DataFrame(
        encoder.transform(X_test[obj_cols]),
        columns=encoder.get_feature_names_out(obj_cols),
        index=X_test.index
    )

    X_train = pd.concat([X_train.drop(columns=obj_cols), X_train_enc], axis=1)
    X_test = pd.concat([X_test.drop(columns=obj_cols), X_test_enc], axis=1)


    # =========================
    # 7. 스케일링 X ->  DecisionTree, RandomForest, XGBoost, LightGBM
    # X_train_unscaled/X_test_unscaled
    # =========================

    X_train_unscaled = X_train.copy()
    X_test_unscaled = X_test.copy()


    # =========================
    # 8. 수치형 변수 스케일링 O -> LogisticRegression, SVM, KNN
    # X_train_scaled/X_test_scaled
    # =========================

    # used_coupon은 0/1 이진 변수이므로 스케일링 제외
    scale_cols = [col for col in num_cols if col != "used_coupon"]

    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])


    # =========================
    # 9. 최종 확인
    # =========================

    print("X_train_unscaled shape:", X_train_unscaled.shape)
    print("X_test_unscaled shape:", X_test_unscaled.shape)
    print("X_train_scaled shape:", X_train_scaled.shape)
    print("X_test_scaled shape:", X_test_scaled.shape)
    print("train/test 컬럼 일치 여부:", X_train_unscaled.columns.equals(X_test_unscaled.columns))
    print("결측치 합계 (unscaled train/test):", X_train_unscaled.isnull().sum().sum(), X_test_unscaled.isnull().sum().sum())
    print("결측치 합계 (scaled train/test):", X_train_scaled.isnull().sum().sum(), X_test_scaled.isnull().sum().sum())

if __name__ == "__main__":
    main()
