import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="이커머스 반품 예측 및 운영 인사이트 대시보드",
    layout="wide"
)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# -----------------------------
# 데이터 로드
# -----------------------------
@st.cache_data
def load_base_data():
    X_tr = pd.read_csv("X_tr_unscaled.csv")
    X_val = pd.read_csv("X_val_unscaled.csv")
    X_test = pd.read_csv("X_test_unscaled.csv")
    y_tr = pd.read_csv("y_tr.csv")
    y_val = pd.read_csv("y_val.csv")
    test_order_id = pd.read_csv("test_order_id.csv")

    metrics_df = pd.DataFrame([
        {"model": "LightGBM", "accuracy": 0.5673, "f1": 0.4915, "roc_auc": 0.5927, "note": "ROC-AUC 최고"},
        {"model": "CatBoost", "accuracy": 0.5680, "f1": 0.4909, "roc_auc": 0.5926, "note": "Accuracy 최고"},
        {"model": "XGBoost", "accuracy": 0.5658, "f1": 0.4999, "roc_auc": 0.5879, "note": "F1 최고"},
        {"model": "Logistic Regression", "accuracy": 0.5656, "f1": 0.4853, "roc_auc": 0.5876, "note": "선형 기준 모델"},
    ])

    return X_tr, X_val, X_test, y_tr, y_val, test_order_id, metrics_df


@st.cache_data
def load_optional_prediction_file():
    """
    있으면 자동 로드:
    - val_predictions.csv
    - validation_predictions.csv
    - pred_val.csv
    """
    candidate_files = [
        "val_predictions.csv",
        "validation_predictions.csv",
        "pred_val.csv"
    ]

    for file_name in candidate_files:
        if os.path.exists(file_name):
            return pd.read_csv(file_name), file_name

    return None, None


@st.cache_data
def load_optional_feature_importance():
    """
    있으면 자동 로드:
    - feature_importance.csv
    """
    file_name = "feature_importance.csv"
    if os.path.exists(file_name):
        return pd.read_csv(file_name), file_name
    return None, None


# -----------------------------
# 전처리 함수
# -----------------------------
def decode_onehot(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    if not cols:
        return pd.Series(["unknown"] * len(df), index=df.index)
    return df[cols].idxmax(axis=1).str.replace(prefix + "_", "", regex=False)


def add_derived_columns(df):
    df = df.copy()

    df["discount_bin"] = pd.cut(
        df["discount_percent"],
        bins=[-0.1, 10, 30, 50, 100],
        labels=["0-10", "10-30", "30-50", "50+"]
    )

    df["delay_bin"] = pd.cut(
        df["delivery_delay_days"],
        bins=[-0.1, 1, 3, 100],
        labels=["0-1", "2-3", "4+"]
    )

    df["past_return_bin"] = pd.cut(
        df["past_return_rate"],
        bins=[-0.01, 0.2, 0.5, 1.0],
        labels=["Low", "Medium", "High"]
    )

    if "pred_prob" in df.columns:
        df["risk_level"] = pd.cut(
            df["pred_prob"],
            bins=[0.0, 0.3, 0.7, 1.0],
            labels=["Low", "Medium", "High"],
            include_lowest=True
        )

    return df


def merge_prediction_columns(val_df, pred_df):
    """
    예측 파일에 pred_prob / pred_label 있으면 validation 데이터에 붙임
    길이가 같으면 index 기준 결합
    order_id가 있으면 order_id 기준 결합 시도
    """
    result = val_df.copy()

    if pred_df is None:
        return result

    pred_cols = pred_df.columns.tolist()

    # order_id 기준 병합 가능할 때
    if "order_id" in result.columns and "order_id" in pred_df.columns:
        merge_cols = ["order_id"]
        for c in ["pred_prob", "pred_label"]:
            if c in pred_cols:
                merge_cols.append(c)
        result = result.merge(pred_df[merge_cols], on="order_id", how="left")
        return result

    # 길이가 같으면 index 기준으로 붙이기
    if len(result) == len(pred_df):
        if "pred_prob" in pred_cols:
            result["pred_prob"] = pred_df["pred_prob"].values
        if "pred_label" in pred_cols:
            result["pred_label"] = pred_df["pred_label"].values
        return result

    return result


# -----------------------------
# 데이터 불러오기
# -----------------------------
X_tr, X_val, X_test, y_tr, y_val, test_order_id, metrics_df = load_base_data()
pred_df, pred_file_name = load_optional_prediction_file()
importance_df, importance_file_name = load_optional_feature_importance()

# validation 분석용 데이터프레임
val_df = X_val.copy()
val_df["returned"] = y_val["returned"].values

# 원핫 복원
val_df["product_category_label"] = decode_onehot(val_df, "product_category")
val_df["device_type_label"] = decode_onehot(val_df, "device_type")
val_df["shipping_method_label"] = decode_onehot(val_df, "shipping_method")
val_df["payment_method_label"] = decode_onehot(val_df, "payment_method")

# 예측 결과 있으면 병합
val_df = merge_prediction_columns(val_df, pred_df)

# 파생 컬럼 추가
val_df = add_derived_columns(val_df)

best_auc = metrics_df.loc[metrics_df["roc_auc"].idxmax()]
best_f1 = metrics_df.loc[metrics_df["f1"].idxmax()]
best_acc = metrics_df.loc[metrics_df["accuracy"].idxmax()]

# -----------------------------
# 제목
# -----------------------------
st.title("이커머스 반품 예측 및 운영 인사이트 대시보드")
st.caption("전처리 데이터와 검증 결과를 바탕으로 반품 패턴과 운영 인사이트를 분석합니다.")

# -----------------------------
# 사이드바 필터
# -----------------------------
st.sidebar.header("필터")

category_options = ["전체"] + sorted(val_df["product_category_label"].dropna().unique().tolist())
shipping_options = ["전체"] + sorted(val_df["shipping_method_label"].dropna().unique().tolist())
device_options = ["전체"] + sorted(val_df["device_type_label"].dropna().unique().tolist())
payment_options = ["전체"] + sorted(val_df["payment_method_label"].dropna().unique().tolist())

selected_category = st.sidebar.selectbox("카테고리", category_options)
selected_shipping = st.sidebar.selectbox("배송방식", shipping_options)
selected_device = st.sidebar.selectbox("디바이스", device_options)
selected_payment = st.sidebar.selectbox("결제수단", payment_options)
selected_coupon = st.sidebar.selectbox("쿠폰 사용 여부", ["전체", "사용", "미사용"])

price_range = st.sidebar.slider(
    "상품 가격 범위",
    min_value=int(val_df["product_price"].min()),
    max_value=int(val_df["product_price"].max()),
    value=(int(val_df["product_price"].min()), int(val_df["product_price"].max()))
)

filtered_df = val_df.copy()

if selected_category != "전체":
    filtered_df = filtered_df[filtered_df["product_category_label"] == selected_category]

if selected_shipping != "전체":
    filtered_df = filtered_df[filtered_df["shipping_method_label"] == selected_shipping]

if selected_device != "전체":
    filtered_df = filtered_df[filtered_df["device_type_label"] == selected_device]

if selected_payment != "전체":
    filtered_df = filtered_df[filtered_df["payment_method_label"] == selected_payment]

if selected_coupon == "사용":
    filtered_df = filtered_df[filtered_df["used_coupon"] == 1]
elif selected_coupon == "미사용":
    filtered_df = filtered_df[filtered_df["used_coupon"] == 0]

filtered_df = filtered_df[
    (filtered_df["product_price"] >= price_range[0]) &
    (filtered_df["product_price"] <= price_range[1])
]

# -----------------------------
# 상단 KPI
# -----------------------------
st.subheader("핵심 지표")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Train 데이터 수", f"{len(X_tr):,}")
k2.metric("Validation 데이터 수", f"{len(filtered_df):,}")
k3.metric("Test 데이터 수", f"{len(X_test):,}")
k4.metric("Validation 실제 반품률", f"{filtered_df['returned'].mean():.1%}")

if "pred_prob" in filtered_df.columns:
    k5, k6 = st.columns(2)
    k5.metric("평균 예측 확률", f"{filtered_df['pred_prob'].mean():.1%}")
    k6.metric("고위험 주문 비율", f"{(filtered_df['pred_prob'] >= 0.7).mean():.1%}")

st.markdown("---")

# -----------------------------
# 모델 성능 비교
# -----------------------------
st.subheader("모델 성능 비교")

left, right = st.columns([1.2, 1])

with left:
    st.dataframe(
        metrics_df.sort_values("roc_auc", ascending=False),
        use_container_width=True,
        hide_index=True
    )

with right:
    metric_plot = metrics_df.set_index("model")[["accuracy", "f1", "roc_auc"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    metric_plot.plot(kind="bar", ax=ax)
    ax.set_xlabel("모델")
    ax.set_ylabel("점수")
    ax.set_title("모델별 성능 비교")
    plt.xticks(rotation=20)
    st.pyplot(fig)

c1, c2, c3 = st.columns(3)
c1.metric("최고 ROC-AUC", best_auc["model"], f"{best_auc['roc_auc']:.4f}")
c2.metric("최고 F1", best_f1["model"], f"{best_f1['f1']:.4f}")
c3.metric("최고 Accuracy", best_acc["model"], f"{best_acc['accuracy']:.4f}")

st.info(
    "운영 관점에서는 ROC-AUC가 가장 높은 LightGBM을 대표 모델로 설명하고, "
    "분류 균형 관점에서는 F1이 가장 높은 XGBoost를 함께 비교하는 것이 자연스럽습니다."
)

st.markdown("---")

# -----------------------------
# 타깃 분포
# -----------------------------
st.subheader("반품 타깃 분포")

dist1, dist2 = st.columns(2)

with dist1:
    dist_df = pd.DataFrame({
        "dataset": ["Train", "Validation"],
        "return_rate": [y_tr["returned"].mean(), filtered_df["returned"].mean()]
    })
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(dist_df["dataset"], dist_df["return_rate"])
    ax.set_ylabel("반품률")
    ax.set_title("Train / Validation 반품률")
    st.pyplot(fig)

with dist2:
    val_counts = filtered_df["returned"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["0", "1"], val_counts.values)
    ax.set_xlabel("returned")
    ax.set_ylabel("건수")
    ax.set_title("Validation 타깃 분포")
    st.pyplot(fig)

st.markdown("---")

# -----------------------------
# 실제 반품 패턴 분석
# -----------------------------
st.subheader("Validation 실제 데이터 기반 반품 패턴 분석")

v1, v2, v3, v4 = st.columns(4)
v1.metric("분석 대상 주문 수", f"{len(filtered_df):,}")
v2.metric("실제 반품률", f"{filtered_df['returned'].mean():.1%}")
v3.metric("평균 할인율", f"{filtered_df['discount_percent'].mean():.1f}%")
v4.metric("평균 배송지연일", f"{filtered_df['delivery_delay_days'].mean():.2f}")

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    category_rate = (
        filtered_df.groupby("product_category_label", as_index=False)["returned"]
        .mean()
        .sort_values("returned", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(category_rate["product_category_label"], category_rate["returned"])
    ax.set_title("카테고리별 실제 반품률")
    ax.set_xlabel("카테고리")
    ax.set_ylabel("반품률")
    plt.xticks(rotation=20)
    st.pyplot(fig)

with row1_col2:
    shipping_rate = (
        filtered_df.groupby("shipping_method_label", as_index=False)["returned"]
        .mean()
        .sort_values("returned", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(shipping_rate["shipping_method_label"], shipping_rate["returned"])
    ax.set_title("배송방식별 실제 반품률")
    ax.set_xlabel("배송방식")
    ax.set_ylabel("반품률")
    plt.xticks(rotation=20)
    st.pyplot(fig)

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    discount_rate = (
        filtered_df.groupby("discount_bin", as_index=False, observed=False)["returned"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(discount_rate["discount_bin"].astype(str), discount_rate["returned"])
    ax.set_title("할인율 구간별 실제 반품률")
    ax.set_xlabel("할인율 구간")
    ax.set_ylabel("반품률")
    st.pyplot(fig)

with row2_col2:
    delay_rate = (
        filtered_df.groupby("delivery_delay_days", as_index=False)["returned"]
        .mean()
        .sort_values("delivery_delay_days")
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(delay_rate["delivery_delay_days"], delay_rate["returned"], marker="o")
    ax.set_title("배송 지연일수별 실제 반품률")
    ax.set_xlabel("배송 지연일수")
    ax.set_ylabel("반품률")
    st.pyplot(fig)

row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    past_return_rate = (
        filtered_df.groupby("past_return_bin", as_index=False, observed=False)["returned"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(past_return_rate["past_return_bin"].astype(str), past_return_rate["returned"])
    ax.set_title("과거 반품률 구간별 실제 반품률")
    ax.set_xlabel("과거 반품률 구간")
    ax.set_ylabel("반품률")
    st.pyplot(fig)

with row3_col2:
    coupon_rate = (
        filtered_df.groupby("used_coupon", as_index=False)["returned"]
        .mean()
    )
    coupon_rate["coupon_label"] = coupon_rate["used_coupon"].map({0: "미사용", 1: "사용"})
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(coupon_rate["coupon_label"], coupon_rate["returned"])
    ax.set_title("쿠폰 사용 여부별 실제 반품률")
    ax.set_xlabel("쿠폰 사용 여부")
    ax.set_ylabel("반품률")
    st.pyplot(fig)

st.markdown("---")

# -----------------------------
# 예측 기반 리스크 분석
# -----------------------------
st.subheader("예측 기반 리스크 분석")

if "pred_prob" in filtered_df.columns:
    risk_left, risk_right = st.columns(2)

    with risk_left:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(filtered_df["pred_prob"], bins=20)
        ax.axvline(0.7, linestyle="--")
        ax.set_title("예측 확률 분포")
        ax.set_xlabel("pred_prob")
        ax.set_ylabel("건수")
        st.pyplot(fig)

    with risk_right:
        risk_counts = (
            filtered_df["risk_level"]
            .value_counts()
            .reindex(["Low", "Medium", "High"])
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(risk_counts.index.astype(str), risk_counts.values)
        ax.set_title("리스크 레벨 분포")
        ax.set_xlabel("Risk Level")
        ax.set_ylabel("주문 수")
        st.pyplot(fig)

    compare_cols = st.columns(3)

    with compare_cols[0]:
        risk_delay = filtered_df.groupby("risk_level", observed=False)["delivery_delay_days"].mean().reindex(["Low", "Medium", "High"])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(risk_delay.index.astype(str), risk_delay.values)
        ax.set_title("리스크별 평균 배송지연")
        ax.set_xlabel("Risk Level")
        ax.set_ylabel("평균 배송지연")
        st.pyplot(fig)

    with compare_cols[1]:
        risk_discount = filtered_df.groupby("risk_level", observed=False)["discount_percent"].mean().reindex(["Low", "Medium", "High"])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(risk_discount.index.astype(str), risk_discount.values)
        ax.set_title("리스크별 평균 할인율")
        ax.set_xlabel("Risk Level")
        ax.set_ylabel("평균 할인율")
        st.pyplot(fig)

    with compare_cols[2]:
        risk_past_return = filtered_df.groupby("risk_level", observed=False)["past_return_rate"].mean().reindex(["Low", "Medium", "High"])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(risk_past_return.index.astype(str), risk_past_return.values)
        ax.set_title("리스크별 평균 과거 반품률")
        ax.set_xlabel("Risk Level")
        ax.set_ylabel("평균 과거 반품률")
        st.pyplot(fig)

else:
    st.warning(
        "예측 결과 파일이 없어 pred_prob 기반 리스크 분석은 아직 표시할 수 없습니다. "
        "val_predictions.csv 파일이 추가되면 자동으로 연결됩니다."
    )

st.markdown("---")

# -----------------------------
# 주요 영향 변수
# -----------------------------
st.subheader("주요 영향 변수")

if importance_df is not None and {"feature", "importance"}.issubset(importance_df.columns):
    plot_df = importance_df.sort_values("importance", ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_df["feature"], plot_df["importance"])
    ax.set_title("주요 영향 변수 TOP 10")
    ax.set_xlabel("Importance")
    st.pyplot(fig)

    st.caption(f"사용한 파일: {importance_file_name}")
else:
    st.info(
        "feature_importance.csv 파일이 없어서 주요 영향 변수 시각화는 생략했습니다. "
        "feature, importance 컬럼이 있는 파일을 추가하면 자동으로 표시됩니다."
    )

st.markdown("---")

# -----------------------------
# 운영 인사이트
# -----------------------------
st.subheader("핵심 운영 인사이트")

insight_1 = filtered_df.loc[filtered_df["delivery_delay_days"] >= 2, "returned"].mean()
insight_2 = filtered_df.loc[filtered_df["discount_percent"] >= 40, "returned"].mean()
insight_3 = filtered_df.loc[filtered_df["past_return_rate"] >= 0.5, "returned"].mean()
insight_4 = filtered_df.loc[filtered_df["used_coupon"] == 1, "returned"].mean()

i1, i2 = st.columns(2)

with i1:
    st.markdown(f"""
- 배송 지연 **2일 이상** 주문의 실제 반품률: **{insight_1:.1%}**
- 할인율 **40% 이상** 주문의 실제 반품률: **{insight_2:.1%}**
""")

with i2:
    st.markdown(f"""
- 과거 반품률 **0.5 이상 고객**의 실제 반품률: **{insight_3:.1%}**
- 쿠폰 사용 주문의 실제 반품률: **{insight_4:.1%}**
""")

st.markdown("""
### 실행 전략 제안
- 배송 지연이 잦은 주문군은 물류 우선관리 대상으로 설정합니다.
- 과거 반품률이 높은 고객군은 구매 전 안내를 강화합니다.
- 고할인 상품은 상세페이지와 기대치 관리가 필요합니다.
- 반품률이 높은 카테고리는 상품 정보, 옵션, 리뷰 품질을 점검합니다.
""")

st.markdown("---")

# -----------------------------
# 데이터 미리보기
# -----------------------------
st.subheader("Validation 데이터 미리보기")

preview_candidates = [
    "customer_age",
    "product_price",
    "discount_percent",
    "product_rating",
    "past_purchase_count",
    "past_return_rate",
    "delivery_delay_days",
    "session_length_minutes",
    "num_product_views",
    "used_coupon",
    "product_category_label",
    "device_type_label",
    "shipping_method_label",
    "payment_method_label",
    "returned",
    "pred_prob",
    "pred_label",
    "risk_level",
]

preview_cols = [c for c in preview_candidates if c in filtered_df.columns]

st.dataframe(
    filtered_df[preview_cols].head(30),
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# 하단 안내
# -----------------------------
st.success("대시보드가 정상적으로 로드되었습니다.")

if pred_file_name:
    st.caption(f"예측 결과 파일 연결됨: {pred_file_name}")
else:
    st.caption("예측 결과 파일 없음: val_predictions.csv를 추가하면 pred_prob 기반 분석이 활성화됩니다.")