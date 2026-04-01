import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="반품 예측 모델 결과 대시보드",
    layout="wide"
)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# -----------------------------
# 모델 성능 결과 (직접 입력)
# -----------------------------
metrics_df = pd.DataFrame([
    {"model": "Decision Tree", "accuracy": 0.5650, "f1": 0.4945, "roc_auc": 0.5882, "note": "기준 트리 모델"},
    {"model": "XGBoost", "accuracy": 0.5658, "f1": 0.4999, "roc_auc": 0.5879, "note": "F1 최고"},
    {"model": "LightGBM", "accuracy": 0.5673, "f1": 0.4915, "roc_auc": 0.5927, "note": "ROC-AUC 최고"},
    {"model": "CatBoost", "accuracy": 0.5680, "f1": 0.4909, "roc_auc": 0.5926, "note": "Accuracy 최고"},
    {"model": "Logistic Regression", "accuracy": 0.5656, "f1": 0.4853, "roc_auc": 0.5876, "note": "선형 기준 모델"},
])

# -----------------------------
# 컬럼 정보
# -----------------------------
feature_groups = pd.DataFrame({
    "변수 구분": ["고객 정보", "상품 정보", "이력 정보", "행동 정보", "디바이스", "배송", "결제"],
    "사용 변수": [
        "customer_age",
        "product_price, discount_percent, product_rating, product_category_*",
        "past_purchase_count, past_return_rate",
        "session_length_minutes, num_product_views, used_coupon",
        "device_type_desktop, device_type_mobile, device_type_tablet",
        "delivery_delay_days, shipping_method_*",
        "payment_method_*"
    ]
})

columns_info = [
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
    "device_type_desktop",
    "device_type_mobile",
    "device_type_tablet",
    "product_category_beauty",
    "product_category_clothing",
    "product_category_electronics",
    "product_category_home",
    "product_category_sports",
    "product_category_toys",
    "shipping_method_express",
    "shipping_method_same_day",
    "shipping_method_standard",
    "payment_method_apple_pay",
    "payment_method_credit_card",
    "payment_method_debit_card",
    "payment_method_paypal",
]

# -----------------------------
# 최고 성능 모델 추출
# -----------------------------
best_auc = metrics_df.loc[metrics_df["roc_auc"].idxmax()]
best_f1 = metrics_df.loc[metrics_df["f1"].idxmax()]
best_acc = metrics_df.loc[metrics_df["accuracy"].idxmax()]

# -----------------------------
# 제목
# -----------------------------
st.title("이커머스 반품 예측 모델 성능 대시보드")
st.caption("Validation 결과를 기반으로 각 모델의 성능을 비교하고 운영 인사이트를 정리한 발표용 대시보드")

# -----------------------------
# KPI
# -----------------------------
st.subheader("핵심 성능 요약")

k1, k2, k3 = st.columns(3)
k1.metric("최고 ROC-AUC 모델", best_auc["model"], f"{best_auc['roc_auc']:.4f}")
k2.metric("최고 F1 모델", best_f1["model"], f"{best_f1['f1']:.4f}")
k3.metric("최고 Accuracy 모델", best_acc["model"], f"{best_acc['accuracy']:.4f}")

st.markdown("---")

# -----------------------------
# 모델 성능 비교 표 + 차트
# -----------------------------
st.subheader("모델별 Validation 성능 비교")

left, right = st.columns([1.2, 1])

with left:
    st.dataframe(
        metrics_df.sort_values("roc_auc", ascending=False),
        use_container_width=True,
        hide_index=True
    )

with right:
    chart_metric = st.selectbox(
        "비교 지표 선택",
        ["accuracy", "f1", "roc_auc"],
        format_func=lambda x: {
            "accuracy": "Accuracy",
            "f1": "F1 Score",
            "roc_auc": "ROC-AUC"
        }[x]
    )

    sorted_df = metrics_df.sort_values(chart_metric, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(sorted_df["model"], sorted_df[chart_metric])
    ax.set_title(f"모델별 {chart_metric.upper()} 비교")
    ax.set_xlabel("모델")
    ax.set_ylabel(chart_metric.upper())
    plt.xticks(rotation=20)
    st.pyplot(fig)

st.info(
    "운영 관점에서는 ROC-AUC가 가장 높은 LightGBM을 대표 모델로 설명할 수 있고, "
    "분류 균형 관점에서는 F1 Score가 가장 높은 XGBoost를 함께 비교하는 것이 적절합니다."
)

st.markdown("---")

# -----------------------------
# 지표별 상세 비교
# -----------------------------
st.subheader("지표별 상세 비교")

m1, m2, m3 = st.columns(3)

with m1:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(metrics_df["model"], metrics_df["accuracy"])
    ax.set_title("Accuracy 비교")
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=25)
    st.pyplot(fig)

with m2:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(metrics_df["model"], metrics_df["f1"])
    ax.set_title("F1 Score 비교")
    ax.set_ylabel("F1")
    plt.xticks(rotation=25)
    st.pyplot(fig)

with m3:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(metrics_df["model"], metrics_df["roc_auc"])
    ax.set_title("ROC-AUC 비교")
    ax.set_ylabel("ROC-AUC")
    plt.xticks(rotation=25)
    st.pyplot(fig)

st.markdown("---")

# -----------------------------
# 모델별 해석
# -----------------------------
st.subheader("모델별 해석")

for _, row in metrics_df.iterrows():
    st.markdown(
        f"- **{row['model']}**: Accuracy **{row['accuracy']:.4f}**, "
        f"F1 **{row['f1']:.4f}**, ROC-AUC **{row['roc_auc']:.4f}** "
        f"→ {row['note']}"
    )

st.markdown("---")

# -----------------------------
# 사용 변수 구조
# -----------------------------
st.subheader("모델 입력 변수 구조")

col1, col2 = st.columns([1, 1])

with col1:
    st.dataframe(feature_groups, use_container_width=True, hide_index=True)

with col2:
    st.markdown("### 전체 컬럼 목록")
    st.code(", ".join(columns_info))

st.markdown("---")

# -----------------------------
# 예상 주요 분석 포인트
# -----------------------------
st.subheader("예상 주요 분석 포인트")

a1, a2 = st.columns(2)

with a1:
    st.markdown("""
### 반품에 영향을 줄 가능성이 높은 변수
- `delivery_delay_days`
- `past_return_rate`
- `discount_percent`
- `product_price`
- `product_category_*`
- `used_coupon`
""")

with a2:
    st.markdown("""
### 운영 관점에서 해석 가능한 포인트
- 배송 지연이 반품 증가로 이어지는지
- 고할인 상품에서 반품률이 높아지는지
- 과거 반품 이력이 재반품 가능성을 높이는지
- 특정 카테고리에 반품이 집중되는지
- 쿠폰 사용 여부가 반품 행동과 관련 있는지
""")

st.markdown("---")

# -----------------------------
# 차트 구성 제안
# -----------------------------
st.subheader("추가 시각화 구성 제안")

chart_plan = pd.DataFrame({
    "차트명": [
        "카테고리별 반품률",
        "배송 지연일수별 반품률",
        "할인율 구간별 반품률",
        "과거 반품률 구간별 반품률",
        "쿠폰 사용 여부별 반품률",
        "예측 확률 분포",
        "고위험 주문 비율",
        "주요 영향 변수 TOP 10"
    ],
    "목적": [
        "상품군별 반품 패턴 파악",
        "배송 이슈와 반품의 관계 분석",
        "프로모션과 반품의 관계 분석",
        "고객 이력과 반품의 관계 분석",
        "쿠폰 사용과 반품 행동 비교",
        "고위험군 분포 파악",
        "운영 우선관리 대상 식별",
        "모델 설명력 강화"
    ]
})

st.dataframe(chart_plan, use_container_width=True, hide_index=True)

st.markdown("---")

# -----------------------------
# 최종 결론
# -----------------------------
st.subheader("최종 결론")

st.markdown(f"""
- **Accuracy 기준 최고 모델**은 **{best_acc['model']} ({best_acc['accuracy']:.4f})** 입니다.
- **F1 Score 기준 최고 모델**은 **{best_f1['model']} ({best_f1['f1']:.4f})** 입니다.
- **ROC-AUC 기준 최고 모델**은 **{best_auc['model']} ({best_auc['roc_auc']:.4f})** 입니다.

이번 프로젝트에서는 단순 정확도보다 **반품 여부를 얼마나 잘 구분해내는가**가 중요하므로,  
**ROC-AUC가 가장 높은 LightGBM**을 대표 모델로 선정하는 것이 타당합니다.

다만 실제 반품 고객을 놓치지 않는 관점에서는 **F1 Score가 가장 높은 XGBoost**도 함께 비교 대상으로 제시할 수 있습니다.
""")

st.markdown("---")

# -----------------------------
# 운영 전략 제안
# -----------------------------
st.subheader("운영 전략 제안")

st.markdown("""
- 배송 지연 가능성이 높은 주문군은 사전 관리 대상으로 설정합니다.
- 과거 반품 이력이 높은 고객군은 구매 전 안내를 강화합니다.
- 고할인 상품은 상세페이지와 기대치 관리를 보완합니다.
- 반품률이 높은 카테고리는 리뷰, 옵션, 설명 정보 품질을 개선합니다.
- 예측 확률 파일이 추가되면 고위험 주문 우선관리 리스트까지 확장할 수 있습니다.
""")

st.success("이 앱은 CSV 데이터셋 없이도 실행되는 발표용 버전입니다.")