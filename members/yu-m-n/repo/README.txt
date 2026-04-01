[전처리 데이터셋 설명]

이 파일은 전처리 후 저장한 csv 파일들이 각각 어떤 데이터인지, 그리고 어떤 모델에 사용하는지 정리한 문서입니다.

0. test_order_id.csv 
 - 테스트 데이터의 주문 식별자(order_id)만 저장한 파일입니다.
 - 추후 예측값과 결합하여 제출 파일을 만들 때 사용합니다.

1. X_tr_unscaled.csv
- 학습용(train) feature 데이터입니다.
- 스케일링을 적용하지 않은 버전입니다.
- 트리 기반 모델 학습에 사용합니다.
- 사용 모델 예시: DecisionTree, RandomForest, XGBoost, LightGBM

2. X_val_unscaled.csv
- 검증용(validation) feature 데이터입니다.
- 스케일링을 적용하지 않은 버전입니다.
- 트리 기반 모델의 검증에 사용합니다.
- 사용 모델 예시: DecisionTree, RandomForest, XGBoost, LightGBM

3. X_test_unscaled.csv
- 테스트용(test) feature 데이터입니다.
- 스케일링을 적용하지 않은 버전입니다.
- 트리 기반 모델로 최종 예측할 때 사용합니다.
- 현재 단계에서는 제출 파일 생성용이 아니라, 예측 입력용 데이터입니다.

4. X_tr_scaled.csv
- 학습용(train) feature 데이터입니다.
- StandardScaler를 적용한 버전입니다.
- 스케일링이 필요한 선형/거리 기반 모델 학습에 사용합니다.
- 사용 모델 예시: LogisticRegression, SVM, KNN

5. X_val_scaled.csv
- 검증용(validation) feature 데이터입니다.
- StandardScaler를 적용한 버전입니다.
- 스케일링이 필요한 선형/거리 기반 모델의 검증에 사용합니다.
- 사용 모델 예시: LogisticRegression, SVM, KNN

6. X_test_scaled.csv
- 테스트용(test) feature 데이터입니다.
- StandardScaler를 적용한 버전입니다.
- 스케일링이 필요한 선형/거리 기반 모델로 최종 예측할 때 사용합니다.
- 현재 단계에서는 제출 파일 생성용이 아니라, 예측 입력용 데이터입니다.

7. y_tr.csv
- 학습용(train) 타깃값 데이터입니다.
- X_tr_unscaled.csv 또는 X_tr_scaled.csv와 함께 사용합니다.
- 값은 returned(반품 여부)입니다.

8. y_val.csv
- 검증용(validation) 타깃값 데이터입니다.
- X_val_unscaled.csv 또는 X_val_scaled.csv와 함께 사용합니다.
- 값은 returned(반품 여부)입니다.

[사용 방법 요약]
- 트리 기반 모델을 사용할 경우:
  X_tr_unscaled.csv, X_val_unscaled.csv, X_test_unscaled.csv, y_tr.csv, y_val.csv 사용

- 선형/거리 기반 모델을 사용할 경우:
  X_tr_scaled.csv, X_val_scaled.csv, X_test_scaled.csv, y_tr.csv, y_val.csv 사용

[주의 사항]
- X_test 데이터에는 타깃값(returned)이 없습니다.
- y_tr, y_val은 각각 학습용/검증용 정답값입니다.
- 현재는 전처리된 데이터셋을 팀원에게 전달하는 단계이며, 제출 파일 생성은 나중에 별도로 진행할 예정입니다.
