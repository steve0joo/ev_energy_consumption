"""serve.predict 회귀 검증 (UI 없이 headless).

명세 Phase 9 Step 0의 최소 검증 4종 + 헤드라인 CSV 로드.
"""

import math

from serve.predict import (
    FEATURES_B,
    _model,
    feature_meta,
    headline_metrics,
    predict,
    predict_fixed,
    to_business,
)


def test_model_schema_order():
    # 모델 입력 스키마와 FEATURES_B 순서 일치.
    assert list(_model.feature_names_in_) == FEATURES_B


def test_predict_at_mean_in_physical_range():
    # 평균 입력 예측이 타깃 물리 범위(11.62~35.00) 안에 든다.
    meta = feature_meta()
    x = predict({f: meta[f]["mean"] for f in FEATURES_B})
    assert 11.62 <= x <= 35.00


def test_predict_fixed_is_constant():
    # 고정값(Dummy)은 ≈24.14이며 호출마다 동일한 상수.
    a = predict_fixed()
    b = predict_fixed()
    assert abs(a - 24.14) <= 0.5
    assert a == b


def test_to_business():
    r = to_business(20, 100, 60, 300)
    assert math.isclose(r["energy_kwh"], 20.0)
    assert math.isclose(r["cost_won"], 6000.0)
    assert math.isclose(r["soc_drop_pct"], 100 / 3)  # 33.333...
    assert math.isclose(r["range_km"], 300.0)


def test_headline_metrics_from_csv():
    # 헤드라인 숫자는 CSV 실측값(하드코딩 아님).
    m = headline_metrics()
    assert math.isclose(m["rmse"], 1.9427)
    assert math.isclose(m["mae"], 1.5783)
    assert math.isclose(m["improvement_rmse_pct"], 47.2)
    assert math.isclose(m["improvement_mae_pct"], 47.3)
