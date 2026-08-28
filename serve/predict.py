"""UI와 분리된 얇은 추론 모듈 (ADR-005 서빙 레이어).

학습은 하지 않는다. 기존 pkl(Model B Linear·Dummy)을 로드만 하고,
헤드라인 숫자는 artifacts/metrics/*.csv에서 읽는다(하드코딩 금지, CLAUDE.md 규칙 4).
Streamlit을 import하지 않는다 → pytest로 단독 검증 가능(serve/test_predict.py).
"""

from pathlib import Path

import joblib
import pandas as pd

# 파일 위치 기준 루트(앱·pytest 모두 루트에서 실행되나 cwd 의존 회피).
ROOT = Path(__file__).resolve().parent.parent

# 순서 고정 — 모델 feature_names_in_ 과 일치해야 함(아래 assert로 강제).
FEATURES_B = [
    "trip_distance_km",
    "road_grade_pct",
    "speed_kmh",
    "ambient_temp_C",
    "payload_kg",
    "tire_pressure_bar",
]

_MODEL_PATH = ROOT / "artifacts" / "models" / "final_model_B_linear.pkl"
_DUMMY_PATH = ROOT / "artifacts" / "models" / "B_dummy.pkl"
_DATA_PATH = ROOT / "data" / "ev_energy_consumption.csv"
_BUSINESS_METRICS_PATH = ROOT / "artifacts" / "metrics" / "business_metrics.csv"

# 로드만 한다(재학습 금지). Dummy = 현재 내비게이션의 고정 평균값 안내.
_model = joblib.load(_MODEL_PATH)
_dummy = joblib.load(_DUMMY_PATH)

assert list(_model.feature_names_in_) == FEATURES_B, (
    f"model schema mismatch: {list(_model.feature_names_in_)} != {FEATURES_B}"
)


def _to_frame(features: dict) -> pd.DataFrame:
    """features dict → columns=FEATURES_B 순서의 1행 DataFrame."""
    return pd.DataFrame([[features[f] for f in FEATURES_B]], columns=FEATURES_B)


def feature_meta() -> dict:
    """슬라이더용 메타 — data/ev_energy_consumption.csv에서 산출.

    반환: {feature: {"min": float, "mean": float, "max": float}} (FEATURES_B 6개)."""
    df = pd.read_csv(_DATA_PATH, usecols=FEATURES_B)
    return {
        f: {
            "min": float(df[f].min()),
            "mean": float(df[f].mean()),
            "max": float(df[f].max()),
        }
        for f in FEATURES_B
    }


def predict(features: dict) -> float:
    """features(dict, FEATURES_B 6키) → 예측 전비(kWh/100km) 스칼라."""
    return float(_model.predict(_to_frame(features))[0])


def predict_fixed() -> float:
    """현재 내비게이션 방식(고정 평균값). Dummy는 입력 무관 상수(≈24.14)를 반환."""
    return float(_dummy.predict(_to_frame({f: 0.0 for f in FEATURES_B}))[0])


def headline_metrics() -> dict:
    """artifacts/metrics/business_metrics.csv에서 읽어 반환(하드코딩 아님)."""
    df = pd.read_csv(_BUSINESS_METRICS_PATH)
    vals = dict(zip(df["지표"], df["값"]))
    return {
        "rmse": float(vals["test RMSE"]),
        "mae": float(vals["test MAE"]),
        "improvement_rmse_pct": float(vals["RMSE 개선율(고정값 대비)"]),
        "improvement_mae_pct": float(vals["MAE 개선율(고정값 대비)"]),
    }


def to_business(consumption: float, distance_km: float,
                battery_kwh: float, price_per_kwh: float) -> dict:
    """업무 지표 환산(REPORT §8 산식)."""
    energy_kwh = consumption / 100 * distance_km
    cost_won = energy_kwh * price_per_kwh
    soc_drop_pct = energy_kwh / battery_kwh * 100
    range_km = battery_kwh / consumption * 100
    return {
        "energy_kwh": energy_kwh,
        "cost_won": cost_won,
        "soc_drop_pct": soc_drop_pct,
        "range_km": range_km,
    }
