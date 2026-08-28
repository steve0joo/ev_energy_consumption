"""전기차 전비 예측 — Streamlit MVP 앱 (ADR-005, 명세서 Phase 9).

경로 계획 시점 확보 가능한 6개 변수(Model B)로 구간 전비를 예측하고,
업무 지표(에너지·요금·도착 SOC·주행가능거리)와 고정값 대비 개선을 보여준다.
예측·업무환산 로직은 전부 serve.predict 호출(단일 출처, 재구현 금지).
"""

from pathlib import Path

import streamlit as st

from serve.predict import (
    FEATURES_B,
    feature_meta,
    headline_metrics,
    predict,
    predict_fixed,
    to_business,
)

# 프로젝트 루트 탐색(CLAUDE.md 규약) — 앱은 루트에서 실행되지만 cwd 의존을 피한다.
ROOT = Path(__file__).resolve().parent
while not (ROOT / "requirements.txt").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

# 슬라이더 라벨(한글 병기) — feature_meta()의 6개 변수와 키가 일치해야 함.
_LABELS = {
    "trip_distance_km": "주행거리 trip_distance_km (km)",
    "road_grade_pct": "도로 경사 road_grade_pct (%)",
    "speed_kmh": "속도 speed_kmh (km/h)",
    "ambient_temp_C": "외기온 ambient_temp_C (°C)",
    "payload_kg": "적재중량 payload_kg (kg)",
    "tire_pressure_bar": "타이어 공기압 tire_pressure_bar (bar)",
}


def render_predictor():
    """대화형 예측 화면 — 사이드바 입력 → 메인 결과·고정값 대비.

    Step 2가 st.tabs로 감쌀 수 있도록 렌더링을 이 함수로 분리한다.
    """
    meta = feature_meta()
    metrics = headline_metrics()

    # --- 사이드바: 조건 입력 (6개 슬라이더) ---
    st.sidebar.header("주행 조건 입력")
    inputs = {}
    for f in FEATURES_B:
        m = meta[f]
        inputs[f] = st.sidebar.slider(
            _LABELS[f],
            min_value=float(m["min"]),
            max_value=float(m["max"]),
            value=float(m["mean"]),
        )

    # --- 사이드바: 업무 가정값 (사용자 조절 가능, ADR-005) ---
    st.sidebar.header("업무 가정값")
    battery_kwh = st.sidebar.number_input(
        "배터리 용량 (kWh)", min_value=1.0, value=60.0, step=1.0
    )
    price_per_kwh = st.sidebar.number_input(
        "충전 단가 (원/kWh)", min_value=0.0, value=300.0, step=10.0
    )

    distance_km = inputs["trip_distance_km"]

    # --- 메인: 예측 결과 ---
    st.title("전기차 구간 전비 예측 (Model B)")
    st.caption("경로 선택 전 확보 가능한 6개 조건만으로 구간 전비를 예측합니다.")

    consumption = predict(inputs)
    rmse = metrics["rmse"]

    st.header("예측 결과")
    st.metric("예측 전비", f"{consumption:.2f} kWh/100km")
    st.caption(f"±{rmse:.2f} kWh/100km (test RMSE 기준)")

    biz = to_business(consumption, distance_km, battery_kwh, price_per_kwh)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("소요 에너지", f"{biz['energy_kwh']:.1f} kWh")
    c2.metric("예상 요금", f"{biz['cost_won']:,.0f} 원")
    c3.metric("도착 SOC", f"{100 - biz['soc_drop_pct']:.1f} %", help="시작 100% 가정")
    c4.metric("주행가능거리", f"{biz['range_km']:.0f} km")

    # --- 메인: 고정값 대비 (개선율 스토리) ---
    st.header("고정값 대비")
    fixed = predict_fixed()
    fixed_biz = to_business(fixed, distance_km, battery_kwh, price_per_kwh)

    energy_diff = biz["energy_kwh"] - fixed_biz["energy_kwh"]
    cost_diff = biz["cost_won"] - fixed_biz["cost_won"]

    st.markdown(
        f"현재 방식(고정 평균 **{fixed:.2f}** kWh/100km)이라면 이 경로에 "
        f"**{fixed_biz['energy_kwh']:.1f} kWh** / **{fixed_biz['cost_won']:,.0f} 원**으로 안내합니다. "
        f"조건 기반 예측은 **{biz['energy_kwh']:.1f} kWh** / **{biz['cost_won']:,.0f} 원** "
        f"(에너지 {energy_diff:+.1f} kWh, 요금 {cost_diff:+,.0f} 원 차이)."
    )

    st.info(
        f"test 기준 고정값 대비 오차 약 **{metrics['improvement_rmse_pct']:.1f}%** 감소 "
        f"(RMSE {metrics['rmse']:.4f}, 고정값 대비 개선율)."
    )


def render_report():
    """정적 해석 탭 — 기존 리포트 산출물(그림 6장·중요도 표)을 임베드만 한다.

    새 그림·수치를 생성하지 않는다. 캡션·표는 REPORT.md §3·§7과
    notes/p6_interpret.md의 실측값 전사(가법 추가).
    """
    fig_dir = ROOT / "artifacts" / "figures"

    st.title("리포트 / 해석")
    st.caption("이 모델이 왜 이렇게 예측하는지 — EDA 결론과 오차 해석(REPORT §3·§7 실측).")

    # --- EDA 그림 4장 (캡션 = REPORT §3 결론 4문장) ---
    st.header("EDA 핵심 결과")
    st.image(
        str(fig_dir / "01_target_dist.png"),
        caption=(
            "결론 ① 타깃이 11.62~35.00으로 약 3배 퍼져 있고(60kWh 기준 주행거리 171~516km) "
            "→ 고정 평균값(24.1) 하나로 안내하는 현재 방식은 구조적으로 틀리다."
        ),
    )
    st.image(
        str(fig_dir / "02_corr_heatmap.png"),
        caption=(
            "결론 ② 타깃을 가장 강하게 지배하는 변수는 road_grade_pct(r=+0.50) > payload_kg(+0.47) "
            "> speed_kmh(+0.45) 순이며, 변수 간 완전상관 쌍이 없어 다중공선성 위험은 낮다."
        ),
    )
    st.image(
        str(fig_dir / "03_speed_vs_target.png"),
        caption=(
            "결론 ③ speed–전비 곡률은 약하다(2차항 계수 +0.00007, R² 0.20→0.20) "
            "→ 2차 파생의 이득이 크지 않아 선형 가정이 대체로 성립."
        ),
    )
    st.image(
        str(fig_dir / "04_temp_vs_target.png"),
        caption=(
            "결론 ④ 최저온 구간(-10~-5°C) 평균 전비 26.0이 고온 구간(30~35°C, 23.5) 대비 +10% 높다 "
            "→ 저온 난방 부하가 전비를 크게 끌어올린다(ambient_temp_C가 공조 부하의 대리변수)."
        ),
    )

    # --- 오차 해석 그림 2장 (캡션 = REPORT §7) ---
    st.header("오차 해석")
    st.image(
        str(fig_dir / "05_perm_importance.png"),
        caption=(
            "Permutation importance — road_grade_pct > payload_kg > speed_kmh 순. "
            "계수의 부호·크기가 모두 물리 상식과 일치 → 계수 자체가 곧 해석."
        ),
    )
    st.image(
        str(fig_dir / "06_residuals.png"),
        caption=(
            "잔차 진단 — 예측 3분위별 잔차 std 비 1.08(대체로 등분산). "
            "상위 25%(고전비) 구간 평균 잔차 +1.139·과소예측 75.1% → 안전마진/quantile 보정 권고."
        ),
    )

    # --- 변수 중요도 표 (REPORT §7 / p6_interpret.md 실측 전사) ---
    st.subheader("변수 중요도 (permutation importance · 계수)")
    st.table(
        [
            {"변수": "road_grade_pct", "importance": "0.5029", "계수": "+1.860", "물리 해석": "오르막 위치에너지 부하 ↑ → 전비 ↑"},
            {"변수": "payload_kg", "importance": "0.4412", "계수": "+1.750", "물리 해석": "적재중량(관성·구름저항) ↑ → 전비 ↑"},
            {"변수": "speed_kmh", "importance": "0.3860", "계수": "+1.660", "물리 해석": "속도(공기저항) ↑ → 전비 ↑"},
            {"변수": "ambient_temp_C", "importance": "0.0517", "계수": "-0.604", "물리 해석": "외기온 ↑ → 난방 부하 ↓ → 전비 ↓ (공조 대리변수)"},
            {"변수": "trip_distance_km", "importance": "0.0480", "계수": "+0.547", "물리 해석": "100km당 정규화값 → 기여 미미"},
        ]
    )
    st.caption(
        "importance = test R² 감소분(permutation, n_repeats=10, random_state=42). "
        "계수 부호로 전비 증감 방향 표기."
    )


# 메인: 예측(기존)과 리포트/해석(가법)을 탭으로 분리.
tab_pred, tab_report = st.tabs(["예측", "리포트/해석"])
with tab_pred:
    render_predictor()
with tab_report:
    render_report()
