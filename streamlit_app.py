import pandas as pd
import numpy as np
from scipy.stats import f_oneway
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder


CSV_PATH = "exp - 설문지 응답 시트1 (1).csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # --- 기본 컬럼 찾기 ---
    age_col = [c for c in df.columns if "[1-2]" in c][0]
    gender_col = [c for c in df.columns if "[1-1]" in c][0]
    education_col = [c for c in df.columns if "[1-3]" in c][0]

    # 나이 숫자 변환
    df[age_col] = pd.to_numeric(
        df[age_col].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce",
    )

    # 연령대 그룹 함수 (3그룹)
    def get_age_group(age):
        if pd.isna(age):
            return np.nan
        age = int(age)
        if age <= 25:
            return "20대 초반"
        if 26 <= age <= 29:
            return "20대 중후반"
        if age >= 30:
            return "30대 이상"
        return "기타"

    # 연령대 그룹 함수 (2그룹)
    def get_age_group_redefined(age):
        if pd.isna(age):
            return np.nan
        age = int(age)
        if age <= 29:
            return "20대"
        if age >= 30:
            return "30대 이상"
        return "기타"

    df["Age_Group_3"] = df[age_col].apply(get_age_group)
    df["Age_Group_2"] = df[age_col].apply(get_age_group_redefined)

    # --- 섹터별 문항 정의 ---
    cols = df.columns.tolist()

    def find_cols(prefix_list):
        found = []
        for prefix in prefix_list:
            matched = [c for c in cols if f"[{prefix}]" in c]
            found.extend(matched)
        return found

    # SNS 이용량 매핑
    sns_freq_col = find_cols(["2-3"])[0]
    sns_time_col = find_cols(["2-4"])[0]

    sns_freq_mapping = {
        "필요시 검색": 1,
        "가끔씩 (2~5회)": 2,
        "수시로 접속 (10회 이상)": 3,
        "거의 습관적으로 자주 (30회 이상)": 4,
        "하루 종일 접속해놓고 있다": 5,
    }

    sns_time_mapping = {
        "10분 미만": 0.5,
        "30분~1시간 미만": 0.75,
        "1~2시간 미만": 1.5,
        "2~4시간 미만": 3,
        "4~5시간 미만": 4.5,
        "5시간 이상": 6,
    }

    df["SNS_Freq_Numeric"] = df[sns_freq_col].map(sns_freq_mapping)
    df["SNS_Time_Numeric"] = df[sns_time_col].map(sns_time_mapping)

    # 섹터 매핑 (노트북 로직과 동일)
    sector_map = {
        "SNS 이용량": ["SNS_Freq_Numeric", "SNS_Time_Numeric"],
        "충동구매 성향": find_cols(
            ["3-1", "3-2", "3-3", "3-4", "3-5", "3-28", "3-29"]
        ),
        "사회적 비교": find_cols(["3-7", "3-8", "3-9", "3-10", "3-11"]),
        "쇼핑 후회/태도": find_cols(
            ["3-13", "3-14", "3-15", "3-16", "3-25", "3-26", "3-27"]
        ),
        "광고 인식/신뢰": find_cols(
            [f"3-{i}" for i in range(17, 25)]
            + [f"4-{i}" for i in range(1, 13)]
        ),
        "구매 의도": find_cols([f"4-{i}" for i in range(13, 28)]),
    }

    factor_names = []
    for name, columns in sector_map.items():
        if not columns:
            continue
        df[f"{name}_Mean"] = df[columns].mean(axis=1)
        factor_names.append(f"{name}_Mean")

    return df, factor_names, age_col, gender_col, education_col


def main():
    st.set_page_config(
        page_title="설문 분석 대시보드",
        layout="wide",
    )

    st.title("설문 데이터 분석 대시보드")
    st.markdown(
        "연령 / 성별 / 학력에 따라 **요인 점수 분포와 평균**, "
        "그리고 요인 간 **상관관계**를 한눈에 살펴볼 수 있는 대시보드입니다."
    )

    try:
        df, factor_names, age_col, gender_col, education_col = load_data(CSV_PATH)
    except FileNotFoundError:
        st.error(f"CSV 파일을 찾을 수 없습니다: `{CSV_PATH}`")
        st.stop()

    factor_cols = [f for f in factor_names if f in df.columns]
    factor_labels = {col: col.replace("_Mean", "") for col in factor_cols}

    # --- 사이드바 설정 ---
    st.sidebar.header("설정")
    
    # 20대 초반 랜덤 샘플링 옵션 추가
    use_sampling = st.sidebar.checkbox("🎲 20대 초반 30명 랜덤 샘플링 적용")

    group_options = {
        "연령 (3그룹: 20대 초반/중후반/30대 이상)": "Age_Group_3",
        "연령 (2그룹: 20대 / 30대 이상)": "Age_Group_2",
        "성별": gender_col,
        "학력": education_col,
    }

    group_label = st.sidebar.selectbox(
        "그룹 기준 선택",
        options=list(group_options.keys()),
    )
    group_col = group_options[group_label]

    factor_choice_label = st.sidebar.selectbox(
        "요인 선택",
        options=["전체 요인"] + list(factor_labels.values()),
    )

    # label -> 실제 컬럼명 역변환
    label_to_col = {v: k for k, v in factor_labels.items()}
    factor_col = None if factor_choice_label == "전체 요인" else label_to_col[
        factor_choice_label
    ]

    st.sidebar.markdown("---")
    st.sidebar.markdown("데이터 파일: ")
    st.sidebar.code(CSV_PATH, language="text")

    # --- 메인 영역 ---
    # 샘플링 적용 로직
    if use_sampling:
        # 20대 초반 그룹 식별 (Age_Group_3 기준)
        group_target = "20대 초반"
        if "Age_Group_3" in df.columns:
            target_mask = df["Age_Group_3"] == group_target
            other_mask = df["Age_Group_3"] != group_target
            
            target_df = df[target_mask]
            other_df = df[other_mask]
            
            if len(target_df) > 30:
                # 30명 랜덤 샘플링 (고정 시드 사용 X -> 매번 다르게, 필요시 random_state=42 추가 가능)
                sampled_target = target_df.sample(n=30, random_state=42) 
                df = pd.concat([sampled_target, other_df], ignore_index=True)
                st.sidebar.success(f"✅ 20대 초반 {len(target_df)}명 → 30명 샘플링 완료")
            else:
                st.sidebar.warning(f"⚠️ 20대 초반 인원이 {len(target_df)}명이라 샘플링하지 않음")

    if group_col:
        group_df = df.dropna(subset=[group_col])
    else:
        group_df = df.copy()

    st.subheader("표본 개요")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("전체 표본 수", len(df))
    with c2:
        st.metric("유효 표본 수 (선택 그룹 기준)", len(group_df))
    with c3:
        st.metric("요인 수", len(factor_cols))

    if group_col:
        st.markdown(f"#### 선택한 그룹: **{group_label}**")
        st.dataframe(
            group_df[group_col].value_counts().to_frame("표본 수"),
        )

    # --- 요인 시각화 ---
    st.markdown("---")

    if factor_col is None:
        st.subheader("그룹별 요인 분포 비교")

        plot_df = group_df[[group_col] + factor_cols].copy()
        long_df = plot_df.melt(
            id_vars=group_col,
            value_vars=factor_cols,
            var_name="요인",
            value_name="점수",
        )
        long_df["요인"] = long_df["요인"].map(factor_labels)

        # 통계 요약 테이블
        st.markdown("#### 📊 그룹별 요약 통계")
        summary_stats = []
        for factor in factor_cols:
            factor_name = factor_labels[factor]
            for group_name in plot_df[group_col].dropna().unique():
                group_data = plot_df[plot_df[group_col] == group_name][factor].dropna()
                summary_stats.append({
                    "요인": factor_name,
                    "그룹": group_name,
                    "평균": round(group_data.mean(), 2),
                    "표준편차": round(group_data.std(), 2),
                    "중앙값": round(group_data.median(), 2),
                    "최소값": round(group_data.min(), 2),
                    "최대값": round(group_data.max(), 2),
                    "표본수": len(group_data),
                })
        
        summary_df = pd.DataFrame(summary_stats)
        st.dataframe(summary_df, use_container_width=True, height=250)

        # 탭으로 여러 시각화 제공
        tab1, tab2, tab3 = st.tabs(["📦 박스플롯", "🎻 바이올린 플롯", "📊 평균 비교"])
        
        with tab1:
            st.markdown("##### 그룹별 요인 점수 분포 (박스플롯)")
            fig_box = px.box(
                long_df,
                x="요인",
                y="점수",
                color=group_col,
                title=f"{group_label}별 요인 점수 분포",
                points="outliers",
            )
            fig_box.update_layout(
                xaxis_title="요인",
                yaxis_title="점수 (1~5점)",
                height=500,
                showlegend=True,
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        with tab2:
            st.markdown("##### 그룹별 요인 점수 분포 (바이올린 플롯)")
            fig_violin = px.violin(
                long_df,
                x="요인",
                y="점수",
                color=group_col,
                box=True,
                points="all",
                title=f"{group_label}별 요인 점수 분포 (바이올린)",
            )
            fig_violin.update_layout(
                xaxis_title="요인",
                yaxis_title="점수 (1~5점)",
                height=500,
                showlegend=True,
            )
            st.plotly_chart(fig_violin, use_container_width=True)
        
        with tab3:
            st.markdown("##### 그룹별 요인 평균 비교 (막대 그래프)")
            fig_bar = px.bar(
                long_df,
                x="요인",
                y="점수",
                color=group_col,
                barmode="group",
                title=f"{group_label}별 요인 평균 비교",
            )
            fig_bar.update_layout(
                xaxis_title="요인",
                yaxis_title="평균 점수 (1~5점)",
                height=500,
                showlegend=True,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # ANOVA 결과 요약
        st.markdown("#### 📈 통계 검정 결과 (ANOVA)")
        anova_results = []
        for factor in factor_cols:
            factor_name = factor_labels[factor]
            group_values = [
                plot_df[plot_df[group_col] == g][factor].dropna().values
                for g in plot_df[group_col].dropna().unique()
            ]
            valid_groups = [g for g in group_values if len(g) >= 2]
            
            if len(valid_groups) >= 2:
                f_val, p_val = f_oneway(*valid_groups)
                significance = "✅ 유의" if p_val < 0.05 else "❌ 비유의"
                anova_results.append({
                    "요인": factor_name,
                    "F 통계량": round(f_val, 4),
                    "P-value": round(p_val, 4),
                    "유의성 (α=0.05)": significance,
                })
        
        if anova_results:
            anova_df = pd.DataFrame(anova_results)
            st.dataframe(anova_df, use_container_width=True)
    else:
        st.subheader(f"단일 요인 분포: **{factor_choice_label}**")

        plot_df = group_df[[group_col, factor_col]].dropna()

        c1, c2 = st.columns([2, 1])
        with c1:
            fig_box = px.box(
                plot_df,
                x=group_col,
                y=factor_col,
                points="all",
                title=f"{group_label}별 '{factor_choice_label}' 분포 (Boxplot)",
            )
            fig_box.update_layout(
                xaxis_title=group_label,
                yaxis_title="점수 (1~5점)",
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with c2:
            mean_by_group = (
                plot_df.groupby(group_col)[factor_col]
                .agg(["mean", "std", "count"])
                .round(2)
            )
            st.write("**그룹별 요약 통계**")
            st.dataframe(mean_by_group)

        # ANOVA (선택한 요인에 대해 그룹 차이 검증)
        st.markdown("##### ANOVA (선택 요인 기준 그룹 간 차이 검증)")
        group_values = [
            g[factor_col].dropna().values
            for _, g in plot_df.groupby(group_col)
        ]
        valid_groups = [g for g in group_values if len(g) >= 2]

        if len(valid_groups) >= 2:
            f_val, p_val = f_oneway(*valid_groups)
            st.write(f"- F 통계량: `{f_val:.4f}`")
            st.write(f"- P-value: `{p_val:.4f}`")
            if p_val < 0.05:
                st.success("통계적으로 유의미한 차이 있음 (P < 0.05)")
            else:
                st.info("통계적으로 유의미한 차이 없음 (P ≥ 0.05)")
        else:
            st.warning("ANOVA를 수행하기에 그룹별 표본 수가 부족합니다.")

    # --- 상관관계 히트맵 ---
    st.markdown("---")
    st.subheader("요인 간 상관관계 (전체 표본 기준)")

    corr = df[factor_cols].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        title="요인 간 상관관계 히트맵",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # --- 머신러닝 예측 섹션 ---
    st.markdown("---")
    st.header("🤖 AI 구매 의도 예측 시뮬레이션")
    st.markdown("머신러닝(Random Forest)을 사용하여 사용자 특성에 따른 **구매 의도**를 예측합니다.")

    # ML 데이터 준비
    ml_df = df.copy()
    
    # 범주형 변수 인코딩
    le_gender = LabelEncoder()
    # 결측치 처리 (최빈값 또는 '기타')
    ml_df[gender_col] = ml_df[gender_col].fillna("기타")
    ml_df["Gender_Code"] = le_gender.fit_transform(ml_df[gender_col].astype(str))
    
    # 입력 변수(X)와 타겟 변수(y) 설정
    # 나이, 성별, SNS 이용량, 충동구매 성향, 사회적 비교, 쇼핑 후회/태도, 광고 인식/신뢰 -> 구매 의도 예측
    feature_cols = [
        "SNS_Freq_Numeric", "SNS_Time_Numeric", 
        "충동구매 성향_Mean", "사회적 비교_Mean", 
        "쇼핑 후회/태도_Mean", "광고 인식/신뢰_Mean"
    ]
    # 나이 컬럼이 숫자형인지 확인하고 추가
    if age_col in ml_df.columns:
        ml_df[age_col] = ml_df[age_col].fillna(ml_df[age_col].median()) # 결측치 중앙값 대체
        feature_cols.insert(0, age_col)
    
    feature_cols.append("Gender_Code")
    
    target_col = "구매 의도_Mean"
    
    # 결측치 제거
    ml_data = ml_df[feature_cols + [target_col]].dropna()
    
    if len(ml_data) > 10:
        X = ml_data[feature_cols]
        y = ml_data[target_col]
        
        # 모델 학습
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 성능 평가
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"**모델 예측 정확도 (R² Score): {r2:.2f}**")
            st.caption("1.0에 가까울수록 예측이 정확합니다.")
            
            # 변수 중요도 시각화
            importances = model.feature_importances_
            feature_names_display = [
                c.replace("_Mean", "").replace("_Numeric", "") 
                for c in feature_cols
            ]
            # 성별, 나이 이름 다듬기
            feature_names_display = [
                "성별" if "Gender" in f else 
                "나이" if "연령" in f or "1-2" in f else f 
                for f in feature_names_display
            ]
            
            imp_df = pd.DataFrame({
                "Feature": feature_names_display,
                "Importance": importances
            }).sort_values("Importance", ascending=True)
            
            fig_imp = px.bar(
                imp_df, 
                x="Importance", 
                y="Feature", 
                orientation='h',
                title="구매 의도에 영향을 미치는 요인 (중요도)",
                color="Importance",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
        with c2:
            st.subheader("🎛️ 내 구매 의도 예측해보기")
            st.markdown("아래 슬라이더를 조절하여 가상의 사용자 프로필을 만들어보세요.")
            
            # 사용자 입력 받기
            input_data = {}
            
            if age_col in feature_cols:
                input_data[age_col] = st.slider("나이", 10, 60, 24)
            
            gender_opt = st.radio("성별", le_gender.classes_, horizontal=True)
            input_data["Gender_Code"] = le_gender.transform([gender_opt])[0]
            
            input_data["SNS_Freq_Numeric"] = st.slider("SNS 접속 빈도 (1:적음 ~ 5:많음)", 1.0, 5.0, 3.0, 0.5)
            input_data["SNS_Time_Numeric"] = st.slider("SNS 이용 시간 (시간)", 0.0, 6.0, 2.0, 0.5)
            
            input_data["충동구매 성향_Mean"] = st.slider("충동구매 성향 점수", 1.0, 5.0, 3.0, 0.1)
            input_data["사회적 비교_Mean"] = st.slider("사회적 비교 점수", 1.0, 5.0, 3.0, 0.1)
            input_data["쇼핑 후회/태도_Mean"] = st.slider("쇼핑 후회/태도 점수", 1.0, 5.0, 3.0, 0.1)
            input_data["광고 인식/신뢰_Mean"] = st.slider("광고 인식/신뢰 점수", 1.0, 5.0, 3.0, 0.1)
            
            # 예측 수행
            input_df = pd.DataFrame([input_data], columns=feature_cols)
            prediction = model.predict(input_df)[0]
            
            st.divider()
            st.markdown(f"### 🔮 예측된 구매 의도 점수: **{prediction:.2f} / 5.0**")
            
            if prediction >= 4.0:
                st.balloons()
                st.success("구매 가능성이 매우 높습니다! 🚀")
            elif prediction >= 3.0:
                st.info("구매를 고려할 가능성이 있습니다. 🤔")
            else:
                st.warning("구매 가능성이 낮습니다. 📉")
                
    else:
        st.warning("머신러닝을 수행하기에 데이터가 충분하지 않습니다.")


if __name__ == "__main__":
    main()


