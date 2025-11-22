import pandas as pd
import numpy as np
from scipy.stats import f_oneway
import plotly.express as px
import streamlit as st


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
        st.subheader("그룹별 요인 평균 비교")

        plot_df = group_df[[group_col] + factor_cols].copy()
        long_df = plot_df.melt(
            id_vars=group_col,
            value_vars=factor_cols,
            var_name="요인",
            value_name="점수",
        )
        long_df["요인"] = long_df["요인"].map(factor_labels)

        mean_table = (
            long_df.pivot_table(
                index=group_col,
                columns="요인",
                values="점수",
                aggfunc="mean",
            )
            .round(2)
            .sort_index()
        )

        st.write("**그룹별 요인 평균 (1~5점 척도)**")
        st.dataframe(mean_table, use_container_width=False)

        fig_bar = px.bar(
            long_df,
            x="요인",
            y="점수",
            color=group_col,
            barmode="group",
            title=f"{group_label}별 요인 평균 비교",
            range_y=[1, 5],
        )
        fig_bar.update_layout(
            xaxis_title="요인",
            yaxis_title="평균 점수 (1~5점)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
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


if __name__ == "__main__":
    main()


