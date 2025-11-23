import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CSV_PATH = "exp - 설문지 응답 시트1 (2).csv"


@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)

    # 기본 컬럼 찾기
    age_col = [c for c in df.columns if "[1-2]" in c][0]
    gender_col = [c for c in df.columns if "[1-1]" in c][0]
    education_col = [c for c in df.columns if "[1-3]" in c][0]

    # 나이 숫자 변환
    df[age_col] = pd.to_numeric(
        df[age_col].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    )

    # 연령대 그룹 (3그룹)
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

    df["Age_Group"] = df[age_col].apply(get_age_group)

    # 유틸리티 함수
    cols = df.columns.tolist()

    def find_cols(prefix_list):
        found = []
        for prefix in prefix_list:
            matched = [c for c in cols if f"[{prefix}]" in c]
            found.extend(matched)
        return found

    # SNS 이용률 매핑
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

    # 6개 요인 정의 (논문 분석과 동일)
    sector_map = {
        "SNS 이용률": ["SNS_Freq_Numeric", "SNS_Time_Numeric"],
        "충동구매 성향": find_cols(["3-1", "3-2", "3-3", "3-4", "3-5", "3-28", "3-29"]),
        "사회적 비교": find_cols(["3-7", "3-8", "3-9", "3-10", "3-11"]),
        "쇼핑 후회/태도": find_cols(
            ["3-13", "3-14", "3-15", "3-16", "3-25", "3-26", "3-27"]
        ),
        "광고 인식/신뢰": find_cols(
            [f"3-{i}" for i in range(17, 25)] + [f"4-{i}" for i in range(1, 13)]
        ),
        "구매 의도": find_cols([f"4-{i}" for i in range(13, 28)]),
    }

    factor_to_col = {}
    for name, columns in sector_map.items():
        if not columns:
            continue
        col_name = f"{name}_Mean"
        df[col_name] = df[columns].mean(axis=1)
        factor_to_col[name] = col_name

    # 광고 개수
    ad_cols = [c for c in df.columns if "[2-6]" in c]
    if ad_cols:
        ad_col = ad_cols[0]
        df["Ad_Count_Numeric"] = pd.to_numeric(df[ad_col], errors="coerce")

    return df, factor_to_col, age_col, gender_col, education_col


def main():
    st.set_page_config(page_title="논문 분석 대시보드", layout="wide")

    st.title("📊 소셜커머스 설문 분석 대시보드")
    st.markdown(
        """
        **연령대별 6개 요인 비교 분석** (SNS 이용률 제외)
        - 충동구매 성향 / 사회적 비교 / 쇼핑 후회·태도 / 광고 인식·신뢰 / 구매 의도
        """
    )

    try:
        df, factor_to_col, age_col, gender_col, education_col = load_data(CSV_PATH)
    except FileNotFoundError:
        st.error(f"CSV 파일을 찾을 수 없습니다: `{CSV_PATH}`")
        st.stop()

    # SNS 이용률 제외한 5개 요인만 분석 대상
    analysis_factors = [name for name in factor_to_col.keys() if name != "SNS 이용률"]
    analysis_cols = [factor_to_col[name] for name in analysis_factors]

    # --- 사이드바 ---
    st.sidebar.header("🔧 분석 설정")

    # 20대 초반 샘플링 옵션
    use_sampling = st.sidebar.checkbox("🎲 20대 초반 30명 랜덤 샘플링")

    if use_sampling:
        target_mask = df["Age_Group"] == "20대 초반"
        other_mask = df["Age_Group"] != "20대 초반"
        target_df = df[target_mask]
        other_df = df[other_mask]

        if len(target_df) > 30:
            sampled_target = target_df.sample(n=30, random_state=42)
            df = pd.concat([sampled_target, other_df], ignore_index=True)
            st.sidebar.success(f"✅ 20대 초반 {len(target_df)}명 → 30명 샘플링")
        else:
            st.sidebar.warning(f"⚠️ 20대 초반 인원이 {len(target_df)}명")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**데이터 파일:**")
    st.sidebar.code(CSV_PATH, language="text")

    # --- 메인: 표본 개요 ---
    st.subheader("📌 표본 개요")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("전체 표본 수", len(df))
    with col2:
        st.metric("분석 요인 수 (SNS 이용률 제외)", len(analysis_factors))
    with col3:
        valid_age_count = df["Age_Group"].notna().sum()
        st.metric("유효 연령대 표본", valid_age_count)

    st.markdown("##### 연령대별 표본 분포")
    age_counts = df["Age_Group"].value_counts().sort_index()
    st.dataframe(age_counts.to_frame("표본 수"), use_container_width=False)

    # --- [1] 연령대별 6개 요인 평균 ---
    st.markdown("---")
    st.header("📊 [1] 연령대별 요인 평균 (SNS 이용률 제외)")

    group_means = df.groupby("Age_Group")[analysis_cols].mean().round(2)
    group_means_display = group_means.copy()
    group_means_display.columns = [c.replace("_Mean", "") for c in group_means_display.columns]

    st.dataframe(group_means_display, use_container_width=True)

    # 막대 그래프로 시각화
    long_df = group_means_display.reset_index().melt(
        id_vars="Age_Group", var_name="요인", value_name="평균 점수"
    )

    fig_bar = px.bar(
        long_df,
        x="요인",
        y="평균 점수",
        color="Age_Group",
        barmode="group",
        title="연령대별 요인 평균 비교",
        category_orders={"Age_Group": ["20대 초반", "20대 중후반", "30대 이상"]},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_bar.update_layout(
        xaxis_title="요인",
        yaxis_title="평균 점수 (1~5점)",
        height=450,
        legend_title="연령대",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- [2] 각 연령대별 상위 3개 요인 + 전체 순위 ---
    st.markdown("---")
    st.header("🏆 [2] 각 연령대별 상위 3개 요인 (SNS 이용률 제외)")

    for group in ["20대 초반", "20대 중후반", "30대 이상"]:
        if group not in group_means.index:
            continue

        row = group_means.loc[group]
        renamed = row.rename(
            index={factor_to_col[name]: name for name in analysis_factors}
        )
        sorted_row = renamed.sort_values(ascending=False)
        top3 = sorted_row.head(3)

        st.subheader(f"■ {group}")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**상위 3개 요인:**")
            for i, (fname, val) in enumerate(top3.items(), start=1):
                st.markdown(f"**{i}.** {fname}: `{val:.2f}`")

        with col2:
            st.markdown("**전체 요인 순위 (내림차순):**")
            rank_df = pd.DataFrame(
                {
                    "순위": range(1, len(sorted_row) + 1),
                    "요인": sorted_row.index,
                    "평균 점수": sorted_row.values,
                }
            )
            st.dataframe(rank_df, hide_index=True, use_container_width=True)

    # --- [3] 연령대별 광고 개수 평균 ---
    st.markdown("---")
    st.header("📺 [3] 연령대별 광고 개수 평균 ([2-6] 문항)")

    if "Ad_Count_Numeric" in df.columns:
        ad_means = df.groupby("Age_Group")["Ad_Count_Numeric"].mean().round(2)
        ad_means_df = ad_means.to_frame("평균 광고 개수")
        st.dataframe(ad_means_df, use_container_width=False)

        # 막대 그래프
        fig_ad = px.bar(
            ad_means.reset_index(),
            x="Age_Group",
            y="Ad_Count_Numeric",
            title="연령대별 평균 광고 개수",
            labels={"Age_Group": "연령대", "Ad_Count_Numeric": "평균 광고 개수"},
            color="Age_Group",
            category_orders={"Age_Group": ["20대 초반", "20대 중후반", "30대 이상"]},
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_ad.update_layout(
            xaxis_title="연령대",
            yaxis_title="평균 광고 개수 (개)",
            height=400,
            showlegend=False,
        )
        st.plotly_chart(fig_ad, use_container_width=True)
    else:
        st.warning("⚠️ 광고 개수 데이터([2-6])를 찾을 수 없습니다.")

    # --- [4] 집단별 상관계수 ---
    st.markdown("---")
    st.header("🔗 [4] 집단별 요인 간 상관관계")

    corr_group_options = {
        "전체 표본": None,
        "20대 초반": "20대 초반",
        "20대 중후반": "20대 중후반",
        "30대 이상": "30대 이상",
    }

    corr_choice = st.selectbox(
        "상관계수를 계산할 집단 선택", options=list(corr_group_options.keys()), index=0
    )
    corr_filter = corr_group_options[corr_choice]

    if corr_filter is None:
        corr_df = df
        subtitle = "전체 표본"
    else:
        corr_df = df[df["Age_Group"] == corr_filter]
        subtitle = corr_filter

    st.caption(f"선택된 표본 수: **{len(corr_df)}명**")

    if len(corr_df) < 5:
        st.warning("상관계수를 계산하기에 표본 수가 너무 적습니다 (5명 미만).")
    else:
        corr = corr_df[analysis_cols].corr()
        # 요인명 간소화
        corr.index = [c.replace("_Mean", "") for c in corr.index]
        corr.columns = [c.replace("_Mean", "") for c in corr.columns]

        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            title=f"요인 간 상관계수 히트맵 ({subtitle})",
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- [추가] 박스플롯 비교 ---
    st.markdown("---")
    st.header("📦 [추가] 연령대별 요인 분포 (박스플롯)")

    plot_df = df[["Age_Group"] + analysis_cols].dropna()
    long_plot_df = plot_df.melt(
        id_vars="Age_Group", value_vars=analysis_cols, var_name="요인", value_name="점수"
    )
    long_plot_df["요인"] = long_plot_df["요인"].str.replace("_Mean", "")

    fig_box = px.box(
        long_plot_df,
        x="요인",
        y="점수",
        color="Age_Group",
        title="연령대별 요인 점수 분포 (박스플롯)",
        category_orders={"Age_Group": ["20대 초반", "20대 중후반", "30대 이상"]},
        points="outliers",
    )
    fig_box.update_layout(
        xaxis_title="요인",
        yaxis_title="점수 (1~5점)",
        height=500,
        legend_title="연령대",
    )
    st.plotly_chart(fig_box, use_container_width=True)


if __name__ == "__main__":
    main()
