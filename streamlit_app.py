import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os

CSV_PATH = "exp - 설문지 응답 시트1 (2).csv"


def process_dataframe(df):
    """데이터프레임을 처리하여 요인 계산"""
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

    # 역문항 처리 (알미사.ipynb와 동일)
    scale_max = 5
    reverse_items = ["3-6", "3-12"]

    for item in reverse_items:
        col_name = find_cols([item])
        if col_name:
            c = col_name[0]
            if f"{c}_rev" not in df.columns:
                df[f"{c}_rev"] = (scale_max + 1) - df[c]

    # 7개 요인 정의 (알미사.ipynb와 동일)
    sector_map = {
        "충동구매 성향": find_cols(["3-1", "3-2", "3-3", "3-4", "3-5"])
        + ([f"{find_cols(['3-6'])[0]}_rev"] if find_cols(["3-6"]) else []),
        "사회적 비교": find_cols(["3-7", "3-8", "3-9", "3-10", "3-11"])
        + ([f"{find_cols(['3-12'])[0]}_rev"] if find_cols(["3-12"]) else []),
        "가격 민감도": find_cols(["3-13", "3-14", "3-15", "3-16", "3-17"]),
        "플랫폼 편의성": find_cols(["3-18", "3-19", "3-20", "3-21"]),
        "서비스 신뢰/후회": find_cols(["3-22", "3-23", "3-24", "3-25", "3-26", "3-27"]),
        "예산/자기통제": find_cols(["3-28", "3-29", "3-30", "3-31", "3-32"]),
        "메타광고 종합 반응": find_cols([f"4-{i}" for i in range(1, 28)]),
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


@st.cache_data
def load_data(path: str):
    """파일 경로에서 데이터 로딩"""
    # 현재 스크립트 위치 기준으로 경로 설정
    if not os.path.isabs(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, path)
    
    df = pd.read_csv(path)
    return process_dataframe(df)


def load_data_from_upload(uploaded_file):
    """업로드된 파일에서 데이터 로딩"""
    df = pd.read_csv(uploaded_file)
    return process_dataframe(df)


def main():
    st.set_page_config(page_title="논문 분석 대시보드", layout="wide")

    st.title("📊 소셜커머스 설문 분석 대시보드")
    st.markdown(
        """
        ### 연령대별 7개 요인 비교 분석
        - 충동구매 성향 / 사회적 비교 / 가격 민감도 / 플랫폼 편의성 / 서비스 신뢰·후회 / 예산·자기통제 / **메타광고 종합 반응**
        """
    )

    try:
        df, factor_to_col, age_col, gender_col, education_col = load_data(CSV_PATH)
    except FileNotFoundError as e:
        st.error(f"❌ CSV 파일을 찾을 수 없습니다: `{CSV_PATH}`")
        st.info("현재 작업 디렉토리: " + os.getcwd())
        st.info("스크립트 위치: " + os.path.dirname(os.path.abspath(__file__)))
        
        # 파일 업로드 옵션 제공
        st.markdown("---")
        st.subheader("📁 CSV 파일 직접 업로드")
        uploaded_file = st.file_uploader("설문 데이터 CSV 파일을 업로드하세요", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df, factor_to_col, age_col, gender_col, education_col = load_data_from_upload(uploaded_file)
                st.success("✅ 파일 업로드 성공!")
            except Exception as upload_error:
                st.error(f"파일 처리 중 오류: {upload_error}")
                st.stop()
        else:
            st.stop()
    except Exception as e:
        st.error(f"데이터 로딩 중 오류 발생: {e}")
        st.stop()

    # 전체 6개 요인 (SNS 이용률 포함)
    analysis_factors = list(factor_to_col.keys())
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
        st.metric("분석 요인 수", len(analysis_factors))
    with col3:
        valid_age_count = df["Age_Group"].notna().sum()
        st.metric("유효 연령대 표본", valid_age_count)

    st.markdown("### 📋 연령대별 표본 분포")
    age_counts = df["Age_Group"].value_counts().sort_index()
    
    # 표본 분포를 더 크게 표시
    age_df = age_counts.to_frame("표본 수").reset_index()
    age_df.columns = ["연령대", "표본 수"]
    st.dataframe(age_df, width="stretch", hide_index=True, height=150)

    # --- [1] 연령대별 6개 요인 평균 ---
    st.markdown("---")
    st.header("📊 [1] 연령대별 요인 평균")

    group_means = df.groupby("Age_Group")[analysis_cols].mean().round(2)
    group_means_display = group_means.copy()
    group_means_display.columns = [c.replace("_Mean", "") for c in group_means_display.columns]

    # 표를 더 크고 읽기 쉽게
    st.markdown("#### 📈 평균 점수 요약표")
    try:
        st.dataframe(
            group_means_display.style.format("{:.2f}").background_gradient(cmap="RdYlGn", axis=1),
            width="stretch",
            height=200
        )
    except ImportError:
        # matplotlib 없으면 plain 표 출력
        st.dataframe(
            group_means_display.style.format("{:.2f}"),
            width="stretch",
            height=200
        )

    # 막대 그래프로 시각화 (크고 명확하게)
    st.markdown("#### 📊 시각화: 연령대별 요인 비교")
    long_df = group_means_display.reset_index().melt(
        id_vars="Age_Group", var_name="요인", value_name="평균 점수"
    )

    fig_bar = px.bar(
        long_df,
        x="요인",
        y="평균 점수",
        color="Age_Group",
        barmode="group",
        title="<b>연령대별 요인 평균 비교</b>",
        category_orders={"Age_Group": ["20대 초반", "20대 중후반", "30대 이상"]},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_bar.update_layout(
        xaxis_title="<b>요인</b>",
        yaxis_title="<b>평균 점수 (1~5점)</b>",
        height=550,
        legend_title="<b>연령대</b>",
        font=dict(size=16),
        title_font_size=20,
        xaxis=dict(tickfont=dict(size=14)),
        yaxis=dict(tickfont=dict(size=14)),
        legend=dict(font=dict(size=14)),
    )
    st.plotly_chart(fig_bar, width="stretch")

    # --- [2] 각 연령대별 상위 3개 요인 + 전체 순위 ---
    st.markdown("---")
    st.header("🏆 [2] 각 연령대별 상위 3개 요인")

    for group in ["20대 초반", "20대 중후반", "30대 이상"]:
        if group not in group_means.index:
            continue

        row = group_means.loc[group]
        renamed = row.rename(
            index={factor_to_col[name]: name for name in analysis_factors}
        )
        sorted_row = renamed.sort_values(ascending=False)
        top3 = sorted_row.head(3)

        st.markdown(f"### 📍 {group}")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### 🥇 상위 3개 요인")
            for i, (fname, val) in enumerate(top3.items(), start=1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                st.markdown(f"### {medal} **{i}위. {fname}**")
                st.markdown(f"#### `{val:.2f}점`")
                st.markdown("")

        with col2:
            st.markdown("#### 📊 전체 요인 순위 (내림차순)")
            rank_df = pd.DataFrame(
                {
                    "순위": range(1, len(sorted_row) + 1),
                    "요인": sorted_row.index,
                    "평균 점수": sorted_row.values,
                }
            )
            try:
                st.dataframe(
                    rank_df.style.format({"평균 점수": "{:.2f}"}).background_gradient(
                        subset=["평균 점수"], cmap="YlGn"
                    ),
                    hide_index=True,
                    width="stretch",
                    height=320
                )
            except ImportError:
                st.dataframe(
                    rank_df.style.format({"평균 점수": "{:.2f}"}),
                    hide_index=True,
                    width="stretch",
                    height=320
                )
        
        st.markdown("---")

    # --- [3] 연령대별 광고 개수 평균 ---
    st.markdown("---")
    st.header("📺 [3] 연령대별 광고 개수 평균")
    st.caption("릴스 30개를 볼 때 노출되는 메타 광고 개수 ([2-6] 문항)")

    if "Ad_Count_Numeric" in df.columns:
        ad_means = df.groupby("Age_Group")["Ad_Count_Numeric"].mean().round(2)
        ad_means_df = ad_means.to_frame("평균 광고 개수").reset_index()
        ad_means_df.columns = ["연령대", "평균 광고 개수"]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 📊 평균값 요약")
            try:
                st.dataframe(
                    ad_means_df.style.format({"평균 광고 개수": "{:.2f}개"}).background_gradient(
                        subset=["평균 광고 개수"], cmap="Blues"
                    ),
                    width="stretch",
                    hide_index=True,
                    height=180
                )
            except ImportError:
                st.dataframe(
                    ad_means_df.style.format({"평균 광고 개수": "{:.2f}개"}),
                    width="stretch",
                    hide_index=True,
                    height=180
                )

        with col2:
            # 막대 그래프 (크고 명확하게)
            fig_ad = px.bar(
                ad_means_df,
                x="연령대",
                y="평균 광고 개수",
                title="<b>연령대별 평균 광고 개수</b>",
                color="연령대",
                category_orders={"연령대": ["20대 초반", "20대 중후반", "30대 이상"]},
                color_discrete_sequence=px.colors.qualitative.Pastel,
                text="평균 광고 개수",
            )
            fig_ad.update_traces(texttemplate='%{text:.2f}개', textposition='outside', textfont_size=16)
            fig_ad.update_layout(
                xaxis_title="<b>연령대</b>",
                yaxis_title="<b>평균 광고 개수 (개)</b>",
                height=450,
                showlegend=False,
                font=dict(size=16),
                title_font_size=20,
                xaxis=dict(tickfont=dict(size=14)),
                yaxis=dict(tickfont=dict(size=14)),
            )
            st.plotly_chart(fig_ad, width="stretch")
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
        "📌 상관계수를 계산할 집단 선택",
        options=list(corr_group_options.keys()),
        index=0,
    )
    corr_filter = corr_group_options[corr_choice]

    if corr_filter is None:
        corr_df = df
        subtitle = "전체 표본"
    else:
        corr_df = df[df["Age_Group"] == corr_filter]
        subtitle = corr_filter

    st.info(f"📊 선택된 표본 수: **{len(corr_df)}명**")

    if len(corr_df) < 5:
        st.warning("⚠️ 상관계수를 계산하기에 표본 수가 너무 적습니다 (5명 미만).")
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
            title=f"<b>요인 간 상관계수 히트맵 ({subtitle})</b>",
        )
        fig_corr.update_layout(
            height=600,
            font=dict(size=16),
            title_font_size=20,
            xaxis=dict(tickfont=dict(size=14)),
            yaxis=dict(tickfont=dict(size=14)),
        )
        fig_corr.update_traces(textfont_size=14)
        st.plotly_chart(fig_corr, width="stretch")

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
    st.plotly_chart(fig_box, width="stretch")


if __name__ == "__main__":
    main()
