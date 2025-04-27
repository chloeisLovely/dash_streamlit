
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.title("🎈 나만의 데이터 시각화 대시보드!")

# 1. 데이터 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요.", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state['data'] = df
    st.write('데이터 미리보기:')
    st.dataframe(df)

    # 2. Plotly로 인터랙티브 시각화
    st.subheader('📊 Plotly 인터랙티브 그래프')
    x = st.selectbox('X축 선택', df.columns)
    y = st.selectbox('Y축 선택', df.columns)
    color = st.selectbox('컬러 그룹 선택', df.columns)

    fig = px.scatter(df, x=x, y=y, color=color, hover_data=df.columns)
    st.plotly_chart(fig)

    # 3. AI 모델 학습 (Linear Regression)
    st.subheader('🤖 모델 학습 및 저장')
    target = st.selectbox('타겟 컬럼 (예측 대상)', df.columns)
    features = st.multiselect('특징 컬럼들 (입력 변수)', [col for col in df.columns if col != target])

    if st.button('모델 학습하기'):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        X = df[features]
        y = df[target]
        model.fit(X, y)
        joblib.dump(model, 'model.pkl')
        st.success('모델 학습 및 저장 완료!')

    # 4. 사용자 입력 → 직접 예측하기
    st.subheader('🧑‍💻 직접 입력해서 예측하기')
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(f'{feature} 입력값', value=0.0)

    if st.button('예측하기'):
        model = joblib.load('model.pkl')
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        st.success(f'예측 결과: {prediction[0]}')

    # 5. 데이터 및 예측 결과 다운로드
    st.subheader('⬇️ 데이터 다운로드')
    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="현재 데이터 CSV 다운로드",
        data=csv,
        file_name='다운로드된_데이터.csv',
        mime='text/csv',
    )
