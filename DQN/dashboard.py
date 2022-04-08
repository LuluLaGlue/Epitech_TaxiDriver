import streamlit as st
import pandas as pd

### WOULD BE NICE TO ADD MORE INFO

st.set_page_config(layout="wide")

if __name__ == "__main__":
    data = pd.read_csv("./models.csv", sep=";")
    data["name"] = data["name"].apply(
        lambda x: x.split("DQN_")[1].split(".")[0])
    st.title("Dashboard")
    st.subheader("Parameter's influence on mean reward and win rate")
    st.write(data)
    data_t = data.T
    data_t.columns = data_t.iloc[0]
    data_t = data_t.drop("name")

    data_r = data_t.copy()
    data_r = data_r.iloc[-2]
    data_wr = data_t.copy()
    data_wr = data_wr.iloc[-1]

    st.subheader("Raw Data")
    a, b = st.columns(2)

    with a:
        st.write("Influence of parameters on mean reward")
        st.bar_chart(data_r, height=500)
    with b:
        st.write("Influence of parameters on win rate")
        st.bar_chart(data_wr, height=500)

    st.subheader("Standardized Data")
    c, d = st.columns(2)

    data_r = (data_r - data_r.mean()) / data_r.std()
    data_wr = (data_wr - data_wr.mean()) / data_wr.std()
    with c:
        st.write("Influence of parameters on mean reward")
        st.bar_chart(data_r, height=500)
    with d:
        st.write("Influence of parameters on win rate")
        st.bar_chart(data_wr, height=500)
