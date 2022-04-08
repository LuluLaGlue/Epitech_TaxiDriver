from PIL import Image
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
    data_t = data.T
    data_t.columns = data_t.iloc[0]
    data_t = data_t.drop("name")

    data_r = data_t.copy()
    data_r = data_r.iloc[-2]
    data_wr = data_t.copy()
    data_wr = data_wr.iloc[-1]

    st.subheader("Raw Reward and Win Rate")
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

    st.title("Training Graphs")

    model_selector = st.selectbox("Select a Param",
                                  ("Reference", "Batch Size", "Epsilon",
                                   "Gamma", "Learning Rate", "Update Rate"))
    if model_selector == "Reference":
        image = [
            "./models/reference/DQN_reference_graph.png",
            "./models/reference_2/DQN_reference_2_graph.png"
        ]
    elif model_selector == "Batch Size":
        image = [
            "./models/batch_low/DQN_batch_low_graph.png",
            "./models/batch_high/DQN_batch_high_graph.png"
        ]
    elif model_selector == "Epsilon":
        image = [
            "./models/eps_low/DQN_eps_low_graph.png",
            "./models/eps_high/DQN_eps_high_graph.png"
        ]
    elif model_selector == "Gamma":
        image = ["./models/gamma_low/DQN_gamma_low_graph.png"]
    elif model_selector == "Learning Rate":
        image = [
            "./models/lr_low/DQN_lr_low_graph.png",
            "./models/lr_high/DQN_lr_high_graph.png",
            "./models/lr_const/DQN_lr_const_graph.png"
        ]
    elif model_selector == "Update Rate":
        image = [
            "./models/update_low/DQN_update_low_graph.png",
            "./models/update_high/DQN_update_high_graph.png"
        ]
    else:
        image = []

    if len(image) > 0:
        cols = st.columns(len(image))

        for index, col in enumerate(cols):
            with col:
                current_image = Image.open(image[index])
                st.write(image[index].split("DQN_")[1].split(".")[0].split(
                    "_graph")[0])
                st.image(current_image)
    else:
        st.error("No image found")

    st.subheader("Raw Data")
    st.write(data)
