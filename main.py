import streamlit as st

st.title("Qurio Streamlit App")
st.write("Hello from Streamlit running on Azure Web App!")

name = st.text_input("Enter your name:")
if name:
    st.success(f"Welcome, {name}!")
