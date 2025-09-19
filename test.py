import streamlit as st
st.title("Test Hello App")
st.write("Dette er en test – virker Streamlit?")
import streamlit as st

st.title("Hello Streamlit 👋")
st.write("Hvis du kan læse dette, så virker Streamlit korrekt!")

x = st.slider("Vælg et tal", 0, 100, 50)
st.write("Du valgte:", x)
