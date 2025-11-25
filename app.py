import streamlit as st

st.set_page_config(
    page_title="Setup Predictor",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Setup Predictor")
st.markdown("Welcome to your Streamlit app!")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    st.markdown("This is a starter Streamlit application.")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Column 1")
    name = st.text_input("Enter your name", value="")
    if name:
        st.success(f"Hello, {name}!")

with col2:
    st.subheader("Column 2")
    number = st.slider("Select a number", 0, 100, 50)
    st.write(f"You selected: {number}")

# Example of using session state
if "counter" not in st.session_state:
    st.session_state.counter = 0

if st.button("Click me!"):
    st.session_state.counter += 1

st.metric("Button clicks", st.session_state.counter)

