import streamlit as st, ollama
MODEL = "gemma3:4b"

st.title("Gemma 3 via Ollama")
prompt = st.text_area("Ask Gemma 3", "Explain CPI vs SPI in 3 lines.")
if st.button("Run"):
    with st.spinner("Thinking..."):
        r = ollama.chat(model=MODEL, messages=[{"role":"user","content": prompt}])
    st.write(r["message"]["content"])
