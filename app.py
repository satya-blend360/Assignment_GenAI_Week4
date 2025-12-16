import streamlit as st
import requests

# ❗ TEMPORARY: API key hard-coded (for testing only)
API_KEY = "YOUR_API_KEY_HERE"

st.set_page_config(page_title="Gemini Simple App")

st.title("🤖 Gemini AI – Simple Demo")

prompt = st.text_input(
    "Ask something",
    value="Explain how AI works in a few words"
)

if st.button("Generate"):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.0-flash:generateContent"
    )

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": API_KEY,
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code != 200:
            st.error(response.json().get("error", {}).get("message", "API Error"))
        else:
            data = response.json()
            answer = data["candidates"][0]["content"]["parts"][0]["text"]
            st.success("Response")
            st.write(answer)

    except Exception as e:
        st.error(f"Request failed: {e}")
