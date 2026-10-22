import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import streamlit as st
import pandas as pd
from pipeline.spec_sensor.sensor import SpecSensor

st.set_page_config(page_title="SpecSense App", layout="wide")
st.title("🛠️ SpecSense AI – Intelligent Product Recommendations")

sensor = SpecSensor()

st.header("Analyze the Product Query")
user_input = st.text_area(
    "Enter your product requirement:",
    height=150,
    placeholder="Example: Need a lightweight laptop under ₹30k for coding and battery life..."
)
    
if st.button("Analyze Query", type="primary"):
    if user_input.strip():
        with st.spinner("Analyzing query..."):
            try:
                result = sensor.detect(user_input)

                st.subheader("Recommended Product")
                st.success(result.get("recommended_product", "Unknown"))

                st.subheader("Meets")
                for item in result.get("meets", []):
                    st.write(f"✅ {item}")

                st.subheader("Partially Meets")
                for item in result.get("partially_meets", []):
                    st.write(f"⚠️ {item}")

                st.subheader("Trade-offs")
                for item in result.get("trade_offs", []):
                    st.write(f"❌ {item}")

                st.subheader("Customer Reviews")
                for review in result.get("review_snippets", []):
                    st.write(f"💬 {review}")

                st.subheader("Alternatives")
                for alt in result.get("alternatives", []):
                    st.write(f"🔄 {alt}")

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    else:
        st.warning("Please enter a product query to analyze.")

