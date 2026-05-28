import streamlit as st
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(layout="wide", page_title="KOL Atlas Evaluation")
st.title("🔬 KOL Entity Extraction & Analysis Dashboard")

try:
    
    with open("kol_profiles.json", "r") as f:
        data = json.load(f)
        
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Extracted Profiles & Influence Scores")
        for item in data:
            with st.expander(f"👤 {item['name']} — Influence Score: {item['influence_score']}/100"):
                st.markdown(f"**Affiliation:** {item['affiliation']}")
                st.markdown(f"**Keywords:** {', '.join(item['top_keywords'])}")
                st.markdown(f"**Metrics:** H-Index: {item['h_index']} | Citations: {item['citations']}")
                st.markdown(f"**Source URL:** [Google Scholar Link]({item['source_url']})")
                st.caption(f"**Extraction Confidence:** {item['confidence_scores']}")

    with col2:
        st.subheader("Semantic Similarity Matrix")
        st.image("similarity_matrix.png", use_container_width=True)
        
        
        st.subheader("🤖 LLM Comparative Insight")
        
        api_key = os.getenv("GEMINI_API_KEY")
        
        if st.button("Generate Comparison Summary"):
            if not api_key:
                st.error("API Key not found! Make sure it is in your .env file.")
            else:
                with st.spinner("Gemini is analyzing the profiles..."):
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-3.5-flash')
                        
                        prompt = f"""
                        You are an expert medical science analyst. Compare these two Key Opinion Leaders based on their research fields.
                        Researcher 1: {data[0]['name']} - Fields: {', '.join(data[0]['top_keywords'])}
                        Researcher 2: {data[1]['name']} - Fields: {', '.join(data[1]['top_keywords'])}
                        
                        Write a strict, professional 3-sentence summary detailing where their research overlaps and how they might collaborate.
                        """
                        
                        response = model.generate_content(prompt)
                        st.success("Analysis Complete!")
                        st.write(response.text)
                        
                    except Exception as e:
                        st.error(f"API Error: {e}")
            
except FileNotFoundError:
    st.error("Please run main.py first to generate the JSON and Matrix image!")