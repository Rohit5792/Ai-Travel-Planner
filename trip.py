import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ──────────────────────────────
# Load environment variables
# ──────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ──────────────────────────────
# Streamlit UI
# ──────────────────────────────
st.set_page_config(page_title="AI Travel Planner", page_icon="🧳")
st.title("🧭 AI Multi-Day Travel Itinerary Planner")
st.write("Plan your trip itinerary by entering your city, interests, and number of days you'd like to travel.")

# ──────────────────────────────
# Form input section
# ──────────────────────────────
with st.form("planner_form"):
    city = st.text_input("Enter the city name for your trip")
    interests = st.text_input("Enter your interests (comma-separated)")
    num_days = st.number_input("Number of days for your trip", min_value=1, max_value=30, value=1)
    submitted = st.form_submit_button("Generate Itinerary")

# ──────────────────────────────
# Itinerary generation logic
# ──────────────────────────────
if submitted:
    if not GROQ_API_KEY:
        st.error("❌ Missing GROQ_API_KEY. Please set it in your environment or .env file.")
    elif not city or not interests:
        st.warning("⚠️ Please fill in City and Interests fields.")
    else:
        try:
            # Initialize Groq LLM
            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model="llama-3.3-70b-versatile",
                temperature=0.4
            )

            # Define the itinerary prompt
            itinerary_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a helpful travel assistant. Create a {num_days}-day travel itinerary for {city}, "
                 "tailored to the user's interests: {interests}. "
                 "For each day, list morning, afternoon, and evening activities with time slots. "
                 "At the end of each day, suggest a suitable area or neighborhood for an overnight stay."),
                ("human", "Plan my {num_days}-day trip.")
            ])

            # Generate itinerary
            response = llm.invoke(
                itinerary_prompt.format_messages(city=city, interests=interests, num_days=num_days)
            )


            # Display result
            st.subheader("📅 Your Multi-Day Itinerary")
            st.markdown(response.content)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
