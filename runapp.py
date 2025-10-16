import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import json

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Google Gemini API Configuration ---
# Use Streamlit's secrets management for the API key
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("⚠️ Google API Key not found. Please add it to your Streamlit secrets.")

# --- Helper Functions ---

# Function to call Gemini and parse the structured data
def analyze_comments_with_gemini(df_comments):
    """
    Sends comments to Gemini and expects a structured JSON array as a response.
    This function uses a prompt designed for row-by-row analysis.
    """
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    
    # Create a string of all comments to send in one batch
    # Assuming the user's CSV has a 'comment_text' column
    comment_list_str = "\n".join(df_comments['comment_text'].dropna().tolist())

    system_prompt = """
    You are a high-performance data processing AI. Your sole function is to receive a list of user comments, analyze each one, and return an array of structured JSON objects.

    For every single comment you process, you MUST return a single, clean JSON object with the following exact keys. Your final output MUST be a JSON array of these objects. Provide NO text outside the JSON array.

    OUTPUT STRUCTURE FOR EACH COMMENT:
    {
      "language": "<string, e.g., 'Malay', 'English', 'Chinese', 'Mixed'>",
      "sentiment_label": "<string, 'Positive', 'Negative', 'Neutral'>",
      "sentiment_score": <float, from -1.0 to 1.0>,
      "primary_topic": "<string, the main subject of the comment>",
      "mentioned_party": "<string, political party if mentioned, otherwise null>",
      "mentioned_leader": "<string, political leader if mentioned, otherwise null>"
    }

    YOUR ANALYSIS LOGIC FOR EACH COMMENT:
    1.  Language Identification: Determine the primary language.
    2.  Sentiment Analysis: Assign a `sentiment_score` from -1.0 to 1.0 and a `sentiment_label` ('Positive', 'Negative', 'Neutral').
    3.  Topic Identification: Concisely identify the single `primary_topic`.
    4.  Strict Entity Recognition: Scan for explicit mentions of Malaysian political parties or leaders. If found, populate the relevant field. Otherwise, it MUST be `null`.
    """

    full_prompt = f"{system_prompt}\n\nAnalyze the following comments:\n---\n{comment_list_str}"
    
    try:
        response = model.generate_content(full_prompt)
        # Clean up the response to ensure it's valid JSON
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
        analyzed_data = json.loads(cleaned_response)
        return pd.DataFrame(analyzed_data)
    except Exception as e:
        st.error(f"An error occurred during AI analysis: {e}")
        st.error(f"Gemini's raw response was: {response.text}")
        return None

# --- Streamlit App UI ---

st.title("📊 Social Media Sentiment Analysis Engine")
st.markdown("Upload a CSV file with a `comment_text` column to begin analysis.")

# 1. File Uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the user's data
    try:
        df_original = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    if 'comment_text' not in df_original.columns:
        st.error("Error: The CSV file must contain a column named 'comment_text'.")
        st.stop()

    st.success(f"Successfully loaded {len(df_original)} comments. Click the button below to start the analysis.")

    # 2. Analysis Trigger Button
    if st.button("🚀 Analyze Comments with Gemini AI", type="primary"):
        with st.spinner("🧠 Gemini is analyzing the comments... This may take a moment."):
            # Perform the analysis
            df_analyzed = analyze_comments_with_gemini(df_original)

        if df_analyzed is not None:
            st.session_state['df_analyzed'] = df_analyzed # Save to session state
            st.success("✅ Analysis complete!")

# 3. Display Dashboard if data is available in session state
if 'df_analyzed' in st.session_state:
    df = st.session_state['df_analyzed']

    st.header("📈 Dashboard")

    # --- Row 1: Key Metrics ---
    total_comments = len(df)
    avg_sentiment = df['sentiment_score'].mean()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Comments Analyzed", f"{total_comments:,}")
    col2.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")

    # --- Row 2: Sentiment & Language Distribution ---
    st.subheader("Sentiment and Language Breakdown")
    col1, col2 = st.columns(2)

    with col1:
        sentiment_counts = df['sentiment_label'].value_counts()
        fig_sentiment = px.pie(
            sentiment_counts, 
            values=sentiment_counts.values, 
            names=sentiment_counts.index, 
            title="Overall Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map={'Positive':'green', 'Negative':'red', 'Neutral':'grey'}
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

    with col2:
        language_counts = df['language'].value_counts()
        fig_lang = px.bar(
            language_counts, 
            x=language_counts.index, 
            y=language_counts.values, 
            title="Language Distribution",
            labels={'x': 'Language', 'y': 'Number of Comments'}
        )
        st.plotly_chart(fig_lang, use_container_width=True)
        
    # --- Row 3: Topic Analysis ---
    st.subheader("Analysis by Topic")
    topic_sentiment = df.groupby('primary_topic').agg(
        comment_count=('primary_topic', 'count'),
        avg_sentiment=('sentiment_score', 'mean')
    ).sort_values(by='comment_count', ascending=False).reset_index()

    fig_topics = px.bar(
        topic_sentiment.head(10), # Show top 10 topics
        x='primary_topic',
        y='comment_count',
        color='avg_sentiment',
        color_continuous_scale=px.colors.diverging.RdYlGn,
        color_continuous_midpoint=0,
        title="Top 10 Topics by Mention Count (Colored by Sentiment)",
        labels={'primary_topic': 'Topic', 'comment_count': 'Number of Comments', 'avg_sentiment': 'Avg. Sentiment'}
    )
    st.plotly_chart(fig_topics, use_container_width=True)

    # --- Row 4: Political Analysis ---
    st.subheader("Political Analysis (Based on Explicit Mentions)")
    col1, col2 = st.columns(2)

    with col1:
        party_sentiment = df.dropna(subset=['mentioned_party']).groupby('mentioned_party').agg(
            mentions=('mentioned_party', 'count'),
            avg_sentiment=('sentiment_score', 'mean')
        ).sort_values(by='mentions', ascending=False).reset_index()
        
        fig_party = px.bar(
            party_sentiment,
            x='mentioned_party',
            y='mentions',
            color='avg_sentiment',
            color_continuous_scale=px.colors.diverging.RdYlGn,
            color_continuous_midpoint=0,
            title="Political Party Mentions & Sentiment",
            labels={'mentioned_party': 'Party', 'mentions': 'Number of Mentions'}
        )
        st.plotly_chart(fig_party, use_container_width=True)
    
    with col2:
        leader_sentiment = df.dropna(subset=['mentioned_leader']).groupby('mentioned_leader').agg(
            mentions=('mentioned_leader', 'count'),
            avg_sentiment=('sentiment_score', 'mean')
        ).sort_values(by='mentions', ascending=False).reset_index()
        
        fig_leader = px.bar(
            leader_sentiment,
            x='mentioned_leader',
            y='mentions',
            color='avg_sentiment',
            color_continuous_scale=px.colors.diverging.RdYlGn,
            color_continuous_midpoint=0,
            title="Political Leader Mentions & Sentiment",
            labels={'mentioned_leader': 'Leader', 'mentions': 'Number of Mentions'}
        )
        st.plotly_chart(fig_leader, use_container_width=True)

    # --- Data Explorer ---
    st.subheader("Explore the Analyzed Data")
    st.dataframe(df)