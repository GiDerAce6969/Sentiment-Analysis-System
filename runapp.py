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
def analyze_comments_with_gemini(df_comments, comment_column):
    """
    Sends comments to Gemini and expects a structured JSON array as a response.
    This function uses a prompt designed for row-by-row analysis.
    """
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    
    # MODIFICATION: Use the user-selected column
    comment_list_str = "\n".join(df_comments[comment_column].dropna().astype(str).tolist())

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
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
        analyzed_data = json.loads(cleaned_response)
        
        # Combine original data with analyzed data
        df_analyzed = pd.DataFrame(analyzed_data)
        
        # Reset index to ensure a clean join
        df_comments_reset = df_comments.reset_index(drop=True)
        df_analyzed_reset = df_analyzed.reset_index(drop=True)
        
        df_final = pd.concat([df_comments_reset, df_analyzed_reset], axis=1)
        
        return df_final
    except Exception as e:
        st.error(f"An error occurred during AI analysis: {e}")
        # MODIFICATION: Show the raw response to help debug Gemini issues
        if 'response' in locals() and hasattr(response, 'text'):
            st.error(f"Gemini's raw response was: {response.text}")
        return None

# --- Streamlit App UI ---

st.title("📊 Social Media Sentiment Analysis Engine")
st.markdown("Upload a CSV or Excel file with comments to begin analysis.")

# 1. MODIFICATION: File Uploader to accept CSV and Excel
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])
comment_column = None

if uploaded_file is not None:
    # MODIFICATION: Load the user's data based on file type
    try:
        if uploaded_file.name.endswith('.csv'):
            df_original = pd.read_csv(uploaded_file)
        else:
            df_original = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
        
    st.success(f"Successfully loaded {len(df_original)} rows from `{uploaded_file.name}`.")

    # 2. MODIFICATION: Let the user select the comment column
    st.markdown("### Please select the column that contains the comments:")
    available_columns = df_original.columns.tolist()
    comment_column = st.selectbox(
        "Select Comment Column",
        options=available_columns,
        index=0 # Default to the first column
    )
    
    st.info(f"You have selected **'{comment_column}'** as the comment column.")

    # 3. Analysis Trigger Button
    if st.button("🚀 Analyze Comments with Gemini AI", type="primary"):
        with st.spinner("🧠 Gemini is analyzing the comments... This may take a moment."):
            # Perform the analysis
            df_analyzed = analyze_comments_with_gemini(df_original, comment_column)

        if df_analyzed is not None:
            st.session_state['df_analyzed'] = df_analyzed # Save to session state
            st.success("✅ Analysis complete!")

# 4. Display Dashboard if data is available in session state
if 'df_analyzed' in st.session_state:
    df = st.session_state['df_analyzed']
    
    # The rest of the dashboard code remains the same...
    st.header("📈 Dashboard")

    # --- Row 1: Key Metrics ---
    total_comments = len(df)
    avg_sentiment = df['sentiment_score'].mean()
    
    col1, col2 = st.columns(2)
    col1.metric("Total Comments Analyzed", f"{total_comments:,}")
    col2.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")

    # --- Row 2: Sentiment & Language Distribution ---
    st.subheader("Sentiment and Language Breakdown")
    col1, col2 = st.columns(2)

    with col1:
        sentiment_counts = df['sentiment_label'].value_counts()
        fig_sentiment = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index, 
                               title="Overall Sentiment Distribution", color=sentiment_counts.index,
                               color_discrete_map={'Positive':'green', 'Negative':'red', 'Neutral':'grey'})
        st.plotly_chart(fig_sentiment, use_container_width=True)

    with col2:
        language_counts = df['language'].value_counts()
        fig_lang = px.bar(language_counts, x=language_counts.index, y=language_counts.values,
                          title="Language Distribution", labels={'x': 'Language', 'y': 'Number of Comments'})
        st.plotly_chart(fig_lang, use_container_width=True)
        
    # --- Row 3: Topic Analysis ---
    st.subheader("Analysis by Topic")
    topic_sentiment = df.groupby('primary_topic').agg(
        comment_count=('primary_topic', 'count'),
        avg_sentiment=('sentiment_score', 'mean')
    ).sort_values(by='comment_count', ascending=False).reset_index()

    fig_topics = px.bar(topic_sentiment.head(10), x='primary_topic', y='comment_count', color='avg_sentiment',
                        color_continuous_scale=px.colors.diverging.RdYlGn, color_continuous_midpoint=0,
                        title="Top 10 Topics by Mention Count (Colored by Sentiment)",
                        labels={'primary_topic': 'Topic', 'comment_count': 'Number of Comments', 'avg_sentiment': 'Avg. Sentiment'})
    st.plotly_chart(fig_topics, use_container_width=True)

    # --- Row 4: Political Analysis ---
    st.subheader("Political Analysis (Based on Explicit Mentions)")
    col1, col2 = st.columns(2)

    with col1:
        party_sentiment = df.dropna(subset=['mentioned_party']).groupby('mentioned_party').agg(
            mentions=('mentioned_party', 'count'), avg_sentiment=('sentiment_score', 'mean')
        ).sort_values(by='mentions', ascending=False).reset_index()
        
        fig_party = px.bar(party_sentiment, x='mentioned_party', y='mentions', color='avg_sentiment',
                           color_continuous_scale=px.colors.diverging.RdYlGn, color_continuous_midpoint=0,
                           title="Political Party Mentions & Sentiment",
                           labels={'mentioned_party': 'Party', 'mentions': 'Number of Mentions'})
        st.plotly_chart(fig_party, use_container_width=True)
    
    with col2:
        leader_sentiment = df.dropna(subset=['mentioned_leader']).groupby('mentioned_leader').agg(
            mentions=('mentioned_leader', 'count'), avg_sentiment=('sentiment_score', 'mean')
        ).sort_values(by='mentions', ascending=False).reset_index()
        
        fig_leader = px.bar(leader_sentiment, x='mentioned_leader', y='mentions', color='avg_sentiment',
                            color_continuous_scale=px.colors.diverging.RdYlGn, color_continuous_midpoint=0,
                            title="Political Leader Mentions & Sentiment",
                            labels={'mentioned_leader': 'Leader', 'mentions': 'Number of Mentions'})
        st.plotly_chart(fig_leader, use_container_width=True)

    # --- Data Explorer ---
    st.subheader("Explore the Full Analyzed Data")
    # MODIFICATION: Display the combined DataFrame
    st.dataframe(df)
