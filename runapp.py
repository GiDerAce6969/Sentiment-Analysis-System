import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import json
import time # Import the time module for handling delays

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

def analyze_comments_in_batches(df_comments, comment_column, batch_size=100):
    """
    Analyzes comments in batches with robust error handling and retries to ensure completion.
    """
    # MODIFICATION: Changed the model name to the requested "2.5 Pro" version.
    # WARNING: This model name is speculative and will not work until Google releases it.
    # Use 'gemini-1.5-pro-latest' for current functionality.
    model = genai.GenerativeModel('gemini-2.5-pro')
    
    all_analyzed_data = []

    system_prompt = """
    You are a high-performance, batch-processing AI data analyst. Your task is to receive a list of numbered user comments and return a JSON array where each object corresponds to the analysis of a single comment.

    Your final output MUST be a single, clean JSON array containing one object for each comment I provide. Provide NO text outside this JSON array.

    OUTPUT STRUCTURE FOR EACH COMMENT:
    {
      "comment_index": <integer, the original index of the comment I provide>,
      "language": "<string, e.g., 'Malay', 'English', 'Chinese', 'Mixed'>",
      "sentiment_label": "<string, 'Positive', 'Negative', 'Neutral'>",
      "sentiment_score": <float, from -1.0 to 1.0>,
      "primary_topic": "<string, the main subject of the comment>",
      "mentioned_party": "<string, political party if mentioned, otherwise null>",
      "mentioned_leader": "<string, political leader if mentioned, otherwise null>"
    }

    YOUR ANALYSIS LOGIC FOR EACH COMMENT:
    1.  Language Identification: Determine the primary language.
    2.  Sentiment Analysis: Assign a `sentiment_score` and `sentiment_label`.
    3.  Topic Identification: Concisely identify the single `primary_topic`.
    4.  Strict Entity Recognition: Scan for explicit mentions of Malaysian political parties or leaders. If a name is mentioned, populate the field. Otherwise, it MUST be `null`.
    """

    comments_to_process = df_comments[comment_column].dropna().astype(str).tolist()
    total_comments = len(comments_to_process)
    num_batches = (total_comments + batch_size - 1) // batch_size

    # Add a progress bar and status text
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(0, total_comments, batch_size):
        batch = comments_to_process[i:i + batch_size]
        
        formatted_batch = "\n".join([f'{i+j}: "{comment}"' for j, comment in enumerate(batch)])
        full_prompt = f"{system_prompt}\n\nAnalyze the following batch of comments. Ensure your output array has exactly {len(batch)} objects, one for each comment index from {i} to {i + len(batch) - 1}.\n---\n{formatted_batch}"
        
        # Implement retry logic
        retries = 3
        delay = 5  # Start with a 5-second delay
        
        for attempt in range(retries):
            try:
                current_batch_num = i // batch_size + 1
                status_text.info(f"Analyzing batch {current_batch_num}/{num_batches} ({len(batch)} comments)... Attempt {attempt + 1}/{retries}")
                
                response = model.generate_content(full_prompt)
                cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
                batch_results = json.loads(cleaned_response)
                
                if len(batch_results) != len(batch):
                    st.warning(f"Warning: AI returned {len(batch_results)} results for a batch of {len(batch)} comments (Batch {current_batch_num}). Results for this batch may be misaligned.")
                
                all_analyzed_data.extend(batch_results)
                
                # Update progress bar
                progress_bar.progress((i + len(batch)) / total_comments)
                
                break # If successful, break the retry loop

            except Exception as e:
                status_text.error(f"Error on batch {current_batch_num}, attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    status_text.warning(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    st.error(f"Batch {current_batch_num} failed after {retries} attempts. Skipping this batch and continuing.")
                    break # Stop retrying this batch and move to the next
    
    status_text.success("All batches processed!")

    if not all_analyzed_data:
        st.error("No data was successfully analyzed. Please check the errors above.")
        return None

    try:
        df_analyzed = pd.DataFrame(all_analyzed_data)
        df_comments_reset = df_comments.reset_index(drop=True)
        df_final = df_comments_reset.join(df_analyzed.set_index('comment_index'))
        return df_final
    except Exception as e:
        st.error(f"Failed to merge original data with AI results: {e}")
        return None


# --- Streamlit App UI ---

st.title("📊 Social Media Sentiment Analysis Engine")
st.markdown("Upload a CSV or Excel file with comments to begin analysis.")

# Initialize session state variables
if 'df_analyzed' not in st.session_state:
    st.session_state['df_analyzed'] = None
if 'df_original' not in st.session_state:
    st.session_state['df_original'] = None

uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_original = pd.read_csv(uploaded_file)
        else:
            df_original = pd.read_excel(uploaded_file)
        st.session_state['df_original'] = df_original
        # Clear previous analysis results when a new file is uploaded
        st.session_state['df_analyzed'] = None 
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.session_state['df_original'] = None

if st.session_state['df_original'] is not None:
    df_original = st.session_state['df_original']
    st.success(f"Successfully loaded {len(df_original)} rows from `{uploaded_file.name}`.")

    st.markdown("### 1. Select the column that contains the comments:")
    available_columns = df_original.columns.tolist()
    comment_column = st.selectbox(
        "Select Comment Column",
        options=available_columns,
        index=0
    )
    
    st.markdown("### 2. Configure Analysis Batch Size")
    batch_size = st.slider(
        "Comments per API Call (Batch Size)", 
        min_value=10, 
        max_value=200, 
        value=100, 
        step=10,
        help="Higher values use fewer API calls (cheaper) but may take longer per call. Lower values are faster for smaller datasets but cost more."
    )
    
    st.info(f"You have selected **'{comment_column}'** as the comment column. The analysis will run in batches of **{batch_size}**.")

    if st.button("🚀 Analyze Comments with Gemini AI", type="primary"):
        df_analyzed = analyze_comments_in_batches(df_original, comment_column, batch_size)

        if df_analyzed is not None:
            st.session_state['df_analyzed'] = df_analyzed
            st.success("✅ Analysis complete!")
            st.rerun() # Rerun the script to display the dashboard immediately
        else:
            st.error("Analysis failed to complete. Please check the error messages above.")

if st.session_state['df_analyzed'] is not None:
    df = st.session_state['df_analyzed']
    
    st.header("📈 Dashboard")
    
    total_comments = len(df)
    avg_sentiment = df['sentiment_score'].dropna().mean()
    
    col1, col2 = st.columns(2)
    col1.metric("Total Comments Analyzed", f"{total_comments:,}")
    col2.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")

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
        
    st.subheader("Analysis by Topic")
    topic_sentiment = df.dropna(subset=['primary_topic']).groupby('primary_topic').agg(
        comment_count=('primary_topic', 'count'),
        avg_sentiment=('sentiment_score', 'mean')
    ).sort_values(by='comment_count', ascending=False).reset_index()

    fig_topics = px.bar(topic_sentiment.head(10), x='primary_topic', y='comment_count', color='avg_sentiment',
                        color_continuous_scale=px.colors.diverging.RdYlGn, color_continuous_midpoint=0,
                        title="Top 10 Topics by Mention Count (Colored by Sentiment)",
                        labels={'primary_topic': 'Topic', 'comment_count': 'Number of Comments', 'avg_sentiment': 'Avg. Sentiment'})
    st.plotly_chart(fig_topics, use_container_width=True)

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

    st.subheader("Explore the Full Analyzed Data")
    st.dataframe(df)
