import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import json
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Google Gemini API Configuration ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("⚠️ Google API Key not found. Please add it to your Streamlit secrets.")

# --- Helper Functions (No changes here) ---
def analyze_comments_in_batches(df_comments, comment_column, batch_size=100):
    # This function remains the same as the previous robust version
    # It uses 'gemini-1.5-pro-latest' for current functionality
    model = genai.GenerativeModel('gemini-2.5-pro')
    
    all_analyzed_data = []
    system_prompt = """
    You are a high-performance, batch-processing AI data analyst. Your task is to receive a list of numbered user comments and return a JSON array where each object corresponds to the analysis of a single comment.
    Your final output MUST be a single, clean JSON array containing one object for each comment I provide. Provide NO text outside this JSON array.
    OUTPUT STRUCTURE FOR EACH COMMENT:
    { "comment_index": <integer>, "language": "<string>", "sentiment_label": "<string>", "sentiment_score": <float>, "primary_topic": "<string>", "mentioned_party": <string, null if none>, "mentioned_leader": <string, null if none> }
    YOUR ANALYSIS LOGIC FOR EACH COMMENT:
    1. Language Identification.
    2. Sentiment Analysis (score and label).
    3. Topic Identification.
    4. Strict Entity Recognition (only explicit mentions).
    """
    comments_to_process = df_comments[comment_column].dropna().astype(str).tolist()
    total_comments = len(comments_to_process)
    num_batches = (total_comments + batch_size - 1) // batch_size
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(0, total_comments, batch_size):
        batch = comments_to_process[i:i + batch_size]
        formatted_batch = "\n".join([f'{i+j}: "{comment}"' for j, comment in enumerate(batch)])
        full_prompt = f"{system_prompt}\n\nAnalyze the following batch:\n---\n{formatted_batch}"
        retries = 3
        delay = 5
        for attempt in range(retries):
            try:
                current_batch_num = i // batch_size + 1
                status_text.info(f"Analyzing batch {current_batch_num}/{num_batches}... Attempt {attempt + 1}")
                response = model.generate_content(full_prompt)
                cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
                batch_results = json.loads(cleaned_response)
                all_analyzed_data.extend(batch_results)
                progress_bar.progress((i + len(batch)) / total_comments)
                break
            except Exception as e:
                status_text.error(f"Error on batch {current_batch_num}, attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    st.error(f"Batch {current_batch_num} failed after {retries} attempts. Skipping.")
                    break
    
    status_text.success("All batches processed!")
    if not all_analyzed_data: return None
    try:
        df_analyzed = pd.DataFrame(all_analyzed_data)
        df_comments_reset = df_comments.reset_index(drop=True)
        df_final = df_comments_reset.join(df_analyzed.set_index('comment_index'))
        return df_final
    except Exception as e:
        st.error(f"Failed to merge results: {e}")
        return None

# --- Streamlit App UI (MODIFIED FOR ROBUSTNESS) ---

st.title("📊 Social Media Sentiment Analysis Engine")
st.markdown("Upload a CSV or Excel file with comments to begin analysis.")

# Function to clear old results when a new file is uploaded
def clear_results():
    if 'df_analyzed' in st.session_state:
        del st.session_state['df_analyzed']
    if 'df_original' in st.session_state:
        del st.session_state['df_original']

uploaded_file = st.file_uploader(
    "Choose a file", 
    type=['csv', 'xlsx', 'xls'],
    on_change=clear_results # This callback clears state on new upload
)

# This block now only runs if a file is actively uploaded.
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.session_state['df_original'] = df
        st.success(f"Successfully loaded {len(df)} rows from `{uploaded_file.name}`.")

    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.session_state.pop('df_original', None)

# This block is now separate and only depends on session state.
# This makes it safe during script re-runs.
if 'df_original' in st.session_state and st.session_state['df_original'] is not None:
    df_original = st.session_state['df_original']

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
        max_value=50, # Reduced for stability on free hosting
        value=25, 
        step=5,
        help="Smaller values are more stable on platforms with limited memory. Start small."
    )
    
    st.info(f"You have selected **'{comment_column}'** as the comment column. The analysis will run in batches of **{batch_size}**.")

    if st.button("🚀 Analyze Comments with Gemini AI", type="primary"):
        df_analyzed = analyze_comments_in_batches(df_original, comment_column, batch_size)

        if df_analyzed is not None:
            st.session_state['df_analyzed'] = df_analyzed
            st.success("✅ Analysis complete!")
            st.rerun() 
        else:
            st.error("Analysis failed to complete. Please check the error messages above.")

# This final block for displaying the dashboard is also now safe.
if 'df_analyzed' in st.session_state and st.session_state['df_analyzed'] is not None:
    df_result = st.session_state['df_analyzed']
    
    st.header("📈 Dashboard")
    
    # ... (The entire dashboard display code is the same as before)
    # ... (Metrics, Pie charts, Bar charts, Dataframe display)
    total_comments = len(df_result)
    avg_sentiment = df_result['sentiment_score'].dropna().mean()
    
    col1, col2 = st.columns(2)
    col1.metric("Total Comments Analyzed", f"{total_comments:,}")
    col2.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")

    # ... (Paste the rest of your charting code here)
    st.subheader("Sentiment and Language Breakdown")
    col1, col2 = st.columns(2)

    with col1:
        sentiment_counts = df_result['sentiment_label'].value_counts()
        fig_sentiment = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index, 
                               title="Overall Sentiment Distribution", color=sentiment_counts.index,
                               color_discrete_map={'Positive':'green', 'Negative':'red', 'Neutral':'grey'})
        st.plotly_chart(fig_sentiment, use_container_width=True)

    with col2:
        language_counts = df_result['language'].value_counts()
        fig_lang = px.bar(language_counts, x=language_counts.index, y=language_counts.values,
                          title="Language Distribution", labels={'x': 'Language', 'y': 'Number of Comments'})
        st.plotly_chart(fig_lang, use_container_width=True)
        
    st.subheader("Explore the Full Analyzed Data")
    st.dataframe(df_result)
