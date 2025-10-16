import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import json
import time

# --- Page Configuration ---
st.set_page_config(page_title="Malaysian Sentiment Analysis Engine", layout="wide", page_icon="🇲🇾")

# --- Security and Configuration ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("⚠️ Google API Key not found. Please add it to your Streamlit secrets.")

# --- Malaysian Political Dictionary ---
MALAYSIAN_POLITICAL_DICTIONARY = {
    "leaders": {
        "anwar": "Anwar Ibrahim", "pmx": "Anwar Ibrahim", "zahid": "Zahid Hamidi",
        "hajiji": "Hajiji Noor", "shafie": "Shafie Apdal", "bung moktar": "Bung Moktar Radin",
        "mahathir": "Mahathir Mohamad", "muhyiddin": "Muhyiddin Yassin", "hadi": "Hadi Awang",
        "madanon": "Anwar Ibrahim", "lebai": "Hadi Awang"
    },
    "parties": {
        "ph": "Pakatan Harapan", "bn": "Barisan Nasional", "umno": "UMNO",
        "pn": "Perikatan Nasional", "pas": "PAS", "grs": "Gabungan Rakyat Sabah",
        "warisan": "Parti Warisan Sabah", "dap": "DAP"
    }
}

# --- STAGE 1: BATCH COMMENT ENRICHMENT ---
def enrich_comments_with_gemini(df_comments, comment_column, batch_size=100):
    # ==================================================================
    # === MODEL NAME SET EXACTLY AS REQUESTED ==========================
    # ==================================================================
    # WARNING: This 'gemini-2.5-pro' model is speculative and will fail with a
    # "model not found" error until Google officially releases it.
    # To make the app functional today, use 'gemini-1.5-pro-latest'.
    model = genai.GenerativeModel('gemini-2.5-pro')
    
    leader_list = list(MALAYSIAN_POLITICAL_DICTIONARY["leaders"].keys())
    party_list = list(MALAYSIAN_POLITICAL_DICTIONARY["parties"].keys())
    
    system_prompt_stage1 = f"""
    You are a high-performance, parallel-processing AI data enrichment service. Your task is to receive a batch of raw user comments and transform EACH one into a structured JSON object. Your analysis assumes the comments are within the context of Trending Malaysian News and Social Topics. For every single comment you process, you MUST return a single, clean JSON object with the following exact keys. Your final output for the entire task MUST be an array of these JSON objects.

    OUTPUT STRUCTURE FOR EACH COMMENT:
    {{
      "comment_index": <integer>,
      "language": "<string, e.g., 'Malay', 'Chinese', 'Indian', 'English', 'Mixed'>",
      "sentiment_score": <float, from -1.0 to 1.0>,
      "primary_topic": "<string, the main subject of the comment, identified by you>",
      "inferred_region": "<string, 'Peninsular', 'Borneo', or 'Unknown'>",
      "inferred_race": "<string, e.g., 'Malay', 'Chinese', 'Indian', 'Native Sabah/Sarawak', 'Other/Unknown'>",
      "mentioned_entities": ["<array of proper names of any explicitly mentioned Malaysian political leaders or parties from the provided lists, otherwise an empty array []>"]
    }}

    YOUR ANALYSIS LOGIC FOR EACH INDIVIDUAL COMMENT:
    1. Language Identification.
    2. Sentiment Scoring (assign a float score).
    3. Topic Identification (dynamically identify the main subject).
    4. Demographic Inference (Regional and Racial, be cautious, use 'Unknown' if ambiguous).
    5. Strict Entity Recognition: Scan for explicit mentions of entities. Map any found slang or acronyms to their proper names. Leaders List: {leader_list}. Parties List: {party_list}.
    """
    
    comments_to_process = df_comments[comment_column].dropna().astype(str).tolist()
    total_comments = len(comments_to_process)
    num_batches = (total_comments + batch_size - 1) // batch_size
    progress_bar = st.progress(0, text="Starting Stage 1: Comment Enrichment...")
    all_analyzed_data = []

    for i in range(0, total_comments, batch_size):
        batch = comments_to_process[i:i + batch_size]
        formatted_batch = "\n".join([f'{i+j}: "{comment}"' for j, comment in enumerate(batch)])
        full_prompt = f"{system_prompt_stage1}\n\nAnalyze the following batch of comments:\n---\n{formatted_batch}"
        
        retries = 3; delay = 5
        for attempt in range(retries):
            try:
                current_batch_num = i // batch_size + 1
                progress_text = f"Stage 1: Enriching comment batch {current_batch_num}/{num_batches}..."
                progress_bar.progress(i / total_comments, text=progress_text)
                response = model.generate_content(full_prompt)
                cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
                batch_results = json.loads(cleaned_response)
                all_analyzed_data.extend(batch_results)
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    st.error(f"Batch {current_batch_num} failed enrichment after {retries} attempts. Skipping.")
                    break
    
    progress_bar.progress(1.0, text="Stage 1 Enrichment Complete!")
    if not all_analyzed_data: return None
    try:
        df_analyzed = pd.DataFrame(all_analyzed_data)
        df_analyzed['sentiment_label'] = df_analyzed['sentiment_score'].apply(
            lambda x: 'Positive' if x > 0.2 else ('Negative' if x < -0.2 else 'Neutral')
        )
        df_comments_reset = df_comments.reset_index(drop=True)
        df_final = df_comments_reset.join(df_analyzed.set_index('comment_index'))
        return df_final
    except Exception as e:
        st.error(f"Failed to merge enrichment results: {e}")
        return None

# --- STAGE 2: AGGREGATED ANALYSIS ---
def generate_ai_summary(df_enriched, main_topic):
    model = genai.GenerativeModel('gemini-2.5-pro') # Using the specified model
    
    # Explode the dataframe to have one row per entity mention
    df_entities = df_enriched.explode('mentioned_entities').dropna(subset=['mentioned_entities'])
    leader_mentions = df_entities[df_entities['mentioned_entities'].isin(MALAYSIAN_POLITICAL_DICTIONARY['leaders'].values())]
    party_mentions = df_entities[df_entities['mentioned_entities'].isin(MALAYSIAN_POLITICAL_DICTIONARY['parties'].values())]

    # Calculate statistics to send to the model
    summary_data = {
        "topic_of_analysis": main_topic,
        "total_comments": len(df_enriched),
        "overall_sentiment_score": df_enriched['sentiment_score'].mean(),
        "sentiment_distribution_percentage": df_enriched['sentiment_label'].value_counts(normalize=True).mul(100).round(1).to_dict(),
        "leader_sentiment_analysis": leader_mentions.groupby('mentioned_entities')['sentiment_score'].agg(['mean', 'count']).rename(columns={'mean': 'avg_sentiment', 'count': 'mentions'}).to_dict('index'),
        "party_sentiment_analysis": party_mentions.groupby('mentioned_entities')['sentiment_score'].agg(['mean', 'count']).rename(columns={'mean': 'avg_sentiment', 'count': 'mentions'}).to_dict('index')
    }

    # ==================================================================
    # === MODIFIED PROMPT FOR DEEPER POLITICAL ANALYSIS ================
    # ==================================================================
    system_prompt_stage2 = """
    You are an expert Malaysian political data scientist. Your task is to interpret a JSON summary of pre-analyzed comment data and generate a final, human-readable report in Markdown.

    **ANALYSIS REPORT**

    **Topic of Analysis:** <The user-specified topic>

    **1. Overall Sentiment Dashboard**
    *   **Total Comments Analyzed:** <Integer count>
    *   **Overall Sentiment:** <'Positive', 'Negative', 'Neutral'>
    *   **Average Sentiment Score:** <Float from -1.0 to 1.0>
    *   **Sentiment Distribution:**
        *   Positive: <Percentage>%
        *   Negative: <Percentage>%
        *   Neutral: <Percentage>%
    *   **Key Insight:** <A one-sentence summary explaining the main driver of the sentiment.>

    **2. Political Entity Analysis**
    Based on the provided data, perform a detailed analysis of mentioned political leaders and parties.

    *   **Leader Sentiment & Support:**
        For each leader, calculate a 'Support Rate' (percentage of positive mentions). List the top 3-5 leaders.
        - **<Leader 1 Name>:**
          - **Average Sentiment:** <Score from -1.0 to 1.0>
          - **Support Rate:** <Calculated Percentage>%
          - **Total Mentions:** <Integer count>
        - **<Leader 2 Name>:**
          - **Average Sentiment:** <Score from -1.0 to 1.0>
          - **Support Rate:** <Calculated Percentage>%
          - **Total Mentions:** <Integer count>

    *   **Party Sentiment & Support:**
        For each party, calculate a 'Support Rate'. List the top 3-5 parties.
        - **<Party 1 Name>:**
          - **Average Sentiment:** <Score from -1.0 to 1.0>
          - **Support Rate:** <Calculated Percentage>%
          - **Total Mentions:** <Integer count>
    
    *   **Key Political Insight:** <A one-sentence summary identifying the most discussed political figure/party and the overall tone of the political conversation.>
    """
    
    full_prompt = f"{system_prompt_stage2}\n\nHere is the pre-processed data summary:\n---\n{json.dumps(summary_data, indent=2)}\n\nGenerate the full report based ONLY on this data. Calculate Support Rate as the percentage of mentions that are positive (assume score > 0.2 is positive)."
    
    response = model.generate_content(full_prompt)
    return response.text


# --- Streamlit UI ---

st.title("🇲🇾 Malaysian Social Issue & News Sentiment Engine")

if 'df_enriched' not in st.session_state:
    st.session_state.df_enriched = None

with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df_original = df
            st.success(f"Loaded {len(df)} comments.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    if 'df_original' in st.session_state:
        df_original = st.session_state.df_original
        st.header("2. Configure Analysis")
        comment_column = st.selectbox("Select the comment column:", options=df_original.columns.tolist())
        datetime_column = st.selectbox("Select the date/timestamp column:", options=[None] + df_original.columns.tolist())
        
        # ==================================================================
        # === MODIFIED BATCH SIZE SLIDER ===================================
        # ==================================================================
        batch_size = st.slider(
            "Comments per API Call (Batch Size)", 
            min_value=50, 
            max_value=1000, 
            value=500, 
            step=50,
            help="Larger batches reduce API calls and cost but use more memory. Adjust based on performance."
        )
        
        if st.button("Start Full Analysis", type="primary"):
            st.session_state.df_enriched = None
            with st.spinner("Stage 1: Enriching all comments with AI. This may take time..."):
                enriched_data = enrich_comments_with_gemini(df_original, comment_column, batch_size)
            if enriched_data is not None:
                if datetime_column:
                    enriched_data[datetime_column] = pd.to_datetime(enriched_data[datetime_column], errors='coerce')
                st.session_state.df_enriched = enriched_data
                st.success("Enrichment complete! Dashboard is ready.")
                st.rerun()
            else:
                st.error("Enrichment failed.")

# --- MAIN DASHBOARD AREA ---
if st.session_state.df_enriched is not None:
    df = st.session_state.df_enriched

    st.header("Overall Topic and Sentiment Analysis")
    
    main_topic = "N/A"
    if 'primary_topic' in df.columns and not df['primary_topic'].isnull().all():
        unique_topics = list(set(df['primary_topic'].dropna().tolist()))
        topics_for_prompt = ', '.join(unique_topics[:50])
        topic_prompt = f"Based on this list of topics extracted from comments, what is the single main subject of discussion? Topics: {topics_for_prompt}"
        topic_model = genai.GenerativeModel('gemini-2.5-pro') # Using the specified model
        try:
            main_topic = topic_model.generate_content(topic_prompt).text.strip()
        except Exception as e:
            st.warning(f"Could not determine main topic from AI: {e}")
            main_topic = "Analysis of Uploaded Comments"
        st.subheader(f"Main Topic Determined by AI: **{main_topic}**")
    else:
        st.warning("Could not determine main topic because 'primary_topic' column was not generated.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Comments", f"{len(df):,}")
    if 'sentiment_score' in df.columns and not df['sentiment_score'].isnull().all():
        avg_sentiment = df['sentiment_score'].mean()
        col2.metric("Average Sentiment", f"{avg_sentiment:.2f}")
        sentiment_label = "Positive" if avg_sentiment > 0.2 else "Negative" if avg_sentiment < -0.2 else "Neutral"
        col3.metric("Overall Sentiment", sentiment_label)
    
    st.markdown("---")
    
    # The rest of the dashboard visualization code remains the same...
    st.header("Political and Demographic Insights")
    # ... (Charts for party mentions, leader mentions, race, region) ...

    # Time Series Analysis
    if datetime_column and datetime_column in df.columns and not df[datetime_column].isnull().all():
        st.header("Time Series: Discussion Trend")
        # ... (Time series chart) ...

    # Generate AI summary
    st.markdown("---")
    st.header("Generate AI-Powered Executive Summary")
    if st.button("Generate Summary"):
        with st.spinner("Gemini is crafting the final report..."):
            summary_report = generate_ai_summary(df, main_topic)
            st.markdown(summary_report)

else:
    st.info("Awaiting data upload and analysis to begin.")
