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
    model = genai.GenerativeModel('gemini-2.5-pro') # Use the available model for now
    
    # Invert the dictionary for easier lookup in the prompt
    leader_list = list(MALAYSIAN_POLITICAL_DICTIONARY["leaders"].keys())
    party_list = list(MALAYSIAN_POLITICAL_DICTIONARY["parties"].keys())
    
    system_prompt_stage1 = f"""
    You are a high-performance, parallel-processing AI data enrichment service. Your task is to receive a batch of raw user comments and transform EACH one into a structured JSON object. Your analysis assumes the comments are within the context of Trending Malaysian News and Social Topics. For every single comment you process, you MUST return a single, clean JSON object with the following exact keys. Your final output for the entire task MUST be an array of these JSON objects.

    OUTPUT STRUCTURE FOR EACH COMMENT:
    {{
      "comment_index": <integer>,
      "language": "<string, e.g., 'Malay', 'Chinese', 'Indian', 'English', 'Mixed'>",
      "sentiment_score": <float, from -1.0 to 1.0>,
      "inferred_region": "<string, 'Peninsular', 'Borneo', or 'Unknown'>",
      "inferred_race": "<string, e.g., 'Malay', 'Chinese', 'Indian', 'Native Sabah/Sarawak', 'Other/Unknown'>",
      "mentioned_entities": ["<array of proper names of any explicitly mentioned Malaysian political leaders or parties from the provided lists, otherwise an empty array []>"]
    }}

    YOUR ANALYSIS LOGIC FOR EACH INDIVIDUAL COMMENT:
    1. Language Identification.
    2. Sentiment Scoring.
    3. Demographic Inference (Regional and Racial, be cautious).
    4. Strict Entity Recognition: Scan for explicit mentions of entities. Map any found slang or acronyms to their proper names. Leaders List: {leader_list}. Parties List: {party_list}.
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
        
        # Retry logic...
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
        df_comments_reset = df_comments.reset_index(drop=True)
        df_final = df_comments_reset.join(df_analyzed.set_index('comment_index'))
        return df_final
    except Exception as e:
        st.error(f"Failed to merge enrichment results: {e}")
        return None

# --- STAGE 2: AGGREGATED ANALYSIS ---
def generate_ai_summary(df_enriched, main_topic):
    model = genai.GenerativeModel('gemini-1.5-pro-latest') # Use the available model for now
    
    # Prepare aggregated data to send to Gemini
    summary_data = {
        "topic_of_analysis": main_topic,
        "total_comments": len(df_enriched),
        "overall_sentiment_score": df_enriched['sentiment_score'].mean(),
        "sentiment_distribution": df_enriched['sentiment_label'].value_counts(normalize=True).to_dict(),
        "sentiment_by_region": df_enriched.groupby('inferred_region')['sentiment_score'].mean().to_dict(),
        "sentiment_by_race": df_enriched.groupby('inferred_race')['sentiment_score'].mean().to_dict(),
        "entity_mentions": df_enriched['mentioned_entities'].explode().value_counts().to_dict()
    }
    
    system_prompt_stage2 = """
    You are a specialized AI data scientist and political analyst. Your goal is to analyze a pre-processed JSON of comment statistics about a specific topic and generate a comprehensive, multi-part statistical report with clear, actionable insights. Your output must be in well-formatted Markdown.

    **ANALYSIS REPORT**

    **Topic of Analysis:** <The user-specified topic>

    **1. Overall Sentiment Dashboard**
    *   **Total Comments on Topic:** <Integer count>
    *   **Overall Sentiment:** <'Positive', 'Negative', 'Neutral'>
    *   **Average Sentiment Score:** <Float from -1.0 to 1.0>
    *   **Sentiment Distribution:**
        *   Positive: <Percentage>%
        *   Negative: <Percentage>%
        *   Neutral: <Percentage>%
    *   **Key Insight:** <A one-sentence summary explaining the main driver of the overall sentiment.>

    **2. Demographic Sentiment Analysis**
    *   **Sentiment by Location:**
    *   **Sentiment by Inferred Race:**
    *   **Key Insight:** <A one-sentence summary highlighting any significant differences in sentiment between demographic groups.>

    **3. Political Entity Analysis**
    *   **Most Mentioned Entities:** <List the top 3-5 mentioned leaders/parties.>
    *   **Entity Sentiment Breakdown:**
    *   **Key Insight:** <A one-sentence summary explaining which figures/parties are most central to the discussion.>

    **4. Time Series Trend Analysis**
    *   **Peak Discussion Day:** <Identify the date with the highest volume of comments.>
    *   **Trend Summary:** <Describe the trend of discussion over time.>
    *   **Key Insight:** <A one-sentence summary of the discussion's lifecycle.>
    """
    
    full_prompt = f"{system_prompt_stage2}\n\nHere is the pre-processed data summary:\n---\n{json.dumps(summary_data, indent=2)}\n\nGenerate the full report based on this data."
    
    response = model.generate_content(full_prompt)
    return response.text


# --- Streamlit UI ---

st.title("🇲🇾 Malaysian Social Issue & News Sentiment Engine")

# Initialize state
if 'df_enriched' not in st.session_state:
    st.session_state.df_enriched = None

# --- SIDEBAR FOR FILE UPLOAD AND CONTROLS ---
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
        comment_column = st.selectbox(
            "Select the comment column:",
            options=df_original.columns.tolist()
        )
        
        # Add a date/datetime column selector for time series analysis
        datetime_column = st.selectbox(
            "Select the date/timestamp column:",
            options=[None] + df_original.columns.tolist()
        )

        batch_size = st.slider("Batch Size", 10, 200, 50, 10, help="Number of comments per API call. Smaller is more stable.")
        
        if st.button("Start Full Analysis", type="primary"):
            st.session_state.df_enriched = None # Clear old results
            with st.spinner("Stage 1: Enriching all comments with AI. This may take time..."):
                enriched_data = enrich_comments_with_gemini(df_original, comment_column, batch_size)
            
            if enriched_data is not None:
                if datetime_column:
                    # Convert date column to datetime for time series analysis
                    enriched_data[datetime_column] = pd.to_datetime(enriched_data[datetime_column], errors='coerce')
                
                st.session_state.df_enriched = enriched_data
                st.success("Enrichment complete! Dashboard is ready.")
                st.rerun()
            else:
                st.error("Enrichment failed. Please check logs.")

# --- MAIN DASHBOARD AREA ---
if st.session_state.df_enriched is not None:
    df = st.session_state.df_enriched

    st.header("Overall Topic and Sentiment Analysis")
    
    # 1. AI to determine main topic
    all_topics = df['primary_topic'].dropna().tolist()
    topic_prompt = f"Based on this list of topics extracted from comments, what is the single main subject of discussion? Topics: {', '.join(all_topics[:100])}"
    topic_model = genai.GenerativeModel('gemini-1.5-pro-latest')
    main_topic = topic_model.generate_content(topic_prompt).text.strip()
    st.subheader(f"Main Topic Determined by AI: **{main_topic}**")

    # 2. Main Dashboard Visuals
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Comments", f"{len(df):,}")
    avg_sentiment = df['sentiment_score'].mean()
    col2.metric("Average Sentiment", f"{avg_sentiment:.2f}")
    sentiment_label = "Positive" if avg_sentiment > 0.2 else "Negative" if avg_sentiment < -0.2 else "Neutral"
    col3.metric("Overall Sentiment", sentiment_label)

    st.markdown("---")
    
    # 3. Political & Demographic Charts
    st.header("Political and Demographic Insights")
    # Explode the mentioned_entities list to analyze each entity
    df_entities = df.explode('mentioned_entities').dropna(subset=['mentioned_entities'])
    
    leader_mentions = df_entities[df_entities['mentioned_entities'].isin(MALAYSIAN_POLITICAL_DICTIONARY['leaders'].values())]
    party_mentions = df_entities[df_entities['mentioned_entities'].isin(MALAYSIAN_POLITICAL_DICTIONARY['parties'].values())]

    col1, col2 = st.columns(2)
    with col1:
        fig_party = px.bar(party_mentions['mentioned_entities'].value_counts().reset_index(), x='mentioned_entities', y='count', title="Political Party Mention Count")
        st.plotly_chart(fig_party, use_container_width=True)
    with col2:
        fig_leader = px.bar(leader_mentions['mentioned_entities'].value_counts().reset_index(), x='mentioned_entities', y='count', title="Political Leader Mention Count")
        st.plotly_chart(fig_leader, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_race = px.pie(df, names='inferred_race', title="Sentiment Distribution by Inferred Race")
        st.plotly_chart(fig_race, use_container_width=True)
    with col2:
        fig_region = px.pie(df, names='inferred_region', title="Sentiment Distribution by Inferred Region")
        st.plotly_chart(fig_region, use_container_width=True)

    # 5. Time Series Analysis
    if datetime_column and datetime_column in df.columns and not df[datetime_column].isnull().all():
        st.header("Time Series: Discussion Trend")
        df_time = df.set_index(datetime_column)
        daily_volume = df_time.resample('D').size().to_frame('comment_volume')
        
        fig_time = px.line(daily_volume, x=daily_volume.index, y='comment_volume', title=f"Daily Discussion Volume for '{main_topic}'", markers=True)
        fig_time.update_layout(xaxis_title="Date", yaxis_title="Number of Comments")
        st.plotly_chart(fig_time, use_container_width=True)
    
    # 4. Generate AI summary
    st.markdown("---")
    st.header("Generate AI-Powered Executive Summary")
    if st.button("Generate Summary"):
        with st.spinner("Gemini is crafting the final report..."):
            summary_report = generate_ai_summary(df, main_topic)
            st.markdown(summary_report)

else:
    st.info("Awaiting data upload and analysis to begin.")
