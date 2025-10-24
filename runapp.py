import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import json
import time
import os
from dotenv import load_dotenv

# --- 1. Page & Environment Configuration ---
load_dotenv()
st.set_page_config(page_title="Malaysian Sentiment Analysis Engine", layout="wide", page_icon="🇲🇾")

# --- API Key Configuration and Verification ---
api_key_loaded = False
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = st.secrets["GOOGLE_API_KEY"]
    if api_key:
        genai.configure(api_key=api_key)
        api_key_loaded = True
    else:
        st.error("⚠️ Google API Key not found.")
except Exception as e:
    st.error(f"⚠️ API Key Error: {e}")

# --- 2. Dictionaries & Constants ---
MALAYSIAN_POLITICAL_DICTIONARY = {
    "leaders": {
        "anwar": "Anwar Ibrahim", "pmx": "Anwar Ibrahim", "madanon": "Anwar Ibrahim",
        "wan azizah": "Wan Azizah Wan Ismail", "rafizi": "Rafizi Ramli", "anthony loke": "Anthony Loke Siew Fook",
        "gobind": "Gobind Singh Deo", "lim guan eng": "Lim Guan Eng", "lce": "Lim Guan Eng",
        "mat sabu": "Mohamad Sabu", "zahid": "Ahmad Zahid Hamidi", "zahid komedi": "Ahmad Zahid Hamidi",
        "tok mat": "Mohamad Hasan", "ismail sabri": "Ismail Sabri Yaakob", "wee ka siong": "Wee Ka Siong",
        "muhyiddin": "Muhyiddin Yassin", "my": "Muhyiddin Yassin", "abah": "Muhyiddin Yassin",
        "hadi": "Hadi Awang", "lebai": "Hadi Awang", "azmin ali": "Azmin Ali",
        "hamzah": "Hamzah Zainudin", "sanusi": "Muhammad Sanusi Md Nor", "hajiji": "Hajiji Noor",
        "jeffrey kitingan": "Jeffrey Kitingan", "abang jo": "Abang Johari Openg",
        "shafie": "Shafie Apdal", "mahathir": "Mahathir Mohamad", "tun m": "Mahathir Mohamad",
        "syed saddiq": "Syed Saddiq Syed Abdul Rahman", "najib": "Najib Razak", "bossku": "Najib Razak"
    },
    "parties": {
        "ph": "Pakatan Harapan", "bn": "Barisan Nasional", "pn": "Perikatan Nasional",
        "grs": "Gabungan Rakyat Sabah", "gps": "Gabungan Parti Sarawak", "pkr": "Parti Keadilan Rakyat (PKR)",
        "dap": "DAP", "amanah": "Parti Amanah Negara (Amanah)", "umno": "UMNO", "mca": "MCA",
        "mic": "MIC", "pas": "PAS", "bersatu": "Parti Pribumi Bersatu Malaysia (Bersatu)",
        "gerakan": "Parti Gerakan Rakyat Malaysia (Gerakan)", "warisan": "Parti Warisan Sabah (Warisan)", "muda": "MUDA"
    }
}

# --- 3. AI & Data Processing Functions ---

# --- STAGE 1: BATCH COMMENT ENRICHMENT (MODIFIED for ABSA) ---
def enrich_comments_with_gemini(df_comments, comment_column, batch_size=100):
    model = genai.GenerativeModel('gemini-2.5-pro')
    leader_list = list(MALAYSIAN_POLITICAL_DICTIONARY["leaders"].keys())
    party_list = list(MALAYSIAN_POLITICAL_DICTIONARY["parties"].keys())
    
    # NEW FEATURE: Aspect-Based Sentiment Analysis (ABSA) prompt
    system_prompt_stage1 = f"""
    You are a high-performance AI data enrichment service. Your task is to receive a batch of raw user comments and transform EACH one into a structured JSON object. Your analysis assumes the comments are within the context of Trending Malaysian News and Social Topics. Your final output MUST be an array of these JSON objects.

    OUTPUT STRUCTURE FOR EACH COMMENT:
    {{
      "comment_index": <integer>,
      "language": "<string>",
      "overall_sentiment_score": <float, from -1.0 to 1.0>,
      "inferred_region": "<string, 'Peninsular', 'Borneo', or 'Unknown'>",
      "inferred_race": "<string, e.g., 'Malay', 'Chinese', 'Indian', 'Other/Unknown'>",
      "mentioned_entities": [
          {{
              "entity_name": "<string, proper name of the entity>",
              "entity_sentiment_score": <float, sentiment towards THIS entity>
          }}
      ]
    }}

    YOUR ANALYSIS LOGIC FOR EACH INDIVIDUAL COMMENT:
    1. Language Identification.
    2. Overall Sentiment Scoring (float score for the whole comment).
    3. Demographic Inference (Regional and Racial, be cautious).
    4. Strict Entity Recognition and Aspect-Based Sentiment:
       - Scan for explicit mentions of entities. Map slang/acronyms to their proper names. Leaders: {leader_list}. Parties: {party_list}.
       - For EACH entity found, determine the specific sentiment towards THAT entity within the comment and assign it an 'entity_sentiment_score'.
       - If no entities are found, return an empty array `[]` for 'mentioned_entities'.
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
                if attempt < retries - 1: time.sleep(delay); delay *= 2
                else: st.error(f"Batch {current_batch_num} failed: {e}. Skipping.")
                    
    progress_bar.progress(1.0, text="Stage 1 Enrichment Complete!")
    if not all_analyzed_data: return None
    try:
        df_analyzed = pd.DataFrame(all_analyzed_data)
        df_analyzed['sentiment_label'] = df_analyzed['overall_sentiment_score'].apply(
            lambda x: 'Positive' if x > 0.2 else ('Negative' if x < -0.2 else 'Neutral')
        )
        df_comments_reset = df_comments.reset_index(drop=True)
        df_final = df_comments_reset.join(df_analyzed.set_index('comment_index'))
        return df_final
    except Exception as e:
        st.error(f"Failed to merge enrichment results: {e}")
        return None

# --- 4. Streamlit Application UI ---
st.title("🇲🇾 Malaysian Social Issue & News Sentiment Engine")

if 'df_enriched' not in st.session_state:
    st.session_state.df_enriched = None

with st.sidebar:
    st.header("1. Upload Data")
    uploaded_files = st.file_uploader("Upload one or more files", type=['csv', 'xlsx'], accept_multiple_files=True)
    
    if uploaded_files:
        df_list = []
        for file in uploaded_files:
            try:
                df_temp = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
                df_list.append(df_temp)
            except Exception as e: st.error(f"Error reading '{file.name}': {e}")
        if df_list:
            st.session_state.df_original = pd.concat(df_list, ignore_index=True)
            st.success(f"Loaded {len(st.session_state.df_original)} total comments.")

    if 'df_original' in st.session_state:
        df_original = st.session_state.df_original
        st.header("2. Configure Analysis")
        comment_column = st.selectbox("Select comment column:", options=df_original.columns.tolist())
        datetime_column = st.selectbox("Select date/timestamp column:", options=[None] + df_original.columns.tolist())
        batch_size = st.slider("Batch Size", 50, 1000, 250, 50, help="Comments per API call.")
        
        if st.button("Start Full Analysis", type="primary"):
            if not api_key_loaded: st.error("Cannot start: Google API Key is not configured.")
            else:
                st.session_state.df_enriched = None
                with st.spinner("Stage 1: Enriching all comments..."):
                    enriched_data = enrich_comments_with_gemini(df_original, comment_column, batch_size)
                if enriched_data is not None:
                    if datetime_column and datetime_column in enriched_data.columns:
                        enriched_data[datetime_column] = pd.to_datetime(enriched_data[datetime_column], errors='coerce', utc=True)
                    st.session_state.df_enriched = enriched_data
                    st.success("Enrichment complete!")
                    st.rerun()
                else: st.error("Enrichment failed.")

# --- MAIN DASHBOARD AREA ---
if st.session_state.df_enriched is not None:
    df = st.session_state.df_enriched

    # ==================================================================
    # === NEW FEATURE 1: INTERACTIVE SIDEBAR FILTERS ===================
    # ==================================================================
    with st.sidebar:
        st.header("3. Dashboard Filters")
        
        # Filter by Sentiment
        sentiment_options = df['sentiment_label'].unique()
        selected_sentiments = st.multiselect("Filter by Sentiment:", options=sentiment_options, default=sentiment_options)
        
        # Filter by Language
        language_options = df['language'].unique()
        selected_languages = st.multiselect("Filter by Language:", options=language_options, default=language_options)
        
        # Filter by Race
        race_options = df['inferred_race'].unique()
        selected_races = st.multiselect("Filter by Inferred Race:", options=race_options, default=race_options)

        # Apply filters to create a new dataframe for visualization
        df_filtered = df[
            df['sentiment_label'].isin(selected_sentiments) &
            df['language'].isin(selected_languages) &
            df['inferred_race'].isin(selected_races)
        ]
        st.info(f"Displaying {len(df_filtered)} of {len(df)} comments based on filters.")

    st.header("High-Level Summary")
    st.subheader("Overall Sentiment of Filtered Comments")

    col1, col2, col3 = st.columns(3)
    col1.metric("Filtered Comments", f"{len(df_filtered):,}")
    if 'overall_sentiment_score' in df_filtered.columns and not df_filtered['overall_sentiment_score'].isnull().all():
        avg_sentiment = df_filtered['overall_sentiment_score'].mean()
        col2.metric("Average Sentiment", f"{avg_sentiment:.2f}")
        sentiment_label = "Positive" if avg_sentiment > 0.2 else "Negative" if avg_sentiment < -0.2 else "Neutral"
        col3.metric("Overall Sentiment", sentiment_label)
    
    st.markdown("---")
    st.header("Detailed Analysis Dashboard")
    tab1, tab2, tab3 = st.tabs(["📊 Sentiment Breakdown", "🏛️ Political Analysis", "📈 Temporal Analysis"])

    # All charts from here on will use df_filtered
    with tab1:
        st.subheader("Sentiment & Demographic Distribution")
        # ... (visualization code is identical)
    with tab2:
        st.subheader("Political Entity Analysis (Aspect-Based)")
        if 'mentioned_entities' in df_filtered.columns:
            # NEW FEATURE: Process ABSA data
            df_entities = df_filtered.explode('mentioned_entities').dropna(subset=['mentioned_entities'])
            df_entities = pd.concat([df_entities.drop(['mentioned_entities'], axis=1), df_entities['mentioned_entities'].apply(pd.Series)], axis=1)

            leader_mentions = df_entities[df_entities['entity_name'].isin(MALAYSIAN_POLITICAL_DICTIONARY['leaders'].values())]
            party_mentions = df_entities[df_entities['entity_name'].isin(MALAYSIAN_POLITICAL_DICTIONARY['parties'].values())]

            col1, col2 = st.columns(2)
            with col1:
                if not party_mentions.empty:
                    party_stats = party_mentions.groupby('entity_name')['entity_sentiment_score'].agg(['count', 'mean']).reset_index()
                    fig = px.bar(party_stats.sort_values('count', ascending=False), 
                                 x='entity_name', y='count', color='mean',
                                 title="Party Mentions & Specific Sentiment",
                                 labels={'entity_name': 'Party', 'count': 'Mentions', 'mean': 'Avg. Sentiment'})
                    st.plotly_chart(fig, use_container_width=True)
                    # NEW FEATURE: DRILL-DOWN
                    with st.expander("🔍 Show Comments Mentioning Parties"):
                        st.dataframe(party_mentions)
                else: st.info("No political parties explicitly mentioned in filtered data.")
            with col2:
                if not leader_mentions.empty:
                    leader_stats = leader_mentions.groupby('entity_name')['entity_sentiment_score'].agg(['count', 'mean']).reset_index()
                    fig = px.bar(leader_stats.sort_values('count', ascending=False), 
                                 x='entity_name', y='count', color='mean',
                                 title="Leader Mentions & Specific Sentiment",
                                 labels={'entity_name': 'Leader', 'count': 'Mentions', 'mean': 'Avg. Sentiment'})
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("🔍 Show Comments Mentioning Leaders"):
                        st.dataframe(leader_mentions)
                else: st.info("No political leaders explicitly mentioned in filtered data.")
        else: st.warning("Mentioned Entities column not found.")

    with tab3:
        st.subheader("Temporal Analysis")
        if datetime_column and datetime_column in df_filtered.columns and not df_filtered[datetime_column].isnull().all():
            df_time = df_filtered.set_index(datetime_column).sort_index()
            
            st.markdown("#### Sentiment Momentum (Rate of Change)")
            time_freq = st.selectbox("Select Time Aggregation:", ["Daily (D)", "Weekly (W)", "Monthly (M)"], index=1)
            freq_code = time_freq.split(" ")[1][1]
            sentiment_over_time = df_time.resample(freq_code)['overall_sentiment_score'].mean().to_frame('average_sentiment')
            sentiment_over_time['momentum'] = sentiment_over_time['average_sentiment'].diff()
            
            fig_mom = go.Figure()
            fig_mom.add_trace(go.Bar(
                x=sentiment_over_time.index, y=sentiment_over_time['momentum'],
                marker_color=['#00CC96' if v > 0 else '#EF553B' for v in sentiment_over_time['momentum']]
            ))
            fig_mom.update_layout(title=f"{time_freq} Sentiment Momentum",
                                  xaxis_title="Date", yaxis_title="Change in Sentiment Score")
            st.plotly_chart(fig_mom, use_container_width=True)
            with st.expander("🔍 Show Momentum Data"):
                st.dataframe(sentiment_over_time)
        else: st.info("No valid date column selected for Temporal Analysis.")

    st.markdown("---")
    st.header("Explore Full Filtered Dataset")
    st.dataframe(df_filtered)

else:
    st.info("Awaiting data upload and analysis to begin.")
