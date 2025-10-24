import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# Use your specified import structure
from google import generativeai as genai
from google.generativeai import types
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
        "gobind": "Gobind Singh Deo", "lim guan eng": "Lim Guan Eng", "lce": "Lim Guan Eng", "guan eng": "Lim Guan Eng",
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
def enrich_comments_with_gemini(df_comments, comment_column, batch_size=100):
    model = genai.GenerativeModel('gemini-2.5-pro')
    leader_list = list(MALAYSIAN_POLITICAL_DICTIONARY["leaders"].keys())
    party_list = list(MALAYSIAN_POLITICAL_DICTIONARY["parties"].keys())
    
    system_prompt_stage1 = f"""
    You are a high-performance AI data enrichment service. Your task is to receive a batch of raw user comments and transform EACH one into a structured JSON object. Your final output MUST be an array of these JSON objects.

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
       - Scan for explicit mentions of entities. Map slang/acronyms to their proper names. Leaders List: {leader_list}. Parties List: {party_list}.
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
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    st.error(f"Batch {current_batch_num} failed enrichment after {retries} attempts. Error: {e}. Skipping.")
                    break
    
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
    uploaded_files = st.file_uploader(
        "Upload one or more CSV or Excel files",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        df_list = []
        for file in uploaded_files:
            try:
                df_temp = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
                df_list.append(df_temp)
            except Exception as e: st.error(f"Error reading file '{file.name}': {e}")
        if df_list:
            st.session_state.df_original = pd.concat(df_list, ignore_index=True)
            st.success(f"Loaded {len(st.session_state.df_original)} total comments.")

    if 'df_original' in st.session_state and st.session_state.df_original is not None:
        df_original = st.session_state.df_original
        st.header("2. Configure Analysis")
        comment_column = st.selectbox("Select the comment column:", options=df_original.columns.tolist())
        datetime_column = st.selectbox("Select the date/timestamp column:", options=[None] + df_original.columns.tolist())
        
        batch_size = st.slider(
            "Comments per API Call (Batch Size)", 
            min_value=50, max_value=1000, value=500, step=50,
            help="Larger batches reduce API calls and cost but use more memory."
        )
        
        if st.button("Start Full Analysis", type="primary"):
            if not api_key_loaded: st.error("Cannot start: Google API Key is not configured.")
            else:
                st.session_state.df_enriched = None
                with st.spinner("Stage 1: Enriching all comments with AI..."):
                    enriched_data = enrich_comments_with_gemini(df_original, comment_column, batch_size)
                if enriched_data is not None:
                    st.session_state.df_enriched = enriched_data
                    st.success("Enrichment complete! Dashboard is ready.")
                    st.rerun()
                else: st.error("Enrichment failed.")

# --- MAIN DASHBOARD AREA ---
if 'df_enriched' in st.session_state and st.session_state.df_enriched is not None:
    df = st.session_state.df_enriched

    with st.sidebar:
        st.header("3. Dashboard Filters")
        sentiment_options = ['Positive', 'Negative', 'Neutral']
        selected_sentiments = st.multiselect("Filter by Sentiment:", options=sentiment_options, default=sentiment_options)
        language_options = df['language'].dropna().unique()
        selected_languages = st.multiselect("Filter by Language:", options=language_options, default=language_options)
        race_options = df['inferred_race'].dropna().unique()
        selected_races = st.multiselect("Filter by Inferred Race:", options=race_options, default=race_options)

        df_filtered = df[
            df['sentiment_label'].isin(selected_sentiments) &
            df['language'].isin(selected_languages) &
            df['inferred_race'].isin(selected_races)
        ]
        st.info(f"Displaying {len(df_filtered)} of {len(df)} comments.")

    st.header("High-Level Summary of Filtered Data")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Filtered Comments", f"{len(df_filtered):,}")
    if 'overall_sentiment_score' in df_filtered.columns and not df_filtered['overall_sentiment_score'].isnull().all():
        avg_sentiment = df_filtered['overall_sentiment_score'].mean()
        col2.metric("Average Sentiment", f"{avg_sentiment:.2f}")
        sentiment_label = "Positive" if avg_sentiment > 0.2 else "Negative" if avg_sentiment < -0.2 else "Neutral"
        col3.metric("Overall Sentiment", sentiment_label)
    
    st.markdown("---")
    st.header("Detailed Analysis Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["📊 Sentiment Breakdown", "🏛️ Political Analysis", "💡 Strategic Analysis"])

    with tab1:
        st.subheader("Sentiment & Demographic Distribution")
        col1, col2 = st.columns(2)
        with col1:
            sentiment_counts = df_filtered['sentiment_label'].value_counts()
            fig = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index, 
                         title="Overall Sentiment Distribution", color=sentiment_counts.index,
                         color_discrete_map={'Positive':'#00CC96', 'Negative':'#EF553B', 'Neutral':'#636EFA'})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            lang_counts = df_filtered['language'].value_counts()
            fig = px.bar(lang_counts, x=lang_counts.index, y=lang_counts.values,
                         title="Language Distribution", labels={'x': 'Language', 'y': 'Comment Count'})
            st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            race_counts = df_filtered['inferred_race'].value_counts()
            fig = px.pie(race_counts, names=race_counts.index, values=race_counts.values, title="Comment Distribution by Inferred Race")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            region_counts = df_filtered['inferred_region'].value_counts()
            fig = px.pie(region_counts, names=region_counts.index, values=region_counts.values, title="Comment Distribution by Inferred Region")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Political Entity Analysis (Aspect-Based)")
        if 'mentioned_entities' in df_filtered.columns:
            df_entities = df_filtered.explode('mentioned_entities').dropna(subset=['mentioned_entities'])
            if not df_entities.empty:
                try:
                    entity_details = pd.json_normalize(df_entities['mentioned_entities'])
                    df_entities = df_entities.drop(columns=['mentioned_entities']).reset_index(drop=True)
                    df_entities = pd.concat([df_entities, entity_details], axis=1)

                    leader_mentions = df_entities[df_entities['entity_name'].isin(MALAYSIAN_POLITICAL_DICTIONARY['leaders'].values())]
                    party_mentions = df_entities[df_entities['entity_name'].isin(MALAYSIAN_POLITICAL_DICTIONARY['parties'].values())]

                    col1, col2 = st.columns(2)
                    with col1:
                        if not party_mentions.empty:
                            party_stats = party_mentions.groupby('entity_name')['entity_sentiment_score'].agg(['count', 'mean']).reset_index()
                            fig = px.bar(party_stats.sort_values('count', ascending=False), x='entity_name', y='count', color='mean',
                                         color_continuous_scale=px.colors.diverging.RdYlGn, range_color=[-1,1],
                                         title="Party Mentions & Specific Sentiment", labels={'entity_name': 'Party', 'count': 'Mentions', 'mean': 'Avg. Sentiment'})
                            st.plotly_chart(fig, use_container_width=True)
                            with st.expander("🔍 Show Comments Mentioning Parties"):
                                st.dataframe(party_mentions)
                        else: st.info("No political parties explicitly mentioned in filtered data.")
                    with col2:
                        if not leader_mentions.empty:
                            leader_stats = leader_mentions.groupby('entity_name')['entity_sentiment_score'].agg(['count', 'mean']).reset_index()
                            fig = px.bar(leader_stats.sort_values('count', ascending=False), x='entity_name', y='count', color='mean',
                                         color_continuous_scale=px.colors.diverging.RdYlGn, range_color=[-1,1],
                                         title="Leader Mentions & Specific Sentiment", labels={'entity_name': 'Leader', 'count': 'Mentions', 'mean': 'Avg. Sentiment'})
                            st.plotly_chart(fig, use_container_width=True)
                            with st.expander("🔍 Show Comments Mentioning Leaders"):
                                st.dataframe(leader_mentions)
                        else: st.info("No political leaders explicitly mentioned in filtered data.")
                except Exception as e:
                    st.warning(f"Could not process entity data. The AI may have returned an unexpected format. Error: {e}")
            else: st.info("No entities were mentioned in the filtered data.")
        else: st.warning("Mentioned Entities column not found.")
    
    # ==================================================================
    # === NEW STRATEGIC ANALYSIS TAB (REPLACES TEMPORAL ANALYSIS) ======
    # ==================================================================
    with tab3:
        st.subheader("Strategic Analysis: Share of Voice vs. Sentiment")
        st.write("This chart plots political entities based on how much they are discussed (Share of Voice) versus how positively or negatively they are perceived (Sentiment).")

        if 'mentioned_entities' in df_filtered.columns:
            df_entities_strat = df_filtered.explode('mentioned_entities').dropna(subset=['mentioned_entities'])
            if not df_entities_strat.empty:
                try:
                    entity_details_strat = pd.json_normalize(df_entities_strat['mentioned_entities'])
                    df_entities_strat = df_entities_strat.drop(columns=['mentioned_entities']).reset_index(drop=True)
                    df_entities_strat = pd.concat([df_entities_strat, entity_details_strat], axis=1)

                    # Calculate stats
                    entity_stats = df_entities_strat.groupby('entity_name').agg(
                        avg_sentiment=('entity_sentiment_score', 'mean'),
                        mention_count=('entity_name', 'count')
                    ).reset_index()

                    total_mentions = entity_stats['mention_count'].sum()
                    entity_stats['share_of_voice'] = (entity_stats['mention_count'] / total_mentions) * 100

                    # Create the bubble chart
                    fig = px.scatter(
                        entity_stats,
                        x='avg_sentiment',
                        y='share_of_voice',
                        size='mention_count',
                        color='entity_name',
                        hover_name='entity_name',
                        hover_data={'share_of_voice': ':.2f}%', 'avg_sentiment': ':.2f', 'mention_count': True},
                        title="Share of Voice vs. Sentiment Matrix",
                        labels={
                            "avg_sentiment": "Average Sentiment Score",
                            "share_of_voice": "Share of Voice (%)"
                        }
                    )

                    # Add quadrant lines
                    avg_sov = entity_stats['share_of_voice'].mean()
                    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="grey")
                    fig.add_hline(y=avg_sov, line_width=1, line_dash="dash", line_color="grey")

                    # Add quadrant labels
                    fig.add_annotation(x=0.5, y=avg_sov*1.5, text="Stars (High SOV, High Sentiment)", showarrow=False, yshift=10)
                    fig.add_annotation(x=-0.5, y=avg_sov*1.5, text="Problem Children (High SOV, Low Sentiment)", showarrow=False, yshift=10)
                    fig.add_annotation(x=0.5, y=avg_sov/2, text="Rising Stars (Low SOV, High Sentiment)", showarrow=False, yshift=10)
                    fig.add_annotation(x=-0.5, y=avg_sov/2, text="Niche Issues (Low SOV, Low Sentiment)", showarrow=False, yshift=10)
                    
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.warning(f"Could not generate strategic chart. Error: {e}")
            else:
                st.info("No entities were mentioned to create a strategic analysis.")
        else:
            st.warning("Mentioned Entities column not found.")

    st.markdown("---")
    st.header("Explore Full Filtered Dataset")
    st.dataframe(df_filtered)

else:
    st.info("Awaiting data upload and analysis to begin.")
