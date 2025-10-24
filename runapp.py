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
      "sentiment_score": <float, from -1.0 to 1.0>,
      "primary_topic": "<string>",
      "inferred_region": "<string, 'Peninsular', 'Borneo', or 'Unknown'>",
      "inferred_race": "<string, e.g., 'Malay', 'Chinese', 'Indian', 'Other/Unknown'>",
      "mentioned_entities": ["<array of proper names of any explicitly mentioned entities from the provided lists>"]
    }}

    YOUR ANALYSIS LOGIC FOR EACH INDIVIDUAL COMMENT:
    1. Language Identification.
    2. Sentiment Scoring (float score).
    3. Topic Identification (main subject).
    4. Demographic Inference (Regional and Racial, be cautious, use 'Unknown' if ambiguous).
    5. Strict Entity Recognition: Map any found slang/acronyms to their proper names. Leaders List: {leader_list}. Parties List: {party_list}.
    """
    
    comments_to_process = df_comments[comment_column].dropna().astype(str).tolist()
    total_comments = len(comments_to_process)
    num_batches = (total_comments + batch_size - 1) // batch_size
    progress_bar = st.progress(0, text="Starting Stage 1: Comment Enrichment...")
    all_analyzed_data = []
    for i in range(0, total_comments, batch_size):
        batch = comments_to_process[i:i + batch_size]
        formatted_batch = "\n".join([f'{i+j}: "{comment}"' for j, comment in enumerate(batch)])
        full_prompt = f"{system_prompt_stage1}\n\nAnalyze the following batch:\n---\n{formatted_batch}"
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
                    st.error(f"Batch {current_batch_num} failed: {e}. Skipping.")
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
                if file.name.endswith('.csv'): df_temp = pd.read_csv(file)
                else: df_temp = pd.read_excel(file)
                df_list.append(df_temp)
            except Exception as e: st.error(f"Error reading file '{file.name}': {e}")
        if df_list:
            df_original = pd.concat(df_list, ignore_index=True)
            st.session_state.df_original = df_original
            st.success(f"Loaded {len(df_original)} total comments.")

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
                    if datetime_column and datetime_column in enriched_data.columns:
                        enriched_data[datetime_column] = pd.to_datetime(enriched_data[datetime_column], errors='coerce')
                    st.session_state.df_enriched = enriched_data
                    st.success("Enrichment complete! Dashboard is ready.")
                    st.rerun()
                else: st.error("Enrichment failed.")

# --- MAIN DASHBOARD AREA ---
if st.session_state.df_enriched is not None:
    df = st.session_state.df_enriched

    st.header("High-Level Summary")
    
    main_topic = "N/A"
    if 'primary_topic' in df.columns and not df['primary_topic'].isnull().all():
        unique_topics = list(set(df['primary_topic'].dropna().tolist()))
        topics_for_prompt = ', '.join(unique_topics[:50])
        topic_prompt = f"Based on this list of topics, what is the single main subject of discussion? Topics: {topics_for_prompt}"
        try:
            topic_model = genai.GenerativeModel('gemini-2.5-pro')
            main_topic = topic_model.generate_content(topic_prompt).text.strip()
        except Exception:
            main_topic = "Analysis of Uploaded Comments"
    st.subheader(f"Main Topic Determined by AI: **{main_topic}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Comments Analyzed", f"{len(df):,}")
    if 'sentiment_score' in df.columns and not df['sentiment_score'].isnull().all():
        avg_sentiment = df['sentiment_score'].mean()
        col2.metric("Average Sentiment", f"{avg_sentiment:.2f}")
        sentiment_label = "Positive" if avg_sentiment > 0.2 else "Negative" if avg_sentiment < -0.2 else "Neutral"
        col3.metric("Overall Sentiment", sentiment_label)
    
    st.markdown("---")
    
    st.header("Detailed Analysis Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["📊 Sentiment Breakdown", "🏛️ Political Analysis", "📈 Temporal Analysis"])

    with tab1:
        st.subheader("Sentiment & Demographic Distribution")
        col1, col2 = st.columns(2)
        with col1:
            if 'sentiment_label' in df.columns:
                sentiment_counts = df['sentiment_label'].value_counts()
                fig = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index, 
                             title="Overall Sentiment Distribution", color=sentiment_counts.index,
                             color_discrete_map={'Positive':'#00CC96', 'Negative':'#EF553B', 'Neutral':'#636EFA'})
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if 'language' in df.columns:
                lang_counts = df['language'].value_counts()
                fig = px.bar(lang_counts, x=lang_counts.index, y=lang_counts.values,
                             title="Language Distribution", labels={'x': 'Language', 'y': 'Comment Count'})
                st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            if 'inferred_race' in df.columns:
                race_counts = df['inferred_race'].value_counts()
                fig = px.pie(race_counts, names=race_counts.index, values=race_counts.values, title="Comment Distribution by Inferred Race")
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if 'inferred_region' in df.columns:
                region_counts = df['inferred_region'].value_counts()
                fig = px.pie(region_counts, names=region_counts.index, values=region_counts.values, title="Comment Distribution by Inferred Region")
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Political Entity Analysis")
        if 'mentioned_entities' in df.columns and 'sentiment_score' in df.columns:
            df_entities = df.explode('mentioned_entities').dropna(subset=['mentioned_entities'])
            leader_mentions = df_entities[df_entities['mentioned_entities'].isin(MALAYSIAN_POLITICAL_DICTIONARY['leaders'].values())]
            party_mentions = df_entities[df_entities['mentioned_entities'].isin(MALAYSIAN_POLITICAL_DICTIONARY['parties'].values())]
            col1, col2 = st.columns(2)
            with col1:
                if not party_mentions.empty:
                    party_stats = party_mentions.groupby('mentioned_entities')['sentiment_score'].agg(['count', 'mean']).reset_index()
                    fig = px.bar(party_stats.sort_values('count', ascending=False), 
                                 x='mentioned_entities', y='count', color='mean',
                                 color_continuous_scale=px.colors.diverging.RdYlGn, range_color=[-1,1],
                                 title="Political Party Mentions & Average Sentiment",
                                 labels={'mentioned_entities': 'Party', 'count': 'Total Mentions', 'mean': 'Avg. Sentiment'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No political parties were explicitly mentioned.")
            with col2:
                if not leader_mentions.empty:
                    leader_stats = leader_mentions.groupby('mentioned_entities')['sentiment_score'].agg(['count', 'mean']).reset_index()
                    fig = px.bar(leader_stats.sort_values('count', ascending=False), 
                                 x='mentioned_entities', y='count', color='mean',
                                 color_continuous_scale=px.colors.diverging.RdYlGn, range_color=[-1,1],
                                 title="Political Leader Mentions & Average Sentiment",
                                 labels={'mentioned_entities': 'Leader', 'count': 'Total Mentions', 'mean': 'Avg. Sentiment'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No political leaders were explicitly mentioned.")
        else:
            st.warning("Mentioned Entities column not found for Political Analysis.")

    with tab3:
        st.subheader("Temporal Analysis: Trends, Events, and Anomalies")
        
        if datetime_column and datetime_column in df.columns and pd.api.types.is_datetime64_any_dtype(df[datetime_column]):
            df_time = df.set_index(datetime_column).sort_index()
            
            st.markdown("#### Real-Time Trend Detection (Volume Spikes)")
            daily_volume = df_time.resample('D').size().to_frame('comment_volume')
            volume_mean = daily_volume['comment_volume'].mean()
            volume_std = daily_volume['comment_volume'].std()
            anomaly_threshold = volume_mean + (2 * volume_std)
            anomalous_days = daily_volume[daily_volume['comment_volume'] > anomaly_threshold]
            fig_anomaly = go.Figure()
            fig_anomaly.add_trace(go.Scatter(x=daily_volume.index, y=daily_volume['comment_volume'], mode='lines', name='Daily Volume'))
            fig_anomaly.add_trace(go.Scatter(x=anomalous_days.index, y=anomalous_days['comment_volume'], mode='markers', 
                                             marker=dict(color='red', size=10, symbol='x'), name='Significant Spike'))
            fig_anomaly.update_layout(title="Daily Discussion Volume with Anomaly Detection",
                                      xaxis_title="Date", yaxis_title="Number of Comments")
            st.plotly_chart(fig_anomaly, use_container_width=True)
            if not anomalous_days.empty:
                st.write("Potential Key Event Dates (High Volume):")
                st.dataframe(anomalous_days)

            st.markdown("---")
            st.markdown("#### Event Impact Tracking")
            
            # Use only unique dates for the selectbox options
            available_dates = sorted(df_time.index.date)
            unique_dates = sorted(list(set(available_dates)))

            event_date = st.selectbox(
                "Select an Event Date to Analyze:",
                options=unique_dates,
                index=len(unique_dates)//2,
                format_func=lambda date: date.strftime('%Y-%m-%d')
            )
            days_window = st.slider("Select Time Window (in days) before/after event:", 1, 30, 7)
            
            if event_date:
                # Convert the user-selected naive date to a tz-naive datetime object
                event_dt = pd.to_datetime(event_date)
                
                # Convert the main dataframe's index to tz-naive for compatibility
                df_time_naive = df_time.tz_convert(None) if df_time.index.tz is not None else df_time

                start_of_before_period = event_dt - pd.Timedelta(days=days_window)
                end_of_before_period = event_dt - pd.Timedelta(seconds=1) # End right before the event day
                
                start_of_after_period = event_dt
                end_of_after_period = event_dt + pd.Timedelta(days=days_window)

                before_period = df_time_naive[start_of_before_period : end_of_before_period]
                after_period = df_time_naive[start_of_after_period : end_of_after_period]
                
                sentiment_before = before_period['sentiment_score'].mean() if not before_period.empty else 0
                sentiment_after = after_period['sentiment_score'].mean() if not after_period.empty else 0
                
                col1, col2 = st.columns(2)
                col1.metric(f"{days_window}-Day Avg. Sentiment BEFORE Event", f"{sentiment_before:.3f}")
                col2.metric(f"{days_window}-Day Avg. Sentiment AFTER Event", f"{sentiment_after:.3f}", delta=f"{sentiment_after - sentiment_before:.3f}")

                combined_period_df = pd.concat([before_period, after_period])

                if not combined_period_df.empty:
                    sentiment_shift_df = combined_period_df.resample('D')['sentiment_score'].mean().to_frame('average_sentiment')
                    fig_impact = px.line(sentiment_shift_df, x=sentiment_shift_df.index, y='average_sentiment',
                                         title=f"Sentiment Shift Around {event_date.strftime('%Y-%m-%d')}", markers=True)
                    
                    fig_impact.add_vline(x=event_dt, line_width=3, line_dash="dash", line_color="red", annotation_text="Event Date")
                    
                    fig_impact.update_layout(yaxis=dict(range=[-1,1]))
                    st.plotly_chart(fig_impact, use_container_width=True)
        else:
            st.info("No valid date/timestamp column was selected to perform Temporal Analysis.")
    
    st.markdown("---")
    st.header("Explore Full Enriched Dataset")
    st.dataframe(df)

else:
    st.info("Awaiting data upload and analysis to begin.")
