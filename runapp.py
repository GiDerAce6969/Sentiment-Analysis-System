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

# --- EXPANDED Malaysian Political Dictionary ---
MALAYSIAN_POLITICAL_DICTIONARY = {
    "leaders": {
        # Pakatan Harapan (PH) & Allies
        "anwar": "Anwar Ibrahim", "pmx": "Anwar Ibrahim", "madanon": "Anwar Ibrahim",
        "wan azizah": "Wan Azizah Wan Ismail",
        "rafizi": "Rafizi Ramli",
        "anthony loke": "Anthony Loke Siew Fook",
        "gobind": "Gobind Singh Deo",
        "lim guan eng": "Lim Guan Eng", "lce": "Lim Guan Eng", "guan eng": "Lim Guan Eng",
        "mat sabu": "Mohamad Sabu",
        
        # Barisan Nasional (BN)
        "zahid": "Ahmad Zahid Hamidi", "zahid komedi": "Ahmad Zahid Hamidi",
        "tok mat": "Mohamad Hasan",
        "ismail sabri": "Ismail Sabri Yaakob",
        "wee ka siong": "Wee Ka Siong",

        # Perikatan Nasional (PN)
        "muhyiddin": "Muhyiddin Yassin", "my": "Muhyiddin Yassin", "abah": "Muhyiddin Yassin",
        "hadi": "Hadi Awang", "lebai": "Hadi Awang",
        "azmin ali": "Azmin Ali",
        "hamzah": "Hamzah Zainudin",
        "sanusi": "Muhammad Sanusi Md Nor",

        # Gabungan Rakyat Sabah (GRS) & Sarawak (GPS)
        "hajiji": "Hajiji Noor",
        "jeffrey kitingan": "Jeffrey Kitingan",
        "abang jo": "Abang Johari Openg",
        
        # Other Key Figures
        "shafie": "Shafie Apdal",
        "mahathir": "Mahathir Mohamad", "tun m": "Mahathir Mohamad",
        "syed saddiq": "Syed Saddiq Syed Abdul Rahman",
        "najib": "Najib Razak", "bossku": "Najib Razak"
    },
    "parties": {
        # Main Coalitions
        "ph": "Pakatan Harapan",
        "bn": "Barisan Nasional",
        "pn": "Perikatan Nasional",
        "grs": "Gabungan Rakyat Sabah",
        "gps": "Gabungan Parti Sarawak",

        # Component Parties
        "pkr": "Parti Keadilan Rakyat (PKR)",
        "dap": "DAP",
        "amanah": "Parti Amanah Negara (Amanah)",
        "umno": "UMNO",
        "mca": "MCA",
        "mic": "MIC",
        "pas": "PAS",
        "bersatu": "Parti Pribumi Bersatu Malaysia (Bersatu)",
        "gerakan": "Parti Gerakan Rakyat Malaysia (Gerakan)",
        "warisan": "Parti Warisan Sabah (Warisan)",
        "muda": "MUDA"
    }
}

# --- STAGE 1: BATCH COMMENT ENRICHMENT ---
def enrich_comments_with_gemini(df_comments, comment_column, batch_size=100):
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
    model = genai.GenerativeModel('gemini-2.5-pro')
    
    df_entities = df_enriched.explode('mentioned_entities').dropna(subset=['mentioned_entities'])
    leader_mentions = df_entities[df_entities['mentioned_entities'].isin(MALAYSIAN_POLITICAL_DICTIONARY['leaders'].values())]
    party_mentions = df_entities[df_entities['mentioned_entities'].isin(MALAYSIAN_POLITICAL_DICTIONARY['parties'].values())]

    summary_data = {
        "topic_of_analysis": main_topic,
        "total_comments": len(df_enriched),
        "overall_sentiment_score": df_enriched['sentiment_score'].mean(),
        "sentiment_distribution_percentage": df_enriched['sentiment_label'].value_counts(normalize=True).mul(100).round(1).to_dict(),
        "leader_sentiment_analysis": leader_mentions.groupby('mentioned_entities')['sentiment_score'].agg(['mean', 'count']).rename(columns={'mean': 'avg_sentiment', 'count': 'mentions'}).to_dict('index'),
        "party_sentiment_analysis": party_mentions.groupby('mentioned_entities')['sentiment_score'].agg(['mean', 'count']).rename(columns={'mean': 'avg_sentiment', 'count': 'mentions'}).to_dict('index')
    }

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
    
    uploaded_files = st.file_uploader(
        "Upload one or more CSV or Excel files",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        df_list = []
        for file in uploaded_files:
            try:
                if file.name.endswith('.csv'):
                    df_temp = pd.read_csv(file)
                else:
                    df_temp = pd.read_excel(file)
                df_list.append(df_temp)
            except Exception as e:
                st.error(f"Error reading file '{file.name}': {e}")
        
        if df_list:
            df_original = pd.concat(df_list, ignore_index=True)
            st.session_state.df_original = df_original
            st.success(f"Loaded and combined {len(uploaded_files)} file(s) with a total of {len(df_original)} comments.")

    if 'df_original' in st.session_state and st.session_state.df_original is not None:
        df_original = st.session_state.df_original
        st.header("2. Configure Analysis")
        comment_column = st.selectbox("Select the comment column:", options=df_original.columns.tolist())
        datetime_column = st.selectbox("Select the date/timestamp column:", options=[None] + df_original.columns.tolist())
        
        batch_size = st.slider(
            "Comments per API Call (Batch Size)", 
            min_value=50, max_value=1000, value=500, step=50,
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
        topic_model = genai.GenerativeModel('gemini-2.5-pro')
        try:
            main_topic = topic_model.generate_content(topic_prompt).text.strip()
        except Exception as e:
            st.warning(f"Could not determine main topic from AI: {e}")
            main_topic = "Analysis of Uploaded Comments"
        st.subheader(f"Main Topic Determined by AI: **{main_topic}**")
    else:
        st.warning("Could not determine the main topic because the 'primary_topic' column was not generated during AI analysis.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Comments", f"{len(df):,}")
    if 'sentiment_score' in df.columns and not df['sentiment_score'].isnull().all():
        avg_sentiment = df['sentiment_score'].mean()
        col2.metric("Average Sentiment", f"{avg_sentiment:.2f}")
        sentiment_label = "Positive" if avg_sentiment > 0.2 else "Negative" if avg_sentiment < -0.2 else "Neutral"
        col3.metric("Overall Sentiment", sentiment_label)
    
    st.markdown("---")
    
    st.header("Dashboard Insights")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Sentiment Breakdown", "Topic Analysis", "Political Analysis", "Time Series"])

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
            else:
                st.warning("Sentiment Label column not found.")
        with col2:
            if 'language' in df.columns:
                lang_counts = df['language'].value_counts()
                fig = px.bar(lang_counts, x=lang_counts.index, y=lang_counts.values,
                             title="Language Distribution", labels={'x': 'Language', 'y': 'Comment Count'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Language column not found.")

        col1, col2 = st.columns(2)
        with col1:
            if 'inferred_race' in df.columns:
                race_counts = df['inferred_race'].value_counts()
                fig = px.pie(race_counts, names=race_counts.index, values=race_counts.values, title="Comment Distribution by Inferred Race")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Inferred Race column not found.")
        with col2:
            if 'inferred_region' in df.columns:
                region_counts = df['inferred_region'].value_counts()
                fig = px.pie(region_counts, names=region_counts.index, values=region_counts.values, title="Comment Distribution by Inferred Region")
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Top Topics Discussed")
        if 'primary_topic' in df.columns and 'sentiment_score' in df.columns:
            topic_sentiment = df.groupby('primary_topic').agg(
                comment_count=('primary_topic', 'count'),
                avg_sentiment=('sentiment_score', 'mean')
            ).sort_values(by='comment_count', ascending=False).reset_index()

            fig = px.bar(topic_sentiment.head(15), x='primary_topic', y='comment_count', color='avg_sentiment',
                         color_continuous_scale=px.colors.diverging.RdYlGn, range_color=[-1,1],
                         title="Top 15 Topics by Mention Count (Colored by Average Sentiment)",
                         labels={'primary_topic': 'Topic', 'comment_count': 'Number of Comments', 'avg_sentiment': 'Avg. Sentiment'})
            fig.update_layout(xaxis_title="Topic", yaxis_title="Number of Comments")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Primary Topic or Sentiment Score columns not found.")

    with tab3:
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
            st.warning("Mentioned Entities column not found.")

    with tab4:
        st.subheader("Time Series Analysis")
        if datetime_column and datetime_column in df.columns and not df[datetime_column].isnull().all():
            df_time = df.set_index(datetime_column)
            
            time_freq = st.selectbox("Select Time Aggregation:", ["Daily (D)", "Weekly (W)", "Monthly (M)"], index=0)
            freq_code = time_freq.split(" ")[1][1] 
            
            volume_over_time = df_time.resample(freq_code).size().to_frame('comment_volume')
            sentiment_over_time = df_time.resample(freq_code)['sentiment_score'].mean().to_frame('average_sentiment')
            
            fig_vol = px.line(volume_over_time, x=volume_over_time.index, y='comment_volume', 
                              title=f"{time_freq} Discussion Volume for '{main_topic}'", markers=True)
            fig_vol.update_layout(xaxis_title="Date", yaxis_title="Number of Comments")
            st.plotly_chart(fig_vol, use_container_width=True)
            
            fig_sent = px.line(sentiment_over_time, x=sentiment_over_time.index, y='average_sentiment',
                               title=f"{time_freq} Average Sentiment for '{main_topic}'", markers=True)
            fig_sent.update_layout(xaxis_title="Date", yaxis_title="Average Sentiment Score", yaxis=dict(range=[-1,1]))
            st.plotly_chart(fig_sent, use_container_width=True)
        else:
            st.info("No valid date/timestamp column was selected to perform a time series analysis.")

    st.markdown("---")
    st.header("Generate AI-Powered Executive Summary")
    if st.button("Generate Summary"):
        with st.spinner("Gemini is crafting the final report..."):
            summary_report = generate_ai_summary(df, main_topic)
            st.markdown(summary_report)
    
    st.markdown("---")
    st.header("Explore Full Enriched Dataset")
    st.dataframe(df)

else:
    st.info("Awaiting data upload and analysis to begin.")
