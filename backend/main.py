# python_backend/main.py

# --- Core FastAPI & Stdlib Imports ---
import os
import re
import json
import asyncio
from dotenv import load_dotenv

# --- Third-party Library Imports ---
import pandas as pd
import google.generativeai as genai
import openai
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- SQLAlchemy Imports ---
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey, Text, func
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base

# ==============================================================================
# 1. CONFIGURATION AND INITIALIZATION
# ==============================================================================
load_dotenv()
app = FastAPI(title="Multi-Model Database Analysis API Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Database Configuration ---
DB_CONFIG = {
    "host": os.getenv("DB_HOST"), "name": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"), "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT")
}
if not all(DB_CONFIG.values()):
    raise RuntimeError("FATAL: Not all database environment variables are set. Please check your .env file.")
DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['name']}"

try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    print("✅ Database connection configured successfully.")
except Exception as e:
    print(f"❌ FATAL: Could not connect to the database. Error: {e}")
    exit()

# --- AI Model and API Key Configuration ---
MODEL_MAP = { "gemini-pro": "gemini-2.5-pro", "openai-gpt5-mini": "gpt-5-mini" }
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Google Gemini API Key configured.")
else:
    print("⚠️ Gemini API Key not found in .env file.")
if OPENAI_API_KEY:
    print("✅ OpenAI API Key configured.")
else:
    print("⚠️ OpenAI API Key not found in .env file.")

# ==============================================================================
# 2. DATABASE ORM MODELS (CORRECTED TO MATCH YOUR SCHEMA)
# ==============================================================================
class TopicToScrape(Base):
    __tablename__ = 'topic_to_scrapes'
    id = Column(Integer, primary_key=True)
    title = Column(String)
    status = Column(String)
    context = Column(Text)
    start_date = Column(DateTime(timezone=True))
    end_date = Column(DateTime(timezone=True))
    deleted_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    facebook_comments = relationship("FacebookComment", back_populates="topic")
    tiktok_comments = relationship("TiktokComment", back_populates="topic")
    youtube_comments = relationship("YoutubeComment", back_populates="topic")

class ExtractedKeyword(Base):
    __tablename__ = 'extracted_keywords'
    id = Column(Integer, primary_key=True)
    keyword = Column(String)
    entity_type = Column(String)
    topic_to_scrape_id = Column(Integer, ForeignKey('topic_to_scrapes.id'))
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

class FacebookComment(Base):
    __tablename__ = 'facebook_comments'
    id = Column(Integer, primary_key=True)
    topic_to_scrape_id = Column(Integer, ForeignKey('topic_to_scrapes.id'))
    sentiment_analysis_status = Column(String, default='pending')
    created_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True))
    topic = relationship("TopicToScrape", back_populates="facebook_comments")
    snapshots = relationship("FacebookCommentSnapshot", back_populates="comment")

class FacebookCommentSnapshot(Base):
    __tablename__ = 'facebook_comment_snapshots'
    id = Column(Integer, primary_key=True)
    facebook_comment_id = Column(Integer, ForeignKey('facebook_comments.id'), nullable=False)
    fb_comment_text = Column(Text)
    fb_date_created = Column(DateTime(timezone=True))
    comment = relationship("FacebookComment", back_populates="snapshots")

class TiktokComment(Base):
    __tablename__ = 'tiktok_comments'
    id = Column(Integer, primary_key=True)
    topic_to_scrape_id = Column(Integer, ForeignKey('topic_to_scrapes.id'))
    sentiment_analysis_status = Column(String, default='pending')
    created_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True))
    topic = relationship("TopicToScrape", back_populates="tiktok_comments")
    snapshots = relationship("TiktokCommentSnapshot", back_populates="comment")

class TiktokCommentSnapshot(Base):
    __tablename__ = 'tiktok_comment_snapshots'
    id = Column(Integer, primary_key=True)
    tiktok_comment_id = Column(Integer, ForeignKey('tiktok_comments.id'), nullable=False)
    ttk_comment_text = Column(Text)
    ttk_date_created = Column("ttk_post_date_created", DateTime(timezone=True))
    comment = relationship("TiktokComment", back_populates="snapshots")

class YoutubeComment(Base):
    __tablename__ = 'youtube_comments'
    id = Column(Integer, primary_key=True)
    topic_to_scrape_id = Column(Integer, ForeignKey('topic_to_scrapes.id'))
    sentiment_analysis_status = Column(String, default='pending')
    created_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True))
    topic = relationship("TopicToScrape", back_populates="youtube_comments")
    snapshots = relationship("YoutubeCommentSnapshot", back_populates="comment")

class YoutubeCommentSnapshot(Base):
    __tablename__ = 'youtube_comment_snapshots'
    id = Column(Integer, primary_key=True)
    youtube_comment_id = Column(Integer, ForeignKey('youtube_comments.id'), nullable=False)
    yt_comment_text = Column(Text)
    yt_date = Column(String)
    comment = relationship("YoutubeComment", back_populates="snapshots")

class SentimentValue(Base):
    __tablename__ = 'sentiment_values'
    id = Column(Integer, primary_key=True)
    polarity_value = Column(Float)
    language = Column(String)
    model_used = Column(String)
    sentimentable_id = Column(Integer)
    sentimentable_type = Column(String)
    extracted_keyword_id = Column(Integer, ForeignKey('extracted_keywords.id'), nullable=True)
    topic_to_scrape_id = Column(Integer, ForeignKey('topic_to_scrapes.id'))
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

# This is safe to run as it won't drop existing tables.
Base.metadata.create_all(bind=engine)
print("✅ SQLAlchemy ORM models synchronized with schema.")

# ==============================================================================
# 3. Pydantic Models & DB Dependency
# ==============================================================================
class SummaryRequest(BaseModel):
    summary_data: str
    model_choice: str

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# ==============================================================================
# 4. HELPER & AI ANALYSIS FUNCTIONS
# ==============================================================================
from knowledge_base_for_prompt import get_entity_map_for_prompt

def clean_text_for_api(text: str) -> str:
    if not isinstance(text, str): return ""
    cleaned = re.sub(r'[\n\r\t]+', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned[:1500]

async def analyze_batch_with_gemini(batch_df: pd.DataFrame, batch_num: int, model_id: str):
    # This function is correct and requires no changes.
    pass # Omitted for brevity, but you should paste the full function here.

async def analyze_batch_with_openai(batch_df: pd.DataFrame, batch_num: int, model_id: str):
    # This function is correct and requires no changes.
    pass # Omitted for brevity, but you should paste the full function here.

# ==============================================================================
# 5. API ENDPOINTS
# ==============================================================================
SOURCE_MAP = {
    'FacebookComment': {'main': FacebookComment, 'snapshot': FacebookCommentSnapshot, 'date_field': 'fb_date_created', 'text_field': 'fb_comment_text', 'fk_field': 'facebook_comment_id'},
    'TiktokComment':   {'main': TiktokComment,   'snapshot': TiktokCommentSnapshot,   'date_field': 'ttk_date_created', 'text_field': 'ttk_comment_text', 'fk_field': 'tiktok_comment_id'},
    'YoutubeComment':  {'main': YoutubeComment,  'snapshot': YoutubeCommentSnapshot,  'date_field': 'yt_date', 'text_field': 'yt_comment_text', 'fk_field': 'youtube_comment_id'},
}

@app.get("/topics/")
async def get_topics(db: Session = Depends(get_db)):
    topics = db.query(TopicToScrape).order_by(TopicToScrape.created_at.desc()).all()
    if not topics: return [{"id": 0, "title": "No topics found in database"}]
    return [{"id": topic.id, "title": topic.title} for topic in topics]

@app.get("/analyze-timeseries/{topic_id}")
async def analyze_timeseries_endpoint(topic_id: int, freq: str = 'D', db: Session = Depends(get_db)):
    """
    CORRECTED: Fetches timestamps ONLY from comments marked 'completed' to align with the dashboard.
    """
    all_dates = []
    for source_info in SOURCE_MAP.values():
        main_model, snapshot_model, date_field, fk_field = source_info['main'], source_info['snapshot'], source_info['date_field'], source_info['fk_field']
        dates = db.query(getattr(snapshot_model, date_field)).join(main_model, getattr(snapshot_model, fk_field) == main_model.id).filter(main_model.topic_to_scrape_id == topic_id, main_model.sentiment_analysis_status == 'completed').all()
        all_dates.extend([d for d in dates])
    if not all_dates: return []
    ts = pd.Series(pd.to_datetime(all_dates, errors='coerce')).dropna()
    if ts.empty: return []
    counts = ts.value_counts().resample(freq).sum().reset_index(name='count')
    counts.columns = ['timestamp', 'count']
    counts['date'] = counts['timestamp'].dt.strftime('%Y-%m-%d')
    return counts[['date', 'count']].to_dict(orient='records')

@app.get("/dashboard-data/{topic_id}")
async def get_dashboard_data(topic_id: int, db: Session = Depends(get_db)):
    """
    DEFINITIVE FIX: Uses a direct JOIN to robustly fetch data for completed comments.
    """
    topic = db.query(TopicToScrape).filter(TopicToScrape.id == topic_id).first()
    if not topic: raise HTTPException(status_code=404, detail="Topic not found")

    print(f"📊 Fetching dashboard data for COMPLETED comments in topic: '{topic.title}'")
    all_sentiments = []
    
    # Iterate through each source and run a direct JOIN query
    for source_name, source_info in SOURCE_MAP.items():
        main_model = source_info['main']
        query_results = (
            db.query(SentimentValue)
            .join(main_model, SentimentValue.sentimentable_id == main_model.id)
            .filter(
                main_model.topic_to_scrape_id == topic_id,
                main_model.sentiment_analysis_status == 'completed',
                SentimentValue.sentimentable_type == source_name
            )
            .all()
        )
        all_sentiments.extend(query_results)

    print(f"Found {len(all_sentiments)} sentiment records for completed comments.")

    if not all_sentiments:
        raise HTTPException(status_code=404, detail="No 'completed' analysis data found for this topic.")
    
    keywords_map = {kw.id: kw for kw in db.query(ExtractedKeyword).filter(ExtractedKeyword.topic_to_scrape_id == topic_id).all()}
    per_comment_data, per_entity_data = [], []

    for sentiment in all_sentiments:
        if sentiment.extracted_keyword_id is None:
            per_comment_data.append({"overall_sentiment_score": sentiment.polarity_value, "language": sentiment.language})
        else:
            keyword = keywords_map.get(sentiment.extracted_keyword_id)
            if keyword:
                per_entity_data.append({"entity_name": keyword.keyword, "entity_type": keyword.entity_type, "entity_sentiment_score": sentiment.polarity_value})
    
    return {"per_comment_data": per_comment_data, "per_entity_data": per_entity_data}

@app.post("/analyze-comments/{topic_id}")
async def analyze_comments_endpoint(topic_id: int, model_choice: str = 'gemini-pro', db: Session = Depends(get_db)):
    # This endpoint is correct and requires no changes. Omitted for brevity.
    pass