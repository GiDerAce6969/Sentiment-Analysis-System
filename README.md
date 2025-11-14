# Hybrid Political Sentiment Analysis Dashboard

This project is a full-stack application designed to analyze the sentiment of social media comments related to political topics. It features a React frontend for data visualization and a Python (FastAPI) backend that leverages large language models (like Google Gemini and OpenAI GPT) for sentiment analysis and entity extraction. The data is stored and retrieved from a Supabase PostgreSQL database.

## Features

- **Dynamic Topic Selection:** Load and select different analysis topics from the database.
- **Comprehensive Dashboard:** Visualize key metrics like total comments, average sentiment, sentiment distribution, and top-mentioned political leaders and parties.
- **Time Series Analysis:** View comment volume over time (Daily, Weekly, Monthly).
- **AI-Powered Analysis:** Use a button to trigger backend analysis of new, unprocessed comments using your choice of AI model.
- **Scalable Backend:** The backend processes comments in concurrent batches for efficiency.
- **Persistent Storage:** All analysis results are stored in a PostgreSQL database (Supabase).

## Tech Stack

- **Frontend:** React, Axios, Plotly.js
- **Backend:** Python, FastAPI, SQLAlchemy, Uvicorn
- **AI:** Google Gemini, OpenAI GPT
- **Database:** Supabase (PostgreSQL)

---

## 🚀 Setup and Installation

Follow these steps to get the project running locally.

### Step 1: Set up Supabase Database

1.  **Create a Supabase Project:**
    - Go to [supabase.com](https://supabase.com) and create a new project.
    - Save your database password securely.

2.  **Get Database Credentials:**
    - In your Supabase project, go to **Project Settings** > **Database**.
    - Find your connection details: `Host`, `Database name`, `User`, `Port`, and the `password` you set.

3.  **Create Tables and Insert Sample Data:**
    - Go to the **SQL Editor** in your Supabase project.
    - Copy the entire content of the `supabase/schema_and_sample_data.sql` file from this repository.
    - Paste it into the SQL Editor and click **RUN**. This will create all the necessary tables and add sample data to make the dashboard work instantly.

### Step 2: Set up Backend

1.  **Navigate to the backend folder:**
    ```bash
    cd backend
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create an environment file:**
    - Rename the `.env.example` file to `.env`.
    - Fill in the file with your credentials from Supabase and your AI API keys.

    ```env
    # Supabase DB Credentials
    DB_HOST=your-supabase-host
    DB_NAME=postgres
    DB_USER=postgres
    DB_PASSWORD=your-supabase-db-password
    DB_PORT=5432

    # AI API Keys
    GOOGLE_API_KEY=your-google-api-key
    OPENAI_API_KEY=your-openai-api-key
    ```

5.  **Run the backend server:**
    ```bash
    uvicorn main:app --reload
    ```
    The backend will be running at `http://127.0.0.1:8000`.

### Step 3: Set up Frontend

1.  **Open a new terminal** and navigate to the frontend folder:
    ```bash
    cd frontend
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Run the frontend development server:**
    ```bash
    npm start
    ```
    The React app will open in your browser at `http://localhost:3000`.

## How to Use the Demo

1.  The dashboard will automatically load and display the data for the "Sabah Election" topic, which includes a few pre-analyzed ("completed") comments.
2.  You will see that some metrics are populated, but others may be low.
3.  Click the **"Analyze New Comments"** button. The backend will process the sample "pending" comments.
4.  After the analysis is complete, the dashboard will automatically refresh and display the newly analyzed data, and all the charts will update.
