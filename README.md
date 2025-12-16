# 🛍️ Retail Insights Assistant - Complete Project Documentation

## 📋 Project Overview

A production-ready GenAI chatbot that provides intelligent analytics for Amazon sales data, powered by Google's Gemini AI. Includes comprehensive scalability architecture for handling 100GB+ datasets.

### 🎯 Key Features

✅ **AI-Powered Chat Interface** - Natural language queries using Gemini Pro  
✅ **Interactive Visualizations** - Plotly charts for data exploration  
✅ **Comprehensive Analytics** - Revenue, geographical, product insights  
✅ **Scalable Architecture** - Ready for 100GB+ data with BigQuery/Spark  
✅ **Real-time Processing** - Instant responses with smart caching  
✅ **Beautiful UI** - Modern Streamlit interface with React components

---

## 🏗️ Project Structure

```
retail-insights-assistant/
│
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration management
├── data_processor.py               # Data processing & analytics
├── ai_assistant.py                 # Gemini AI integration
├── visualizations.py               # Plotly visualizations
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (create this)
│
├── scalability/                   # 100GB scalability implementation
│   ├── data_ingestion_pipeline.py # Spark ETL pipeline
│   ├── bigquery_connector.py      # BigQuery integration
│   ├── semantic_search.py         # Vector DB + LangChain
│   ├── caching_layer.py           # Redis caching
│   └── kubernetes/                # K8s deployment configs
│       ├── deployment.yaml
│       ├── service.yaml
│       └── hpa.yaml
│
├── docs/                          # Documentation
│   ├── architecture.md            # Scalability architecture
│   ├── api_reference.md           # API documentation
│   └── deployment_guide.md        # Deployment instructions
│
├── tests/                         # Unit tests
│   ├── test_data_processor.py
│   ├── test_ai_assistant.py
│   └── test_integration.py
│
└── data/                          # Data directory
    ├── raw/                       # Raw CSV files
    ├── processed/                 # Cleaned data
    └── sample/                    # Sample datasets
```

---

## ⚡ Quick Start Guide

### Prerequisites

- Python 3.9+
- Gemini API Key ([Get it here](https://makersuite.google.com/app/apikey))
- 4GB+ RAM
- Internet connection

### 1️⃣ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/retail-insights-assistant.git
cd retail-insights-assistant

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 3️⃣ Run the Application

```bash
# Start Streamlit app
streamlit run streamlit_app.py

# The app will open in your browser at http://localhost:8501
```

### 4️⃣ Upload Your Data

1. Click "Upload CSV" in the sidebar
2. Select your `Cleaned_Amazon_Sale_Report.csv`
3. Wait for processing to complete
4. Start asking questions!

---

## 📊 Sample Queries to Try

### Revenue Analysis
- "What is the total revenue for 2022?"
- "Which month had the highest sales?"
- "Show me quarterly revenue trends"

### Geographical Insights
- "Which state performed best in Q3?"
- "What are the top 5 cities by revenue?"
- "Compare Maharashtra vs Karnataka performance"

### Product Analytics
- "What are the top selling categories?"
- "Which product sizes are most popular?"
- "Show me category-wise revenue breakdown"

### Business Metrics
- "What's the average order value?"
- "What is the cancellation rate?"
- "How does B2B compare to B2C sales?"

---

## 🎨 Screenshots & Examples

### 1. Chat Interface
```
User: Which region performed best in Q3?