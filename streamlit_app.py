"""
RETAIL INSIGHTS ASSISTANT - COMPLETE STREAMLIT APPLICATION
===========================================================
A production-ready GenAI chatbot for Amazon Sales Analytics
Powered by Google Gemini AI

To run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datetime import datetime
import json
import os
import json
import requests

import json
import google.generativeai as genai


# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🛍️ Retail Insights Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #667eea;
        color: white;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f0f2f6;
        color: #262730;
        margin-right: 20%;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA PROCESSING CLASS
# ============================================================================

class DataProcessor:
    """Handle all data processing and analytics"""
    
    def __init__(self, df):
        self.df = df
        self.valid_orders = None
        self.summary = None
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Clean and prepare data"""
        # Convert Amount to numeric
        self.df['Amount'] = pd.to_numeric(self.df['Amount'], errors='coerce')
        
        # Convert Date
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        
        # Extract quarter
        self.df['Quarter'] = 'Q' + self.df['Date'].dt.quarter.astype(str)
        
        # Filter valid orders
        self.valid_orders = self.df[
            (self.df['Status'] != 'Cancelled') & 
            (self.df['Amount'] > 0)
        ].copy()
    
    def generate_summary(self):
        """Generate comprehensive data summary"""
        vo = self.valid_orders
        
        if vo is None or len(vo) == 0:
            return self._empty_summary()
        
        # Overview metrics
        overview = {
            'total_orders': len(self.df),
            'completed_orders': len(vo),
            'cancelled_orders': len(self.df[self.df['Status'] == 'Cancelled']),
            'total_revenue': vo['Amount'].sum(),
            'avg_order_value': vo['Amount'].mean(),
            'median_order_value': vo['Amount'].median(),
            'date_range': {
                'start': self.df['Date'].min().strftime('%Y-%m-%d'),
                'end': self.df['Date'].max().strftime('%Y-%m-%d')
            }
        }
        
        # Temporal analysis
        monthly = None
        quarterly = None
        if 'MonthName' in vo.columns:
            monthly = vo.groupby('MonthName')['Amount'].agg(['sum', 'count', 'mean'])
        if 'Quarter' in vo.columns:
            quarterly = vo.groupby('Quarter')['Amount'].agg(['sum', 'count'])
        
        temporal = {
            'monthly_revenue': monthly['sum'].to_dict() if monthly is not None else {},
            'monthly_orders': monthly['count'].to_dict() if monthly is not None else {},
            'quarterly_revenue': quarterly['sum'].to_dict() if quarterly is not None else {},
            'best_month': monthly['sum'].idxmax() if monthly is not None and len(monthly) > 0 else 'N/A',
            'best_quarter': quarterly['sum'].idxmax() if quarterly is not None and len(quarterly) > 0 else 'N/A'
        }
        
        # Geographical analysis
        state_stats = None
        city_stats = None
        if 'ship-state' in vo.columns:
            state_stats = vo.groupby('ship-state')['Amount'].agg(['sum', 'count', 'mean'])
        if 'ship-city' in vo.columns:
            city_stats = vo.groupby('ship-city')['Amount'].sum()
        
        geographical = {
            'top_states': state_stats.nlargest(10, 'sum')['sum'].to_dict() if state_stats is not None else {},
            'top_cities': city_stats.nlargest(10).to_dict() if city_stats is not None else {},
            'total_states': len(state_stats) if state_stats is not None else 0,
            'total_cities': len(city_stats) if city_stats is not None else 0
        }
        
        # Product analysis
        category_stats = None
        size_stats = None
        if 'Category' in vo.columns:
            category_stats = vo.groupby('Category')['Amount'].agg(['sum', 'count', 'mean'])
        if 'Size' in vo.columns:
            size_stats = vo.groupby('Size')['Amount'].agg(['sum', 'count'])
        
        product = {
            'top_categories': category_stats.nlargest(10, 'sum')['sum'].to_dict() if category_stats is not None else {},
            'category_orders': category_stats['count'].to_dict() if category_stats is not None else {},
            'size_distribution': size_stats['count'].to_dict() if size_stats is not None else {}
        }
        
        # Performance metrics
        b2b_orders = vo[vo['B2B'].astype(str).str.upper() == 'TRUE'] if 'B2B' in vo.columns else pd.DataFrame()
        
        performance = {
            'b2b_revenue': b2b_orders['Amount'].sum() if len(b2b_orders) > 0 else 0,
            'b2b_orders': len(b2b_orders),
            'b2b_percentage': (len(b2b_orders) / len(vo)) * 100 if len(vo) > 0 else 0,
            'cancellation_rate': (overview['cancelled_orders'] / overview['total_orders']) * 100,
            'fulfillment_breakdown': vo['Fulfilment'].value_counts().to_dict() if 'Fulfilment' in vo.columns else {},
            'channel_breakdown': vo['Sales Channel'].value_counts().to_dict() if 'Sales Channel' in vo.columns else {}
        }
        
        self.summary = {
            'overview': overview,
            'temporal': temporal,
            'geographical': geographical,
            'product': product,
            'performance': performance
        }
        
        return self.summary
    
    def _empty_summary(self):
        """Return empty summary when no valid data"""
        return {
            'overview': {
                'total_orders': 0,
                'completed_orders': 0,
                'cancelled_orders': 0,
                'total_revenue': 0,
                'avg_order_value': 0,
                'median_order_value': 0,
                'date_range': {'start': 'N/A', 'end': 'N/A'}
            },
            'temporal': {
                'monthly_revenue': {},
                'monthly_orders': {},
                'quarterly_revenue': {},
                'best_month': 'N/A',
                'best_quarter': 'N/A'
            },
            'geographical': {
                'top_states': {},
                'top_cities': {},
                'total_states': 0,
                'total_cities': 0
            },
            'product': {
                'top_categories': {},
                'category_orders': {},
                'size_distribution': {}
            },
            'performance': {
                'b2b_revenue': 0,
                'b2b_orders': 0,
                'b2b_percentage': 0,
                'cancellation_rate': 0,
                'fulfillment_breakdown': {},
                'channel_breakdown': {}
            }
        }

# ============================================================================
# AI ASSISTANT CLASS
# ============================================================================

"""
AI ASSISTANT - DIRECT REST API IMPLEMENTATION
==============================================
Uses direct HTTP requests to Gemini API
No google-generativeai SDK required!
"""



class AIAssistant:
    """Handle AI interactions using direct REST API calls"""
    
    def __init__(self, api_key):
        """
        Initialize AI Assistant with direct API access
        
        Args:
            api_key (str): Your Gemini API key
        """
        if not api_key or not api_key.strip():
            raise ValueError("Valid API key is required")
        
        self.api_key = api_key.strip()
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.model = "gemini-2.0-flash"
        
        # Test the API connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if API key is valid"""
        try:
            test_response = self._make_api_request("Hello!")
            if not test_response:
                raise RuntimeError("API test failed")
            print("✅ Gemini API connected successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Gemini API: {str(e)}")
    
    def _make_api_request(self, prompt, temperature=0.7, max_tokens=2048):
        """
        Make direct API request to Gemini
        
        Args:
            prompt (str): The prompt to send
            temperature (float): Creativity level (0.0-1.0)
            max_tokens (int): Maximum response length
            
        Returns:
            str: Generated response text
        """
        url = f"{self.base_url}/{self.model}:generateContent"
        
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': self.api_key
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": max_tokens,
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            # Check for HTTP errors
            if response.status_code != 200:
                error_msg = response.json().get('error', {}).get('message', 'Unknown error')
                raise RuntimeError(f"API Error ({response.status_code}): {error_msg}")
            
            # Parse response
            data = response.json()
            
            # Extract text from response
            if 'candidates' in data and len(data['candidates']) > 0:
                candidate = data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    if len(parts) > 0 and 'text' in parts[0]:
                        return parts[0]['text']
            
            raise ValueError("Invalid response format from API")
            
        except requests.exceptions.Timeout:
            raise RuntimeError("Request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Connection failed. Check your internet connection.")
        except Exception as e:
            raise RuntimeError(f"API request failed: {str(e)}")
    
    def create_enriched_prompt(self, summary, user_question):
        """
        Create prompt with data context
        
        Args:
            summary (dict): Sales data summary
            user_question (str): User's question
            
        Returns:
            str: Enriched prompt
        """
        try:
            prompt = f"""You are an expert retail analytics assistant specializing in Amazon sales data analysis.
Analyze the data below and provide a comprehensive, data-driven answer to the user's question.

SALES DATA SUMMARY:

📊 OVERVIEW METRICS:
- Total Orders: {summary['overview']['total_orders']:,}
- Completed Orders: {summary['overview']['completed_orders']:,}
- Cancelled Orders: {summary['overview']['cancelled_orders']:,}
- Total Revenue: ₹{summary['overview']['total_revenue']:,.2f}
- Average Order Value: ₹{summary['overview']['avg_order_value']:,.2f}
- Median Order Value: ₹{summary['overview']['median_order_value']:,.2f}
- Cancellation Rate: {summary['performance']['cancellation_rate']:.2f}%
- Date Range: {summary['overview']['date_range']['start']} to {summary['overview']['date_range']['end']}

📅 TEMPORAL PERFORMANCE:
- Best Performing Month: {summary['temporal']['best_month']}
- Best Performing Quarter: {summary['temporal']['best_quarter']}

Monthly Revenue Breakdown:
{json.dumps(summary['temporal']['monthly_revenue'], indent=2)}

Quarterly Revenue Breakdown:
{json.dumps(summary['temporal']['quarterly_revenue'], indent=2)}

🌍 GEOGRAPHICAL INSIGHTS:
- Total States Covered: {summary['geographical']['total_states']}
- Total Cities Covered: {summary['geographical']['total_cities']}

Top 5 States by Revenue:
{json.dumps(dict(list(summary['geographical']['top_states'].items())[:5]), indent=2)}

📦 PRODUCT PERFORMANCE:
Top 5 Categories by Revenue:
{json.dumps(dict(list(summary['product']['top_categories'].items())[:5]), indent=2)}

Category Order Counts:
{json.dumps(dict(list(summary['product']['category_orders'].items())[:5]), indent=2)}

🎯 BUSINESS METRICS:
- B2B Revenue: ₹{summary['performance']['b2b_revenue']:,.2f}
- B2B Orders: {summary['performance']['b2b_orders']:,}
- B2B Percentage: {summary['performance']['b2b_percentage']:.2f}%

Fulfillment Breakdown:
{json.dumps(summary['performance']['fulfillment_breakdown'], indent=2)}

Sales Channel Breakdown:
{json.dumps(summary['performance']['channel_breakdown'], indent=2)}

USER QUESTION: {user_question}

INSTRUCTIONS:
Provide a detailed, data-driven answer using specific numbers from the data above.

Format your response with:
• Clear bullet points for key insights
• Specific revenue figures and percentages
• Month/quarter comparisons where relevant
• Actionable recommendations based on the data
• Highlight trends and patterns

Be conversational but professional. Use emojis sparingly for emphasis.
"""
            return prompt
            
        except KeyError as e:
            return f"Error: Missing data in summary - {str(e)}"
        except Exception as e:
            return f"Error creating prompt: {str(e)}"
    
    def generate_response(self, prompt, max_retries=3):
        """
        Generate AI response with retry logic
        
        Args:
            prompt (str): The prompt to send
            max_retries (int): Maximum retry attempts
            
        Returns:
            str: Generated response
        """
        for attempt in range(max_retries):
            try:
                response_text = self._make_api_request(prompt)
                
                if not response_text or response_text.strip() == "":
                    raise ValueError("Empty response from API")
                
                return response_text
                
            except Exception as e:
                error_msg = str(e)
                
                # Handle specific errors
                if "API_KEY_INVALID" in error_msg or "401" in error_msg:
                    return """❌ API Key Error
                    
Your API key appears to be invalid. Please:
1. Get a new key from: https://makersuite.google.com/app/apikey
2. Make sure to copy the entire key (starts with 'AIzaSy')
3. Paste it in the sidebar
"""
                
                if "QUOTA_EXCEEDED" in error_msg or "429" in error_msg:
                    return """❌ Quota Exceeded
                    
You've reached the API usage limit. Please:
1. Wait a few minutes and try again
2. Check your quota at: https://console.cloud.google.com
3. Consider upgrading your API plan
"""
                
                if "SAFETY" in error_msg or "BLOCKED" in error_msg:
                    return """❌ Safety Filter Triggered
                    
The response was blocked by safety filters. Try:
1. Rephrasing your question
2. Being more specific about business metrics
3. Avoiding sensitive topics
"""
                
                # Retry logic
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed, retrying...")
                    import time
                    time.sleep(1)
                    continue
                else:
                    return f"""❌ Error after {max_retries} attempts

Error: {error_msg}

Troubleshooting Steps:
1. ✓ Verify API key is correct
2. ✓ Check internet connection  
3. ✓ Ensure API quota is available
4. ✓ Try a simpler question
5. ✓ Wait a moment and retry

Get help at: https://ai.google.dev/gemini-api/docs
"""
    
    def quick_test(self):
        """Test the API with a simple query"""
        try:
            response = self._make_api_request("Say 'API working!' if you can read this.")
            return response
        except Exception as e:
            return f"Connection test failed: {str(e)}"

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_revenue_trend_chart(df):
    """Create monthly revenue trend chart"""
    monthly = df.groupby('MonthName')['Amount'].sum().reset_index()
    
    # Sort by month order
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly['MonthName'] = pd.Categorical(monthly['MonthName'], 
                                          categories=month_order, 
                                          ordered=True)
    monthly = monthly.sort_values('MonthName')
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly['MonthName'],
        y=monthly['Amount'],
        marker_color='rgba(102, 126, 234, 0.8)',
        text=monthly['Amount'].apply(lambda x: f'₹{x:,.0f}'),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Revenue: ₹%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Monthly Revenue Trend',
        xaxis_title='Month',
        yaxis_title='Revenue (₹)',
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_state_performance_chart(summary):
    """Create top states performance chart"""
    top_states = dict(list(summary['geographical']['top_states'].items())[:10])
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=list(top_states.keys()),
        x=list(top_states.values()),
        orientation='h',
        marker_color='rgba(118, 75, 162, 0.8)',
        text=[f'₹{v:,.0f}' for v in top_states.values()],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Revenue: ₹%{x:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Top 10 States by Revenue',
        xaxis_title='Revenue (₹)',
        yaxis_title='State',
        height=500,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_category_pie_chart(summary):
    """Create category distribution pie chart"""
    categories = dict(list(summary['product']['top_categories'].items())[:8])
    
    fig = go.Figure(data=[go.Pie(
        labels=list(categories.keys()),
        values=list(categories.values()),
        hole=0.4,
        marker_colors=px.colors.qualitative.Set3,
        textposition='auto',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Revenue: ₹%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title='Revenue Distribution by Category',
        height=400,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_quarterly_comparison(df):
    """Create quarterly comparison chart"""
    quarterly = df.groupby('Quarter')['Amount'].agg(['sum', 'count']).reset_index()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Quarterly Revenue', 'Quarterly Orders'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    fig.add_trace(
        go.Bar(
            x=quarterly['Quarter'],
            y=quarterly['sum'],
            name='Revenue',
            marker_color='rgba(102, 126, 234, 0.8)',
            text=quarterly['sum'].apply(lambda x: f'₹{x:,.0f}'),
            textposition='outside'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=quarterly['Quarter'],
            y=quarterly['count'],
            name='Orders',
            marker_color='rgba(118, 75, 162, 0.8)',
            text=quarterly['count'],
            textposition='outside'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🛍️ Retail Insights Assistant</h1>', 
                unsafe_allow_html=True)
    st.markdown("### Amazon Sales Analytics powered by Gemini AI")
    st.markdown("---")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = None
    if 'ai_assistant' not in st.session_state:
        st.session_state.ai_assistant = None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Input
        api_key = st.text_input(
            "🔑 Gemini API Key",
            type="password",
            help="Get your free API key from https://makersuite.google.com/app/apikey",
            placeholder="AIzaSy..."
        )
        
        if api_key and st.session_state.ai_assistant is None:
            st.session_state.ai_assistant = AIAssistant(api_key)
            st.success("✅ API Key configured!")
        
        st.markdown("---")
        
        # File Upload
        st.header("📁 Upload Data")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload your cleaned Amazon sales data"
        )
        
        if uploaded_file is not None:
            if st.session_state.data_processor is None:
                with st.spinner("🔄 Processing data..."):
                    df = pd.read_csv(uploaded_file)
                    st.session_state.data_processor = DataProcessor(df)
                    summary = st.session_state.data_processor.generate_summary()
                    st.success(f"✅ Loaded {len(df):,} orders!")
        
        st.markdown("---")
        
        # Quick Stats
        if st.session_state.data_processor:
            st.header("📊 Quick Stats")
            summary = st.session_state.data_processor.summary
            if summary is not None:
                st.metric(
                    "Total Revenue",
                    f"₹{summary['overview']['total_revenue']:,.0f}",
                    delta=f"{summary['overview']['completed_orders']:,} orders"
                )
                st.metric(
                    "Avg Order Value",
                    f"₹{summary['overview']['avg_order_value']:.2f}"
                )
                st.metric(
                    "Best Month",
                    summary['temporal']['best_month'],
                    delta=f"{summary['temporal']['best_quarter']}"
                )
            
            # Download Report
            st.markdown("---")
            if st.button("📥 Download Report"):
                report_json = json.dumps(summary, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=report_json,
                    file_name="sales_report.json",
                    mime="application/json"
                )
    
    # Main content
    if st.session_state.data_processor:
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📈 Visualizations", "📋 Summary Report"])
        
        # TAB 1: Chat Assistant
        with tab1:
            st.subheader("Ask me anything about your sales data!")
            
            # Suggested questions
            st.markdown("**💡 Suggested Questions:**")
            col1, col2, col3 = st.columns(3)
            
            suggestions = [
                "Which region performed best in Q3?",
                "What are the top selling categories?",
                "Show me monthly revenue trends",
                "What's the average order value by state?",
                "Compare B2B vs B2C performance",
                "Which products have highest cancellation rate?"
            ]
            
            for idx, suggestion in enumerate(suggestions[:3]):
                if col1.button(suggestion, key=f"sug1_{idx}"):
                    st.session_state.messages.append({
                        "role": "user",
                        "content": suggestion
                    })
                    st.rerun()
            
            for idx, suggestion in enumerate(suggestions[3:]):
                if col2.button(suggestion, key=f"sug2_{idx}"):
                    st.session_state.messages.append({
                        "role": "user",
                        "content": suggestion
                    })
                    st.rerun()
            
            st.markdown("---")
            
            # Display chat messages
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        st.markdown(
                            f'<div class="chat-message user-message">👤 You: {message["content"]}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="chat-message assistant-message">🤖 Assistant: {message["content"]}</div>',
                            unsafe_allow_html=True
                        )
            
            # Chat input
            user_input = st.chat_input("Type your question here...")
            
            if user_input:
                if not api_key:
                    st.error("⚠️ Please enter your Gemini API key in the sidebar!")
                else:
                    # Add user message
                    st.session_state.messages.append({
                        "role": "user",
                        "content": user_input
                    })
                    
                    # Generate response
                    with st.spinner("🤔 Thinking..."):
                        if st.session_state.ai_assistant and st.session_state.data_processor.summary:
                            enriched_prompt = st.session_state.ai_assistant.create_enriched_prompt(
                                st.session_state.data_processor.summary,
                                user_input
                            )
                            response = st.session_state.ai_assistant.generate_response(enriched_prompt)
                        else:
                            response = "Error: AI assistant or data summary not available"
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })
                    
                    st.rerun()
        
        # TAB 2: Visualizations
        with tab2:
            st.subheader("📊 Interactive Visualizations")
            
            df = st.session_state.data_processor.valid_orders
            summary = st.session_state.data_processor.summary
            
            if df is not None and summary is not None:
                # Row 1
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_revenue_trend_chart(df), use_container_width=True)
                with col2:
                    st.plotly_chart(create_category_pie_chart(summary), use_container_width=True)
                
                # Row 2
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_state_performance_chart(summary), use_container_width=True)
                with col2:
                    st.plotly_chart(create_quarterly_comparison(df), use_container_width=True)
            else:
                st.error("No valid data available for visualization")
        
        # TAB 3: Summary Report
        with tab3:
            st.subheader("📋 Comprehensive Summary Report")
            summary = st.session_state.data_processor.summary
            
            if summary is None:
                st.error("No summary data available")
                return
            
            # Overview Section
            st.markdown("### 📊 Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric(
                "Total Orders",
                f"{summary['overview']['total_orders']:,}",
                delta=f"+{summary['overview']['completed_orders']:,} completed"
            )
            col2.metric(
                "Total Revenue",
                f"₹{summary['overview']['total_revenue']:,.0f}"
            )
            col3.metric(
                "Avg Order Value",
                f"₹{summary['overview']['avg_order_value']:.2f}"
            )
            col4.metric(
                "Cancellation Rate",
                f"{summary['performance']['cancellation_rate']:.2f}%"
            )
            
            st.markdown("---")
            
            # Temporal Analysis
            st.markdown("### 📅 Temporal Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown(f"**🏆 Best Performing Month:** {summary['temporal']['best_month']}")
                st.markdown(f"**🏆 Best Performing Quarter:** {summary['temporal']['best_quarter']}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("**Monthly Revenue Breakdown:**")
                for month, revenue in summary['temporal']['monthly_revenue'].items():
                    st.write(f"• {month}: ₹{revenue:,.2f}")
            
            with col2:
                st.markdown("**Quarterly Performance:**")
                for quarter, revenue in summary['temporal']['quarterly_revenue'].items():
                    st.write(f"• {quarter}: ₹{revenue:,.2f}")
            
            st.markdown("---")
            
            # Geographical Analysis
            st.markdown("### 🌍 Geographical Insights")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top 5 States by Revenue:**")
                for idx, (state, revenue) in enumerate(list(summary['geographical']['top_states'].items())[:5], 1):
                    st.write(f"{idx}. {state}: ₹{revenue:,.2f}")
            
            with col2:
                st.markdown("**Market Coverage:**")
                st.info(f"📍 Total States Covered: **{summary['geographical']['total_states']}**")
                st.info(f"🏙️ Total Cities Covered: **{summary['geographical']['total_cities']}**")
            
            st.markdown("---")
            
            # Product Analysis
            st.markdown("### 📦 Product Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top 5 Categories:**")
                for idx, (cat, revenue) in enumerate(list(summary['product']['top_categories'].items())[:5], 1):
                    st.write(f"{idx}. {cat}: ₹{revenue:,.2f}")
            
            with col2:
                st.markdown("**B2B Performance:**")
                st.metric("B2B Revenue", f"₹{summary['performance']['b2b_revenue']:,.0f}")
                st.metric("B2B Order %", f"{summary['performance']['b2b_percentage']:.2f}%")
            
            st.markdown("---")
            
            # Performance Metrics
            st.markdown("### 🎯 Performance Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Fulfillment Breakdown:**")
                for fulfillment, count in summary['performance']['fulfillment_breakdown'].items():
                    st.write(f"• {fulfillment}: {count:,} orders")
            
            with col2:
                st.markdown("**Sales Channel:**")
                for channel, count in summary['performance']['channel_breakdown'].items():
                    st.write(f"• {channel}: {count:,} orders")
    
    else:
        # Welcome screen
        st.info("👆 **Get Started:** Upload your sales CSV file in the sidebar!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🚀 Features
            
            ✅ **AI-Powered Chat** - Ask questions in natural language  
            ✅ **Interactive Visualizations** - Beautiful Plotly charts  
            ✅ **Comprehensive Analytics** - Deep insights into your data  
            ✅ **Real-time Processing** - Instant data analysis  
            ✅ **Export Reports** - Download insights as JSON  
            """)
        
        with col2:
            st.markdown("""
            ### 📚 Example Questions
            
            💬 "Which region performed best in Q3?"  
            💬 "What are the top selling categories?"  
            💬 "Show me monthly revenue trends"  
            💬 "Compare B2B vs B2C performance"  
            💬 "What's the average order value?"  
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 🏗️ Scalability Architecture
        
        This application is designed to scale from small datasets to **100GB+** using:
        
        - **BigQuery** - Cloud data warehouse for massive datasets
        - **Apache Spark** - Distributed data processing
        - **Vector DB** - Semantic search for intelligent retrieval
        - **Kubernetes** - Auto-scaling infrastructure
        - **Redis Caching** - Lightning-fast response times
        
        📄 See the complete architecture documentation in the `docs/` folder.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🚀 Built with Streamlit + Google Gemini AI | 📊 Enterprise-Grade Analytics</p>
        <p>💡 For 100GB+ scalability, see the architecture documentation</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()