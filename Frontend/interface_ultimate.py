import streamlit as st
import pandas as pd
# Reload trigger
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Add Backend to path for RAG imports
backend_path = os.path.join(os.path.dirname(__file__), '..', 'Backend')
sys.path.insert(0, backend_path)

# Try to import RAG system and ETL pipeline
try:
    from rag_system import FraudRAG
    from etl_pipeline import run_etl_pipeline
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(
    page_title="FraudGuard AI - Ultimate",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown(
"""
<style>
:root {
    --bg-main:#060b16;
    --bg-card:rgba(17,24,39,0.55);
    --glass-bg:rgba(23,32,50,0.35);
    --glass-border:rgba(255,255,255,0.1);
    --neon-blue:#3abdf8;
    --neon-purple:#9b82ff;
    --neon-pink:#ff4fa3;
    --text-primary:#f8fafc;
    --text-secondary:#a1a5b3;
}

.stApp {
    background: radial-gradient(circle at 30% 10%, #0c1221 0%, #060b16 50%, #04070f 100%);
    color: var(--text-primary);
}

/* ===========================
     HERO SECTION UPGRADE
   =========================== */
.hero {
    padding:30px 36px;
    border-radius:24px;
    background: linear-gradient(140deg, rgba(14,22,38,0.85), rgba(6,12,24,0.85));
    border:1px solid var(--glass-border);
    box-shadow:0 0 40px rgba(58,189,248,0.15);
    backdrop-filter: blur(12px);
}

.hero-title {
    font-size:36px;
    font-weight:800;
    letter-spacing:0.5px;
}

.hero-title span {
    background: linear-gradient(135deg, var(--neon-blue), var(--neon-purple));
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.hero-badge {
    padding:6px 16px;
    font-size:14px;
    border-radius:999px;
    background:rgba(34,197,94,0.2);
    border:1px solid rgba(34,197,94,0.5);
}

.rag-badge {
    background:rgba(156,110,255,0.18);
    border:1px solid rgba(156,110,255,0.45);
}

/* ===========================
    METRIC CARDS (GLOW)
   =========================== */
.metric-card {
    background: var(--glass-bg);
    border:1px solid var(--glass-border);
    border-radius:18px;
    padding:22px;
    transition:0.28s ease;
    backdrop-filter: blur(14px);
    box-shadow:0 0 15px rgba(0,0,0,0.35);
}

.metric-card:hover {
    transform: translateY(-6px);
    box-shadow:0 0 22px rgba(58,189,248,0.35), 
               0 0 18px rgba(155,130,255,0.25);
    border-color: rgba(155,130,255,0.4);
}

/* ===========================
     TABS (pill buttons)
   =========================== */
.stTabs [data-baseweb="tab-list"] {
    background:rgba(255,255,255,0.06);
    padding:8px;
    border-radius:14px;
}

.stTabs [data-baseweb="tab"] {
    padding:12px 22px;
    border-radius:10px;
    font-weight:600;
    color:var(--text-secondary);
    transition:0.2s;
}

.stTabs [data-baseweb="tab"]:hover {
    background:rgba(58,189,248,0.12);
    color:var(--neon-blue);
}

.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg, rgba(58,189,248,0.22), rgba(155,130,255,0.22));
    border:1px solid rgba(58,189,248,0.35);
    color:var(--neon-blue) !important;
}

/* ===========================
     BUTTONS (neon glow)
   =========================== */
.stButton > button {
    background: linear-gradient(135deg, var(--neon-blue), var(--neon-purple));
    color:white;
    border:none;
    border-radius:10px;
    padding:12px 26px;
    font-weight:700;
    box-shadow:0 4px 12px rgba(58,189,248,0.4);
    transition:0.3s;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.03);
    box-shadow:0 0 25px rgba(58,189,248,0.55),
               0 0 25px rgba(155,130,255,0.45);
}

/* ===========================
   CHAT / TABLE / GENERAL
   =========================== */
.stChatMessage {
    background: var(--bg-card);
    border:1px solid var(--glass-border);
    border-radius:16px;
    padding:18px;
    backdrop-filter:blur(10px);
}

.stMarkdown p {
    color: var(--text-primary);
}

/* Scrollbar */
::-webkit-scrollbar-thumb {
    background:linear-gradient(var(--neon-blue), var(--neon-purple));
    border-radius:6px;
}

</style>

""", unsafe_allow_html=True)

# -----------------------------
# HERO SECTION
# -----------------------------
st.markdown(f"""
<div class="hero">
    <div class="hero-title">
        🛡️ FraudGuard: <span>AI Claims Analyst</span>
        <span class="hero-badge">ETL + RAG powered</span>
        {'<span class="rag-badge">Semantic Search Enabled</span>' if RAG_AVAILABLE else ''}
    </div>
    <div class="hero-sub">
        Advanced fraud detection with AI-powered analysis and semantic search. Ask questions like 
        <code>"Show me suspicious claims in Cardiology"</code> or search for similar fraud patterns.
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("🔑 Google Gemini API Key", type="password", help="Enter your Gemini API key")
if api_key:
    st.sidebar.caption(f"✅ Key active (ends in ...{api_key[-4:]})")

# RAG Settings
if RAG_AVAILABLE:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 RAG Settings")
    use_rag = st.sidebar.checkbox("Enable Semantic Search", value=True)
    rag_top_k = st.sidebar.slider("Similar cases to show", 3, 10, 5)
else:
    use_rag = False
    st.sidebar.warning("⚠️ RAG unavailable. Install: `pip install sentence-transformers faiss-cpu`")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📤 Data Management")
uploaded_claims = st.sidebar.file_uploader("Upload Claims (CSV)", type="csv")
uploaded_providers = st.sidebar.file_uploader("Upload Providers (CSV)", type="csv")

if st.sidebar.button("🚀 Process New Data"):
    if uploaded_claims:
        with st.spinner("Processing new data..."):
            try:
                # Save uploaded files temporarily
                base_dir = os.path.dirname(os.path.abspath(__file__))
                backend_dir = os.path.join(base_dir, '..', 'Backend')
                claims_path = os.path.join(backend_dir, 'claims.csv')
                providers_path = os.path.join(backend_dir, 'providers.csv')
                
                with open(claims_path, "wb") as f:
                    f.write(uploaded_claims.getbuffer())
                
                if uploaded_providers:
                    with open(providers_path, "wb") as f:
                        f.write(uploaded_providers.getbuffer())
                
                # Run ETL
                st.sidebar.info("Running ETL pipeline...")
                
                # Capture stdout to show progress
                import io
                from contextlib import redirect_stdout
                f = io.StringIO()
                with redirect_stdout(f):
                    run_etl_pipeline(claims_path, providers_path)
                
                st.sidebar.success("✅ ETL Completed!")
                
                # Clear RAG index to force rebuild
                rag_index_path = os.path.join(backend_dir, 'rag_index')
                if os.path.exists(rag_index_path):
                    import shutil
                    shutil.rmtree(rag_index_path)
                    st.sidebar.info("🗑️ Cleared old RAG index")
                
                # Clear cache and rerun
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
                
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
    else:
        st.sidebar.warning("Please upload a claims file first.")

st.sidebar.markdown("---")

# Load data
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'Backend', 'processed_claims_etl.csv')
    
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    
    data_path_ml = os.path.join(base_dir, '..', 'Backend', 'processed_claims_ml.csv')
    if os.path.exists(data_path_ml):
        return pd.read_csv(data_path_ml)
    
    return None

df = load_data()

if df is None:
    st.error("🚨 Could not find processed claims data!")
    st.stop()

# Sidebar stats
total_claims = len(df)
fraud_claims = df['rule_based_fraud'].sum() if 'rule_based_fraud' in df.columns else 0
fraud_rate = (fraud_claims / total_claims * 100) if total_claims > 0 else 0

st.sidebar.markdown("### 📊 Quick Stats")
st.sidebar.metric("Total Claims", f"{total_claims:,}")
st.sidebar.metric("Fraud Cases", f"{fraud_claims:,}", f"{fraud_rate:.1f}%")

if 'amount' in df.columns:
    total_amount = df['amount'].sum()
    st.sidebar.metric("Total Amount", f"${total_amount:,.0f}")

st.sidebar.markdown("---")
st.sidebar.success("✅ System Online")

# Initialize RAG
@st.cache_resource
def init_rag():
    if not RAG_AVAILABLE:
        return None
    
    try:
        rag = FraudRAG()
        rag_index_path = os.path.join(os.path.dirname(__file__), '..', 'Backend', 'rag_index')
        
        if os.path.exists(rag_index_path):
            rag.load_index(rag_index_path)
            return rag
        else:
            with st.spinner("Building RAG index (first time only)..."):
                rag.build_index(df, save_path=rag_index_path)
            return rag
    except Exception as e:
        st.sidebar.error(f"RAG init failed: {e}")
        return None

rag_system = init_rag() if use_rag and RAG_AVAILABLE else None

# -----------------------------
# TABS
# -----------------------------
if RAG_AVAILABLE and rag_system:
    tabs = st.tabs(["📊 Overview", "💬 AI Chat", "🔍 Semantic Search", "📈 Analytics"])
else:
    tabs = st.tabs(["📊 Overview", "💬 AI Chat", "📈 Analytics"])

# TAB 1: OVERVIEW
with tabs[0]:
    st.markdown("### 📊 Fraud Detection Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Claims</div>
            <div class="metric-value">{total_claims:,}</div>
            <div class="metric-delta positive">Active</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Fraud Detected</div>
            <div class="metric-value">{fraud_claims:,}</div>
            <div class="metric-delta negative">{fraud_rate:.1f}% rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if 'amount' in df.columns:
            fraud_amount = df[df['rule_based_fraud'] == True]['amount'].sum() if 'rule_based_fraud' in df.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Fraud Amount</div>
                <div class="metric-value">${fraud_amount:,.0f}</div>
                <div class="metric-delta negative">Blocked</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        unique_providers = df['provider_id'].nunique() if 'provider_id' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Providers</div>
            <div class="metric-value">{unique_providers}</div>
            <div class="metric-delta positive">Monitored</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### 🎯 Fraud by Specialty")
        if 'specialty' in df.columns and 'rule_based_fraud' in df.columns:
            fraud_by_spec = df[df['rule_based_fraud'] == True].groupby('specialty').size().reset_index(name='count')
            fraud_by_spec = fraud_by_spec.sort_values('count', ascending=False)
            
            fig = px.bar(
                fraud_by_spec,
                x='specialty',
                y='count',
                color='count',
                color_continuous_scale='Reds',
                labels={'specialty': 'Specialty', 'count': 'Fraud Cases'}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#F3F4F6',
                showlegend=False,
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col_chart2:
        st.markdown("#### 💰 Amount Distribution")
        if 'amount' in df.columns and 'rule_based_fraud' in df.columns:
            fig = px.box(
                df,
                x='rule_based_fraud',
                y='amount',
                color='rule_based_fraud',
                labels={'rule_based_fraud': 'Fraud Status', 'amount': 'Claim Amount ($)'},
                color_discrete_map={True: '#EF4444', False: '#22C55E'}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#F3F4F6',
                showlegend=False,
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 🚨 Recent Fraud Cases")
    if 'rule_based_fraud' in df.columns:
        fraud_df = df[df['rule_based_fraud'] == True].head(10)
        display_cols = ['claim_id', 'specialty', 'amount', 'fraud_reasons', 'status']
        display_cols = [col for col in display_cols if col in fraud_df.columns]
        st.dataframe(fraud_df[display_cols], use_container_width=True, height=300)

# TAB 2: AI CHAT
with tabs[1]:
    st.markdown("#### 💬 AI Fraud Investigator")
    st.markdown("""
    Ask natural language questions about the claims data.
    """)
    
    # Example Query Buttons
    col_q1, col_q2, col_q3 = st.columns(3)
    query_to_run = None
    
    with col_q1:
        if st.button("🔍 Top 10 Suspicious", use_container_width=True):
            query_to_run = "Show me the top 10 suspicious claims"
    
    with col_q2:
        if st.button("📊 Fraud by Specialty", use_container_width=True):
            query_to_run = "What's the fraud rate by specialty?"
            
    with col_q3:
        if st.button("🏥 High Risk Providers", use_container_width=True):
            query_to_run = "List providers with highest fraud risk"

    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "image" in msg:
                st.image(msg["image"])
    
    if api_key:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=api_key,
                temperature=0
            )
            agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                allow_dangerous_code=True,
                handle_parsing_errors=True
            )
            
            query = st.chat_input("Ask a question about the claims data...")
            
            if query_to_run:
                query = query_to_run
            
            if query:
                st.session_state.messages.append({"role": "user", "content": query})
                with st.chat_message("user"):
                    st.write(query)
                
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Analyzing..."):
                        prompt = query + " If you draw a chart, save it as 'chart.png'. Do not use plt.show()."
                        
                        try:
                            response = agent.invoke(prompt)
                            output = response["output"]
                            st.write(output)
                            
                            msg_data = {"role": "assistant", "content": output}
                            
                            if os.path.exists("chart.png"):
                                st.image("chart.png")
                                new_name = f"chart_{len(st.session_state.messages)}.png"
                                if os.path.exists(new_name):
                                    os.remove(new_name)
                                os.rename("chart.png", new_name)
                                msg_data["image"] = new_name
                            
                            st.session_state.messages.append(msg_data)
                        
                        except Exception as e:
                            error_msg = str(e)
                            if "429" in error_msg or "quota" in error_msg.lower():
                                st.warning("🚦 **API Rate Limit Exceeded**")
                                st.info("You're using the free tier of the Gemini API, which has request limits. Please wait a minute before trying again.")
                            else:
                                st.error(f"❌ Error: {error_msg}")
        
        except Exception as e:
            st.error(f"❌ Connection Error: {str(e)}")
    else:
        st.info("🔑 Please enter your Google Gemini API key in the sidebar.")

# TAB 3: SEMANTIC SEARCH (if RAG available)
if RAG_AVAILABLE and rag_system:
    with tabs[2]:
        st.markdown("#### 🔍 Semantic Fraud Search")
        st.markdown("""
        Use AI-powered semantic search to find similar fraud cases based on meaning, not just keywords.
        """)
        
        # Initialize search query from session state or empty
        if 'search_query' not in st.session_state:
            st.session_state.search_query = ""
        
        # Example query buttons (placed before search input)
        st.markdown("**Quick Search Examples:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Duplicate billing", use_container_width=True):
                st.session_state.search_query = "duplicate billing in cardiology"
                st.rerun()
        with col2:
            if st.button("🔍 Abnormal amounts", use_container_width=True):
                st.session_state.search_query = "abnormally high claim amounts"
                st.rerun()
        with col3:
            if st.button("🔍 Specialty mismatches", use_container_width=True):
                st.session_state.search_query = "wrong specialty for procedure"
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Search input and controls
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input(
                "Search for fraud cases",
                value=st.session_state.search_query,
                placeholder="e.g., 'duplicate billing in cardiology with high amounts'",
                key="search_input"
            )
            # Update session state when user types
            if search_query != st.session_state.search_query:
                st.session_state.search_query = search_query
        
        with col2:
            fraud_only = st.checkbox("Fraud cases only", value=True)
        
        # Perform search if query exists
        if search_query:
            with st.spinner("🔍 Searching..."):
                try:
                    results = rag_system.search_fraud_cases(
                        search_query,
                        top_k=rag_top_k,
                        fraud_only=fraud_only
                    )
                    
                    if results:
                        st.success(f"✅ Found {len(results)} relevant cases")
                        
                        for i, case in enumerate(results, 1):
                            with st.expander(
                                f"#{i} - {case.get('claim_id')} | "
                                f"{case.get('specialty', 'Unknown')} | "
                                f"${case.get('amount', 0):,.2f} | "
                                f"Similarity: {case.get('similarity_score', 0):.1%}"
                            ):
                                col_a, col_b = st.columns(2)
                                
                                with col_a:
                                    st.markdown("**Claim Details**")
                                    st.write(f"• Claim ID: {case.get('claim_id')}")
                                    st.write(f"• Patient: {case.get('patient_id')}")
                                    st.write(f"• Provider: {case.get('provider_id')}")
                                    st.write(f"• Specialty: {case.get('specialty')}")
                                    st.write(f"• Amount: ${case.get('amount', 0):,.2f}")
                                
                                with col_b:
                                    st.markdown("**Fraud Information**")
                                    st.write(f"• Reasons: {case.get('fraud_reasons', 'N/A')}")
                                    if 'name' in case:
                                        st.write(f"• Provider: {case.get('name')}")
                                        st.write(f"• Location: {case.get('location')}")
                    else:
                        st.warning("No matching cases found.")
                
                except Exception as e:
                    st.error(f"Search error: {e}")

# TAB 4 (or 3): ANALYTICS
analytics_tab_index = 3 if (RAG_AVAILABLE and rag_system) else 2
with tabs[analytics_tab_index]:
    st.markdown("### 📈 Advanced Analytics")
    
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        st.markdown("#### 📊 Fraud Reasons Breakdown")
        if 'fraud_reasons' in df.columns:
            fraud_reasons = df[df['rule_based_fraud'] == True]['fraud_reasons'].value_counts().head(10)
            fig = px.pie(
                values=fraud_reasons.values,
                names=fraud_reasons.index,
                color_discrete_sequence=px.colors.sequential.Reds_r
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#F3F4F6',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col_a2:
        st.markdown("#### 🏥 Provider Risk Distribution")
        if 'fraud_risk_score' in df.columns:
            fig = px.histogram(
                df,
                x='fraud_risk_score',
                nbins=30,
                color_discrete_sequence=['#38BDF8']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#F3F4F6',
                xaxis_title='Risk Score',
                yaxis_title='Claims',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 📅 Claims Over Time")
    if 'claim_date' in df.columns:
        df_time = df.copy()
        df_time['claim_date'] = pd.to_datetime(df_time['claim_date'])
        df_time['date'] = df_time['claim_date'].dt.date
        
        daily_claims = df_time.groupby('date').agg({
            'claim_id': 'count',
            'rule_based_fraud': 'sum'
        }).reset_index()
        daily_claims.columns = ['date', 'total_claims', 'fraud_claims']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_claims['date'],
            y=daily_claims['total_claims'],
            name='Total Claims',
            line=dict(color='#38BDF8', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=daily_claims['date'],
            y=daily_claims['fraud_claims'],
            name='Fraud Claims',
            line=dict(color='#EF4444', width=2)
        ))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#F3F4F6',
            xaxis_title='Date',
            yaxis_title='Claims',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 14px;">
    🛡️ FraudGuard AI | ETL + RAG + ML | Built with Streamlit & Google Gemini
</div>
""", unsafe_allow_html=True)
