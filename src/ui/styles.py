"""Custom Dark Glassmorphic CSS style injection for Streamlit."""

import streamlit as st
from config import THEME_COLORS


def inject_glassmorphic_styles():
    """Injects high-end dark glassmorphic styles, custom scrollbars, animated glows, and fonts."""
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Base root variables */
        :root {{
            --bg-main: {THEME_COLORS["background"]};
            --surface-card: {THEME_COLORS["surface"]};
            --primary: {THEME_COLORS["primary"]};
            --primary-glow: {THEME_COLORS["primary_glow"]};
            --secondary: {THEME_COLORS["secondary"]};
            --text-main: {THEME_COLORS["text_primary"]};
            --text-sub: {THEME_COLORS["text_secondary"]};
            --border-color: {THEME_COLORS["border"]};
        }}

        /* App Background */
        .stApp {{
            background: radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.08) 0%, transparent 40%),
                        radial-gradient(circle at 85% 85%, rgba(6, 182, 212, 0.06) 0%, transparent 40%),
                        #080c14 !important;
            font-family: 'Outfit', sans-serif !important;
            color: var(--text-main) !important;
        }}

        /* Header / Hero Title */
        .es-hero-title {{
            font-size: 2.3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #f8fafc 0%, #818cf8 50%, #06b6d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
            letter-spacing: -0.02em;
        }}

        .es-hero-sub {{
            color: #94a3b8;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
            font-weight: 400;
        }}

        /* Glassmorphic Cards */
        .es-card {{
            background: rgba(15, 23, 42, 0.65) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            border: 1px solid rgba(255, 255, 255, 0.07) !important;
            border-radius: 16px !important;
            padding: 1.25rem !important;
            margin-bottom: 1rem !important;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5) !important;
            transition: all 0.25s ease-in-out !important;
        }}

        .es-card:hover {{
            border-color: rgba(99, 102, 241, 0.3) !important;
            box-shadow: 0 12px 35px -8px rgba(99, 102, 241, 0.15) !important;
        }}

        /* Metric Badges */
        .es-metric-box {{
            display: flex;
            flex-direction: column;
            background: rgba(30, 41, 59, 0.45);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            position: relative;
            overflow: hidden;
        }}

        .es-metric-box::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, #6366f1, #06b6d4);
        }}

        .es-metric-label {{
            font-size: 0.8rem;
            font-weight: 500;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .es-metric-value {{
            font-size: 1.6rem;
            font-weight: 700;
            color: #f8fafc;
            font-family: 'JetBrains Mono', monospace;
            margin-top: 0.2rem;
        }}

        /* Status Pills */
        .es-pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}

        .es-pill-active {{
            background: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        .es-pill-idle {{
            background: rgba(148, 163, 184, 0.15);
            color: #94a3b8;
            border: 1px solid rgba(148, 163, 184, 0.25);
        }}

        /* Streamlit Sidebar Customization */
        [data-testid="stSidebar"] {{
            background-color: #0b0f19 !important;
            border-right: 1px solid #1e293b !important;
        }}

        /* Buttons */
        .stButton>button {{
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
            color: white !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.5rem 1.25rem !important;
            transition: all 0.2s ease !important;
        }}

        .stButton>button:hover {{
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.4) !important;
            transform: translateY(-1px) !important;
        }}

        /* Streamlit Text Inputs & Text Areas */
        .stTextArea textarea, .stTextInput input {{
            background: rgba(15, 23, 42, 0.7) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 12px !important;
            color: #f8fafc !important;
            font-family: 'Outfit', sans-serif !important;
            font-size: 1rem !important;
            padding: 0.85rem !important;
            transition: all 0.2s ease !important;
        }}

        .stTextArea textarea:focus, .stTextInput input:focus {{
            border-color: #6366f1 !important;
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.25) !important;
        }}

        /* Streamlit Tabs Customization */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: rgba(15, 23, 42, 0.5);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            color: #94a3b8 !important;
            font-weight: 600;
            padding: 8px 16px;
            background-color: transparent;
            transition: all 0.2s ease;
        }}

        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.3) 0%, rgba(79, 70, 229, 0.4) 100%) !important;
            color: #f8fafc !important;
            border: 1px solid rgba(99, 102, 241, 0.5) !important;
        }}

        /* Custom Scrollbar */
        ::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        ::-webkit-scrollbar-track {{
            background: #080c14;
        }}
        ::-webkit-scrollbar-thumb {{
            background: #1e293b;
            border-radius: 3px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: #334155;
        }}
    </style>
    """, unsafe_allow_html=True)

