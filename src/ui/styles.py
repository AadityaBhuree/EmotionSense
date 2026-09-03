"""Precision Neuro-Affective Instrument styling & CSS design system for Streamlit."""

import streamlit as st
from config import THEME_COLORS


def inject_modern_styles():
    """Injects high-precision neuro-affective instrument styles, technical grid, and typography."""
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');

        /* Base root design tokens */
        :root {{
            --bg-main: {THEME_COLORS["background"]};
            --surface-card: {THEME_COLORS["surface"]};
            --surface-raised: {THEME_COLORS.get("surface_raised", "#171d2b")};
            --surface-subtle: {THEME_COLORS.get("surface_subtle", "#1c2334")};
            --primary: {THEME_COLORS["primary"]};
            --primary-glow: {THEME_COLORS["primary_glow"]};
            --secondary: {THEME_COLORS["secondary"]};
            --text-main: {THEME_COLORS["text_primary"]};
            --text-sub: {THEME_COLORS["text_secondary"]};
            --text-muted: {THEME_COLORS.get("text_muted", "#526079")};
            --border-color: {THEME_COLORS["border"]};
            --border-active: {THEME_COLORS.get("border_active", "#3b82f6")};
            --border-subtle: {THEME_COLORS.get("border_subtle", "rgba(255, 255, 255, 0.07)")};
        }}

        /* App Background: Scientific Slate-Obsidian with calibrated technical grid */
        .stApp {{
            background-color: var(--bg-main) !important;
            background-image: 
                linear-gradient(rgba(255, 255, 255, 0.018) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.018) 1px, transparent 1px) !important;
            background-size: 28px 28px !important;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            color: var(--text-main) !important;
            letter-spacing: -0.01em;
        }}

        /* Typography */
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            letter-spacing: -0.025em !important;
            color: #f8fafc !important;
            font-weight: 700 !important;
        }}

        /* Precision Console Header */
        .es-console-bar {{
            background: var(--surface-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.85rem 1.25rem;
            margin-bottom: 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }}

        .es-console-brand {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .es-console-title {{
            font-size: 1.25rem;
            font-weight: 800;
            color: #f8fafc;
            letter-spacing: -0.03em;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .es-console-sub {{
            font-size: 0.8rem;
            color: var(--text-sub);
            font-weight: 500;
        }}

        .es-telemetry-cluster {{
            display: flex;
            align-items: center;
            gap: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
        }}

        /* Precision Instrument Panels */
        .es-card, .es-panel {{
            background: var(--surface-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            padding: 1.15rem 1.25rem !important;
            margin-bottom: 0.85rem !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
            transition: border-color 0.15s ease, transform 0.15s ease !important;
        }}

        .es-card:hover, .es-panel:hover {{
            border-color: rgba(59, 130, 246, 0.4) !important;
        }}

        /* Section Subheaders */
        .es-section-title {{
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-sub);
            margin-bottom: 0.65rem;
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        /* High-Density Telemetry Metric Gauge */
        .es-metric-box {{
            display: flex;
            flex-direction: column;
            background: var(--surface-raised);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.8rem 1rem;
            position: relative;
            transition: all 0.15s ease;
        }}

        .es-metric-box:hover {{
            background: var(--surface-subtle);
            border-color: rgba(59, 130, 246, 0.35);
        }}

        .es-metric-label {{
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .es-metric-value {{
            font-size: 1.45rem;
            font-weight: 700;
            color: #f8fafc;
            font-family: 'JetBrains Mono', monospace;
            margin-top: 0.25rem;
            letter-spacing: -0.02em;
        }}

        .es-metric-meta {{
            font-size: 0.72rem;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-muted);
            margin-top: 0.2rem;
        }}

        /* Technical Status Badges & Beacons */
        .es-pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 9px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .es-pill-active {{
            background: rgba(16, 185, 129, 0.12);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }}

        .es-pill-warning {{
            background: rgba(245, 158, 11, 0.12);
            color: #f59e0b;
            border: 1px solid rgba(245, 158, 11, 0.35);
        }}

        .es-pill-idle {{
            background: rgba(100, 116, 139, 0.12);
            color: #94a3b8;
            border: 1px solid rgba(100, 116, 139, 0.25);
        }}

        /* Live Pulsing Beacon */
        .es-live-beacon {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 8px #10b981;
            display: inline-block;
            animation: pulse-beacon 1.8s infinite ease-in-out;
        }}

        @keyframes pulse-beacon {{
            0% {{ transform: scale(0.95); opacity: 0.7; }}
            50% {{ transform: scale(1.2); opacity: 1; }}
            100% {{ transform: scale(0.95); opacity: 0.7; }}
        }}

        /* Streamlit Sidebar Customization */
        [data-testid="stSidebar"] {{
            background-color: #0b0f17 !important;
            border-right: 1px solid var(--border-color) !important;
        }}

        [data-testid="stSidebar"] hr {{
            border-color: var(--border-color) !important;
            margin: 1rem 0 !important;
        }}

        /* High-Precision Tactile Buttons */
        .stButton>button {{
            background: #161d2b !important;
            color: #f1f5f9 !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 6px !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            padding: 0.45rem 1rem !important;
            letter-spacing: -0.01em !important;
            transition: all 0.15s ease !important;
        }}

        .stButton>button:hover {{
            background: #1e283c !important;
            border-color: #3b82f6 !important;
            color: #ffffff !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
            transform: translateY(-1px) !important;
        }}

        .stButton>button:active {{
            transform: translateY(0) !important;
        }}

        /* Primary Action Buttons */
        .stButton>button[kind="primary"] {{
            background: #2563eb !important;
            border-color: #3b82f6 !important;
            color: #ffffff !important;
        }}

        .stButton>button[kind="primary"]:hover {{
            background: #1d4ed8 !important;
            border-color: #60a5fa !important;
        }}

        /* Streamlit Text Inputs & Text Areas */
        .stTextArea textarea, .stTextInput input {{
            background: #0f1420 !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 6px !important;
            color: #f1f5f9 !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: 0.92rem !important;
            padding: 0.75rem !important;
            transition: border-color 0.15s ease !important;
        }}

        .stTextArea textarea:focus, .stTextInput input:focus {{
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 1px #3b82f6 !important;
        }}

        /* Segmented Instrument Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background-color: #0d121c;
            padding: 4px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: 4px;
            color: #94a3b8 !important;
            font-weight: 600;
            font-size: 0.85rem;
            padding: 6px 14px;
            background-color: transparent;
            transition: all 0.15s ease;
        }}

        .stTabs [aria-selected="true"] {{
            background: #171f30 !important;
            color: #f8fafc !important;
            border: 1px solid rgba(59, 130, 246, 0.4) !important;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3) !important;
        }}

        /* Monospace Selectboxes & Radios */
        div[data-baseweb="select"] {{
            border-radius: 6px !important;
        }}

        /* Clean Technical Scrollbar */
        ::-webkit-scrollbar {{
            width: 5px;
            height: 5px;
        }}
        ::-webkit-scrollbar-track {{
            background: #0b0e14;
        }}
        ::-webkit-scrollbar-thumb {{
            background: #1f283d;
            border-radius: 2px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: #2d3b55;
        }}
    </style>
    """, unsafe_allow_html=True)


def inject_glassmorphic_styles():
    """Backward compatibility alias for inject_modern_styles()."""
    inject_modern_styles()
