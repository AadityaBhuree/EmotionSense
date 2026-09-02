"""Diagnostic and Clinical Affective Intelligence Report Generator.

Generates standalone formatted HTML and Markdown session diagnostic reports
summarizing emotional trajectory, VAD coordinates, behavioral telemetry,
and detected affective anomalies.
"""

import time
from typing import Dict, Any, Optional, List
from src.core.types import SessionRecord


class DiagnosticReportGenerator:
    """Generates professional diagnostic reports for affective telemetry sessions."""

    @staticmethod
    def generate_markdown_report(
        session: SessionRecord,
        anomalies: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Generates a structured markdown report suitable for clinical/research notes."""
        duration_sec = round((session.end_time or time.time()) - session.start_time, 1)
        minutes = int(duration_sec // 60)
        seconds = int(duration_sec % 60)
        dur_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

        avg_v = session.average_affect.get("valence", 0.0)
        avg_a = session.average_affect.get("arousal", 0.0)
        avg_d = session.average_affect.get("dominance", 0.0)

        # Emotion ranking
        sorted_emotions = sorted(
            session.dominant_emotion_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )

        md = []
        md.append(f"# 🧠 EmotionSense — Session Diagnostic Report")
        md.append(f"**Session ID:** `{session.session_id}` | **Duration:** {dur_str} | **Samples:** {session.samples_count}\n")
        md.append("---")
        md.append("## 1. Executive Telemetry Overview")
        md.append(f"- **Average Valence:** {avg_v:+.2f} ({'Positive' if avg_v > 0 else 'Negative' if avg_v < 0 else 'Neutral'})")
        md.append(f"- **Average Arousal:** {avg_a:+.2f} ({'Activated / High Energy' if avg_a > 0.2 else 'Calm / Deactivated'})")
        md.append(f"- **Average Dominance:** {avg_d:+.2f}")
        md.append(f"- **Mean Engagement Index:** {session.average_engagement * 100:.1f}%")
        md.append(f"- **Mean Visual Attention:** {session.average_attention * 100:.1f}%")
        md.append(f"- **Mean Cognitive Fatigue:** {session.average_fatigue * 100:.1f}%\n")

        md.append("## 2. Dominant Emotion Distribution")
        md.append("| Emotion | Prevalence | Representation |")
        md.append("| :--- | :--- | :--- |")
        for emo, pct in sorted_emotions:
            bars = "█" * int(pct * 20)
            md.append(f"| **{emo.capitalize()}** | {pct * 100:.1f}% | `{bars}` |")
        md.append("")

        if anomalies:
            md.append("## 3. Affective Anomaly & Escalation Sentinel")
            md.append("| Severity | Type | Description | Recommended Action |")
            md.append("| :--- | :--- | :--- | :--- |")
            for a in anomalies:
                sev = a.get("severity", "INFO")
                typ = a.get("anomaly_type", "")
                desc = a.get("description", "")
                rec = a.get("recommended_action", "")
                md.append(f"| `{sev}` | {typ} | {desc} | {rec} |")
            md.append("")

        if session.key_moments:
            md.append("## 4. Key Affective Pivot Moments")
            md.append("| Time (s) | Emotion | Confidence | Valence | Arousal |")
            md.append("| :--- | :--- | :--- | :--- | :--- |")
            for km in session.key_moments:
                t = km.get("relative_time_sec", 0)
                e = km.get("dominant_emotion", "").capitalize()
                c = km.get("confidence", 0)
                v = km.get("valence", 0)
                a = km.get("arousal", 0)
                md.append(f"| +{t}s | {e} | {c:.2f} | {v:+.2f} | {a:+.2f} |")
            md.append("")

        md.append("---")
        md.append("*Generated automatically by EmotionSense Multimodal Telemetry Intelligence.*")
        return "\n".join(md)

    @staticmethod
    def generate_html_report(
        session: SessionRecord,
        anomalies: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Generates a styled, standalone HTML diagnostic report ready for printing or viewing."""
        duration_sec = round((session.end_time or time.time()) - session.start_time, 1)
        dur_str = f"{int(duration_sec // 60)}m {int(duration_sec % 60)}s"

        avg_v = session.average_affect.get("valence", 0.0)
        avg_a = session.average_affect.get("arousal", 0.0)

        sorted_emotions = sorted(
            session.dominant_emotion_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )

        emotion_rows = "".join([
            f"""<tr>
                <td style="font-weight: 600; text-transform: capitalize;">{emo}</td>
                <td>{pct * 100:.1f}%</td>
                <td>
                    <div style="background: rgba(255,255,255,0.1); border-radius: 4px; overflow: hidden; height: 10px; width: 100%;">
                        <div style="background: linear-gradient(90deg, #6366f1, #06b6d4); height: 100%; width: {pct * 100}%;"></div>
                    </div>
                </td>
            </tr>"""
            for emo, pct in sorted_emotions
        ])

        anomaly_rows = ""
        if anomalies:
            for a in anomalies:
                sev = a.get("severity", "INFO")
                badge_bg = "#ef4444" if sev == "CRITICAL" else "#f59e0b" if sev == "WARNING" else "#3b82f6"
                anomaly_rows += f"""<tr>
                    <td><span style="background: {badge_bg}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;">{sev}</span></td>
                    <td style="font-family: monospace;">{a.get('anomaly_type')}</td>
                    <td>{a.get('description')}</td>
                    <td style="color: #94a3b8; font-size: 13px;">{a.get('recommended_action')}</td>
                </tr>"""
        else:
            anomaly_rows = """<tr><td colspan="4" style="text-align: center; color: #10b981; padding: 16px;">✓ No affective anomalies or distress spikes detected during session.</td></tr>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EmotionSense Diagnostic Report — {session.session_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 32px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 32px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}
        .brand {{
            font-size: 24px;
            font-weight: 800;
            background: linear-gradient(135deg, #6366f1, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .meta {{
            color: #94a3b8;
            font-size: 13px;
            text-align: right;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin-bottom: 28px;
        }}
        .card {{
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px;
        }}
        .card-label {{
            font-size: 12px;
            text-transform: uppercase;
            color: #94a3b8;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }}
        .card-value {{
            font-size: 24px;
            font-weight: 700;
            color: #38bdf8;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            font-size: 14px;
        }}
        th, td {{
            padding: 10px 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.07);
            text-align: left;
        }}
        th {{
            color: #94a3b8;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .section-title {{
            font-size: 18px;
            font-weight: 700;
            margin: 28px 0 12px;
            color: #e2e8f0;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .footer {{
            margin-top: 36px;
            padding-top: 16px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
            color: #64748b;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <div class="brand">EmotionSense</div>
                <div style="color: #cbd5e1; font-size: 14px;">Multimodal Affective Intelligence Report</div>
            </div>
            <div class="meta">
                <div><strong>Session:</strong> {session.session_id}</div>
                <div><strong>Duration:</strong> {dur_str} ({session.samples_count} frames)</div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="card-label">Engagement Index</div>
                <div class="card-value">{session.average_engagement * 100:.1f}%</div>
            </div>
            <div class="card">
                <div class="card-label">Visual Attention</div>
                <div class="card-value">{session.average_attention * 100:.1f}%</div>
            </div>
            <div class="card">
                <div class="card-label">Cognitive Fatigue</div>
                <div class="card-value">{session.average_fatigue * 100:.1f}%</div>
            </div>
        </div>

        <div class="section-title">📊 Emotion Prevalence Distribution</div>
        <table>
            <thead>
                <tr>
                    <th style="width: 25%;">Emotion</th>
                    <th style="width: 20%;">Prevalence</th>
                    <th style="width: 55%;">Telemetry Distribution</th>
                </tr>
            </thead>
            <tbody>
                {emotion_rows}
            </tbody>
        </table>

        <div class="section-title">🛡️ Affective Anomaly Sentinel Log</div>
        <table>
            <thead>
                <tr>
                    <th style="width: 12%;">Severity</th>
                    <th style="width: 25%;">Type</th>
                    <th style="width: 38%;">Description</th>
                    <th style="width: 25%;">Mitigation Recommendation</th>
                </tr>
            </thead>
            <tbody>
                {anomaly_rows}
            </tbody>
        </table>

        <div class="footer">
            CONFIDENTIAL & PROPRIETARY &bull; Generated by EmotionSense Multimodal Telemetry Platform
        </div>
    </div>
</body>
</html>"""
        return html
