"""Clinical & Multimodal Affective Intelligence PDF Report Exporter.

Generates publication-quality, multi-page printable PDF reports for clinical,
psychological, research, and talent evaluation workflows using ReportLab.
"""

import io
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from src.core.types import SessionRecord
from src.storage.models import AssessmentType, SessionMetadata, StoredSession


class ClinicalPDFExporter:
    """Generates structured, publication-grade Clinical Affect Diagnostic PDF summaries."""

    @classmethod
    def is_available(cls) -> bool:
        """Returns True if ReportLab is installed and ready."""
        return REPORTLAB_AVAILABLE

    @classmethod
    def generate_pdf(
        cls,
        session: Union[SessionRecord, StoredSession, Dict[str, Any]],
        anomalies: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Union[SessionMetadata, Dict[str, Any]]] = None,
    ) -> bytes:
        """Compiles session telemetry and clinical alerts into printable PDF bytes."""
        if not REPORTLAB_AVAILABLE:
            raise RuntimeError("ReportLab library is required for PDF generation. Run 'pip install reportlab'.")

        # Normalize session data
        if isinstance(session, (SessionRecord, StoredSession)):
            sess_dict = session.to_dict()
        else:
            sess_dict = dict(session)

        session_id = sess_dict.get("session_id", "session_unnamed")
        start_time = sess_dict.get("start_time", time.time())
        end_time = sess_dict.get("end_time") or time.time()
        duration_sec = round(max(0.0, end_time - start_time), 1)
        dur_str = f"{int(duration_sec // 60)}m {int(duration_sec % 60)}s" if duration_sec >= 60 else f"{duration_sec:.1f}s"

        samples_count = sess_dict.get("samples_count", 0)

        # Affect data
        avg_affect = sess_dict.get("average_affect", {})
        if not avg_affect:
            avg_affect = {
                "valence": sess_dict.get("average_valence", 0.0),
                "arousal": sess_dict.get("average_arousal", 0.0),
                "dominance": sess_dict.get("average_dominance", 0.0),
            }
        avg_v = float(avg_affect.get("valence", 0.0))
        avg_a = float(avg_affect.get("arousal", 0.0))
        avg_d = float(avg_affect.get("dominance", 0.0))

        avg_eng = float(sess_dict.get("average_engagement", 0.0))
        avg_fat = float(sess_dict.get("average_fatigue", 0.0))
        avg_att = float(sess_dict.get("average_attention", 0.0))

        dist = sess_dict.get("dominant_emotion_distribution", {})
        key_moments = sess_dict.get("key_moments", [])
        anom_list = anomalies if anomalies is not None else sess_dict.get("anomalies", [])

        # Metadata extraction
        meta_dict: Dict[str, Any] = {}
        if metadata:
            if isinstance(metadata, SessionMetadata):
                meta_dict = metadata.to_dict()
            else:
                meta_dict = dict(metadata)
        elif "metadata" in sess_dict and isinstance(sess_dict["metadata"], dict):
            meta_dict = sess_dict["metadata"]

        subject_id = meta_dict.get("subject_id") or "N/A"
        subject_name = meta_dict.get("subject_name") or "Anonymous Candidate"
        ass_type_val = meta_dict.get("assessment_type") or AssessmentType.GENERAL_AFFECT.value
        ass_type_label = AssessmentType.display_names().get(ass_type_val, ass_type_val.replace("_", " ").title())
        evaluator = meta_dict.get("evaluator") or "EmotionSense Autonomous Agent"
        notes = meta_dict.get("notes") or "Standard automated baseline assessment recording."
        tags = meta_dict.get("tags", [])
        tags_str = ", ".join(tags) if tags else "None"

        # Build Document
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()

        # Custom Palette
        c_dark = colors.HexColor("#0f172a")
        c_primary = colors.HexColor("#1e40af")
        c_green = colors.HexColor("#15803d")
        c_gray_bg = colors.HexColor("#f8fafc")
        c_border = colors.HexColor("#cbd5e1")
        c_muted = colors.HexColor("#64748b")

        # Custom Paragraph Styles
        header_title = ParagraphStyle(
            "DocTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=c_dark,
        )
        header_subtitle = ParagraphStyle(
            "DocSubTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=c_primary,
            spaceAfter=4,
        )
        section_h = ParagraphStyle(
            "SectionHeading",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=c_dark,
            spaceBefore=10,
            spaceAfter=5,
        )
        body_p = ParagraphStyle(
            "BodySmall",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=c_dark,
        )
        body_muted = ParagraphStyle(
            "BodyMuted",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            leading=10,
            textColor=c_muted,
        )
        tbl_hdr = ParagraphStyle(
            "TblHdr",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
        )
        tbl_cell = ParagraphStyle(
            "TblCell",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=c_dark,
        )
        tbl_cell_bold = ParagraphStyle(
            "TblCellBold",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=c_dark,
        )

        elements = []

        # 1. Header & Confidentiality Notice
        dt_str = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph("EMOTIONSENSE NEURO-AFFECTIVE INTELLIGENCE", header_subtitle))
        elements.append(Paragraph("Clinical & Multimodal Affect Diagnostic Report", header_title))
        elements.append(Paragraph(f"Generated: {dt_str} | Session Reference: <b>{session_id}</b>", body_muted))
        elements.append(Spacer(1, 6))

        # Confidentiality Banner
        conf_banner = Table(
            [[Paragraph("<b>CONFIDENTIAL MEDICAL & PSYCHOMETRIC EVALUATION RECORD</b> — RESTRICTED ACCESS", ParagraphStyle(
                "Conf", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=7.5, leading=9, textColor=c_primary, alignment=1
            ))]],
            colWidths=[540],
        )
        conf_banner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#bfdbfe")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        elements.append(conf_banner)
        elements.append(Spacer(1, 8))

        # 2. Subject & Clinical Evaluation Metadata
        elements.append(Paragraph("1. ASSESSMENT & SUBJECT IDENTIFICATION", section_h))
        meta_table_data = [
            [
                Paragraph("<b>Subject Name:</b>", tbl_cell), Paragraph(str(subject_name), tbl_cell),
                Paragraph("<b>Subject / Patient ID:</b>", tbl_cell), Paragraph(str(subject_id), tbl_cell),
            ],
            [
                Paragraph("<b>Assessment Type:</b>", tbl_cell), Paragraph(str(ass_type_label), tbl_cell_bold),
                Paragraph("<b>Evaluator / Clinician:</b>", tbl_cell), Paragraph(str(evaluator), tbl_cell),
            ],
            [
                Paragraph("<b>Session Duration:</b>", tbl_cell), Paragraph(f"{dur_str} ({samples_count} frames)", tbl_cell),
                Paragraph("<b>Classification Tags:</b>", tbl_cell), Paragraph(str(tags_str), tbl_cell),
            ],
            [
                Paragraph("<b>Clinical Notes:</b>", tbl_cell),
                Paragraph(str(notes), ParagraphStyle("NotesCell", parent=tbl_cell, fontSize=7.5, leading=9)),
                Paragraph("", tbl_cell), Paragraph("", tbl_cell),
            ]
        ]
        meta_table = Table(meta_table_data, colWidths=[100, 170, 110, 160])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), c_gray_bg),
            ("BOX", (0, 0), (-1, -1), 0.5, c_border),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
            ("SPAN", (1, 3), (3, 3)),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 6))

        # 3. Affective Dimensions (Russell VAD & Behavioral Indices)
        elements.append(Paragraph("2. AFFECTIVE STATE & BEHAVIORAL TELEMETRY", section_h))

        v_interp = "Positive / Elevated" if avg_v > 0.15 else "Negative / Depressed" if avg_v < -0.15 else "Neutral Baseline"
        a_interp = "High Arousal / Agitated" if avg_a > 0.25 else "Calm / Hypo-Arousal" if avg_a < -0.15 else "Moderate Equilibrium"
        d_interp = "Dominant / Assertive" if avg_d > 0.1 else "Submissive / Compliant" if avg_d < -0.1 else "Balanced"

        vad_table_data = [
            [
                Paragraph("Dimension / Metric", tbl_hdr),
                Paragraph("Quantitative Value", tbl_hdr),
                Paragraph("Scale Reference", tbl_hdr),
                Paragraph("Clinical / Behavioral Interpretation", tbl_hdr),
            ],
            [
                Paragraph("<b>Valence (Pleasantness)</b>", tbl_cell),
                Paragraph(f"{avg_v:+.3f}", tbl_cell_bold),
                Paragraph("[-1.000 to +1.000]", body_muted),
                Paragraph(v_interp, tbl_cell),
            ],
            [
                Paragraph("<b>Arousal (Physiological Activation)</b>", tbl_cell),
                Paragraph(f"{avg_a:+.3f}", tbl_cell_bold),
                Paragraph("[-1.000 to +1.000]", body_muted),
                Paragraph(a_interp, tbl_cell),
            ],
            [
                Paragraph("<b>Dominance (Control / Coping)</b>", tbl_cell),
                Paragraph(f"{avg_d:+.3f}", tbl_cell_bold),
                Paragraph("[-1.000 to +1.000]", body_muted),
                Paragraph(d_interp, tbl_cell),
            ],
            [
                Paragraph("<b>Mean Engagement Index</b>", tbl_cell),
                Paragraph(f"{avg_eng * 100:.1f}%", tbl_cell_bold),
                Paragraph("[0.0% to 100.0%]", body_muted),
                Paragraph("Active Cognitive Focus" if avg_eng > 0.6 else "Mild Disengagement", tbl_cell),
            ],
            [
                Paragraph("<b>Mean Visual Attention Score</b>", tbl_cell),
                Paragraph(f"{avg_att * 100:.1f}%", tbl_cell_bold),
                Paragraph("[0.0% to 100.0%]", body_muted),
                Paragraph("Direct Gaze & Head Posture" if avg_att > 0.6 else "Distracted / Averted Gaze", tbl_cell),
            ],
            [
                Paragraph("<b>Mean Cognitive Fatigue</b>", tbl_cell),
                Paragraph(f"{avg_fat * 100:.1f}%", tbl_cell_bold),
                Paragraph("[0.0% to 100.0%]", body_muted),
                Paragraph("High Fatigue / Eye Strain" if avg_fat > 0.55 else "Nominal Cognitive Stamina", tbl_cell),
            ],
        ]
        vad_table = Table(vad_table_data, colWidths=[150, 95, 105, 190])
        vad_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), c_primary),
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.5, c_border),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_gray_bg]),
        ]))
        elements.append(vad_table)
        elements.append(Spacer(1, 6))

        # 4. Dominant Emotion Spectrum Distribution
        elements.append(Paragraph("3. DOMINANT EMOTION SPECTRUM DISTRIBUTION", section_h))
        sorted_emotions = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        emo_data = [
            [
                Paragraph("Emotion Category", tbl_hdr),
                Paragraph("Relative Share (%)", tbl_hdr),
                Paragraph("Density Bar Indicator", tbl_hdr),
                Paragraph("Affect Classification", tbl_hdr),
            ]
        ]
        for emo, pct in sorted_emotions:
            pct_num = round(pct * 100, 1)
            bar_len = int(pct * 30)
            bar_str = "■" * bar_len + "□" * max(0, 30 - bar_len)
            emo_data.append([
                Paragraph(f"<b>{emo.capitalize()}</b>", tbl_cell),
                Paragraph(f"{pct_num:.1f}%", tbl_cell_bold),
                Paragraph(f"<font color='#0284c7'>{bar_str}</font>", ParagraphStyle("Bar", parent=tbl_cell, fontSize=6.5, leading=8)),
                Paragraph("Primary Affect" if pct == sorted_emotions[0][1] else "Secondary Trait", tbl_cell),
            ])

        if len(emo_data) > 1:
            emo_table = Table(emo_data, colWidths=[120, 90, 180, 150])
            emo_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), c_dark),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.5, c_border),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_gray_bg]),
            ]))
            elements.append(emo_table)
            elements.append(Spacer(1, 6))

        # 5. Affective Anomalies & Escalation Sentinel
        elements.append(Paragraph(f"4. AFFECTIVE ANOMALY SENTINEL ALERTS ({len(anom_list)} EVENTS)", section_h))
        if anom_list:
            anom_data = [
                [
                    Paragraph("Severity", tbl_hdr),
                    Paragraph("Anomaly Category", tbl_hdr),
                    Paragraph("Clinical Presentation", tbl_hdr),
                    Paragraph("Recommended Action / Intervention", tbl_hdr),
                ]
            ]
            for a in anom_list[:8]:
                sev = a.get("severity", "INFO").upper()
                sev_color = "#b91c1c" if sev in ("CRITICAL", "HIGH") else "#b45309" if sev == "MODERATE" else "#0284c7"
                sev_p = Paragraph(f"<b><font color='{sev_color}'>[{sev}]</font></b>", tbl_cell)
                typ_p = Paragraph(str(a.get("anomaly_type", "")).replace("_", " ").title(), tbl_cell_bold)
                desc_p = Paragraph(str(a.get("description", "")), ParagraphStyle("Desc", parent=tbl_cell, fontSize=7.5, leading=9))
                rec_p = Paragraph(str(a.get("recommended_action", "Monitor baseline")), ParagraphStyle("Rec", parent=tbl_cell, fontSize=7.5, leading=9))
                anom_data.append([sev_p, typ_p, desc_p, rec_p])

            anom_table = Table(anom_data, colWidths=[70, 130, 170, 170])
            anom_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), c_primary),
                ("BOX", (0, 0), (-1, -1), 0.5, c_border),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_gray_bg]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            elements.append(anom_table)
        else:
            elements.append(Paragraph("<i>No affective anomalies or escalation distress events detected during this recording session. Affect trajectory maintained baseline stability.</i>", body_p))
        elements.append(Spacer(1, 6))

        # Dyadic Interpersonal Synchrony Section (if dyadic metrics present)
        dyadic_data = sess_dict.get("dyadic_metrics") or meta_dict.get("dyadic_metrics")
        if dyadic_data:
            elements.append(Paragraph("5. DYADIC INTERACTION & INTERPERSONAL SYNCHRONY", section_h))
            dy_table_data = [
                [
                    Paragraph("Dyadic Metric", tbl_hdr),
                    Paragraph("Score / Index", tbl_hdr),
                    Paragraph("Assessment Status", tbl_hdr),
                    Paragraph("Clinical / Behavioral Interpretation", tbl_hdr),
                ],
                [
                    Paragraph("<b>Dyadic Rapport Score</b>", tbl_cell),
                    Paragraph(f"<b>{float(dyadic_data.get('rapport_score', 50.0)):.1f} / 100</b>", tbl_cell_bold),
                    Paragraph(str(dyadic_data.get("resonance_category", "Collaborative")), tbl_cell),
                    Paragraph("Composite index of affective synchrony, turn balance, and mimicry", tbl_cell),
                ],
                [
                    Paragraph("<b>Valence Synchrony (r)</b>", tbl_cell),
                    Paragraph(f"{float(dyadic_data.get('valence_synchrony', 0.0)):+.2f}", tbl_cell_bold),
                    Paragraph("Aligned Trajectory" if float(dyadic_data.get('valence_synchrony', 0.0)) > 0.3 else "Discordant / Independent", tbl_cell),
                    Paragraph("Temporal concordance of emotional pleasantness/positivity", tbl_cell),
                ],
                [
                    Paragraph("<b>Floor Balance</b>", tbl_cell),
                    Paragraph(f"{float(dyadic_data.get('conversational_balance', 1.0)) * 100:.0f}%", tbl_cell_bold),
                    Paragraph(f"Dominance: {dyadic_data.get('dominance_speaker', 'balanced')}", tbl_cell),
                    Paragraph("Equality of conversational floor distribution (50/50 balance)", tbl_cell),
                ],
                [
                    Paragraph("<b>Facial Mimicry Index</b>", tbl_cell),
                    Paragraph(f"{float(dyadic_data.get('mimicry_index', 0.0)) * 100:.0f}%", tbl_cell_bold),
                    Paragraph("Responsive" if float(dyadic_data.get('mimicry_index', 0.0)) > 0.25 else "Reserved", tbl_cell),
                    Paragraph("Lagged micro-expression smile matching (0.5s - 2.5s window)", tbl_cell),
                ],
            ]
            dy_table = Table(dy_table_data, colWidths=[130, 90, 140, 180])
            dy_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), c_primary),
                ("BOX", (0, 0), (-1, -1), 0.5, c_border),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_gray_bg]),
            ]))
            elements.append(dy_table)
            elements.append(Spacer(1, 6))

        # 6. Key Pivot Moments (if any)
        if key_moments:
            elements.append(Paragraph(f"6. KEY AFFECTIVE PIVOT MOMENTS ({len(key_moments[:6])} HIGHEST INTENSITY TRANSITIONS)", section_h))
            km_data = [
                [
                    Paragraph("Time Offset", tbl_hdr),
                    Paragraph("Dominant Affect", tbl_hdr),
                    Paragraph("Confidence", tbl_hdr),
                    Paragraph("Valence", tbl_hdr),
                    Paragraph("Arousal", tbl_hdr),
                ]
            ]
            for km in key_moments[:6]:
                t = km.get("relative_time_sec", 0)
                e = str(km.get("dominant_emotion", "neutral")).capitalize()
                c = float(km.get("confidence", 0.0))
                v = float(km.get("valence", 0.0))
                a = float(km.get("arousal", 0.0))
                km_data.append([
                    Paragraph(f"+{t:.1f}s", tbl_cell_bold),
                    Paragraph(e, tbl_cell),
                    Paragraph(f"{c * 100:.0f}%", tbl_cell),
                    Paragraph(f"{v:+.2f}", tbl_cell),
                    Paragraph(f"{a:+.2f}", tbl_cell),
                ])
            km_table = Table(km_data, colWidths=[80, 120, 100, 120, 120])
            km_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), c_dark),
                ("BOX", (0, 0), (-1, -1), 0.5, c_border),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_gray_bg]),
            ]))
            elements.append(km_table)
            elements.append(Spacer(1, 8))

        # 7. Clinical Sign-Off Block
        signoff_data = [
            [
                Paragraph("<b>Evaluating Clinician / Assessor Signature:</b>", tbl_cell),
                Paragraph("<b>Date of Review:</b>", tbl_cell),
                Paragraph("<b>Audit Status:</b>", tbl_cell),
            ],
            [
                Paragraph("<br/><br/>________________________________________", tbl_cell),
                Paragraph(f"<br/><br/>{datetime.now().strftime('%Y-%m-%d')}", tbl_cell),
                Paragraph("<br/><br/><b>VERIFIED & CERTIFIED</b>", ParagraphStyle("Cert", parent=tbl_cell, textColor=c_green)),
            ]
        ]
        signoff_table = Table(signoff_data, colWidths=[240, 150, 150])
        signoff_table.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, c_border),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(KeepTogether([
            Paragraph("6. CLINICAL AUDIT & VERIFICATION", section_h),
            signoff_table,
            Spacer(1, 4),
            Paragraph("<i>Automated Multimodal Affect Diagnostic Report compiled by EmotionSense Neural Affect Engine. Conforms to Russell Circumplex & FACS Action Unit taxonomies.</i>", body_muted),
        ]))

        doc.build(elements)
        return buf.getvalue()

    @classmethod
    def save_pdf(
        cls,
        session: Union[SessionRecord, StoredSession, Dict[str, Any]],
        output_path: Union[str, Path],
        anomalies: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Union[SessionMetadata, Dict[str, Any]]] = None,
    ) -> Path:
        """Generates and writes clinical PDF report to filesystem path."""
        pdf_bytes = cls.generate_pdf(session, anomalies=anomalies, metadata=metadata)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            f.write(pdf_bytes)
        return out
