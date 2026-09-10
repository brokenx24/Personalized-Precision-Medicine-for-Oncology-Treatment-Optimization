"""Medical NER Visual Entity Highlighting and HTML Generator.
NLP Engineer Module - Stage 03 NLP.
Generates HTML visual mockups and formatted reports highlighting extracted oncology entities in context.
"""

import os
import re
import json

COLOR_MAP = {
    "GENE_MUTATION": "#9b59b6",
    "DRUG": "#3498db",
    "DOSAGE": "#f39c12",
    "ADVERSE_EVENT": "#e74c3c"
}

def highlight_entities(text: str, entities: list) -> str:
    """Returns HTML highlighted text."""
    # Sort entities in reverse order of start position
    sorted_ents = sorted(entities, key=lambda x: x.get("start", 0), reverse=True)
    highlighted = text
    for ent in sorted_ents:
        s = ent.get("start", -1)
        e = ent.get("end", -1)
        etype = ent.get("entity_type", "")
        col = COLOR_MAP.get(etype, "#888888")
        if s != -1 and e != -1 and s < len(highlighted) and e <= len(highlighted):
            span_html = f'<span style="background-color: {col}33; border-bottom: 2px solid {col}; padding: 2px 4px; border-radius: 3px; font-weight: bold;" title="{etype}">{highlighted[s:e]} <small style="color:{col}">[{etype}]</small></span>'
            highlighted = highlighted[:s] + span_html + highlighted[e:]
        else:
            # Fallback substring replace
            ent_text = ent.get("text", "")
            if ent_text and ent_text in highlighted:
                span_html = f'<span style="background-color: {col}33; border-bottom: 2px solid {col}; padding: 2px 4px; border-radius: 3px; font-weight: bold;" title="{etype}">{ent_text} <small style="color:{col}">[{etype}]</small></span>'
                highlighted = highlighted.replace(ent_text, span_html, 1)
    return highlighted

def generate_sample_ner_visualization():
    print("Generating Visual Entity Highlighting Report...")
    sample_notes = [
        {
            "title": "Case 1 — Emergent Immune-Related Toxicity",
            "text": "Patient with metastatic melanoma receiving pembrolizumab 200 mg IV every 3 weeks presented with acute febrile neutropenia and severe grade 3 colitis.",
            "entities": [
                {"text": "pembrolizumab", "entity_type": "DRUG", "start": 39, "end": 52},
                {"text": "200 mg", "entity_type": "DOSAGE", "start": 53, "end": 59},
                {"text": "febrile neutropenia", "entity_type": "ADVERSE_EVENT", "start": 103, "end": 122},
                {"text": "colitis", "entity_type": "ADVERSE_EVENT", "start": 140, "end": 147}
            ]
        },
        {
            "title": "Case 2 — Targeted Biomarker Therapy Consultation",
            "text": "Molecular pathology confirms EGFR L858R mutation in non-small cell lung carcinoma. Initiated osimertinib 80 mg daily with routine surveillance.",
            "entities": [
                {"text": "EGFR L858R", "entity_type": "GENE_MUTATION", "start": 29, "end": 39},
                {"text": "osimertinib", "entity_type": "DRUG", "start": 94, "end": 105},
                {"text": "80 mg", "entity_type": "DOSAGE", "start": 106, "end": 111}
            ]
        }
    ]
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Medical NER In-Context Entity Highlighting</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 24px; background: #f8f9fa; color: #212529; }
        .card { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); padding: 20px; margin-bottom: 20px; border-left: 5px solid #2980b9; }
        h3 { margin-top: 0; color: #2c3e50; }
        .legend { display: flex; gap: 15px; margin-bottom: 20px; }
        .badge { padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; color: white; }
    </style>
</head>
<body>
    <h2>Stage 03 NLP — Medical Named Entity Recognition Highlighting</h2>
    <div class="legend">
        <span class="badge" style="background: #9b59b6;">GENE_MUTATION</span>
        <span class="badge" style="background: #3498db;">DRUG</span>
        <span class="badge" style="background: #f39c12;">DOSAGE</span>
        <span class="badge" style="background: #e74c3c;">ADVERSE_EVENT</span>
    </div>
"""
    for note in sample_notes:
        hl = highlight_entities(note["text"], note["entities"])
        html_content += f"""
    <div class="card">
        <h3>{note['title']}</h3>
        <p style="font-size: 15px; line-height: 1.6;">{hl}</p>
    </div>
"""
    html_content += """
</body>
</html>
"""
    vis_dir = os.path.join("STAGE_03_NLP", "nlp_engineer", "visualizations", "ner")
    os.makedirs(vis_dir, exist_ok=True)
    with open(os.path.join(vis_dir, "entity_highlighting_sample.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
    print("  -> Saved entity_highlighting_sample.html\n")

if __name__ == "__main__":
    generate_sample_ner_visualization()
