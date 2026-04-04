"""
Minimal Flask dashboard: last runtime cycle from JSONL log.

Read-only. No actuator control in this stub (per phased architecture).
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml
from flask import Flask, Response, jsonify

from web_dashboard.auth import apply_optional_basic_auth
from web_dashboard.disease_history import (
    read_last_disease_prediction,
    read_last_disease_predictions,
    register_disease_history_routes,
)
from web_dashboard.disease_predict import register_disease_predict_routes

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.jsonl_io import read_last_jsonl_row  # noqa: E402
from web_dashboard.thingspeak_links import (  # noqa: E402
    collect_configured_links,
    links_section_html,
)

_OPENAPI_PATH = _ROOT / "docs" / "api" / "openapi.yaml"


def _resolve_log_path() -> Path:
    raw = os.getenv("AGROEDGE_RUNTIME_LOG", "logs/runtime_cycles.jsonl")
    p = Path(raw)
    if not p.is_absolute():
        p = _ROOT / p
    return p


def create_app() -> Flask:
    app = Flask(__name__)
    apply_optional_basic_auth(app)
    register_disease_predict_routes(app, _ROOT)
    register_disease_history_routes(app, _ROOT)

    @app.get("/health")
    def health() -> Response:
        return Response("ok\n", mimetype="text/plain", status=200)

    @app.get("/api/openapi.yaml")
    def openapi_yaml() -> Response | tuple[Response, int]:
        if not _OPENAPI_PATH.is_file():
            return jsonify({"ok": False, "error": "openapi_not_found"}), 404
        return Response(
            _OPENAPI_PATH.read_text(encoding="utf-8"),
            mimetype="application/yaml",
        )

    @app.get("/api/openapi.json")
    def openapi_json() -> Response | tuple[Response, int]:
        if not _OPENAPI_PATH.is_file():
            return jsonify({"ok": False, "error": "openapi_not_found"}), 404
        spec = yaml.safe_load(_OPENAPI_PATH.read_text(encoding="utf-8"))
        return jsonify(spec)

    @app.get("/api/thingspeak-links")
    def api_thingspeak_links() -> Response:
        """Read-only ThingSpeak web UI URLs derived from environment channel IDs."""
        links = [
            {"label": link.label, "channel_id": link.channel_id, "url": link.channel_url}
            for link in collect_configured_links()
        ]
        return jsonify({"ok": True, "links": links})

    @app.get("/api/last-cycle")
    def api_last_cycle() -> tuple[Response, int]:
        path = _resolve_log_path()
        row = read_last_jsonl_row(path)
        if row is None:
            return jsonify({"ok": False, "error": "no_log", "path": str(path)}), 503
        return jsonify({"ok": True, "path": str(path), "row": row}), 200

    @app.get("/")
    def index() -> Response:
        path = _resolve_log_path()
        row = read_last_jsonl_row(path)
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        ts_section = links_section_html()
        disease_row, disease_path = read_last_disease_prediction(_ROOT)
        history_rows, _ = read_last_disease_predictions(_ROOT, limit=5)

        def _format_top3(probs) -> str:
            if isinstance(probs, dict) and probs:
                items = sorted(probs.items(), key=lambda kv: float(kv[1]), reverse=True)[:3]
                return "\n".join(f"{k}: {float(v):.3f}" for k, v in items)
            return "—"

        if not history_rows:
            history_html = "<h3>Recent disease predictions</h3><p class='muted'>—</p>"
        else:
            history_items = "".join(
                "<li>"
                f"<b>{r.get('disease_class_name','—')}</b>"
                f" · Model: <code>{r.get('model_version','—')}</code>"
                f" · Inferred: <code>{r.get('inferred_at_utc','—')}</code>"
                f"<br><span class='muted'>Top-3:</span> "
                f"<pre style='margin:0;display:inline-block'>"
                f"{_format_top3(r.get('probabilities'))}</pre>"
                "</li>"
                for r in history_rows
            )
            history_html = "<h3>Recent disease predictions</h3><ol>" + history_items + "</ol>"
        if disease_row is None:
            disease_html = (
                "<h2>Disease inference</h2>"
                "<p>No disease prediction yet. Upload an image to run inference.</p>"
                f"<p class='muted'>Disease log: <code>{disease_path}</code></p>"
                "<p>"
                "Class: <b id='diseaseClass'>—</b> · "
                "Model: <code id='diseaseModel'>—</code><br>"
                "Inferred at: <code id='diseaseInferred'>—</code>"
                "</p>"
                "<h3>Top-3 Probabilities</h3>"
                "<pre id='diseaseTopProbs' class='muted' style='white-space:pre-wrap;margin-top:.25rem'>—</pre>"
                "<form id='diseaseForm' style='margin-top:1rem'>"
                "<label>Upload rice leaf image:</label><br>"
                "<input type='file' id='diseaseImage' accept='image/*' required><br>"
                "<button type='submit'>Predict</button>"
                "</form>"
                "<p id='diseaseStatus' class='muted'></p>"
                "<script>"
                "document.getElementById('diseaseForm').addEventListener('submit', async (e) => {"
                "e.preventDefault();"
                "const elStatus = document.getElementById('diseaseStatus');"
                "elStatus.textContent = 'Submitting...';"
                "const f = document.getElementById('diseaseImage').files[0];"
                "if(!f){ return; }"
                "const r = new FileReader();"
                "r.onload = async () => {"
                "try {"
                "const dataUrl = r.result;"
                "const base64 = dataUrl.includes(',') ? dataUrl.split(',')[1] : dataUrl;"
                "const res = await fetch('/api/disease/predict', {"
                "method: 'POST',"
                "headers: {'Content-Type':'application/json'},"
                "body: JSON.stringify({image_base64: base64})"
                "});"
                "const data = await res.json();"
                "if(data.ok){"
                "document.getElementById('diseaseClass').textContent = data.disease_class_name;"
                "document.getElementById('diseaseModel').textContent = data.model_version;"
                "document.getElementById('diseaseInferred').textContent = data.inferred_at_utc;"
                "elStatus.textContent = 'Predicted: ' + data.disease_class_name;"
                "if(data.probabilities){"
                "const entries = Object.entries(data.probabilities).sort((a,b)=>b[1]-a[1]).slice(0,3);"
                "document.getElementById('diseaseTopProbs').textContent = entries.map("
                "(e)=>e[0] + ': ' + Number(e[1]).toFixed(3)"
                ").join('\\n');"
                "} else {"
                "document.getElementById('diseaseTopProbs').textContent = '—';"
                "}"
                "} else {"
                "elStatus.textContent = 'Error: ' + (data.error || 'unknown');"
                "document.getElementById('diseaseTopProbs').textContent = '—';"
                "}"
                "} catch (err) {"
                "elStatus.textContent = 'Error: ' + String(err);"
                "}"
                "};"
                "r.readAsDataURL(f);"
                "});"
                "</script>"
                f"{history_html}"
            )
        else:
            disease_class = disease_row.get("disease_class_name", "—")
            model_version = disease_row.get("model_version", "—")
            inferred_at = disease_row.get("inferred_at_utc", "—")
            disease_probs = disease_row.get("probabilities")
            if isinstance(disease_probs, dict) and disease_probs:
                items = sorted(disease_probs.items(), key=lambda kv: float(kv[1]), reverse=True)[:3]
                disease_probs_text = "\n".join(f"{k}: {float(v):.3f}" for k, v in items)
            else:
                disease_probs_text = "—"
            disease_html = (
                "<h2>Disease inference — last result</h2>"
                "<p>"
                f"Class: <b id='diseaseClass'>{disease_class}</b> · "
                f"Model: <code id='diseaseModel'>{model_version}</code><br>"
                f"Inferred at: <code id='diseaseInferred'>{inferred_at}</code>"
                "</p>"
                f"<p class='muted'>Disease log: <code>{disease_path}</code></p>"
                "<form id='diseaseForm'>"
                "<label>Upload rice leaf image:</label><br>"
                "<input type='file' id='diseaseImage' accept='image/*' required><br>"
                "<button type='submit'>Predict</button>"
                "</form>"
                "<p id='diseaseStatus' class='muted'></p>"
                "<h3>Top-3 Probabilities</h3>"
                f"<pre id='diseaseTopProbs' class='muted' "
                "style='white-space:pre-wrap;margin-top:.25rem'>"
                f"{disease_probs_text}</pre>"
                "<script>"
                "document.getElementById('diseaseForm').addEventListener('submit', async (e) => {"
                "e.preventDefault();"
                "const elStatus = document.getElementById('diseaseStatus');"
                "elStatus.textContent = 'Submitting...';"
                "const f = document.getElementById('diseaseImage').files[0];"
                "if(!f){ return; }"
                "const r = new FileReader();"
                "r.onload = async () => {"
                "try {"
                "const dataUrl = r.result;"
                "const base64 = dataUrl.includes(',') ? dataUrl.split(',')[1] : dataUrl;"
                "const res = await fetch('/api/disease/predict', {"
                "method: 'POST',"
                "headers: {'Content-Type':'application/json'},"
                "body: JSON.stringify({image_base64: base64})"
                "});"
                "const data = await res.json();"
                "if(data.ok){"
                "document.getElementById('diseaseClass').textContent = data.disease_class_name;"
                "document.getElementById('diseaseModel').textContent = data.model_version;"
                "document.getElementById('diseaseInferred').textContent = data.inferred_at_utc;"
                "elStatus.textContent = 'Predicted: ' + data.disease_class_name;"
                "if(data.probabilities){"
                "const entries = Object.entries(data.probabilities).sort((a,b)=>b[1]-a[1]).slice(0,3);"
                "document.getElementById('diseaseTopProbs').textContent = entries.map("
                "(e)=>e[0] + ': ' + Number(e[1]).toFixed(3)"
                ").join('\\n');"
                "} else {"
                "document.getElementById('diseaseTopProbs').textContent = '—';"
                "}"
                "} else {"
                "elStatus.textContent = 'Error: ' + (data.error || 'unknown');"
                "document.getElementById('diseaseTopProbs').textContent = '—';"
                "}"
                "} catch (err) {"
                "elStatus.textContent = 'Error: ' + String(err);"
                "}"
                "};"
                "r.readAsDataURL(f);"
                "});"
                "</script>"
                f"{history_html}"
            )
        if row is None:
            body = (
                "<!DOCTYPE html><html><head><meta charset='utf-8'><title>AgroEdge</title>"
                "<style>.muted{color:#666} code{font-size:90%}</style></head>"
                "<body><h1>AgroEdge runtime</h1>"
                f"<p>No log yet or empty: <code>{path}</code></p>"
                f"<p>Server time: {now}</p>"
                "<p><a href='/api/last-cycle'>JSON</a> · "
                "<a href='/api/thingspeak-links'>ThingSpeak links</a> · "
                "<a href='/api/openapi.yaml'>OpenAPI (YAML)</a> · "
                "<a href='/api/openapi.json'>OpenAPI (JSON)</a> · "
                "<a href='/api/openapi.yaml#paths/api/disease/predict'>Disease predict (POST)</a></p>"
                f"{ts_section}"
                f"{disease_html}"
                "</body></html>"
            )
            return Response(body, mimetype="text/html")

        decision = row.get("decision") or {}
        action = row.get("action_payload") or {}
        status = row.get("status", "—")
        err = row.get("error")
        rows_html = "".join(
            f"<tr><th>{k}</th><td><pre style='margin:0'>{v}</pre></td></tr>"
            for k, v in [
                ("Log path", str(path)),
                ("Cycle", row.get("cycle_index")),
                ("Timestamp (row)", row.get("timestamp_utc")),
                ("Status", status),
                ("Error", err or "—"),
                ("Should irrigate", decision.get("should_irrigate")),
                ("Approved duration (min)", decision.get("approved_duration_minutes")),
                ("Blocked reason", decision.get("blocked_reason")),
                ("Model probability", decision.get("model_probability")),
                ("Model version", decision.get("model_version")),
                ("Node", action.get("node_id")),
                ("Publish entry id", row.get("publish_entry_id")),
            ]
        )
        html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'><title>AgroEdge</title>"
            "<style>body{font-family:system-ui,sans-serif;max-width:56rem;margin:1.5rem auto}"
            "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:.5rem}"
            "th{text-align:left;background:#f5f5f5;width:14rem}code{font-size:90%}"
            ".muted{color:#666}</style></head>"
            "<body><h1>AgroEdge — last runtime cycle</h1>"
            f"<p>Server time: {now}. <a href='/api/last-cycle'>Raw JSON</a> · "
            "<a href='/api/thingspeak-links'>ThingSpeak links</a> · "
            "<a href='/api/openapi.yaml'>OpenAPI (YAML)</a> · "
            "<a href='/api/openapi.json'>OpenAPI (JSON)</a> · "
            "<a href='/api/openapi.yaml#paths/api/disease/predict'>Disease predict (POST)</a></p>"
            f"{ts_section}"
            f"<table>{rows_html}</table>"
            f"{disease_html}"
            "<h2>Full row (JSON)</h2>"
            f"<pre>{__import__('json').dumps(row, indent=2)}</pre>"
            "</body></html>"
        )
        return Response(html, mimetype="text/html")

    return app


# WSGI entry: `gunicorn 'web_dashboard.app:create_app()'` or flask --app
app = create_app()
