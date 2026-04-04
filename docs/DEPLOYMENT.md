# AgroEdge AI — Raspberry Pi deployment (runtime scheduler)

This guide runs the **continuous runtime loop** as a **systemd** service: poll ThingSpeak → build features → edge model decision → optional irrigation log publish → local JSONL audit log.

## 0. Verify on your dev machine

From the `agroedge_ai/` directory:

```bash
./scripts/verify.sh
```

Installs/updates `.venv`, runs **pytest**, then **ruff** (if `requirements-dev.txt` was installed). Use `SKIP_RUFF=1 ./scripts/verify.sh` to skip linting.

## Prerequisites

- Raspberry Pi OS (or Debian-based) with **Python 3.11+**
- Repo copied to e.g. `/home/pi/agroedge_ai`
- Virtualenv created and dependencies installed: `pip install -r requirements.txt`
- Exported model bundle present: `model_export/irrigation_model/v1.0.0/`
- ThingSpeak channels configured per `configs/thingspeak_channels.yaml`

## 1. Layout on the Pi

```text
/home/pi/agroedge_ai/
  .venv/
  scripts/runtime_scheduler.py
  model_export/irrigation_model/v1.0.0/
  logs/                      # created automatically; used for JSONL cycles
```

## 2. Secrets and environment

Do **not** commit API keys.

```bash
sudo mkdir -p /etc/agroedge
sudo cp deploy/agroedge-runtime.env.example /etc/agroedge/runtime.env
sudo chmod 600 /etc/agroedge/runtime.env
sudo nano /etc/agroedge/runtime.env   # fill ThingSpeak keys and IDs
```

Edit paths in `/etc/agroedge/runtime.env` if your user or home directory is not `pi`.

## 3. Install systemd unit

Adjust **User**, **Group**, **WorkingDirectory**, and **ExecStart** paths in the unit file if your install differs.

```bash
sudo cp deploy/systemd/agroedge-runtime.service /etc/systemd/system/agroedge-runtime.service
sudo systemctl daemon-reload
sudo systemctl enable agroedge-runtime.service
```

**First run:** consider **without** `--publish-log` until the irrigation log channel is verified:

```ini
ExecStart=/home/pi/agroedge_ai/.venv/bin/python .../runtime_scheduler.py --interval-seconds 900 ...
```

Then enable publishing:

```ini
ExecStart=.../runtime_scheduler.py --publish-log --interval-seconds 900 ...
```

Use a **systemd drop-in** to override flags without editing the main unit:

```bash
sudo systemctl edit agroedge-runtime.service
```

## 4. Start and observe

```bash
sudo systemctl start agroedge-runtime.service
sudo systemctl status agroedge-runtime.service
journalctl -u agroedge-runtime.service -f
```

Local cycle log (default):

```text
/home/pi/agroedge_ai/logs/runtime_cycles.jsonl
```

## 5. Health check

After at least one cycle has written a row:

```bash
/home/pi/agroedge_ai/.venv/bin/python scripts/runtime_health_check.py \
  --project-root /home/pi/agroedge_ai \
  --log-path logs/runtime_cycles.jsonl \
  --max-age-seconds 2700
```

- Default `--max-age-seconds` (2700) allows ~45 minutes slack for a **900 s** poll interval.
- Add `--require-ok-status` if you want to fail when the last cycle recorded `status: error`.

Optional **cron** probe (every 30 minutes):

```cron
*/30 * * * * /home/pi/agroedge_ai/.venv/bin/python /home/pi/agroedge_ai/scripts/runtime_health_check.py --project-root /home/pi/agroedge_ai || logger -t agroedge HEALTH_CHECK_FAILED
```

## 6. Log rotation (JSONL)

Install the logrotate snippet (adjust path if not `pi`):

```bash
sudo cp deploy/logrotate/agroedge-runtime-jsonl /etc/logrotate.d/agroedge-runtime-jsonl
sudo logrotate -d /etc/logrotate.d/agroedge-runtime-jsonl
```

`copytruncate` avoids stopping the service while rotating; logs may split one line rarely — acceptable for JSONL operational logs.

## 7. Restart and failure behavior

The unit uses:

- `Restart=on-failure` — process exits non-zero → restart after `RestartSec`
- `StartLimitBurst` — avoids tight restart loops

The scheduler **also** retries ThingSpeak read/write inside each cycle (see `RetryPolicy` in `edge_inference/scheduler.py`).

## 8. Related scripts

| Script | Purpose |
|--------|---------|
| `scripts/runtime_scheduler.py` | Continuous loop |
| `scripts/runtime_decision_cycle.py` | Single cycle (debug); **live mode** uses the same ThingSpeak enrichment as the scheduler (latest-N env rows, lag/delta features). Set `THINGSPEAK_IRRIGATION_CHANNEL_ID` + `THINGSPEAK_IRRIGATION_READ_API_KEY` for irrigation-log context (optional; omitted fields fall back via `runtime_loop.build_feature_payload`). |
| `scripts/runtime_health_check.py` | Log-based liveness |
| `scripts/run_dashboard.py` | Minimal Flask UI (last cycle, dev) |
| `scripts/run_dashboard_gunicorn.sh` | Dashboard via Gunicorn (production) |
| `scripts/pull_thingspeak_training_data.py` | ThingSpeak → Parquet for retraining ([INGESTION.md](INGESTION.md)) |
| `scripts/merge_training_parquet.py` | Concat + shuffle multiple training Parquets |
| `scripts/train_disease_model.py` | Disease bundle sanity train (sklearn, not images) |
| `scripts/disease_inference_smoke_test.py` | Load disease `model.joblib` and classify random vectors |

## 9. Periodic health check (systemd timer)

Install the oneshot service + timer so JSONL freshness is verified every **15 minutes** (adjust in the unit if needed):

```bash
sudo cp deploy/systemd/agroedge-runtime-health.service /etc/systemd/system/
sudo cp deploy/systemd/agroedge-runtime-health.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now agroedge-runtime-health.timer
systemctl list-timers | grep agroedge
journalctl -u agroedge-runtime-health.service -n 20
```

The service uses `--allow-missing-log` so the **first boot** before any cycle does not fail. After you expect steady logging, edit `/etc/systemd/system/agroedge-runtime-health.service` and remove `--allow-missing-log` for stricter monitoring.

## 10. Optional ExecStartPost (delayed probe)

Example drop-in (non-fatal until log exists):

```bash
sudo mkdir -p /etc/systemd/system/agroedge-runtime.service.d/
sudo cp deploy/systemd/agroedge-runtime.service.d/10-execstartpost.conf.example \
      /etc/systemd/system/agroedge-runtime.service.d/10-execstartpost.conf
sudo systemctl daemon-reload
```

The `-` prefix on `ExecStartPost` tells systemd **not** to mark the main service failed if the probe fails.

## 11. Local monitoring dashboard (Flask)

Read-only stub: shows the **last line** of `logs/runtime_cycles.jsonl` (or `AGROEDGE_RUNTIME_LOG`).

### Manual run

```bash
cd /home/pi/agroedge_ai
source .venv/bin/activate
export AGROEDGE_RUNTIME_LOG=logs/runtime_cycles.jsonl
python scripts/run_dashboard.py
```

Open from another device on the LAN: `http://<pi-ip>:5000/`  
Health: `http://<pi-ip>:5000/health`  
JSON: `http://<pi-ip>:5000/api/last-cycle`

### systemd service (Flask dev server)

```bash
sudo cp deploy/systemd/agroedge-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now agroedge-dashboard.service
```

### Production: Gunicorn (recommended)

Install extras and use the WSGI launcher:

```bash
cd /home/pi/agroedge_ai
source .venv/bin/activate
pip install -r requirements.txt -r requirements-prod.txt
chmod +x scripts/run_dashboard_gunicorn.sh
./scripts/run_dashboard_gunicorn.sh
```

Or install the example unit (overrides the Flask-only unit if you name it the same):

```bash
sudo cp deploy/systemd/agroedge-dashboard-gunicorn.service.example \
       /etc/systemd/system/agroedge-dashboard.service
# Edit User/Group/WorkingDirectory/ExecStart paths if not /home/pi/agroedge_ai
sudo systemctl daemon-reload
sudo systemctl restart agroedge-dashboard.service
```

Tune `GUNICORN_WORKERS` / `GUNICORN_THREADS` in the unit or environment. The app factory is exposed as `web_dashboard.app:app` for other WSGI servers too.

**Security:** bind to LAN only or put behind a reverse proxy with auth before exposing to the internet. Default listens on `0.0.0.0:5000` for farm Wi‑Fi use.

**ThingSpeak links:** If `THINGSPEAK_ENV_CHANNEL_ID` / `THINGSPEAK_IRRIGATION_CHANNEL_ID` are set in the process environment (e.g. from `/etc/agroedge/runtime.env`), the dashboard shows read-only links to `https://thingspeak.com/channels/<id>` and exposes JSON at `/api/thingspeak-links`. Override the web host with `AGROEDGE_THINGSPEAK_WEB_BASE` if needed (not the API URL).

## 12. HTTPS reverse proxy (Caddy or nginx)

For remote access, **do not** expose plain HTTP on `0.0.0.0:5000` to the public internet. Terminate TLS on the Pi or a small VPS and proxy to the app bound to **localhost**.

1. Set `FLASK_RUN_HOST=127.0.0.1` (and `FLASK_RUN_PORT=5000`) in the dashboard unit, or use Gunicorn with `--bind 127.0.0.1:5000`.
2. Install **Caddy** or **nginx** and use one of:
   - `deploy/reverse-proxy/Caddyfile.example` — automatic HTTPS (Let’s Encrypt) when DNS points to the host.
   - `deploy/reverse-proxy/nginx-dashboard.conf.example` — TLS paths for certbot-style certs.

Optional: enable **basic auth** in Caddy (see comment in the example) or nginx `auth_basic` for a minimal gate.

**Flask-native Basic auth:** set `AGROEDGE_DASHBOARD_BASIC_USER` and `AGROEDGE_DASHBOARD_BASIC_PASSWORD` in the service environment (see `.env.example`). Endpoints other than `GET /health` return `401` without valid credentials.

Retraining from ThingSpeak is documented in [INGESTION.md](INGESTION.md).
