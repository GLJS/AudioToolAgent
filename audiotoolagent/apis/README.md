# Local API Endpoints

Only two FastAPI apps are required after the clean-up:

```bash
# DeSTA 2.5 (used by the open configuration)
uvicorn audiotoolagent.apis.desta25_api:app --host 0.0.0.0 --port 4004

# AudioFlamingo 3 OpenAI-compatible proxy (used by the closed configuration)
uvicorn audiotoolagent.apis.audioflamingo_api:app --host 0.0.0.0 --port 4010
```

Both servers update `hostnames.txt` on startup so that tool adapters pick up the correct endpoint URL. The helper scripts in `scripts/` start them automatically; the commands above are kept here for reference.
