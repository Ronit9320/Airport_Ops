# AirportOpsEnv

An OpenEnv-compliant environment where an AI agent acts as an Airport Ground Operations Controller.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Test

```bash
# Health check
curl http://localhost:8000/health

# Reset environment
curl -X POST http://localhost:8000/reset?task_id=task1

# Take a step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"flight_id":"FL001","action_type":"hold"}'
```

## Tasks

| Task | Difficulty | Flights | Crises |
|------|------------|---------|--------|
| task1 | Easy | 5 | 1 medevac |
| task2 | Medium | 8 | Fuel emergency + bomb threat |
| task3 | Hard | 15 | Hijacking + runway fire |

## Priority Hierarchy

1. Army / Defense
2. Medevac / Medical
3. Government / VIP
4. Commercial
5. Cargo

**Fuel emergency (< 10 mins) overrides ALL priorities**

## License

MIT
