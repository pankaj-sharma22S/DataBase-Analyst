# Insight AI frontend

Run the API from the project root:

```bash
uvicorn api:app --reload --port 8000
```

Install and run the React client:

```bash
npm install
npm run dev
```

The client uses the real backend fields exposed by `api.py`: `answer`, `validated_sql`, `primary_result`, `plotly_figure`, `diagnostic_report`, `insights`, `final_response`, and LangGraph workflow events.
