# Vireo Audio Support Dashboard

A small local dashboard for monthly ticket volume by category and first-routed team, with optional AI-assisted recategorisation and a human-review check.

## Run

Requirements: Python 3.10 or newer and an internet connection for package installation. The dashboard itself runs locally. AI classification requires an OpenAI API key and may incur provider charges; charts from the existing tags work without a key.

On Windows, from this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The included `data/tickets.csv` is the ticket export. Alternatively, upload a ticket CSV from the dashboard sidebar. The app does not need customer, order, product, or agent-name exports for the requested charts.

## AI classification

Enter an API key and model name in the sidebar. The app does not save the key. Before classification, it asks for consent because ticket text is sent to the configured API. It redacts common email, phone, and long-ID patterns, but names or other identifying details in free text may remain. Review the provider's data terms first.

Start with a small sample to inspect results and provider costs. Labels are cached locally in `ai_labels.csv`; the cache stores ticket IDs, text fingerprints, predicted labels, and confidence scores, not the message text. Choose **All remaining** only after deciding that the usage and cost are acceptable.

## Checking quality

Download the human-review template for AI-labelled tickets, fill in `reviewed_category` using an exact label from the existing tag list, and upload the completed CSV. The dashboard reports agreement and per-category recall on the reviewed sample. This measures the model against human review; agreement with the original intake tags is not treated as proof of correctness.

## Reporting decisions and limitations

- The requested window is January 2025 through June 2026. Rows outside it are excluded and counted in the sidebar.
- Team charts use `assigned_team`, which is the first-routed team, not necessarily the resolving agent's team.
- The existing tags seed the initial category list but are not assumed accurate. AI labels are suggestions until reviewed.
- The response-breach estimate uses channel targets from the support policy and applies ₹350 per breach. Missing or negative timestamp intervals are excluded.
- Legacy transfer blanks are unknown, not zero. The dashboard does not estimate transfer savings from that field.
- Tier 2 volume should not be compared directly with Tier 1 volume for staffing decisions.
- A sample's agreement rate is only an estimate; inspect the sample size and category-level results before using it.
