# Georgia Labor Market Research Dashboard

An interactive research dashboard tracking 25 years of Georgia's labor market against its Southeastern peers and the national economy. Built in Python with Plotly Dash.

**Live demo:** _your-render-url-here.onrender.com_
**Author:** John Brassard

---

## Overview

Six tabs covering 25 years of Georgia labor market data:

1. **Executive Summary** — headline KPIs, key findings, methodology
2. **State Overview** — Georgia vs SE peers vs US (UR, LFPR, payrolls, GA-vs-US spread) with interactive date range filter
3. **Labor Force Demographics** — prime-age participation, education and race/ethnicity gaps with demographic dropdown
4. **Georgia Economic Development** — real GDP growth, per-capita income, industry composition with chart-type toggle
5. **National Labor Market** — Beveridge curve, labor tightness, real wages, ECI, percentile rank
6. **Econometric Models & Research** — OLS with HC3 robust SEs, STL decomposition, AR(1) forecast, shift-share, Welch t-tests

## Headline finding

Georgia has flipped from a +1.5pp unemployment-rate deficit during the Great Recession to consistently matching or beating the national average — a structural improvement, not a cyclical blip. The state now ranks in the top quartile of US states on UR.

## Tech stack

- **Data:** BLS LAUS (Excel), FRED (`fredapi`), BLS CPS, BLS JOLTS, BLS CES, BEA
- **Analysis:** `pandas`, `numpy`, `scipy`, `statsmodels`
- **Visualization:** `plotly` (graph_objects + subplots)
- **App framework:** `dash` + `dash-bootstrap-components`
- **Deployment:** `gunicorn` on Render

## Running locally

```bash
git clone https://github.com/<your-username>/ga-labor-market-dashboard.git
cd ga-labor-market-dashboard

python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set your FRED API key (free from https://fred.stlouisfed.org/docs/api/api_key.html)
export FRED_API_KEY="your_key_here"   # Windows: set FRED_API_KEY=your_key_here

python ga_labor_dashboard_v2_render.py
# Open http://localhost:8050
```

## Deploying to Render

This repo includes `render.yaml` for one-click deploy.

1. Push to GitHub
2. Sign in at [render.com](https://render.com) with your GitHub account
3. New → Blueprint → select this repo → Apply
4. When prompted, paste your FRED API key into the `FRED_API_KEY` environment variable
5. Wait 5-10 minutes for the first build

You'll get a URL like `https://ga-labor-market-dashboard.onrender.com`.

**Free-tier note:** Render's free web services sleep after 15 min of inactivity. First visit after sleep takes ~30 seconds to spin up. Fine for portfolio use.

## Methodology

- **OLS regression:** GA UR modeled on US UR, JOLTS Job Openings Rate, AHE YoY%, and a linear time trend, with HC3 heteroskedasticity-robust standard errors
- **STL decomposition:** Georgia LFPR separated into trend, seasonal, and remainder components (period=12, robust=True)
- **AR(1) forecast:** 12-month-ahead Georgia payroll growth forecast with 95% confidence intervals
- **Shift-share decomposition:** 5-year decomposition of GA employment change into national, industry-mix, and competitive effects
- **Welch's t-tests:** GA UR vs US UR mean comparisons across five business cycles

## Data sources

| Source | What it provides |
|---|---|
| BLS LAUS (Excel, in `data/`) | State-level UR, LFPR, employment counts |
| FRED | State and national series, recession indicators |
| BLS CPS (via FRED) | Demographic LFPR/UR by sex, race/ethnicity, education |
| BLS JOLTS (via FRED) | Job openings, hires, quits, layoffs rates |
| BLS CES (via FRED) | Average hourly earnings, Employment Cost Index |
| BEA (via FRED) | GA real GDP, per-capita personal income |

## License

MIT

## Contact

John Brassard — [LinkedIn](https://www.linkedin.com/in/your-handle)
