# Deploying on Netlify

Read this before running `netlify deploy`, because the honest answer is that
**this app cannot run on Netlify as it runs locally** — and the fix is a shape
change, not a config flag.

## Why the backend can't run there

| The app needs | Netlify gives |
|---|---|
| A long-lived Python process (`uvicorn app.main:app`) | Static files, plus serverless functions in **Node/Go**. There is no first-class Python runtime for a persistent server. |
| A writable disk for SQLite (`data/app.db`) | A read-only filesystem; only `/tmp` is writable, and it is per-invocation and discarded. |
| A background thread pool training for 25 rounds and committing per round | Functions are request-scoped and time-limited; nothing survives the response. |
| An in-process model registry with immutable versions | No durable local storage. |

So there are three real options. This repo implements the first, and supports the
second with one flag.

## Option A — static snapshot (implemented, recommended)

`scripts/build_netlify.py` trains for real, then dumps the API's own JSON
responses into a snapshot that a static frontend replays:

```
python scripts/build_netlify.py          # -> netlify/site/
python -m http.server 8080 --directory netlify/site   # preview at :8080
npx netlify deploy --prod --dir netlify/site
```

Or drag the `netlify/site` folder onto <https://app.netlify.com/drop> — no CLI,
no account setup, and you get a public URL in about ten seconds.

What works in the snapshot:

- **Dashboard, Runs, run drawer** — the three seeded runs with their real
  per-round curves, client class balances, configs and test metrics.
- **Registry** — versions, stages, the promotion gate's recorded decisions.
- **Predict** — and this is the part worth knowing: the model is 31 numbers, so
  the forward pass runs **in the browser** (`frontend/js/demo.js`) over the
  exported rows. Move the threshold slider and the confusion matrix genuinely
  changes. It is not a screenshot of a prediction.

What does not work, by design:

- Training, scoring, registering, promoting, deleting. They throw a clear error
  rather than pretending, and the Train form is disabled with an explanation.
- CSV uploads.
- The snapshot is frozen at build time — new runs appear only when you rebuild
  and redeploy.
- A "Read-only demo · static build" flag is pinned in the sidebar so nobody
  mistakes it for the live system.

The committed `netlify/site/` directory is **generated**. Do not edit it by hand;
edit `frontend/` and re-run the script.

## Option B — static frontend on Netlify, API elsewhere

Keep the full app and host the Python part where it can run: Render, Railway,
Fly.io, a Hugging Face Space, or any VPS. Then build the site pointing at it:

```
python scripts/build_netlify.py --api-base https://your-api.onrender.com
npx netlify deploy --prod --dir netlify/site
```

With `--api-base`, the site is not in demo mode: it sends real requests to that
URL. The backend already allows cross-origin calls (`allow_origins=["*"]` in
`backend/app/main.py`), so no CORS work is needed — but narrow that list to your
Netlify domain before this is anything but a demo.

Serving the API is one command on any Python host:

```bash
cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Note the database is a SQLite file on that host's disk. On a platform with an
ephemeral filesystem, point `FEDFL_DATABASE_URL` at a persistent volume or swap
SQLAlchemy to Postgres — the models are portable.

## Option C — rewrite as Netlify Functions (not done)

Possible, but it is a rewrite, not a deploy: Netlify Functions run Node or Go, so
either the federated core (NumPy logistic regression, FedAvg, DP) gets ported to
JavaScript, or the backend ships as a container image on a Lambda-style runtime,
plus a hosted Postgres for the registry. That trades the project's readable NumPy
implementation for infrastructure. Worth doing if you need a serverless demo;
it is not what this repo is.

## Configuration

`netlify/site/index.html` carries the runtime config the frontend reads at boot:

```html
<script>window.FEDFLOPS_CONFIG = {"demo": true, "apiBase": ""};</script>
```

| Key | Meaning |
|---|---|
| `demo: true` | replay the snapshot; writes are refused |
| `apiBase: "https://…"` | send requests to that origin instead of same-origin |

`netlify.toml` sets `publish = "netlify/site"` with no build command, so the
deploy does not depend on whatever Python the Netlify build image happens to
ship. Security headers and a hash-routing fallback are in there too.

## Checklist

- [ ] `python scripts/build_netlify.py` (add `--api-base …` for Option B)
- [ ] `python -m http.server 8080 --directory netlify/site` and click through all
      five pages
- [ ] `npx netlify deploy --prod --dir netlify/site` (or drag-and-drop)
- [ ] Open the URL: the sidebar should say **Read-only demo · static build**
- [ ] Try the threshold slider on Predict — the matrix should move
