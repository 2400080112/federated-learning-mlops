/**
 * Read-only demo backend, for the static Netlify build.
 *
 * The real app is FastAPI + SQLite, which Netlify cannot run: no Python runtime
 * for a long-lived process, no writable disk, no place for a background training
 * thread. So the static build ships the *results* of a real run as JSON and
 * replays the read paths from it.
 *
 * Predictions are not replayed -- the model is 31 numbers, so the forward pass
 * runs here in the browser against the exported rows. Move the threshold slider
 * and the confusion matrix really does change. Writes (train, promote, delete)
 * refuse with an explanation rather than pretending.
 */

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

export function createDemoApi(demoUrl) {
  const ready = fetch(demoUrl)
    .then((response) => {
      if (!response.ok) throw new Error(`demo data missing: ${demoUrl}`);
      return response.json();
    });

  let store = null;
  ready.then((data) => { store = data; });

  const RO = (what) =>
    new Error(
      `${what} needs the live API. This Netlify build is a read-only demo of a ` +
      "seeded run — run the app locally (uvicorn app.main:app) for the full loop.",
    );

  const withStore = async (fn) => {
    await ready;
    return fn(store);
  };

  const runDetail = (id) => {
    const run = store.runs.find((r) => r.id === id);
    if (!run) throw new Error(`run ${id} not found in the demo snapshot`);
    return run;
  };

  return {
    isDemo: true,

    health: () => withStore(() => ({ status: "demo", version: store.version, database: "static snapshot" })),
    stats: () => withStore(() => store.stats),
    datasetPreview: (limit = 10) => withStore(() => ({
      ...store.dataset,
      rows: store.dataset.rows.slice(0, limit),
    })),
    defaults: () => withStore(() => store.defaults),

    listRuns: (params = {}) => withStore(() => {
      let runs = store.runs.map(({ history, clients, config, ...rest }) => rest);
      if (params.status) runs = runs.filter((r) => r.status === params.status);
      if (params.scored_only) runs = runs.filter((r) => r.is_scored);
      return runs.slice(0, params.limit || 100);
    }),
    getRun: (id) => withStore(() => runDetail(id)),
    createRun: () => { throw RO("Training"); },
    evaluateRun: () => { throw RO("Scoring"); },
    registerRun: () => { throw RO("Registering"); },
    deleteRun: () => { throw RO("Deleting"); },

    listModels: (params = {}) => withStore(() => {
      let models = [...store.models];
      if (params.stage) models = models.filter((m) => m.stage === params.stage);
      return models;
    }),
    productionModel: () => withStore(() => {
      const model = store.models.find((m) => m.stage === "production");
      if (!model) throw new Error("no production model in the demo snapshot");
      return model;
    }),
    promote: () => { throw RO("Promotion"); },
    deleteVersion: () => { throw RO("Deleting"); },

    listPredictions: () => withStore(() => store.predictions),

    /** Real forward pass over the exported rows, in the browser. */
    predict: (payload = {}) => withStore(() => {
      const model = payload.model_version_id
        ? store.models.find((m) => m.id === payload.model_version_id)
        : store.models.find((m) => m.stage === (payload.stage || "production"));
      if (!model) throw new Error("no model in the demo snapshot for that selection");

      const threshold = payload.threshold ?? 0.5;
      const limit = Math.min(payload.limit ?? store.dataset_rows.length, store.dataset_rows.length);
      const rows = store.dataset_rows.slice(0, limit);

      const { params: w, mean, std } = store.model_math;
      const probs = rows.map((row) => {
        let z = w[w.length - 1]; // bias
        for (let i = 0; i < w.length - 1; i += 1) {
          z += w[i] * ((row[i] - mean[i]) / std[i]);
        }
        return sigmoid(z);
      });
      const preds = probs.map((p) => (p >= threshold ? 1 : 0));
      const truth = rows.map((row) => row[store.model_math.target_index]);

      let tp = 0; let tn = 0; let fp = 0; let fn = 0;
      preds.forEach((p, i) => {
        if (p === 1 && truth[i] === 1) tp += 1;
        else if (p === 0 && truth[i] === 0) tn += 1;
        else if (p === 1 && truth[i] === 0) fp += 1;
        else fn += 1;
      });
      const correct = tp + tn;

      const preview = rows.slice(0, 25).map((row, i) => ({
        sample_id: row[store.model_math.sample_id_index],
        target: truth[i],
        prediction_label: preds[i] === 1 ? "benign" : "malignant",
        p_benign: Number(probs[i].toFixed(3)),
      }));

      return {
        id: 0,
        created_at: new Date().toISOString(),
        model_label: `${model.name}:${model.version}`,
        source_label: "bundled dataset (in-browser inference)",
        threshold,
        n_rows: rows.length,
        summary: {
          n_rows: rows.length,
          predicted_benign: preds.filter((p) => p === 1).length,
          predicted_malignant: preds.filter((p) => p === 0).length,
          mean_confidence: Number(
            (probs.reduce((acc, p) => acc + Math.abs(p - 0.5) * 2, 0) / probs.length).toFixed(4),
          ),
          threshold,
          accuracy: Number((correct / rows.length).toFixed(4)),
          confusion: { tp, tn, fp, fn },
          evaluated_against: "target column present in the snapshot",
        },
        preview,
        download_url: "",
      };
    }),

    predictUpload: () => { throw RO("Uploading a CSV"); },
  };
}
