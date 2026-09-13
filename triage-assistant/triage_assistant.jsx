import { useState, useMemo, useRef } from "react";

// ---- Trained model, exported from a real scikit-learn LogisticRegression
// trained on the Wisconsin Breast Cancer Diagnostic dataset. See the
// accompanying train_model.py for how these numbers were produced.
const MODEL = {
  featureNames: [
    "mean radius", "mean texture", "mean perimeter",
    "mean area", "mean concavity", "mean smoothness",
  ],
  labels: [
    { key: "mean radius", label: "Mean radius", unit: "mm", help: "Average distance from center to the tumor edge" },
    { key: "mean texture", label: "Mean texture", unit: "", help: "Variation in grayscale intensity across the cell surface" },
    { key: "mean perimeter", label: "Mean perimeter", unit: "mm", help: "Perimeter of the tumor outline" },
    { key: "mean area", label: "Mean area", unit: "mm\u00B2", help: "Total area of the tumor cross-section" },
    { key: "mean concavity", label: "Mean concavity", unit: "", help: "Severity of concave portions of the contour" },
    { key: "mean smoothness", label: "Mean smoothness", unit: "", help: "Local variation in radius lengths" },
  ],
  mean: [14.067213186813202, 19.247362637362627, 91.55740659340661, 648.5410989010988, 0.08919332241758247, 0.0961674285714285],
  scale: [3.4955321235941827, 4.400447138909897, 24.122678706966134, 344.5652953047693, 0.08165706364246943, 0.013442917175187976],
  coef: [-1.4267309721797272, -1.4560645638961969, -1.2717414569847152, -1.5537828087336476, -1.1323062186239392, -1.5602493052408832],
  intercept: 0.08300033386336135,
  ranges: {
    "mean radius": [6.981, 28.11],
    "mean texture": [9.71, 39.28],
    "mean perimeter": [43.79, 188.5],
    "mean area": [143.5, 2501.0],
    "mean concavity": [0.0, 0.4268],
    "mean smoothness": [0.05263, 0.1634],
  },
  metrics: {
    accuracy: 0.877,
    precisionMalignant: 0.769,
    recallMalignant: 0.952,
    f1Malignant: 0.851,
    testSetSize: 114,
  },
};

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function predict(values) {
  let z = MODEL.intercept;
  MODEL.featureNames.forEach((name, i) => {
    const scaled = (values[name] - MODEL.mean[i]) / MODEL.scale[i];
    z += MODEL.coef[i] * scaled;
  });
  const pBenign = sigmoid(z);
  const pMalignant = 1 - pBenign;
  return { pMalignant, pBenign };
}

function riskBand(pMalignant) {
  if (pMalignant < 0.2) return { label: "Low", color: "#3F6357" };
  if (pMalignant < 0.55) return { label: "Elevated", color: "#B07A2E" };
  return { label: "High", color: "#A8412F" };
}

const INK = "#16232B";
const INK_SOFT = "#4B5A61";
const BG = "#F2F1EC";
const PANEL = "#FFFFFF";
const LINE = "#DDDAD1";
const ACCENT = "#3F6357";

export default function TriageAssistant() {
  const [values, setValues] = useState(() => {
    const v = {};
    MODEL.featureNames.forEach((name, i) => (v[name] = MODEL.mean[i]));
    return v;
  });
  const [explanation, setExplanation] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const requestId = useRef(0);

  const { pMalignant, pBenign } = useMemo(() => predict(values), [values]);
  const band = riskBand(pMalignant);

  function handleSlider(key, val) {
    setValues((prev) => ({ ...prev, [key]: Number(val) }));
    setExplanation("");
    setError("");
  }

  async function getExplanation() {
    const myId = ++requestId.current;
    setLoading(true);
    setError("");
    setExplanation("");
    try {
      const summary = MODEL.labels
        .map((f) => `${f.label}: ${values[f.key].toFixed(3)}${f.unit}`)
        .join(", ");
      const prompt = `You are assisting inside an educational, non-diagnostic demo app that a product manager built to show ML + LLM skills. A logistic regression model trained on the Wisconsin Breast Cancer Diagnostic dataset just scored a set of hypothetical tumor measurements.

Measurements: ${summary}
Model output: ${(pMalignant * 100).toFixed(1)}% probability malignant, ${(pBenign * 100).toFixed(1)}% probability benign.

Write a short (3-4 sentence) plain-language explanation of what this output means and which measurements likely pushed the score in that direction, written for someone without a clinical background. End with one sentence reminding the reader this is a portfolio demo on a public research dataset, not a medical device or diagnosis.`;

      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-6",
          max_tokens: 400,
          messages: [{ role: "user", content: prompt }],
        }),
      });
      const data = await response.json();
      if (myId !== requestId.current) return;
      const text = (data.content || [])
        .map((b) => (b.type === "text" ? b.text : ""))
        .join("\n")
        .trim();
      if (!text) throw new Error("empty response");
      setExplanation(text);
    } catch (e) {
      if (myId === requestId.current) {
        setError("Couldn't generate an explanation right now. Try again in a moment.");
      }
    } finally {
      if (myId === requestId.current) setLoading(false);
    }
  }

  return (
    <div style={{ background: BG, minHeight: "100%", color: INK, fontFamily: "'Iowan Old Style', 'Palatino Linotype', Georgia, serif" }} className="w-full p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        <header className="mb-6">
          <p style={{ color: INK_SOFT, fontFamily: "ui-monospace, monospace", letterSpacing: "0.02em" }} className="text-xs mb-2">
            AI/ML PORTFOLIO PROJECT — HEALTH TECH
          </p>
          <h1 className="text-3xl md:text-4xl mb-2" style={{ fontWeight: 600 }}>
            Tumor risk triage assistant
          </h1>
          <p style={{ color: INK_SOFT, fontFamily: "ui-sans-serif, system-ui" }} className="text-sm max-w-xl leading-relaxed">
            A logistic regression model trained on the Wisconsin Breast Cancer Diagnostic dataset,
            paired with an LLM that translates the score into plain language. Adjust the
            measurements to see the model respond in real time.
          </p>
        </header>

        <div style={{ background: "#EFE6DC", borderLeft: `3px solid ${band.color}` }} className="text-sm px-4 py-3 mb-6" >
          <span style={{ fontFamily: "ui-sans-serif, system-ui" }}>
            Educational demo only, built on a public research dataset. Not a medical device and not
            a substitute for clinical judgment.
          </span>
        </div>

        <div className="grid md:grid-cols-5 gap-6">
          {/* Controls */}
          <div style={{ background: PANEL, border: `1px solid ${LINE}` }} className="md:col-span-3 p-5">
            <h2 style={{ fontFamily: "ui-sans-serif, system-ui", color: INK_SOFT }} className="text-xs uppercase tracking-wide mb-4">
              Measurements
            </h2>
            <div className="space-y-5">
              {MODEL.labels.map((f) => {
                const [min, max] = MODEL.ranges[f.key];
                const step = (max - min) / 200;
                return (
                  <div key={f.key}>
                    <div className="flex justify-between items-baseline mb-1" style={{ fontFamily: "ui-sans-serif, system-ui" }}>
                      <label htmlFor={f.key} className="text-sm" style={{ color: INK }}>
                        {f.label}
                      </label>
                      <span style={{ fontFamily: "ui-monospace, monospace", color: INK_SOFT }} className="text-sm">
                        {values[f.key].toFixed(3)}{f.unit}
                      </span>
                    </div>
                    <input
                      id={f.key}
                      type="range"
                      min={min}
                      max={max}
                      step={step}
                      value={values[f.key]}
                      onChange={(e) => handleSlider(f.key, e.target.value)}
                      style={{ accentColor: ACCENT, width: "100%" }}
                    />
                    <p style={{ color: INK_SOFT, fontFamily: "ui-sans-serif, system-ui" }} className="text-xs mt-1">
                      {f.help}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Result */}
          <div className="md:col-span-2 flex flex-col gap-4">
            <div style={{ background: PANEL, border: `1px solid ${LINE}` }} className="p-5">
              <h2 style={{ fontFamily: "ui-sans-serif, system-ui", color: INK_SOFT }} className="text-xs uppercase tracking-wide mb-3">
                Model output
              </h2>
              <div className="flex items-end justify-between mb-2">
                <span style={{ fontFamily: "ui-monospace, monospace", fontSize: "2.25rem", color: band.color, lineHeight: 1 }}>
                  {(pMalignant * 100).toFixed(1)}%
                </span>
                <span style={{ fontFamily: "ui-sans-serif, system-ui", color: band.color }} className="text-sm mb-1">
                  {band.label} likelihood
                </span>
              </div>
              <div style={{ background: LINE, height: 6, borderRadius: 3, overflow: "hidden" }}>
                <div style={{ width: `${pMalignant * 100}%`, background: band.color, height: "100%" }} />
              </div>
              <p style={{ color: INK_SOFT, fontFamily: "ui-sans-serif, system-ui" }} className="text-xs mt-2">
                predicted probability of a malignant classification
              </p>
            </div>

            <div style={{ background: PANEL, border: `1px solid ${LINE}` }} className="p-5 flex-1">
              <h2 style={{ fontFamily: "ui-sans-serif, system-ui", color: INK_SOFT }} className="text-xs uppercase tracking-wide mb-3">
                Plain-language explanation
              </h2>
              <button
                onClick={getExplanation}
                disabled={loading}
                style={{ background: loading ? LINE : ACCENT, color: loading ? INK_SOFT : "#fff", fontFamily: "ui-sans-serif, system-ui" }}
                className="text-sm px-4 py-2 mb-3 w-full transition-colors"
              >
                {loading ? "Thinking\u2026" : "Explain this result"}
              </button>
              {error && <p className="text-xs" style={{ color: "#A8412F", fontFamily: "ui-sans-serif, system-ui" }}>{error}</p>}
              {explanation && (
                <p style={{ fontFamily: "ui-sans-serif, system-ui", color: INK, lineHeight: 1.6 }} className="text-sm whitespace-pre-wrap">
                  {explanation}
                </p>
              )}
              {!explanation && !loading && !error && (
                <p style={{ fontFamily: "ui-sans-serif, system-ui", color: INK_SOFT }} className="text-sm">
                  Adjust the measurements, then click above to have the model's reasoning
                  translated into plain language.
                </p>
              )}
            </div>
          </div>
        </div>

        <footer style={{ borderTop: `1px solid ${LINE}`, fontFamily: "ui-sans-serif, system-ui", color: INK_SOFT }} className="mt-8 pt-4 text-xs flex flex-wrap gap-x-6 gap-y-1">
          <span>Accuracy: {(MODEL.metrics.accuracy * 100).toFixed(1)}%</span>
          <span>Recall (malignant): {(MODEL.metrics.recallMalignant * 100).toFixed(1)}%</span>
          <span>Precision (malignant): {(MODEL.metrics.precisionMalignant * 100).toFixed(1)}%</span>
          <span>Test set: {MODEL.metrics.testSetSize} cases</span>
        </footer>
      </div>
    </div>
  );
}
