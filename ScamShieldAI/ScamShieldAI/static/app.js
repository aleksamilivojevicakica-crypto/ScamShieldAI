const tabs = document.querySelectorAll(".tabs button");
for (const tab of tabs) {
  tab.addEventListener("click", () => {
    for (const item of tabs) {
      const isSelected = item === tab;
      document.getElementById(item.dataset.target).hidden = !isSelected;
      if (isSelected) {
        item.setAttribute("aria-current", "page");
      } else {
        item.removeAttribute("aria-current");
      }
    }
  });
}

const form = document.getElementById("analyze-form");
const message = document.getElementById("message");
const length = document.getElementById("length");
const button = document.getElementById("analyze-button");
const result = document.getElementById("result");
const error = document.getElementById("error");
const placeholder = document.getElementById("placeholder");

function refreshInput() {
  length.textContent = message.value.length;
  result.hidden = true;
  error.hidden = true;
  placeholder.hidden = false;
}

message.addEventListener("input", refreshInput);
for (const example of form.querySelectorAll(".example")) {
  example.addEventListener("click", () => {
    message.value = example.dataset.example;
    refreshInput();
    message.focus();
  });
}

document.getElementById("defense-example").addEventListener("click", event => {
  document.querySelector('[data-target="analyzer"]').click();
  message.value = event.currentTarget.dataset.example;
  refreshInput();
  message.focus();
});

const percent = new Intl.NumberFormat("en", {
  style: "percent",
  maximumFractionDigits: 1,
});

function display(data) {
  const verdict = document.getElementById("verdict");
  verdict.classList.toggle("flagged", data.flagged);
  document.getElementById("verdict-text").textContent = data.phishing?.flagged ? "Potential phishing detected" : (data.ml_flagged ? "Flagged as possible spam" : "Below the spam threshold");
  document.getElementById("score").textContent = percent.format(data.score);
  document.getElementById("threshold").textContent = `Model spam score · Alert threshold ${percent.format(data.threshold)}`;
  document.getElementById("guidance").textContent = data.flagged
    ? "If the message asks for money or a code, verify the request through an official channel before acting."
    : "The sender could still be impersonating someone. Check unexpected requests for money or codes through an official channel.";

  const phishing = document.getElementById("phishing-signals");
  phishing.hidden = !data.phishing?.flagged;
  if (!phishing.hidden) {
    document.getElementById("phishing-level").textContent = `${data.phishing.level} rule-based risk · separate from the ML spam score`;
    const signalList = document.getElementById("phishing-list");
    signalList.replaceChildren();
    for (const signal of data.phishing.signals) {
      const item = document.createElement("li");
      item.textContent = signal;
      signalList.append(item);
    }
  }

  const restoration = document.getElementById("restoration");
  restoration.hidden = data.restored_text === undefined;
  if (!restoration.hidden) {
    const before = percent.format(data.original_score);
    const after = percent.format(data.score);
    document.getElementById("restoration-score").textContent = `Spam score before: ${before} · After: ${after}`;
    document.getElementById("restoration-text").textContent = data.restored_text;
  }

  const list = document.getElementById("explanation");
  list.replaceChildren();
  for (const reason of data.reasons) {
    const item = document.createElement("li");
    const feature = document.createElement("strong");
    feature.textContent = reason.term;
    const detail = document.createElement("span");
    detail.textContent = `${reason.kind} · ${reason.impact > 0 ? "+" : ""}${reason.impact.toFixed(3)}`;
    detail.classList.toggle("positive", reason.impact > 0);
    item.append(feature, detail);
    list.append(item);
  }
  if (!data.reasons.length) {
    const empty = document.createElement("li");
    empty.textContent = "The model found no recognized features in this text.";
    list.append(empty);
  }
  placeholder.hidden = true;
  result.hidden = false;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const submitted = message.value;
  refreshInput();
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  try {
    const response = await fetch("/analyze", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message: submitted}),
    });
    const data = await response.json();
    if (message.value !== submitted) return;
    if (!response.ok) throw new Error(data.error || "Analysis failed.");
    display(data);
  } catch (problem) {
    if (message.value === submitted) {
      error.textContent = problem.message || "Analysis failed. Try again.";
      error.hidden = false;
    }
  } finally {
    button.disabled = false;
    button.removeAttribute("aria-busy");
  }
});

