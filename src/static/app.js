const promptEl = document.getElementById("prompt");
const generateBtn = document.getElementById("generate");
const statusEl = document.getElementById("status");
const previewEl = document.getElementById("preview");
const sourceEl = document.getElementById("source");

async function generateHtml() {
  const prompt = promptEl.value.trim();
  if (!prompt) {
    statusEl.textContent = "Enter a prompt first.";
    return;
  }

  generateBtn.disabled = true;
  statusEl.textContent = "Generating…";
  sourceEl.textContent = "";
  previewEl.srcdoc = "";

  try {
    // Response is raw text/html, not JSON.
    const response = await fetch("/generate-html", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    const body = await response.text();
    if (!response.ok) {
      throw new Error(body || `HTTP ${response.status}`);
    }
    sourceEl.textContent = body;
    // iframe is sandboxed in index.html — preview only, no script execution.
    previewEl.srcdoc = body;
    statusEl.textContent = `OK · ${response.headers.get("content-type") || "text/html"}`;
  } catch (error) {
    statusEl.textContent = "Generation failed.";
    sourceEl.textContent = String(error.message || error);
  } finally {
    generateBtn.disabled = false;
  }
}

generateBtn.addEventListener("click", generateHtml);
