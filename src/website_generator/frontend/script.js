const form = document.querySelector('#generator');
const button = document.querySelector('#generate');
const status = document.querySelector('#status');
const source = document.querySelector('#source');
const preview = document.querySelector('#preview');

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  button.disabled = true;
  form.setAttribute('aria-busy', 'true');
  button.textContent = 'Generating…';
  status.className = '';
  status.textContent = 'Generating your webpage locally…';
  source.textContent = 'Waiting for generation…';
  preview.srcdoc = '';
  try {
    const response = await fetch('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: document.querySelector('#prompt').value }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Please enter a prompt of 1–4000 characters.');
    source.textContent = data.html;
    preview.srcdoc = `<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"><style>body{font:16px/1.7 system-ui,sans-serif;color:#22332d;padding:24px;overflow-wrap:anywhere}h1,h2,h3{line-height:1.2}section,article{margin:20px 0}a{color:#286a50}</style></head><body>${data.html}</body></html>`;
    status.textContent = 'Generated successfully. Preview and source are ready.';
  } catch (error) {
    status.className = 'error';
    status.textContent = error.message || 'Generation failed. Please try again.';
    source.textContent = 'No HTML returned.';
  } finally {
    button.disabled = false;
    button.textContent = 'Generate webpage →';
    form.removeAttribute('aria-busy');
  }
});
