"""
Trajectory Viewer — A Flask app to visualize VLM Tool Call agent trajectories.

Usage:
    python trajectory_viewer.py                                    # auto-detect ./trajectories/
    python trajectory_viewer.py /path/to/trajectories/latest       # specific trajectory dir
    python trajectory_viewer.py --port 8888                        # custom port

Open http://localhost:5050 in your browser.
"""

import argparse
import glob
import json
import os
import sys

from flask import Flask, Response, abort, redirect, render_template_string, send_file, url_for

app = Flask(__name__)

# Will be set by CLI
TRAJECTORIES_ROOT = ""

# ─────────────────────────────────────────────────────────────────────
# HTML Template (single-file, self-contained)
# ─────────────────────────────────────────────────────────────────────

INDEX_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trajectory Viewer</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text-muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922; --purple: #bc8cff;
    --user-bg: #1c2333; --assistant-bg: #1a1f2e; --tool-bg: #121a12;
    --code-bg: #0d1117; --radius: 10px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6;
  }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }

  .container { max-width: 1100px; margin: 0 auto; padding: 24px 20px; }
  h1 { font-size: 28px; margin-bottom: 8px; }
  .subtitle { color: var(--text-muted); margin-bottom: 32px; font-size: 14px; }

  .run-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 20px 24px; margin-bottom: 12px;
    display: flex; justify-content: space-between; align-items: center;
    transition: border-color 0.15s;
  }
  .run-card:hover { border-color: var(--accent); }
  .run-card .info h3 { font-size: 16px; margin-bottom: 4px; }
  .run-card .info p { font-size: 13px; color: var(--text-muted); }
  .run-card .meta { text-align: right; font-size: 12px; color: var(--text-muted); }
  .run-card .meta .model { color: var(--purple); font-weight: 600; }
  .run-card .meta .steps { color: var(--green); }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 11px; font-weight: 600; margin-left: 8px;
  }
  .badge-steps { background: rgba(63,185,80,0.15); color: var(--green); }
  .badge-images { background: rgba(188,140,255,0.15); color: var(--purple); }
</style>
</head>
<body>
<div class="container">
  <h1>Trajectory Viewer</h1>
  <p class="subtitle">Select a trajectory run to view the full agent trace.</p>
  {% for run in runs %}
  <a href="/view/{{ run.name }}" style="text-decoration:none; color:inherit;">
    <div class="run-card">
      <div class="info">
        <h3>{{ run.name }}
          <span class="badge badge-steps">{{ run.total_steps }} steps</span>
          {% if run.image_count > 0 %}
          <span class="badge badge-images">{{ run.image_count }} images</span>
          {% endif %}
        </h3>
        <p>{{ run.query[:120] }}{% if run.query|length > 120 %}…{% endif %}</p>
        {% if run.final_answer %}
        <p style="margin-top:4px;color:var(--green);">✓ {{ run.final_answer[:100] }}{% if run.final_answer|length > 100 %}…{% endif %}</p>
        {% endif %}
      </div>
      <div class="meta">
        <div class="model">{{ run.model }}</div>
        <div>{{ run.start_time }}</div>
        <div class="steps">{{ run.total_steps }} steps</div>
      </div>
    </div>
  </a>
  {% endfor %}
  {% if not runs %}
  <p style="color:var(--text-muted);">No trajectories found in <code>{{ root }}</code></p>
  {% endif %}
</div>
</body>
</html>
"""


VIEW_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ meta.get('query','Trajectory')[:60] }} — Trajectory Viewer</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text-muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922; --purple: #bc8cff;
    --orange: #f0883e; --cyan: #39d2c0;
    --code-bg: #0d1117; --radius: 10px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6;
  }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  code, pre {
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    font-size: 13px;
  }

  .container { max-width: 1000px; margin: 0 auto; padding: 24px 20px 80px; }

  /* Header */
  .header { margin-bottom: 28px; }
  .header .back { font-size: 13px; color: var(--text-muted); margin-bottom: 12px; display: block; }
  .header h1 { font-size: 22px; line-height: 1.3; margin-bottom: 6px; }
  .header .final-answer {
    background: rgba(63,185,80,0.1); border: 1px solid rgba(63,185,80,0.3);
    border-radius: var(--radius); padding: 14px 18px; margin-top: 16px;
  }
  .header .final-answer .label { color: var(--green); font-weight: 700; font-size: 13px; margin-bottom: 4px; }
  .header .final-answer .text { font-size: 15px; white-space: pre-wrap; }

  /* Metadata grid */
  .meta-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 10px; margin-bottom: 28px;
  }
  .meta-item {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 14px;
  }
  .meta-item .label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .meta-item .value { font-size: 14px; font-weight: 600; margin-top: 2px; word-break: break-all; }
  .meta-item .value.purple { color: var(--purple); }
  .meta-item .value.green { color: var(--green); }
  .meta-item .value.accent { color: var(--accent); }

  /* Timeline */
  .timeline { position: relative; padding-left: 36px; }
  .timeline::before {
    content: ''; position: absolute; left: 14px; top: 0; bottom: 0;
    width: 2px; background: var(--border);
  }

  .step {
    position: relative; margin-bottom: 16px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); overflow: hidden;
    transition: border-color 0.15s;
  }
  .step:hover { border-color: #484f58; }

  .step::before {
    content: ''; position: absolute; left: -28px; top: 18px;
    width: 12px; height: 12px; border-radius: 50%;
    border: 2px solid var(--border); background: var(--bg); z-index: 1;
  }
  .step.role-user::before { border-color: var(--accent); background: var(--accent); }
  .step.role-assistant::before { border-color: var(--purple); background: var(--purple); }
  .step.role-tool::before { border-color: var(--green); background: var(--green); }

  .step-header {
    padding: 10px 16px; display: flex; justify-content: space-between;
    align-items: center; cursor: pointer; user-select: none;
  }
  .step-header:hover { background: rgba(255,255,255,0.02); }
  .step-header .left { display: flex; align-items: center; gap: 10px; }
  .step-header .step-num {
    font-size: 11px; font-weight: 700; color: var(--text-muted);
    background: rgba(255,255,255,0.06); padding: 2px 7px; border-radius: 6px;
    min-width: 28px; text-align: center;
  }
  .step-header .role-badge {
    font-size: 12px; font-weight: 700; padding: 2px 10px;
    border-radius: 12px; text-transform: uppercase; letter-spacing: 0.5px;
  }
  .role-user .role-badge { background: rgba(88,166,255,0.15); color: var(--accent); }
  .role-assistant .role-badge { background: rgba(188,140,255,0.15); color: var(--purple); }
  .role-tool .role-badge { background: rgba(63,185,80,0.15); color: var(--green); }
  .step-header .time { font-size: 11px; color: var(--text-muted); }
  .step-header .preview {
    font-size: 12px; color: var(--text-muted); margin-left: 12px;
    max-width: 500px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .step-header .toggle { font-size: 16px; color: var(--text-muted); transition: transform 0.2s; }
  .step.open .step-header .toggle { transform: rotate(90deg); }

  .step-body { padding: 0 16px 16px; display: none; }
  .step.open .step-body { display: block; }

  /* Content blocks */
  .content-text {
    white-space: pre-wrap; word-break: break-word; font-size: 14px;
    padding: 12px 0; border-bottom: 1px solid var(--border); line-height: 1.7;
  }
  .content-text:last-child { border-bottom: none; }

  .tool-call-block {
    background: rgba(188,140,255,0.06); border: 1px solid rgba(188,140,255,0.2);
    border-radius: 8px; padding: 12px 14px; margin: 10px 0;
  }
  .tool-call-block .fn-name { color: var(--purple); font-weight: 700; font-size: 13px; }
  .tool-call-block .fn-id { color: var(--text-muted); font-size: 11px; margin-left: 8px; }

  .code-block {
    background: var(--code-bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px; margin: 10px 0;
    overflow-x: auto; position: relative;
  }
  .code-block pre { white-space: pre; color: var(--text); }
  .code-block .code-label {
    position: absolute; top: 6px; right: 10px;
    font-size: 10px; color: var(--text-muted); text-transform: uppercase;
    background: rgba(255,255,255,0.06); padding: 1px 6px; border-radius: 4px;
  }

  /* Python syntax highlighting (basic) */
  .kw { color: #ff7b72; }
  .str { color: #a5d6ff; }
  .num { color: #79c0ff; }
  .comment { color: #8b949e; font-style: italic; }
  .fn { color: #d2a8ff; }
  .builtin { color: #ffa657; }

  .image-gallery {
    display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0;
  }
  .image-gallery img {
    max-width: 100%; max-height: 400px; border-radius: 8px;
    border: 1px solid var(--border); cursor: pointer;
    transition: transform 0.15s;
  }
  .image-gallery img:hover { transform: scale(1.02); }
  .image-gallery .single img { max-width: 100%; }

  /* Lightbox */
  .lightbox {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.9);
    z-index: 999; justify-content: center; align-items: center; cursor: zoom-out;
  }
  .lightbox.active { display: flex; }
  .lightbox img { max-width: 95vw; max-height: 95vh; border-radius: 8px; }

  /* Section label */
  .section-label {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;
    color: var(--text-muted); margin: 10px 0 4px; font-weight: 600;
  }

  .error-text { color: var(--red); }
</style>
</head>
<body>
<div class="container">
  <!-- Header -->
  <div class="header">
    <a href="/" class="back">← All Trajectories</a>
    <h1>{{ meta.get('query', 'Trajectory') }}</h1>
    {% if final_answer %}
    <div class="final-answer">
      <div class="label">✓ Final Answer</div>
      <div class="text">{{ final_answer }}</div>
    </div>
    {% endif %}
  </div>

  <!-- Metadata -->
  <div class="meta-grid">
    <div class="meta-item">
      <div class="label">Model</div>
      <div class="value purple">{{ meta.get('model', '—') }}</div>
    </div>
    <div class="meta-item">
      <div class="label">Total Steps</div>
      <div class="value green">{{ meta.get('total_steps', steps|length) }}</div>
    </div>
    <div class="meta-item">
      <div class="label">Start Time</div>
      <div class="value">{{ meta.get('start_time', '—') }}</div>
    </div>
    <div class="meta-item">
      <div class="label">End Time</div>
      <div class="value">{{ meta.get('end_time', '—') }}</div>
    </div>
    <div class="meta-item">
      <div class="label">Max Iterations</div>
      <div class="value accent">{{ meta.get('max_iterations', '—') }}</div>
    </div>
    <div class="meta-item">
      <div class="label">Images</div>
      <div class="value">{{ image_count }}</div>
    </div>
  </div>

  <!-- Timeline -->
  <div class="timeline">
    {% for step in steps %}
    <div class="step role-{{ step.role }} {% if loop.first or loop.last %}open{% endif %}" id="step-{{ step.step }}">
      <div class="step-header" onclick="toggleStep(this)">
        <div class="left">
          <span class="step-num">#{{ step.step }}</span>
          <span class="role-badge">{{ step.role }}</span>
          {% if step.tool_calls %}
            {% for tc in step.tool_calls %}
              <span style="font-size:12px;color:var(--yellow);">→ {{ tc.name or tc.get('function',{}).get('name','') }}</span>
            {% endfor %}
          {% endif %}
          {% if step.code %}
            <span style="font-size:12px;color:var(--cyan);">⌨ code</span>
          {% endif %}
          {% if step.images %}
            <span style="font-size:12px;color:var(--purple);">🖼 {{ step.images|length }}</span>
          {% endif %}
          <span class="preview">{{ (step.content_text or '')[:80] }}</span>
        </div>
        <div style="display:flex;align-items:center;gap:12px;">
          <span class="time">{{ step.timestamp.split('T')[1] if step.timestamp and 'T' in step.timestamp else '' }}</span>
          <span class="toggle">▸</span>
        </div>
      </div>
      <div class="step-body">

        {# ─── User content ─── #}
        {% if step.role == 'user' and step.content_text %}
          <div class="section-label">Query</div>
          <div class="content-text">{{ step.content_text }}</div>
        {% endif %}

        {# ─── Images ─── #}
        {% if step.images %}
          <div class="section-label">Images</div>
          <div class="image-gallery {% if step.images|length == 1 %}single{% endif %}">
            {% for img in step.images %}
              <img src="/image/{{ run_name }}/{{ img }}" alt="{{ img }}" onclick="openLightbox(this.src)" loading="lazy">
            {% endfor %}
          </div>
        {% endif %}

        {# ─── Assistant text ─── #}
        {% if step.role == 'assistant' and step.content_text %}
          <div class="section-label">Response</div>
          <div class="content-text">{{ step.content_text }}</div>
        {% endif %}

        {# ─── Tool calls ─── #}
        {% if step.tool_calls %}
          {% for tc in step.tool_calls %}
          <div class="tool-call-block">
            <div>
              <span class="fn-name">{{ tc.name or tc.get('function',{}).get('name','') }}()</span>
              <span class="fn-id">{{ tc.id or '' }}</span>
            </div>
            {% if tc.arguments %}
              {% set args_str = tc.arguments if tc.arguments is string else tc.arguments|tojson %}
              {% set parsed = parse_args(args_str) %}
              {% if parsed.get('code') %}
                <div class="code-block" style="margin-top:8px;">
                  <span class="code-label">Python</span>
                  <pre>{{ parsed['code'] }}</pre>
                </div>
              {% elif parsed.get('answer') %}
                <div class="content-text" style="color:var(--green);">{{ parsed['answer'] }}</div>
              {% else %}
                <div class="code-block" style="margin-top:8px;">
                  <span class="code-label">Arguments</span>
                  <pre>{{ args_str }}</pre>
                </div>
              {% endif %}
            {% endif %}
          </div>
          {% endfor %}
        {% endif %}

        {# ─── Tool code (from trajectory.json) ─── #}
        {% if step.code %}
          <div class="section-label">Code Executed</div>
          <div class="code-block">
            <span class="code-label">Python</span>
            <pre>{{ step.code }}</pre>
          </div>
        {% endif %}

        {# ─── Tool output ─── #}
        {% if step.role == 'tool' and step.content_text %}
          <div class="section-label">Output</div>
          <div class="content-text {% if '[ERROR]' in (step.content_text or '') or '[Execution Error]' in (step.content_text or '') %}error-text{% endif %}">{{ step.content_text }}</div>
        {% endif %}

      </div>
    </div>
    {% endfor %}
  </div>
</div>

<!-- Lightbox -->
<div class="lightbox" id="lightbox" onclick="closeLightbox()">
  <img id="lightbox-img" src="" alt="zoom">
</div>

<script>
function toggleStep(header) {
  header.parentElement.classList.toggle('open');
}
function openLightbox(src) {
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox').classList.add('active');
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('active');
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeLightbox();
});
// Expand all / Collapse all with keyboard
document.addEventListener('keydown', e => {
  if (e.key === 'e' && !e.ctrlKey && !e.metaKey) {
    document.querySelectorAll('.step').forEach(s => s.classList.add('open'));
  }
  if (e.key === 'c' && !e.ctrlKey && !e.metaKey) {
    document.querySelectorAll('.step').forEach(s => s.classList.remove('open'));
  }
});
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────

def find_trajectory_dirs(root: str):
    """Find all directories containing a trajectory.json file."""
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "trajectory.json" in filenames:
            results.append(dirpath)
    results.sort(key=lambda d: os.path.getmtime(os.path.join(d, "trajectory.json")), reverse=True)
    return results


def load_trajectory(traj_dir: str):
    """Load trajectory.json from a directory."""
    path = os.path.join(traj_dir, "trajectory.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_images(traj_dir: str) -> int:
    """Count images in the trajectory images dir."""
    img_dir = os.path.join(traj_dir, "images")
    if not os.path.isdir(img_dir):
        return 0
    return len([f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))])


def parse_tool_args(args_str: str) -> dict:
    """Parse a JSON arguments string, returning a dict."""
    try:
        return json.loads(args_str)
    except (json.JSONDecodeError, TypeError):
        return {}


# ─────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    traj_dirs = find_trajectory_dirs(TRAJECTORIES_ROOT)
    runs = []
    for d in traj_dirs:
        try:
            data = load_trajectory(d)
            meta = data.get("metadata", {})
            rel_name = os.path.relpath(d, TRAJECTORIES_ROOT)
            runs.append({
                "name": rel_name,
                "query": meta.get("query", "(no query)"),
                "model": meta.get("model", "—"),
                "start_time": meta.get("start_time", "—"),
                "total_steps": meta.get("total_steps", len(data.get("steps", []))),
                "final_answer": data.get("final_answer"),
                "image_count": count_images(d),
            })
        except Exception:
            continue
    return render_template_string(INDEX_TEMPLATE, runs=runs, root=TRAJECTORIES_ROOT)


@app.route("/view/<path:run_name>")
def view_trajectory(run_name: str):
    traj_dir = os.path.join(TRAJECTORIES_ROOT, run_name)
    traj_path = os.path.join(traj_dir, "trajectory.json")
    if not os.path.isfile(traj_path):
        abort(404, f"Trajectory not found: {run_name}")

    data = load_trajectory(traj_dir)
    meta = data.get("metadata", {})
    steps = data.get("steps", [])
    final_answer = data.get("final_answer")
    img_count = count_images(traj_dir)

    return render_template_string(
        VIEW_TEMPLATE,
        meta=meta,
        steps=steps,
        final_answer=final_answer,
        run_name=run_name,
        image_count=img_count,
        parse_args=parse_tool_args,
    )


@app.route("/image/<path:img_path>")
def serve_image(img_path: str):
    """Serve images from trajectory directories."""
    full_path = os.path.join(TRAJECTORIES_ROOT, img_path)
    full_path = os.path.realpath(full_path)
    # Security: ensure the path is under TRAJECTORIES_ROOT
    if not full_path.startswith(os.path.realpath(TRAJECTORIES_ROOT)):
        abort(403)
    if not os.path.isfile(full_path):
        abort(404)
    return send_file(full_path)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    global TRAJECTORIES_ROOT

    parser = argparse.ArgumentParser(description="Trajectory Viewer — visualize VLM agent traces")
    parser.add_argument(
        "trajectory_dir",
        nargs="?",
        default=None,
        help="Path to a trajectories root dir (default: ./trajectories/)",
    )
    parser.add_argument("--port", "-p", type=int, default=5050, help="Port (default: 5050)")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    if args.trajectory_dir:
        candidate = os.path.abspath(args.trajectory_dir)
    else:
        candidate = os.path.abspath("trajectories")

    # If user pointed directly to a trajectory dir (contains trajectory.json),
    # use its parent as root
    if os.path.isfile(os.path.join(candidate, "trajectory.json")):
        TRAJECTORIES_ROOT = os.path.dirname(candidate)
    else:
        TRAJECTORIES_ROOT = candidate

    if not os.path.isdir(TRAJECTORIES_ROOT):
        print(f"Error: Directory not found: {TRAJECTORIES_ROOT}", file=sys.stderr)
        sys.exit(1)

    n = len(find_trajectory_dirs(TRAJECTORIES_ROOT))
    print(f"\n  Trajectory Viewer")
    print(f"  Root: {TRAJECTORIES_ROOT}")
    print(f"  Found: {n} trajectory run(s)")
    print(f"  Open: http://localhost:{args.port}\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
