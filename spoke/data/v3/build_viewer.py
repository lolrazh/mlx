#!/usr/bin/env python3
"""Build an HTML viewer for the v3 dataset from source JSON files."""

import json
from pathlib import Path

V3 = Path(__file__).parent
SOURCE = V3 / "source"

# Category display order and colors (matching viewer.html conventions)
CATEGORIES = [
    ("spell-replace",    "spell"),
    ("self-correction",  "self-correction"),
    ("quote-unquote",    "quote"),
    ("quote-endquote",   "quote"),
    ("at-symbol",        "at-symbol"),
    ("caps",             "caps"),
    ("emphasis",         "emphasis"),
    ("emoji",            "emoji"),
    ("camelcase",        "code-aware"),
]

# How many examples were in v3 BEFORE the patch (for highlighting new ones)
PRE_PATCH_COUNTS = {
    "self-correction": 95,
    "emoji": 48,
    "quote-unquote": 50,
    "camelcase": 43,
    # All others unchanged
}


def load_sources():
    """Load all source JSON files, return list of (category, examples)."""
    result = []
    for cat, _ in CATEGORIES:
        path = SOURCE / f"{cat}.json"
        if path.exists():
            with open(path) as f:
                examples = json.load(f)
            result.append((cat, examples))
    return result


def build_js_data(sources):
    """Build the JS data object from sources."""
    lines = []
    global_id = 0
    for cat, examples in sources:
        pre_count = PRE_PATCH_COUNTS.get(cat, len(examples))
        for i, ex in enumerate(examples):
            global_id += 1
            is_new = i >= pre_count
            inp = json.dumps(ex["input"])
            ideal = json.dumps(ex["ideal"])
            lines.append(
                f'  {{id:{global_id}, cat:"{cat}", input:{inp}, ideal:{ideal}, isNew:{str(is_new).lower()}}}'
            )
    return "[\n" + ",\n".join(lines) + "\n]"


def build_html(js_data, sources):
    total = sum(len(exs) for _, exs in sources)
    new_count = sum(
        max(0, len(exs) - PRE_PATCH_COUNTS.get(cat, len(exs)))
        for cat, exs in sources
    )
    cat_counts = {cat: len(exs) for cat, exs in sources}

    # Build category color CSS classes
    cat_css_map = {}
    for cat, color_class in CATEGORIES:
        cat_css_map[cat] = color_class

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Spoke — v3 Dataset Viewer</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --yellow: #d29922; --red: #f85149; --purple: #bc8cff;
    --orange: #d18616; --pink: #f778ba; --cyan: #39d2c0; --blue: #58a6ff;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; }}

  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}

  header {{ margin-bottom: 24px; }}
  header h1 {{ font-size: 24px; font-weight: 600; margin-bottom: 4px; }}
  header p {{ color: var(--muted); font-size: 14px; }}

  .stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; padding: 12px 16px; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; }}
  .stat {{ font-size: 13px; color: var(--muted); }}
  .stat strong {{ color: var(--text); font-variant-numeric: tabular-nums; }}
  .stat .new {{ color: var(--green); }}

  .filters {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; align-items: center; }}
  .filters label {{ font-size: 13px; color: var(--muted); margin-right: 4px; }}
  .filter-btn {{
    padding: 4px 10px; font-size: 12px; border-radius: 12px;
    border: 1px solid var(--border); background: var(--surface); color: var(--muted);
    cursor: pointer; transition: all 0.15s;
  }}
  .filter-btn:hover {{ border-color: var(--accent); color: var(--text); }}
  .filter-btn.active {{ background: var(--accent); color: #000; border-color: var(--accent); font-weight: 600; }}
  .filter-divider {{ width: 1px; height: 20px; background: var(--border); margin: 0 8px; }}

  .toggle-btn {{
    padding: 4px 10px; font-size: 12px; border-radius: 12px;
    border: 1px solid var(--border); background: var(--surface); color: var(--muted);
    cursor: pointer; transition: all 0.15s;
  }}
  .toggle-btn:hover {{ border-color: var(--green); color: var(--text); }}
  .toggle-btn.active {{ background: #1a3a2a; color: var(--green); border-color: var(--green); font-weight: 600; }}

  .search-bar {{ display: flex; gap: 12px; margin-bottom: 20px; }}
  .search-bar input {{
    flex: 1; padding: 8px 12px; font-size: 14px; background: var(--surface);
    border: 1px solid var(--border); border-radius: 6px; color: var(--text); outline: none;
  }}
  .search-bar input:focus {{ border-color: var(--accent); }}
  .search-bar input::placeholder {{ color: var(--muted); }}

  .example {{
    display: grid; grid-template-columns: 40px 1fr 1fr 140px;
    gap: 0; border: 1px solid var(--border); border-radius: 8px;
    margin-bottom: 8px; background: var(--surface); overflow: hidden;
    transition: border-color 0.15s;
  }}
  .example:hover {{ border-color: #484f58; }}
  .example.highlight {{ border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }}
  .example.is-new {{ border-left: 3px solid var(--green); }}

  .ex-id {{
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 600; color: var(--muted);
    background: rgba(255,255,255,0.02); border-right: 1px solid var(--border);
    padding: 12px 0;
  }}
  .ex-input, .ex-ideal {{ padding: 12px 16px; font-size: 13px; line-height: 1.6; }}
  .ex-input {{ border-right: 1px solid var(--border); }}
  .ex-ideal {{ color: var(--green); }}
  .ex-meta {{
    display: flex; flex-direction: column; gap: 6px; padding: 10px 12px;
    border-left: 1px solid var(--border); align-items: flex-start;
  }}

  .tag {{
    display: inline-block; padding: 1px 7px; font-size: 10px; font-weight: 600;
    border-radius: 8px; white-space: nowrap; letter-spacing: 0.02em;
  }}
  .tag-spell {{ background: #1a2d4a; color: var(--blue); }}
  .tag-self-correction {{ background: #3a2f1a; color: var(--orange); }}
  .tag-quote {{ background: #2d1a3a; color: var(--purple); }}
  .tag-at-symbol {{ background: #1a3a2a; color: var(--green); }}
  .tag-caps {{ background: #3a351a; color: var(--yellow); }}
  .tag-emphasis {{ background: #3a1a2a; color: var(--pink); }}
  .tag-emoji {{ background: #3a351a; color: var(--yellow); }}
  .tag-code-aware {{ background: #1a2a3a; color: #79c0ff; }}

  .new-badge {{
    display: inline-block; padding: 1px 7px; font-size: 10px; font-weight: 700;
    border-radius: 8px; text-transform: uppercase; letter-spacing: 0.05em;
    background: #1a3a2a; color: var(--green);
  }}

  .col-headers {{
    display: grid; grid-template-columns: 40px 1fr 1fr 140px;
    gap: 0; padding: 8px 0; font-size: 11px; font-weight: 600;
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border); margin-bottom: 8px;
  }}
  .col-headers > div {{ padding: 0 16px; }}
  .col-headers > div:first-child {{ padding: 0; text-align: center; }}

  .count-badge {{ font-size: 11px; color: var(--muted); margin-left: 3px; }}

  @media (max-width: 900px) {{
    .example {{ grid-template-columns: 30px 1fr; }}
    .ex-ideal, .ex-meta {{ grid-column: 2; }}
    .ex-ideal {{ border-top: 1px solid var(--border); border-right: none; }}
    .ex-meta {{ flex-direction: row; flex-wrap: wrap; border-left: none; border-top: 1px solid var(--border); }}
  }}

  #result-count {{ font-size: 12px; color: var(--muted); margin-left: auto; }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Spoke — v3 Dataset (Patched)</h1>
    <p>{total} source examples across 9 trigger-matched categories. <span style="color:var(--green)">{new_count} new</span> from T11 failure patch.</p>
  </header>
  <div class="stats" id="stats"></div>
  <div class="filters" id="filters"></div>
  <div class="search-bar">
    <input type="text" id="search" placeholder="Search inputs, ideals, or type #id (e.g. #27)..." />
  </div>
  <div class="col-headers">
    <div>#</div><div>Input (raw transcript)</div><div>Ideal (cleaned output)</div><div>Tags</div>
  </div>
  <div id="examples"></div>
</div>

<script>
const DATA = {js_data};

const CAT_COLORS = {{
  "spell-replace": "spell", "self-correction": "self-correction", "quote-unquote": "quote",
  "quote-endquote": "quote", "at-symbol": "at-symbol", "caps": "caps",
  "emphasis": "emphasis", "emoji": "emoji", "camelcase": "code-aware",
}};

const CAT_COUNTS = {json.dumps(cat_counts)};
const ALL_CATS = Object.keys(CAT_COLORS);

let activeCats = new Set();
let showNewOnly = false;

function escHtml(s) {{
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}}

function renderStats() {{
  const el = document.getElementById("stats");
  const total = DATA.length;
  const newCount = DATA.filter(d => d.isNew).length;
  el.innerHTML = `<span class="stat"><strong>${{total}}</strong> total</span><span class="stat"><strong class="new">${{newCount}}</strong> new (patch)</span><span id="result-count"></span>`;
}}

function renderFilters() {{
  const el = document.getElementById("filters");
  let html = '<label>Category:</label>';
  ALL_CATS.forEach(cat => {{
    const count = CAT_COUNTS[cat] || 0;
    if (!count) return;
    html += `<button class="filter-btn" data-cat="${{cat}}" onclick="toggleCat('${{cat}}')">${{cat}}<span class="count-badge">${{count}}</span></button>`;
  }});
  html += '<div class="filter-divider"></div>';
  html += `<button class="toggle-btn" id="new-toggle" onclick="toggleNew()">new only</button>`;
  el.innerHTML = html;
  updateFilterUI();
}}

function toggleCat(cat) {{
  if (activeCats.has(cat)) activeCats.delete(cat); else activeCats.add(cat);
  updateFilterUI(); render();
}}

function toggleNew() {{
  showNewOnly = !showNewOnly;
  updateFilterUI(); render();
}}

function updateFilterUI() {{
  document.querySelectorAll(".filter-btn").forEach(b => b.classList.toggle("active", activeCats.has(b.dataset.cat)));
  const nb = document.getElementById("new-toggle");
  if (nb) nb.classList.toggle("active", showNewOnly);
}}

function render() {{
  const q = document.getElementById("search").value.trim().toLowerCase();
  const el = document.getElementById("examples");

  const filtered = DATA.filter(d => {{
    if (activeCats.size > 0 && !activeCats.has(d.cat)) return false;
    if (showNewOnly && !d.isNew) return false;
    if (q) {{
      if (q.startsWith("#")) {{
        const num = parseInt(q.slice(1));
        if (!isNaN(num) && d.id !== num) return false;
      }} else {{
        const haystack = (d.input + " " + d.ideal + " " + d.cat).toLowerCase();
        if (!haystack.includes(q)) return false;
      }}
    }}
    return true;
  }});

  const rc = document.getElementById("result-count");
  if (rc) rc.textContent = filtered.length < DATA.length ? `${{filtered.length}} shown` : "";

  el.innerHTML = filtered.map(d => {{
    const hl = q && q.startsWith("#") && d.id === parseInt(q.slice(1)) ? " highlight" : "";
    const newClass = d.isNew ? " is-new" : "";
    const tag = `<span class="tag tag-${{CAT_COLORS[d.cat]}}">${{d.cat}}</span>`;
    const newBadge = d.isNew ? '<span class="new-badge">new</span>' : '';
    return `<div class="example${{hl}}${{newClass}}">
      <div class="ex-id">${{d.id}}</div>
      <div class="ex-input">${{escHtml(d.input)}}</div>
      <div class="ex-ideal">${{escHtml(d.ideal)}}</div>
      <div class="ex-meta">${{tag}}${{newBadge}}</div>
    </div>`;
  }}).join("");
}}

document.getElementById("search").addEventListener("input", render);
renderStats(); renderFilters(); render();
</script>
</body>
</html>'''


def main():
    sources = load_sources()
    js_data = build_js_data(sources)
    html = build_html(js_data, sources)
    out = V3 / "viewer.html"
    out.write_text(html)
    total = sum(len(exs) for _, exs in sources)
    new_count = sum(
        max(0, len(exs) - PRE_PATCH_COUNTS.get(cat, len(exs)))
        for cat, exs in sources
    )
    print(f"Wrote {out} ({total} examples, {new_count} new)")


if __name__ == "__main__":
    main()
