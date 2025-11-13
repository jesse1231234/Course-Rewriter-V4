# streamlit_app.py
# Canvas Course-wide HTML Rewrite (Test Instance)
# - Admin-only Streamlit app for dry-run + bulk apply on Pages, Assignment descriptions, and Discussion topic descriptions
# - Canvas API (test instance), OpenAI Responses API (auto-pick newest model + retries)
# - Iframe freeze/restore (no new iframes), NO sanitizer (write model HTML as-is)
# - Visual PREVIEW (Original vs Proposed) using a built-in DesignPLUS-like "Shim" (CSS+JS)
# - Optional Upload CSS/JS for near-Canvas fidelity in preview (Layer 2)
# - ETA during dry-run
# - Approve All / Unapprove All bulk toggles
# - Precision Directives (micro-language) + few-shot examples for higher precision
# - Large-course resilience: indexing, batching, time budget, resume

import os
import re
import json
import base64
import time
import random
import hashlib
import urllib.parse
from typing import Optional, Tuple, List

import streamlit as st
from canvasapi import Canvas
from openai import OpenAI
from bs4 import BeautifulSoup, NavigableString, Tag
from diff_match_patch import diff_match_patch

# ---------------------- Config & Clients ----------------------

SECRETS = st.secrets if hasattr(st, "secrets") else os.environ

CANVAS_BASE_URL = SECRETS["CANVAS_BASE_URL"]
CANVAS_ACCOUNT_ID = int(SECRETS["CANVAS_ACCOUNT_ID"])
CANVAS_ADMIN_TOKEN = SECRETS["CANVAS_ADMIN_TOKEN"]
OPENAI_API_KEY = SECRETS["OPENAI_API_KEY"]

# Defaults if auto-pick fails or listing isn't permitted
DEFAULT_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-5")    # falls back to 4.1 if unavailable
DEFAULT_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

st.set_page_config(page_title="Canvas Course-wide HTML Rewrite (Test)", layout="wide")

canvas = Canvas(CANVAS_BASE_URL, CANVAS_ADMIN_TOKEN)
account = canvas.get_account(CANVAS_ACCOUNT_ID)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------- Small utils ----------------------

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

# ---------------------- Canvas helpers ----------------------

def find_course_by_code(account, course_code: str) -> List:
    """Search by course_code; prefer exact code match if available."""
    courses = account.get_courses(search_term=course_code)
    matches = [c for c in courses if getattr(c, "course_code", "") == course_code]
    return matches if matches else list(courses)

def list_supported_items(
    course,
    include_pages: bool = True,
    include_assignments: bool = True,
    include_discussions: bool = False
):
    """Enumerate Module Items with optional type filtering (Pages, Assignments, Discussions)."""
    wanted = set()
    if include_pages:
        wanted.add("Page")
    if include_assignments:
        wanted.add("Assignment")
    if include_discussions:
        wanted.update({"Discussion", "DiscussionTopic"})  # tolerate naming variants

    supported = []
    for module in course.get_modules(include_items=True):
        items = list(module.get_module_items()) if not getattr(module, "items", None) else module.items
        for it in items:
            if it.type in wanted:
                supported.append((module, it))
    return supported

def _normalized_item_kind(item_type: str) -> str:
    return "Discussion" if item_type in {"Discussion", "DiscussionTopic"} else item_type


def _item_cache_candidates(item) -> Tuple[str, List[str]]:
    kind = _normalized_item_kind(item.type)
    candidates: List[str] = []

    content_id = getattr(item, "content_id", None)
    if content_id:
        candidates.append(f"{kind}:{content_id}")

    page_id = getattr(item, "page_id", None)
    if page_id:
        candidates.append(f"{kind}:{page_id}")

    page_url = getattr(item, "page_url", None)
    if page_url:
        candidates.append(f"{kind}@url:{page_url}")

    module_item_id = getattr(item, "id", None)
    if module_item_id:
        candidates.append(f"ModuleItem:{module_item_id}")

    deduped = []
    seen = set()
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            deduped.append(cand)
    return kind, deduped


def _get_item_cache_structures():
    cache = st.session_state.setdefault("item_cache", {})
    aliases = st.session_state.setdefault("item_cache_aliases", {})
    return cache, aliases


def _write_item_cache(meta: dict, candidates: List[str]):
    cache, aliases = _get_item_cache_structures()
    cache_key = f"{meta['kind']}:{meta['id']}"
    cache[cache_key] = dict(meta)
    aliases[cache_key] = cache_key
    for cand in candidates:
        aliases[cand] = cache_key
    return cache_key


def _cache_id_from_key(cache_key: str) -> str:
    return cache_key.split(":", 1)[1] if ":" in cache_key else cache_key


def _refresh_item_cache_from_record(record: dict, new_html: str):
    meta = {
        "kind": record["kind"],
        "id": _cache_id_from_key(record["key"]),
        "title": record.get("title"),
        "html": new_html,
    }

    item = record.get("item")
    if record["kind"] == "Page" and item is not None:
        page_url = getattr(item, "page_url", None)
        if page_url:
            meta["url"] = page_url

    if item is not None:
        _, candidates = _item_cache_candidates(item)
    else:
        candidates = []

    _write_item_cache(meta, candidates)


def fetch_item_html(course, item):
    """Fetch the HTML-bearing body for a supported item with session cache."""
    cache, aliases = _get_item_cache_structures()

    kind, candidates = _item_cache_candidates(item)
    for cand in candidates:
        if cand in cache:
            return dict(cache[cand])
        mapped = aliases.get(cand)
        if mapped and mapped in cache:
            return dict(cache[mapped])

    if kind == "Page":
        page = course.get_page(item.page_url)
        meta = {"kind": "Page", "id": page.page_id, "url": page.url, "title": page.title, "html": page.body or ""}
    elif kind == "Assignment":
        asg = course.get_assignment(item.content_id)
        meta = {"kind": "Assignment", "id": asg.id, "title": asg.name, "html": asg.description or ""}
    elif kind == "Discussion":
        topic = course.get_discussion_topic(item.content_id)
        meta = {
            "kind": "Discussion",
            "id": topic.id,
            "title": getattr(topic, "title", f"Discussion {topic.id}"),
            "html": getattr(topic, "message", "") or ""
        }
    else:
        meta = {"kind": kind, "id": getattr(item, "content_id", getattr(item, "id", "unknown")), "html": ""}

    _write_item_cache(meta, candidates)

    return dict(meta)

def apply_update(course, item, new_html: str):
    if item.type == "Page":
        course.get_page(item.page_url).edit(wiki_page={"body": new_html})
    elif item.type == "Assignment":
        course.get_assignment(item.content_id).edit(assignment={"description": new_html})
    elif item.type in {"Discussion", "DiscussionTopic"}:
        topic = course.get_discussion_topic(item.content_id)
        try:
            topic.edit(message=new_html)
        except TypeError:
            topic.edit(**{"message": new_html})

# ---------------------- Iframe freeze/restore ----------------------

PLACEHOLDER_FMT = "⟪IFRAME:{i}⟫"

def protect_iframes(html: str):
    """Replace existing iframes with placeholders and return (frozen_html, mapping, hosts)."""
    soup = BeautifulSoup(html or "", "html.parser")
    mapping, hosts = {}, set()
    i = 1
    for tag in soup.find_all("iframe"):
        src = (tag.get("src") or "").strip()
        host = urllib.parse.urlparse(src).hostname or ""
        if host:
            hosts.add(host.lower())
        placeholder = PLACEHOLDER_FMT.format(i=i)
        mapping[placeholder] = str(tag)
        tag.replace_with(placeholder)
        i += 1
    return str(soup), mapping, hosts

def restore_iframes(html: str, mapping: dict) -> str:
    """Replace placeholders with the original <iframe> markup."""
    for ph, original in mapping.items():
        html = html.replace(ph, original)
    return html

def strip_new_iframes(html: str) -> str:
    """Remove any iframes the model added (we only want to restore originals)."""
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup.find_all("iframe"):
        tag.decompose()
    return str(soup)

# ---------------------- Model-reference helpers ----------------------

def html_to_skeleton(model_html: str, max_nodes: int = 2000, max_text: int = 80) -> str:
    """Build a compact, valid-ish skeleton of the model HTML."""
    soup = BeautifulSoup(model_html or "", "html.parser")

    def keep_attr(k: str) -> bool:
        return k in ("id", "class", "role") or k.startswith("aria-") or k.startswith("data-")

    def fmt_attrs(attrs: dict) -> str:
        parts = []
        for k, v in (attrs or {}).items():
            if not keep_attr(k):
                continue
            if isinstance(v, (list, tuple)):
                v = " ".join(map(str, v))
            v = " ".join(str(v).split())[:120]
            parts.append(f'{k}="{v}"')
        return (" " + " ".join(parts)) if parts else ""

    headings = {"h1","h2","h3","h4","h5","h6","summary","label"}
    count = [0]

    def render(node) -> str:
        if count[0] >= max_nodes:
            return ""
        if isinstance(node, NavigableString):
            return ""
        if not isinstance(node, Tag):
            return ""

        count[0] += 1
        start = f"<{node.name}{fmt_attrs(node.attrs)}>"
        inner = []

        if node.name in headings:
            direct_text = []
            for child in node.children:
                if isinstance(child, NavigableString):
                    direct_text.append(str(child))
            text = " ".join(" ".join(direct_text).split())
            if text:
                inner.append(text[:max_text])

        for child in node.children:
            if isinstance(child, Tag):
                piece = render(child)
                if piece:
                    inner.append(piece)
            if count[0] >= max_nodes:
                break

        end = f"</{node.name}>"
        return start + "".join(inner) + end

    roots = list(soup.body.children) if soup.body else list(soup.children)
    out = []
    for child in roots:
        piece = render(child)
        if piece:
            out.append(piece)
        if count[0] >= max_nodes:
            break

    text = "".join(out)
    text = " ".join(text.split())
    if len(text) > 120_000:
        text = text[:120_000] + " <!-- truncated -->"
    return text

def find_single_course_by_code(account, course_code: str):
    courses = account.get_courses(search_term=course_code)
    return list(courses)

def list_all_pages(course):
    pages = []
    for p in course.get_pages():
        pages.append((getattr(p, "title", p.url), p.url))
    return pages

def list_all_assignments(course):
    items = []
    for a in course.get_assignments():
        items.append((getattr(a, "name", f"Assignment {a.id}"), a.id))
    return items

def fetch_model_item_html(course, kind: str, ident):
    if kind == "Page":
        page = course.get_page(ident)  # ident = page.url
        return page.body or ""
    elif kind == "Assignment":
        asg = course.get_assignment(int(ident))
        return asg.description or ""
    return ""

def image_to_data_url(file) -> Tuple[str, str]:
    if file is None:
        raise ValueError("No image provided")
    mime = file.type or "image/png"
    raw = None
    try:
        raw = bytes(file.getbuffer())
    except Exception:
        pass
    if not raw:
        try:
            file.seek(0)
        except Exception:
            pass
        raw = file.read()
    if not raw:
        raise ValueError("Empty image or no bytes read")
    b64 = base64.b64encode(raw).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"
    return data_url, mime

# ---------------------- Auto-pick newest model ----------------------

def _client_token_fingerprint(client: OpenAI) -> str:
    token = getattr(client, "api_key", None) or getattr(client, "_api_key", None)
    if token:
        return sha256(str(token))
    env_token = os.getenv("OPENAI_API_KEY", "")
    return sha256(env_token) if env_token else "anon"


def _model_record(model) -> Optional[dict]:
    if model is None:
        return None
    if isinstance(model, dict):
        model_id = model.get("id")
        created = model.get("created", 0)
    else:
        model_id = getattr(model, "id", None)
        created = getattr(model, "created", 0)
    if not model_id:
        return None
    return {"id": model_id, "created": created}


@st.cache_data(show_spinner=False, hash_funcs={OpenAI: lambda client: _client_token_fingerprint(client)})
def list_models(client: OpenAI):
    try:
        res = client.models.list()
        data = getattr(res, "data", res)
        serialized = []
        for entry in data:
            record = _model_record(entry)
            if record:
                serialized.append(record)
        return serialized
    except Exception:
        return []


def latest_model_id(
    client: OpenAI,
    pattern: str,
    default_id: str,
    models: Optional[List[dict]] = None,
) -> str:
    if models is None:
        models = list_models(client)
    try:
        matches = [m for m in models if re.search(pattern, m.get("id", ""))]
        if not matches:
            return default_id
        matches.sort(key=lambda m: m.get("created", 0), reverse=True)
        return matches[0]["id"]
    except Exception:
        return default_id

# ---------------------- OpenAI + retries ----------------------

SYSTEM_PROMPT = (
    "You are an expert Canvas HTML editor. Preserve links, anchors/IDs, classes, and data-* attributes. "
    "Placeholders like ⟪IFRAME:n⟫ represent protected iframes—do not add, remove, or reorder them. "
    "Follow the policy. Return only HTML, no explanations."
)

DT_MODES = ["Preserve", "Enhance", "Replace"]

def _create_with_retries(
    client: OpenAI,
    model: str,
    payload_input,
    fallback_model: Optional[str] = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
):
    def _do_call(m: str):
        return client.responses.create(
            model=m,
            input=payload_input,
            temperature=0.2,
            timeout=60,  # seconds; avoid long hangs per item
        )

    def _should_retry(err: Exception) -> bool:
        s = str(err).lower()
        return any(k in s for k in ["server_error", "status code: 5", "timed out", "timeout", "temporarily unavailable"])

    last_err = None

    for attempt in range(max_retries):
        try:
            return _do_call(model)
        except Exception as e:
            last_err = e
            if not _should_retry(e):
                break
            time.sleep(base_delay * (2 ** attempt) + random.random() * 0.5)

    if fallback_model and fallback_model != model:
        for attempt in range(max_retries):
            try:
                return _do_call(fallback_model)
            except Exception as e:
                last_err = e
                if not _should_retry(e):
                    break
                time.sleep(base_delay * (2 ** attempt) + random.random() * 0.5)

    raise last_err

# Few-shot examples to increase determinism without UI sliders
EXAMPLES_TEXT = r"""
EXAMPLE 1 — Banner normalize
INPUT HTML:
<div id="page-banner" class="hero">Welcome</div>

RULES:
- ENFORCE-CLASS: body -> dp-theme--circle-left-1
- BANNER: selector="#page-banner" classes="dp-banner dp-banner--lg" style="min-height:180px;display:flex;align-items:center;"

EXPECTED OUTPUT:
<div id="page-banner" class="dp-banner dp-banner--lg" style="min-height:180px;display:flex;align-items:center;">Welcome</div>

EXAMPLE 2 — Callout normalize
INPUT HTML:
<blockquote class="note">Remember the deadline.</blockquote>

RULES:
- CALLOUTS-ALLOWED: info, warning, success

EXPECTED OUTPUT:
<div class="dp-callout dp-callout--info">Remember the deadline.</div>
"""

def openai_rewrite(
    user_request: str,
    html: str,
    dt_mode: str,
    precision_directives: str,
    model_html_skeleton: Optional[str] = None,
    model_image_data_url: Optional[str] = None,
    model_text_id: str = DEFAULT_TEXT_MODEL,
    model_vision_id: str = DEFAULT_VISION_MODEL,
) -> str:
    """Call OpenAI Responses API to rewrite the HTML with precision directives and few-shot examples."""
    policy = {
        "design_tools_mode": dt_mode,
        "allow_inline_styles": True,
        "block_new_iframes": True,
        "reference_model": ("image" if model_image_data_url else "html" if model_html_skeleton else "none"),
        "reference_usage": "Match layout/sectioning/components; do not copy literal course-specific links or text."
    }

    hard_rules_text = (
        "Hard rules (follow EXACTLY; reason silently; output HTML only):\n"
        + (precision_directives.strip() + "\n" if precision_directives.strip() else "")
        + "- Do not remove existing anchors/IDs/classes/data-* unless conflicting with required theme/classes.\n"
        + "- Do not create new iframes. Respect ⟪IFRAME:n⟫ placeholders.\n"
    )

    blocks = [
        hard_rules_text,
        "Policy: " + json.dumps(policy, ensure_ascii=False),
        "Examples:\n" + EXAMPLES_TEXT.strip(),
        f"DesignTools Mode: {dt_mode}",
        "Rewrite goals (optional): " + (user_request or ""),
    ]
    if model_html_skeleton:
        blocks.append("Model HTML skeleton (follow structure/classes, not text):\n" + model_html_skeleton)
    blocks.append("HTML to rewrite follows:\n" + html)

    if not model_image_data_url:
        prompt = "\n\n".join(blocks)
        payload = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        resp = _create_with_retries(
            client=openai_client,
            model=model_text_id,
            payload_input=payload,
            fallback_model="gpt-4.1",
            max_retries=3,
            base_delay=1.0,
        )
    else:
        try:
            content_parts = [
                {"type": "input_text", "text": "\n\n".join(blocks[:-1])},
                {"type": "input_image", "image_url": {"url": model_image_data_url}},
                {"type": "input_text", "text": blocks[-1]},
            ]
            payload = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content_parts},
            ]
            resp = _create_with_retries(
                client=openai_client,
                model=model_vision_id,
                payload_input=content_parts,
                fallback_model="gpt-4o",
                max_retries=3,
                base_delay=1.0,
            )
        except Exception:
            prompt = "\n\n".join(blocks) + "\n\n(Note: image reference unavailable; proceed using textual model description if any.)"
            payload = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            resp = _create_with_retries(
                client=openai_client,
                model=model_text_id,
                payload_input=payload,
                fallback_model="gpt-4.1",
                max_retries=3,
                base_delay=1.0,
            )

    try:
        return resp.output_text
    except AttributeError:
        return resp.output[0].content[0].text

# ---------------------- Diff ----------------------

def html_diff(old: str, new: str) -> str:
    dmp = diff_match_patch()
    d = dmp.diff_main(old or "", new or "")
    dmp.diff_cleanupSemantic(d)
    return dmp.diff_prettyHtml(d)

# ---------------------- Shim CSS/JS & preview ----------------------

SHIM_CSS = """
:root {
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Lato, Arial, sans-serif;
  --csu-green: #1E4D2B;
  --csu-gold: #C8C372;
  --neutral-900: #262626;
  --border: #e2e8f0;
}
html, body { margin:0; padding:0; font-family:var(--font); line-height:1.5; background:#fff; color:#111; }
#frame { max-width: 980px; margin: 0 auto; padding: 24px; background: #fff; }
h1,h2,h3,h4,h5,h6 { margin: 0.6em 0 0.4em; line-height:1.25; }
p,li { font-size: 16px; }
a { color: #005f85; text-decoration: underline; }
img { max-width: 100%; height: auto; }

/* Banner */
.dp-banner, .dpl-banner, .designplus.banner {
  border-radius: 10px; padding: 16px 20px; color: #fff;
  background: linear-gradient(135deg, var(--csu-green), #0f2a18);
  display:flex; align-items:center; gap: 12px; min-height: 140px;
}
.dp-banner.dp-banner--lg { min-height: 180px; }
.dp-theme--circle-left-1 .dp-banner { background: radial-gradient(circle at left, var(--csu-green), #0f2a18); }

/* Also style dp-header to look like banner */
.dp-header {
  border-radius: 10px; padding: 16px 20px; color: #fff;
  background: linear-gradient(135deg, var(--csu-green), #0f2a18);
  display:flex; align-items:center; gap: 12px; min-height: 140px;
}
.dp-header.dp-banner--lg { min-height: 180px; }
.dp-theme--circle-left-1 .dp-header { background: radial-gradient(circle at left, var(--csu-green), #0f2a18); }

/* Callouts */
.dp-callout, .dpl-callout {
  border-radius: 8px; padding: 12px 14px; border:1px solid var(--border); background:#f9fafb; margin: 12px 0;
}
.dp-callout--info { background:#eef6ff; border-color:#bfdbfe; }
.dp-callout--warning { background:#fff7ed; border-color:#fed7aa; }
.dp-callout--success { background:#ecfdf5; border-color:#a7f3d0; }

/* Buttons */
.dpl-button, .dp-button, a.button {
  display:inline-block; border-radius:6px; padding:10px 14px; text-decoration:none; font-weight:600; border:1px solid transparent;
}
.dpl-button--primary { background: var(--csu-green); color:#fff; }
.dpl-button--ghost { background: transparent; border-color: var(--csu-green); color: var(--csu-green); }

/* Tabs (approx) */
.tabs { margin: 16px 0; }
.tab-list { display:flex; gap:8px; border-bottom:1px solid var(--border); padding-bottom:6px; }
.tab-list button { background:#fff; border:1px solid var(--border); border-bottom:none; padding:8px 12px; border-radius:6px 6px 0 0; cursor:pointer; }
.tab-list button.active { background:#fff; border-color: var(--csu-green); color: var(--csu-green); }
.tab-panel { border:1px solid var(--border); border-radius:0 6px 6px 6px; padding:12px; display:none; }
.tab-panel.active { display:block; }

/* Accordions */
details { border:1px solid var(--border); border-radius:8px; padding:8px 12px; margin:10px 0; }
details > summary { font-weight:600; cursor:pointer; }

/* Tables */
table { border-collapse: collapse; width: 100%; }
.ic-Table th, .ic-Table td { border: 1px solid #e5e7eb; padding: 8px; }
.ic-Table.zebra tr:nth-child(even) { background: #fafafa; }

/* Misc containers */
.designplus, .dp-component, .dpl { border-radius: 6px; padding: 10px; border: 1px dashed #cbd5e1; background: #f8fafc; }
.iframe-placeholder { border:1px dashed #cbd5e1; padding:12px; background:#fbfbfb; color:#555; border-radius:6px; }
"""

SHIM_JS = """
(function(){
  function initTabs(root){
    const tabsets = root.querySelectorAll('.tabs');
    tabsets.forEach(ts => {
      const btns = ts.querySelectorAll('.tab-list button');
      const panels = ts.querySelectorAll('.tab-panel');
      function activate(i){
        btns.forEach((b, idx)=> b.classList.toggle('active', idx===i));
        panels.forEach((p, idx)=> p.classList.toggle('active', idx===i));
      }
      btns.forEach((b, i) => b.addEventListener('click', () => activate(i)));
      if (btns.length && panels.length) activate(0);
    });
  }

  function applyDesignPlusHints(){
    const wrap = document.getElementById('dp-wrapper');
    if (!wrap) return;

    // Apply background image from data-img-url
    const imgUrl = wrap.getAttribute('data-img-url');
    const header = wrap.querySelector('.dp-banner, .dp-header');
    if (imgUrl && header && !header.style.backgroundImage) {
      header.style.backgroundImage = `url("${imgUrl}")`;
      header.style.backgroundSize = 'cover';
      header.style.backgroundPosition = 'center';
    }

    // Apply data-header-class and data-nav-class if present
    const headerClass = wrap.getAttribute('data-header-class');
    if (headerClass && header) {
      headerClass.split(/\\s+/).forEach(c => c && header.classList.add(c));
    }
    const navClass = wrap.getAttribute('data-nav-class');
    const nav = wrap.querySelector('.dp-nav');
    if (navClass && nav) {
      navClass.split(/\\s+/).forEach(c => c && nav.classList.add(c));
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    applyDesignPlusHints();
    initTabs(document);
  });
})();
"""

def neutralize_iframes_for_preview(html: str) -> str:
    """Replace iframes with placeholders (no external loads in preview)."""
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup.find_all("iframe"):
        src = (tag.get("src") or "").strip()
        host = urllib.parse.urlparse(src).hostname or "iframe"
        placeholder = soup.new_tag("div", attrs={"class": "iframe-placeholder"})
        placeholder.string = f"[iframe: {host}]"
        tag.replace_with(placeholder)
    return str(soup)

def preview_html_document(
    inner_html: str,
    use_js_shim: bool,
    extra_css_texts: Optional[List[str]] = None,
    extra_js_texts: Optional[List[str]] = None,
) -> str:
    """Sandboxed preview document with shim and optional uploaded CSS/JS."""
    css_bundle = SHIM_CSS + "\n" + "\n".join(extra_css_texts or [])
    js_blocks = []
    if use_js_shim:
        js_blocks.append(SHIM_JS)
    for t in (extra_js_texts or []):
        js_blocks.append(t)

    safe_inner = inner_html or ""
    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>{css_bundle}</style>
  </head>
  <body>
    <main id="frame">{safe_inner}</main>
    {''.join(f'<script>{js}</script>' for js in js_blocks)}
  </body>
</html>"""

# ---------------------- UI ----------------------

st.title("Canvas Course-wide HTML Rewrite (Test Instance)")

# Defaults for preview extra assets to avoid NameError later
extra_css_texts: List[str] = []
extra_js_texts: List[str] = []

with st.sidebar:
    st.header("Rewrite Setup")

    dt_mode = st.selectbox("DesignTools Mode", DT_MODES, index=1, help="How aggressively to use/change DesignTools patterns.")
    user_request = st.text_area(
        "Rewrite goals (optional, in plain English)",
        value="Improve accessibility; normalize headings; refine layout using DesignPLUS where appropriate; preserve links and existing iframes.",
        height=120,
    )

    st.markdown("**Precision directives (optional, power users)**")
    precision_directives = st.text_area(
        label="Precision directives",
        value=(
            "PRESERVE: anchors, ids, classes, data-*\n"
            "NO-NEW: iframes\n"
            "ENFORCE-CLASS: body -> dp-theme--circle-left-1\n"
            "BANNER: selector=\"#page-banner,.dp-banner\" classes=\"dp-banner dp-banner--lg\" style=\"min-height:180px;display:flex;align-items:center;\" insert-if-missing=true\n"
            "PALETTE-ALLOWED: #1E4D2B, #C8C372, #FFFFFF, #262626\n"
            "PALETTE-MAP: nearest\n"
            "CALLOUTS-ALLOWED: info, warning, success\n"
            "TABS: allow\n"
            "ACCORDIONS: auto\n"
            "HEADINGS: min=2, enforce-hierarchy=true\n"
            "BUTTON-CLASS: dpl-button dpl-button--primary\n"
            "TABLE: class=ic-Table, zebra=true\n"
            "IMAGES: max-width=100%, alt=require\n"
            "DO-NOT-EDIT: \"#course-links, .lti-embed, .assignment-group\""
        ),
        height=170,
        help="Short command-style rules. Leave as-is or tweak.",
        label_visibility="collapsed",
    )

    # -------- Preview options with Layer 2 (Upload CSS/JS) --------
    with st.expander("Preview options"):
        PREVIEW_HEIGHT = st.slider("Preview height (px)", 320, 1200, 520, 20)
        PREVIEW_JS = st.checkbox("Enable shim JS (tabs interactions)", value=True)
        PREVIEW_MODE = st.radio("Fidelity", ["Shim only", "Upload CSS/JS"], horizontal=True)
        if PREVIEW_MODE == "Upload CSS/JS":
            css_files = st.file_uploader("Upload CSS (DesignPLUS/Canvas theme)", type=["css"], accept_multiple_files=True)
            js_files  = st.file_uploader("Upload JS (optional)", type=["js"], accept_multiple_files=True)
            if css_files:
                for f in css_files:
                    try:
                        content = f.read()
                        extra_css_texts.append((content or b"").decode("utf-8", errors="ignore"))
                    except Exception:
                        pass
            if js_files:
                for f in js_files:
                    try:
                        content = f.read()
                        extra_js_texts.append((content or b"").decode("utf-8", errors="ignore"))
                    except Exception:
                        pass
            st.caption("Uploaded assets are inlined into the preview iframe only (not saved to Canvas).")

    # -------- Advanced (incl. large course controls) --------
    with st.expander("Advanced"):
        MAX_INPUT_CHARS = st.number_input("Max item HTML chars sent to model", 5000, 200000, 40000, 5000)
        MAX_MODEL_SKELETON_CHARS = st.number_input("Max model skeleton chars", 5000, 200000, 60000, 5000)

        st.markdown("---")
        st.markdown("**Large course controls**")
        INCLUDE_PAGES = st.checkbox("Include Pages", value=True)
        INCLUDE_ASSIGNMENTS = st.checkbox("Include Assignments", value=True)
        INCLUDE_DISCUSSIONS = st.checkbox("Include Discussions (topic descriptions)", value=True)
        BATCH_SIZE = st.number_input("Batch size (items per run)", 1, 200, 25, 1)
        TIME_BUDGET_MIN = st.number_input("Time budget (minutes, per run)", 1, 120, 10, 1)
        GENERATE_DIFFS = st.checkbox("Generate HTML diffs", value=True, help="Turn off to speed up on very large pages")

    st.markdown("---")
    st.subheader("Model")
    if st.button("Refresh model list", type="secondary", help="Clear the cached model listing and fetch the latest models on next use."):
        list_models.clear()
        st.success("Model list cache cleared. Fetching latest models…")
    mode = st.radio("Model selection", ["Auto (latest)", "Manual"], horizontal=True)
    if mode == "Auto (latest)":
        available_models = list_models(openai_client)
        MODEL_TEXT = latest_model_id(
            openai_client,
            r"^(gpt-5|gpt-4\.1|gpt-4o|o\d)",
            DEFAULT_TEXT_MODEL,
            models=available_models,
        )
        MODEL_VISION = latest_model_id(
            openai_client,
            r"^(gpt-5|gpt-4o|gpt-4\.1|o\d)",
            DEFAULT_VISION_MODEL,
            models=available_models,
        )
    else:
        MODEL_TEXT = st.text_input("Text model id", value=DEFAULT_TEXT_MODEL)
        MODEL_VISION = st.text_input("Vision model id", value=DEFAULT_VISION_MODEL)
    st.caption(f"Using text model: {MODEL_TEXT}")
    st.caption(f"Using vision model: {MODEL_VISION}")

    st.markdown("---")
    st.subheader("Model Reference (optional)")
    ref_kind = st.radio("Reference type", ["None", "Paste HTML", "Upload Image"], horizontal=True)  # removed "Model Course"

    model_html_skeleton = None
    model_image_data_url = None

    if ref_kind == "Paste HTML":
        pasted_html = st.text_area("Paste model HTML here", height=200)
        if pasted_html.strip():
            try:
                model_html_skeleton = html_to_skeleton(pasted_html)
                if len(model_html_skeleton) > MAX_MODEL_SKELETON_CHARS:
                    model_html_skeleton = model_html_skeleton[:MAX_MODEL_SKELETON_CHARS] + " <!-- truncated -->"
                st.caption(f"Model skeleton length: {len(model_html_skeleton)} chars")
                with st.expander("Preview model skeleton"):
                    st.code(model_html_skeleton[:4000])
            except Exception as e:
                st.error(f"Failed to process pasted HTML: {e}")

    elif ref_kind == "Upload Image":
        uploaded_img = st.file_uploader("Upload model page image", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=False)
        if uploaded_img is not None:
            try:
                model_image_data_url, mime = image_to_data_url(uploaded_img)
                st.image(uploaded_img, caption=f"Model reference ({mime})", use_container_width=True)
                st.caption("The image will be passed to the model as a vision reference. If unsupported, we will fall back to text-only.")
            except Exception as e:
                st.error(f"Failed to process image: {e}")

st.subheader("1) Pick Course by Code")
course_code = st.text_input("Course code", help="Exact course code preferred; we'll disambiguate if needed.")
if st.button("Search course"):
    with st.spinner("Searching…"):
        st.session_state["courses"] = find_course_by_code(account, course_code)

courses = st.session_state.get("courses", [])
if courses:
    idx = st.selectbox(
        "Select course",
        options=list(range(len(courses))),
        format_func=lambda i: f"{courses[i].id} · {getattr(courses[i], 'course_code', '')} · {getattr(courses[i], 'name', '')}",
    )
    course = courses[idx]

    st.subheader("2) Dry-run")

    # 2.1 Index items once (fast)
    if st.button("Index course items"):
        with st.spinner("Indexing module items…"):
            idx_items = list_supported_items(
                course,
                include_pages=INCLUDE_PAGES,
                include_assignments=INCLUDE_ASSIGNMENTS,
                include_discussions=INCLUDE_DISCUSSIONS,
            )
        st.session_state["index"] = idx_items
        st.session_state["offset"] = 0
        st.session_state.setdefault("items", [])
        st.session_state.setdefault("drafts", {})
        st.session_state.setdefault("processed_keys", set())
        st.success(f"Indexed {len(idx_items)} items.")

    index = st.session_state.get("index", [])
    offset = st.session_state.get("offset", 0)

    if index:
        remaining = max(0, len(index) - offset)
        st.info(f"Indexed: {len(index)} • Processed: {offset} • Remaining: {remaining}")

        cols = st.columns(3)
        with cols[0]:
            run_batch = st.button(f"Simulate rewrite: Next {min(BATCH_SIZE, remaining) if remaining else BATCH_SIZE}")
        with cols[1]:
            run_all = st.button("Simulate rewrite: All remaining (may be slow)")
        with cols[2]:
            if st.button("Reset dry-run progress"):
                st.session_state["offset"] = 0
                st.session_state["items"] = []
                st.session_state["drafts"] = {}
                st.session_state["processed_keys"] = set()
                st.experimental_rerun()

        # 2.2 Process batch(es)
        if run_batch or run_all:
            start_time = time.time()
            to_process = range(offset, len(index)) if run_all else range(offset, min(offset + BATCH_SIZE, len(index)))

            progress = st.progress(0.0)
            eta_box = st.empty()
            done = 0
            WINDOW = 5
            recent = []

            st.session_state.setdefault("items", [])
            st.session_state.setdefault("drafts", {})
            st.session_state.setdefault("processed_keys", set())

            for i in to_process:
                if (time.time() - start_time) > (TIME_BUDGET_MIN * 60):
                    st.warning("Time budget reached; you can click 'Next' again to resume.")
                    break

                module, it = index[i]
                meta = fetch_item_html(course, it)
                key = f"{meta['kind']}:{meta['id']}"

                if key in st.session_state["processed_keys"]:
                    done += 1
                    denom = (len(index) - offset) if run_all else (min(offset + BATCH_SIZE, len(index)) - offset)
                    progress.progress(min(1.0, done / max(1, denom)))
                    continue

                original = meta["html"] or ""
                frozen, mapping, hosts = protect_iframes(original)

                prepped = frozen if len(frozen) <= MAX_INPUT_CHARS else (frozen[:MAX_INPUT_CHARS] + "\n<!-- truncated for prompt -->")

                t0 = time.time()
                try:
                    rewritten = openai_rewrite(
                        user_request=user_request,
                        html=prepped,
                        dt_mode=dt_mode,
                        precision_directives=precision_directives,
                        model_html_skeleton=model_html_skeleton,
                        model_image_data_url=model_image_data_url,
                        model_text_id=MODEL_TEXT,
                        model_vision_id=MODEL_VISION,
                    )
                except Exception as e:
                    rewritten = original
                    st.error(f"Rewrite failed for [{meta.get('title') or meta.get('url')}] — {e}")

                rewritten_no_new_iframes = strip_new_iframes(rewritten)
                final_html = restore_iframes(rewritten_no_new_iframes, mapping)

                diff_html = html_diff(original, final_html) if GENERATE_DIFFS else "<div>Diff disabled for speed.</div>"

                st.session_state["items"].append({
                    "key": key,
                    "title": meta.get("title") or meta.get("url"),
                    "kind": meta["kind"],
                    "module": getattr(module, "name", ""),
                    "item": it,
                    "original": original,
                    "draft": final_html,
                })
                st.session_state["drafts"][key] = {"diff": diff_html}
                st.session_state["processed_keys"].add(key)

                duration = time.time() - t0
                recent.append(duration)
                if len(recent) > WINDOW:
                    recent.pop(0)
                avg = (sum(recent) / len(recent)) if recent else duration
                remaining_in_this_run = (len(index) - i - 1) if run_all else (min(offset + BATCH_SIZE, len(index)) - i - 1)
                eta_box.info(f"Processed {done+1} · Avg/Item {avg:.1f}s · ETA {max(0,int(remaining_in_this_run * avg))}s")
                done += 1
                denom = (len(index) - offset) if run_all else (min(offset + BATCH_SIZE, len(index)) - offset)
                progress.progress(min(1.0, done / max(1, denom)))

            st.session_state["offset"] = offset + done
            st.success(f"Batch complete. Total processed: {st.session_state['offset']} of {len(index)}.")

    items = st.session_state.get("items", [])
    if items:
        st.subheader("3) Review visual previews, diffs & approve")

        # Bulk toggles
        col1, col2, col3 = st.columns([1,1,3])
        with col1:
            if st.button("Approve All"):
                for rec in items:
                    st.session_state[f"approve_{rec['key']}"] = True
        with col2:
            if st.button("Unapprove All"):
                for rec in items:
                    st.session_state[f"approve_{rec['key']}"] = False
        with col3:
            approved_count = sum(1 for rec in items if st.session_state.get(f"approve_{rec['key']}", False))
            st.write(f"Approved: **{approved_count} / {len(items)}**")

        # Render items
        for _, rec in enumerate(items):
            cb_key = f"approve_{rec['key']}"
            with st.expander(f"[{rec['kind']}] {rec['title']} — {rec['module']}"):
                left, right = st.columns(2)
                with left:
                    st.markdown("**Original (visual)**")
                    original_preview = neutralize_iframes_for_preview(rec["original"])
                    st.components.v1.html(
                        preview_html_document(original_preview, use_js_shim=PREVIEW_JS, extra_css_texts=extra_css_texts, extra_js_texts=extra_js_texts),
                        height=PREVIEW_HEIGHT, scrolling=True
                    )
                with right:
                    st.markdown("**Proposed (visual)**")
                    proposed_preview = neutralize_iframes_for_preview(rec["draft"])
                    st.components.v1.html(
                        preview_html_document(proposed_preview, use_js_shim=PREVIEW_JS, extra_css_texts=extra_css_texts, extra_js_texts=extra_js_texts),
                        height=PREVIEW_HEIGHT, scrolling=True
                    )

                st.markdown("**Diff (proposed vs original)**  \n_Green = insertions, Red = deletions_")
                st.components.v1.html(st.session_state["drafts"][rec["key"]]["diff"], height=240, scrolling=True)

                approved = st.checkbox("Approve this item", key=cb_key, value=st.session_state.get(cb_key, False))
                rec["approved"] = bool(approved)

                with st.expander("Show HTML (original)"):
                    st.code(rec["original"][:2000])
                with st.expander("Show HTML (proposed)"):
                    st.code(rec["draft"][:2000])

        st.write("")
        if st.button("Apply approved changes"):
            applied, failed = 0, 0
            with st.spinner("Applying to Canvas…"):
                for rec in items:
                    cb_key = f"approve_{rec['key']}"
                    if not st.session_state.get(cb_key, False):
                        continue
                    try:
                        apply_update(course, rec["item"], rec["draft"])
                        _refresh_item_cache_from_record(rec, rec["draft"])
                        applied += 1
                    except Exception as e:
                        failed += 1
                        st.error(f"Failed: {rec['title']}: {e}")
            st.success(f"Applied {applied} item(s); {failed} failed.")
else:
    st.info("Search for a course above to begin.")
