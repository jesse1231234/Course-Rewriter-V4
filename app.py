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
DEFAULT_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")

MAX_INPUT_CHARS = 25000
BATCH_SIZE = 10
TIME_BUDGET_SECONDS = 60 * 10  # 10 minutes
GENERATE_DIFFS = True

openai_client = OpenAI(api_key=OPENAI_API_KEY)
canvas = Canvas(CANVAS_BASE_URL, CANVAS_ADMIN_TOKEN)

# ---------------------- Utility: Model Picker ----------------------


def pick_models() -> Tuple[str, str]:
    """Pick best-available text + vision models. Fallback to defaults."""
    try:
        models = openai_client.models.list()
        names = [m.id for m in models.data]
    except Exception:
        names = []

    text = DEFAULT_TEXT_MODEL
    vision = DEFAULT_VISION_MODEL

    preferred_text = ["gpt-5.1", "gpt-5", "gpt-4.1", "gpt-4.1-mini"]
    preferred_vision = ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4o"]

    for m in preferred_text:
        if m in names:
            text = m
            break

    for m in preferred_vision:
        if m in names:
            vision = m
            break

    return text, vision


MODEL_TEXT, MODEL_VISION = pick_models()

# ---------------------- HTML Helpers ----------------------


PLACEHOLDER_FMT = "⟪IFRAME:{i}⟫"


def protect_iframes(html: str) -> Tuple[str, dict, set]:
    """Replace <iframe> tags with placeholders; return html, mapping, and set of hosts."""
    soup = BeautifulSoup(html or "", "html.parser")
    mapping = {}
    hosts = set()
    i = 0
    for tag in soup.find_all("iframe"):
        src = tag.get("src") or ""
        host = urllib.parse.urlparse(src).netloc
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
    """Strip any <iframe> tags the model might have introduced."""
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup.find_all("iframe"):
        tag.decompose()
    return str(soup)


def html_diff(a: str, b: str) -> str:
    """Return an HTML diff (side-by-side) between two HTML strings."""
    dmp = diff_match_patch()
    diffs = dmp.diff_main(a or "", b or "")
    dmp.diff_cleanupSemantic(diffs)

    out = []
    for op, data in diffs:
        data = (
            data.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        if op == dmp.DIFF_INSERT:
            out.append(f"<ins style='background:#e6ffe6;'>{data}</ins>")
        elif op == dmp.DIFF_DELETE:
            out.append(f"<del style='background:#ffe6e6;'>{data}</del>")
        else:
            out.append(f"<span>{data}</span>")

    return "<div style='font-family:monospace;white-space:pre-wrap;word-wrap:break-word;'>" + "".join(out) + "</div>"


# ---------------------- Canvas Helpers ----------------------


def get_courses_for_account(account_id: int):
    account = canvas.get_account(account_id)
    return list(account.get_courses(enrollment_type="teacher"))


def get_course(course_id: int):
    return canvas.get_course(course_id)


def get_course_pages(course):
    return list(course.get_pages())


def get_course_modules(course):
    return list(course.get_modules())


def get_module_items(mod):
    return list(mod.get_module_items())


def get_item_html(course, item) -> Tuple[str, dict]:
    """Get HTML + metadata for a module item (Page, Assignment, Discussion)."""
    meta = {"kind": item.type, "id": item.id, "title": None, "html": ""}

    if item.type == "Page":
        page = course.get_page(item.page_url)
        meta["title"] = page.title
        meta["html"] = page.body or ""
        meta["item"] = page
        meta["page_url"] = item.page_url
    elif item.type == "Assignment":
        asg = course.get_assignment(item.content_id)
        meta["title"] = asg.name
        meta["html"] = asg.description or ""
        meta["item"] = asg
    elif item.type == "Discussion":
        disc = course.get_discussion_topic(item.content_id)
        meta["title"] = disc.title
        meta["html"] = disc.message or ""
        meta["item"] = disc
    else:
        meta["html"] = ""
        meta["item"] = None

    return meta["html"], meta


def update_item_html(meta: dict, new_html: str):
    """Write HTML back to Canvas for a given meta record."""
    kind = meta["kind"]
    item = meta.get("item")
    if item is None:
        return

    if kind == "Page":
        item.edit(wiki_page={"body": new_html})
    elif kind == "Assignment":
        item.edit(assignment={"description": new_html})
    elif kind == "Discussion":
        item.edit(message=new_html)


# ---------------------- Caching Helpers ----------------------


def _cache_id_from_key(key: str) -> str:
    # key looks like "Page:12345" or "Assignment:9876"
    if ":" in key:
        return key.split(":", 1)[1]
    return key


def _item_cache_meta_from_record(record: dict, new_html: str):
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
        _title = getattr(item, "title", None) or getattr(item, "name", None)
        if _title and not meta.get("title"):
            meta["title"] = _title

    return meta


# ---------------------- Precision Directives / Examples ----------------------

EXAMPLES_TEXT = """
Example 1:
Original:
<h2>Objectives</h2>
<ul>
  <li>Learn X</li>
  <li>Practice Y</li>
</ul>

Rewritten (Enhance mode, DesignPLUS-like):
<section class="dp-section dp-section--objectives">
  <h2 class="dp-heading dp-heading--h2">Objectives</h2>
  <ul class="dp-list dp-list--check">
    <li>Learn X</li>
    <li>Practice Y</li>
  </ul>
</section>

Example 2:
Original:
<p><strong>Due:</strong> Sunday at 11:59 pm</p>

Rewritten:
<p class="dp-due-date"><strong>Due:</strong> Sunday at 11:59 pm (MT)</p>
"""

SYSTEM_PROMPT = (
    "You are an expert Canvas HTML editor. Preserve links, anchors/IDs, classes, and data-* attributes. "
    "Placeholders like ⟪IFRAME:n⟫ represent protected iframes—do not add, remove, or reorder them. "
    "Follow the policy. Return only HTML, no explanations."
)

SYSTEM_PROMPT_JSON = (
    "You are an HTML editor for Canvas LMS. You do not return HTML directly. "
    "You return ONLY JSON matching the provided schema for DOM edit operations. "
    "Preserve semantics and accessibility. "
    "Do not remove or reorder ⟪IFRAME:n⟫ placeholders. "
    "Prefer adding classes, wrapping elements, and inserting small snippets over rewriting entire sections."
)

EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "ops": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            "add_class",
                            "remove_class",
                            "replace_class",
                            "wrap",
                            "insert_before",
                            "insert_after",
                            "replace_inner_html",
                            "set_attr",
                            "remove_attr",
                        ],
                    },
                    "selector": {"type": "string"},
                    "value": {"type": "string"},
                    "html": {"type": "string"},
                },
                "required": ["op", "selector"],
                "additionalProperties": True,
            },
        }
    },
    "required": ["ops"],
    "additionalProperties": False,
}

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
            timeout=60,
        )

    attempt = 0
    last_exc = None
    models_to_try = [model]
    if fallback_model and fallback_model != model:
        models_to_try.append(fallback_model)

    for m in models_to_try:
        attempt = 0
        while attempt < max_retries:
            try:
                return _do_call(m)
            except Exception as e:
                last_exc = e
                delay = base_delay * (2**attempt) + random.random()
                time.sleep(delay)
                attempt += 1

    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown error in _create_with_retries")


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
        "reference_usage": "Match layout/sectioning/components; do not copy literal course-specific links or text.",
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
            prompt = "\n\n".join(blocks) + "\n\n(Note: image reference unavailable; ignore.)"
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

    txt = getattr(resp, "output_text", None)
    if not txt and getattr(resp, "output", None):
        try:
            first = resp.output[0]
            if hasattr(first, "content") and first.content:
                part = first.content[0]
                if hasattr(part, "text") and part.text:
                    txt = part.text
        except Exception:
            txt = None

    return txt or html


def openai_plan_edits(
    user_request: str,
    html: str,
    dt_mode: str,
    precision_directives: str,
    model_html_skeleton: Optional[str] = None,
    model_text_id: str = DEFAULT_TEXT_MODEL,
) -> dict:
    """
    Ask OpenAI for a JSON edit plan instead of full rewritten HTML.
    Returns a dict like {"ops": [...]}.
    """
    policy = {
        "design_tools_mode": dt_mode,
        "allow_inline_styles": True,
        "block_new_iframes": True,
        "reference_model": "html" if model_html_skeleton else "none",
        "reference_usage": "Match layout/sectioning/components; do not copy literal course-specific links or text.",
    }

    hard_rules_text = (
        "Hard rules (follow EXACTLY; return JSON only):\n"
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

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_JSON},
        {"role": "user", "content": "\n\n".join(blocks)},
    ]

    resp = _create_with_retries(
        client=openai_client,
        model=model_text_id,
        payload_input=messages,
        fallback_model="gpt-4.1",
        max_retries=3,
        base_delay=1.0,
    )

    txt = getattr(resp, "output_text", None)
    if not txt and getattr(resp, "output", None):
        try:
            # Responses API style content
            first = resp.output[0]
            if hasattr(first, "content") and first.content:
                part = first.content[0]
                if hasattr(part, "text") and part.text:
                    txt = part.text
        except Exception:
            txt = None

    try:
        plan = json.loads(txt) if txt else {}
    except Exception:
        plan = {}

    if not isinstance(plan, dict):
        plan = {}
    plan.setdefault("ops", [])
    return plan


def apply_ops(original_html: str, plan: dict) -> str:
    """
    Apply a simple JSON edit plan (ops with CSS selectors) to the HTML.
    """
    soup = BeautifulSoup(original_html or "", "html.parser")

    def safe_select(selector: str):
        try:
            return soup.select(selector)
        except Exception:
            return []

    for op in plan.get("ops", []):
        op_type = op.get("op")
        selector = op.get("selector", "")
        els = safe_select(selector)
        if not els:
            continue

        if op_type == "add_class":
            classes = (op.get("value") or "").split()
            for el in els:
                existing = el.get("class", []) or []
                el["class"] = list({*existing, *classes})

        elif op_type == "remove_class":
            classes = set((op.get("value") or "").split())
            for el in els:
                existing = el.get("class", []) or []
                el["class"] = [c for c in existing if c not in classes]

        elif op_type == "replace_class":
            classes = (op.get("value") or "").split()
            for el in els:
                el["class"] = classes

        elif op_type == "set_attr":
            # value like "data-kind=callout-info"
            raw = op.get("value") or ""
            if "=" in raw:
                k, v = raw.split("=", 1)
                for el in els:
                    el[k.strip()] = v.strip()

        elif op_type == "remove_attr":
            k = (op.get("value") or "").strip()
            for el in els:
                el.attrs.pop(k, None)

        elif op_type == "insert_before":
            frag = BeautifulSoup(op.get("html") or "", "html.parser")
            for el in els:
                el.insert_before(frag)

        elif op_type == "insert_after":
            frag = BeautifulSoup(op.get("html") or "", "html.parser")
            for el in els:
                el.insert_after(frag)

        elif op_type == "replace_inner_html":
            frag = BeautifulSoup(op.get("html") or "", "html.parser")
            for el in els:
                el.clear()
                for n in list(frag.contents):
                    el.append(n)

        elif op_type == "wrap":
            # value is a class string for a <div>
            classes = (op.get("value") or "").split()
            for el in els:
                wrapper = soup.new_tag("div")
                if classes:
                    wrapper["class"] = classes
                el.wrap(wrapper)

    return str(soup)


# ---------------------- Streamlit UI & Orchestration ----------------------


def _normalized_item_kind(kind: str) -> str:
    mapping = {
        "Page": "Page",
        "Assignment": "Assignment",
        "Discussion": "Discussion",
        "SubHeader": "SubHeader",
    }
    return mapping.get(kind, kind)


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

    return candidates[0] if candidates else f"{kind}:{item.id}", candidates


def _digest(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    st.set_page_config(page_title="Canvas Course HTML Rewriter (Test)", layout="wide")

    st.title("Canvas Course-wide HTML Rewriter (Test Instance)")
    st.markdown(
        """
This tool will pull HTML from a Canvas course (Pages, Assignment descriptions, Discussion topics),
run it through an AI rewriter with your instructions, and **show a side-by-side preview**.

Nothing is written back to Canvas until you explicitly click **Apply to Canvas** on an item.
        """
    )

    # Sidebar: connection + course
    st.sidebar.header("Connection & Course")
    st.sidebar.text(f"Canvas: {CANVAS_BASE_URL}")
    st.sidebar.text(f"OpenAI text model: {MODEL_TEXT}")
    st.sidebar.text(f"OpenAI vision model: {MODEL_VISION}")

    course_id = st.sidebar.text_input("Course ID", value="", help="Numeric Canvas course ID (test instance).")
    dt_mode = st.sidebar.selectbox("DesignTools Mode", DT_MODES, index=1)

    st.sidebar.header("Rewrite Controls")
    user_request = st.sidebar.text_area(
        "Rewrite goals",
        value="Use consistent headings, highlight due dates, and wrap learning objectives in a styled callout.",
        height=120,
    )

    precision_directives = st.sidebar.text_area(
        "Precision Directives (advanced)",
        value="",
        help="Optional micro-language: e.g., 'All h2 must have class dp-heading--h2' or 'Convert any IMPORTANT note into a callout'.",
    )

    st.sidebar.header("Model Reference (optional)")
    model_html = st.sidebar.text_area(
        "Paste Model HTML (optional)",
        value="",
        height=120,
        help="Paste representative HTML from a model page. The AI will try to mimic its structure/classes.",
    )

    model_image_data_url = None
    uploaded_image = st.sidebar.file_uploader(
        "Upload reference screenshot (optional)",
        type=["png", "jpg", "jpeg"],
        help="Screenshot of a model page. The AI may use it as a reference for layout and styling.",
    )
    if uploaded_image is not None:
        b = uploaded_image.read()
        b64 = base64.b64encode(b).decode("ascii")
        mime = "image/png" if uploaded_image.name.lower().endswith(".png") else "image/jpeg"
        model_image_data_url = f"data:{mime};base64,{b64}"
        st.sidebar.image(b, caption="Reference screenshot", use_container_width=True)

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("Index & Rewrite Course", type="primary")

    if "items" not in st.session_state:
        st.session_state["items"] = []
    if "processed_keys" not in st.session_state:
        st.session_state["processed_keys"] = set()
    if "index" not in st.session_state:
        st.session_state["index"] = []
    if "time_started" not in st.session_state:
        st.session_state["time_started"] = None

    if run_button and course_id.strip():
        try:
            cid_int = int(course_id.strip())
        except ValueError:
            st.error("Course ID must be an integer.")
            return

        course = get_course(cid_int)
        st.subheader(f"Course: {course.name} (ID: {course.id})")

        st.session_state["time_started"] = time.time()
        st.session_state["items"] = []
        st.session_state["processed_keys"] = set()

        # Index modules & items
        modules = get_course_modules(course)
        index = []
        for mod in modules:
            items = get_module_items(mod)
            for it in items:
                if it.type not in ("Page", "Assignment", "Discussion"):
                    continue
                html, meta = get_item_html(course, it)
                if not html:
                    continue
                key, candidates = _item_cache_candidates(it)
                index.append(
                    {
                        "key": key,
                        "course_id": course.id,
                        "module_id": mod.id,
                        "module_name": mod.name,
                        "item_id": it.id,
                        "kind": it.type,
                        "title": meta["title"],
                        "html": html,
                        "item": it,
                        "meta": meta,
                    }
                )

        st.session_state["index"] = index

        # Rewrite in batches
        st.info(f"Indexed {len(index)} items. Beginning rewrite in batches of {BATCH_SIZE}...")
        progress = st.progress(0.0)
        status = st.empty()

        model_html_skeleton = None
        if model_html.strip():
            model_html_skeleton = model_html.strip()

        offset = 0
        done = 0
        t_start = time.time()

        while offset < len(index):
            now = time.time()
            if now - t_start > TIME_BUDGET_SECONDS:
                st.warning("Time budget reached; stopping after current batch.")
                break

            batch = index[offset : offset + BATCH_SIZE]
            for record in batch:
                meta = record["meta"]
                module = type("ModuleStub", (), {"name": record["module_name"]})
                it = record["item"]
                key = f"{meta['kind']}:{meta['id']}"

                if key in st.session_state["processed_keys"]:
                    done += 1
                    denom = (len(index) - offset) if True else (
                        min(offset + BATCH_SIZE, len(index)) - offset
                    )
                    progress.progress(min(1.0, done / max(1, denom)))
                    continue

                original = meta["html"] or ""
                frozen, mapping, hosts = protect_iframes(original)

                prepped = frozen if len(frozen) <= MAX_INPUT_CHARS else (
                    frozen[:MAX_INPUT_CHARS] + "\n<!-- truncated for prompt -->"
                )

                t0 = time.time()
                try:
                    plan = openai_plan_edits(
                        user_request=user_request,
                        html=prepped,
                        dt_mode=dt_mode,
                        precision_directives=precision_directives,
                        model_html_skeleton=model_html_skeleton,
                        model_text_id=MODEL_TEXT,
                    )
                    rewritten = apply_ops(frozen, plan)
                except Exception as e:
                    rewritten = original
                    st.error(f"Rewrite (JSON plan) failed for [{meta.get('title') or meta.get('url')}] — {e}")

                rewritten_no_new_iframes = strip_new_iframes(rewritten)
                final_html = restore_iframes(rewritten_no_new_iframes, mapping)

                diff_html = html_diff(original, final_html) if GENERATE_DIFFS else "<div>Diff disabled for speed.</div>"

                st.session_state["items"].append(
                    {
                        "key": key,
                        "title": meta.get("title") or meta.get("url"),
                        "kind": meta["kind"],
                        "module": getattr(module, "name", ""),
                        "item": it,
                        "original": original,
                        "draft": final_html,
                        "diff": diff_html,
                        "elapsed": time.time() - t0,
                    }
                )
                st.session_state["processed_keys"].add(key)

                done += 1
                denom = len(index)
                progress.progress(min(1.0, done / max(1, denom)))
                status.text(f"Processed {done}/{len(index)} items...")

            offset += BATCH_SIZE

        st.success(f"Done. Processed {done} items.")

    # Show existing items (if any)
    if st.session_state["items"]:
        st.subheader("Review & Apply Changes")

        approve_all = st.button("Approve All")
        unapprove_all = st.button("Unapprove All")

        if "approvals" not in st.session_state:
            st.session_state["approvals"] = {}

        if approve_all:
            for item in st.session_state["items"]:
                st.session_state["approvals"][item["key"]] = True
        if unapprove_all:
            for item in st.session_state["items"]:
                st.session_state["approvals"][item["key"]] = False

        for item in st.session_state["items"]:
            key = item["key"]
            approved = st.session_state["approvals"].get(key, False)

            with st.expander(f"[{item['kind']}] {item['title']} — Module: {item['module']}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original HTML**")
                    st.code(item["original"], language="html")
                with col2:
                    st.markdown("**Draft HTML**")
                    st.code(item["draft"], language="html")

                st.markdown("**Diff**")
                st.markdown(item["diff"], unsafe_allow_html=True)

                approved = st.checkbox("Approve for Apply", value=approved, key=f"approve_{key}")
                st.session_state["approvals"][key] = approved

        if st.button("Apply Approved Changes to Canvas", type="primary"):
            applied = 0
            for item in st.session_state["items"]:
                key = item["key"]
                if not st.session_state["approvals"].get(key, False):
                    continue
                meta = _item_cache_meta_from_record(
                    {
                        "kind": item["kind"],
                        "key": key,
                        "title": item["title"],
                        "item": item["item"],
                    },
                    item["draft"],
                )
                update_item_html(meta, item["draft"])
                applied += 1
            st.success(f"Applied {applied} items to Canvas.")
    else:
        st.info("No items yet. Enter a course ID and click **Index & Rewrite Course** to begin.")


if __name__ == "__main__":
    main()
