import json
import random
import os
import sys
import re

# Allow running from any directory by ensuring `src` is on the path
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_ROOT = os.path.join(_HERE, "..", "..")   # up to homework_W5
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from src.first_crew.crew import YelpPredictionCrew

def _balanced_json_at(s: str, start: int) -> str | None:
    """Return one top-level `{ ... }` starting at `start` (string-aware, balanced braces)."""
    if start < 0 or start >= len(s) or s[start] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    for j in range(start, len(s)):
        c = s[j]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start : j + 1]
    return None


def _iter_balanced_json_objects(s: str):
    """Yield every balanced `{...}` substring in document order (handles ```json and multiple blobs)."""
    pos = 0
    while pos < len(s):
        i = s.find("{", pos)
        if i == -1:
            break
        frag = _balanced_json_at(s, i)
        if frag:
            yield frag
            pos = i + len(frag)
        else:
            pos = i + 1


def _loads_json_lenient(json_str: str):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
        return json.loads(fixed)


def _normalize_prediction_dict(obj: dict) -> dict | None:
    """Turn Crew 'Final Answer' wrappers into `{stars, text}` if present."""
    if "input" in obj and isinstance(obj["input"], dict):
        inner = obj["input"]
        if ("stars" in inner or "rating" in inner) and (
            "text" in inner or "review" in inner or "summary" in inner
        ):
            return inner
    if ("stars" in obj or "rating" in obj) and (
        "text" in obj or "review" in obj or "summary" in obj
    ):
        return obj
    return None


def extract_json_robustly(text):
    """
    Parse prediction JSON from messy LLM output: markdown fences, multiple objects, wrappers.
    """
    if not text:
        return None

    best = None
    best_score = -1
    for js in _iter_balanced_json_objects(text):
        try:
            obj = _loads_json_lenient(js)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        cand = _normalize_prediction_dict(obj)
        if not cand:
            continue
        score = 0
        if cand.get("stars") is not None or cand.get("rating") is not None:
            score += 2
        tv = cand.get("text") or cand.get("review") or cand.get("summary")
        if isinstance(tv, str) and tv.strip():
            score += 2 + min(len(tv.strip()) // 200, 3)
        if score > best_score:
            best_score = score
            best = cand
    return best


def _aggregate_crew_output_text(result) -> str:
    """Combine all string outputs Crew may store (final raw + per-task raw + json_dict)."""
    if result is None:
        return ""
    parts: list[str] = []
    r = getattr(result, "raw", None)
    if r:
        parts.append(str(r))
    jd = getattr(result, "json_dict", None)
    if isinstance(jd, dict) and jd:
        try:
            parts.append(json.dumps(jd, ensure_ascii=False))
        except (TypeError, ValueError):
            parts.append(str(jd))
    for t in getattr(result, "tasks_output", None) or []:
        tr = getattr(t, "raw", None)
        if tr:
            parts.append(str(tr))
    return "\n".join(parts)

def run():
    """
    Run the crew using a randomly selected user-item pair from test_review_subset.json.
    """
    print("Starting run...", flush=True)
    test_data_path = os.path.join(os.path.dirname(__file__), '../../data/test_review_subset.json')
    print(f"Loading test reviews from {test_data_path}...", flush=True)
    
    try:
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_reviews = json.load(f)
        print(f"Loaded {len(test_reviews)} test reviews.", flush=True)
            
        if not test_reviews:
            print("No test reviews found in test_review_subset.json")
            return

        # Select a random entry
        random_entry = random.choice(test_reviews)
        selected_user_id = random_entry.get("user_id")
        selected_item_id = random_entry.get("item_id")
        
        print(f"Testing with User ID: {selected_user_id}", flush=True)
        print(f"Testing with Item ID: {selected_item_id}", flush=True)
        
        print("Initializing YelpPredictionCrew...", flush=True)

        inputs = {
            'user_id': selected_user_id,
            'item_id': selected_item_id
        }
        
        print("Kicking off crew...", flush=True)
        result = None
        try:
            result = YelpPredictionCrew().crew().kickoff(inputs=inputs)
        except Exception as e:
            print(f"\n[ERROR] Crew execution failed: {e}", flush=True)
            print("Attempting to recover partial results if possible...")
            raw_output = str(e)
            prediction = extract_json_robustly(raw_output)
        else:
            jd = getattr(result, "json_dict", None)
            if isinstance(jd, dict) and (
                jd.get("stars") is not None or jd.get("rating") is not None
            ):
                prediction = jd
            else:
                prediction = extract_json_robustly(_aggregate_crew_output_text(result))
        
        if prediction and isinstance(prediction, dict):
            # Normalization
            stars = prediction.get("stars") or prediction.get("rating") or 0.0
            text_val = prediction.get("text") or prediction.get("review") or prediction.get("summary") or "No text generated."
            
            if isinstance(text_val, dict):
                text_val = text_val.get("content") or text_val.get("text") or str(text_val)

            report = {
                "stars": float(stars),
                "text": str(text_val)
            }
            report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
            print(f"\n[SUCCESS] Final prediction saved to {report_path}", flush=True)
        else:
            print("\n[FAIL] Could not extract valid JSON from output. Check logs.", flush=True)
            
        print("\nExecution complete.", flush=True)
        
    except FileNotFoundError:
        print(f"Could not find {test_data_path}. Please ensure the paths are correct.")

if __name__ == '__main__':
    run()
