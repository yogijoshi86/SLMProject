"""
Generate benchmark_test_set_proto_only.json from benchmark_test_set (1).json.

Produces structured, deterministic explanations based purely on:
- matched prototype label and description
- cosine distance (embedding geometry)
- failure type (FP vs FN)
- recommended fix per prototype

No LLM API call required. Runs fully offline and is privacy-preserving.
"""

import json
from pathlib import Path

INPUT_PATH  = "/sapmnt/HOME/i852519/SAPDevelop/SLMProject/benchmark_test_set (1).json"
OUTPUT_PATH = "/sapmnt/HOME/i852519/SAPDevelop/SLMProject/benchmark_test_set_proto_only (1).json"

# ── Prototype metadata ────────────────────────────────────────────────────────

PROTOTYPE_META = {
    "prototype_0": {
        "label": "Persona and Role-Based Bypass",
        "description": (
            "Prompts that use a persona, fictional character, or role framing to bypass "
            "safety restrictions. Includes explicit jailbreak personas (Kevin, DAN, NRAF), "
            "academic/researcher role setups, and character descriptions designed to elicit "
            "restricted content."
        ),
        "failure_mode_fp": (
            "Guard over-triggers on legitimate role-play, academic research requests, or "
            "character descriptions that superficially resemble jailbreak patterns."
        ),
        "failure_mode_fn": (
            "Guard misses novel persona names not seen in training data, or subtle role "
            "assignments that do not explicitly state harmful intent."
        ),
        "fix_fp": (
            "Add domain-context exemptions for legitimate academic and creative role "
            "assignments. Distinguish 'roleplay for creative writing' from "
            "'roleplay to bypass restrictions'."
        ),
        "fix_fn": (
            "Add training examples for persona-override jailbreaks with varied character "
            "names. Flag any prompt that instructs the model to adopt an identity with "
            "'no restrictions' or equivalent."
        ),
    },
    "prototype_1": {
        "label": "Fictional Narrative Bypass",
        "description": (
            "Prompts that embed harmful requests inside story, roleplay, or creative writing "
            "framing. Uses 'write a story where...', 'in this fictional world...', or character "
            "dialogue to create distance from the harmful core request."
        ),
        "failure_mode_fp": (
            "Guard over-triggers on benign creative writing, romance fiction, fanfiction, or "
            "fantasy scenarios that share vocabulary with harmful narrative prompts."
        ),
        "failure_mode_fn": (
            "Guard misses harmful content when it is distributed across a long narrative "
            "context, or when the fictional framing is sufficiently elaborate to obscure "
            "the harmful core request."
        ),
        "fix_fp": (
            "Add benign creative writing examples to training. Distinguish narrative framing "
            "as decoration (FP) from narrative framing used as the bypass mechanism (TP)."
        ),
        "fix_fn": (
            "Flag prompts containing 'write a story where [harmful act]' regardless of "
            "fictional framing. Add training examples of harmful content embedded in "
            "elaborate fiction."
        ),
    },
    "prototype_2": {
        "label": "Direct Harmful Content Request",
        "description": (
            "Prompts that directly and explicitly request harmful content, dangerous "
            "information, or illegal material without any fictional or roleplay framing. "
            "Includes requests in multiple languages and requests phrased as "
            "'write about X' rather than 'how to do X'."
        ),
        "failure_mode_fp": (
            "Guard over-triggers on academic, journalistic, medical, or historical queries "
            "that use dangerous-sounding vocabulary in a legitimate context. High false "
            "positive rate on non-English requests."
        ),
        "failure_mode_fn": (
            "Guard misses requests that use euphemistic phrasing, indirect framing, or "
            "non-English languages where the harmful vocabulary differs from training data."
        ),
        "fix_fp": (
            "Add domain-context examples for medical, journalistic, and historical queries. "
            "Improve multilingual coverage to distinguish legitimate foreign-language "
            "requests from harmful ones."
        ),
        "fix_fn": (
            "Add training examples for indirect and euphemistic harmful requests. Expand "
            "non-English training data coverage, particularly for Romance languages."
        ),
    },
    "prototype_3": {
        "label": "Privacy and Sensitive Information Request",
        "description": (
            "Prompts involving real individuals, private data, medical information, financial "
            "details, or other sensitive personal information. Includes requests to write about "
            "named persons, interpret medical data, or access restricted information."
        ),
        "failure_mode_fp": (
            "Guard over-triggers on legitimate journalistic requests, medical queries, "
            "public figure analysis, and statistical questions that incidentally mention "
            "sensitive topics."
        ),
        "failure_mode_fn": (
            "Guard misses privacy violations when harmful intent is implicit, such as "
            "building a profile of a private individual across multiple seemingly innocuous "
            "requests."
        ),
        "fix_fp": (
            "Distinguish public figures from private individuals. Add exemptions for clearly "
            "journalistic, educational, or medical professional contexts."
        ),
        "fix_fn": (
            "Add training examples for indirect privacy violations and multi-step "
            "information aggregation attacks."
        ),
    },
}


# ── Explanation builder ───────────────────────────────────────────────────────

def build_prototype_explanation(case: dict) -> str:
    trt       = case["treatment"]
    proto_key = trt["matched_prototype"]
    proto     = PROTOTYPE_META.get(proto_key, {})
    failure   = case["failure_type"]
    label     = trt.get("prototype_label", proto_key)
    cos_dist  = trt.get("cosine_distance", "n/a")
    cos_sim   = trt.get("similarity_score", "n/a")
    categories = trt.get("guard_categories", [])
    cats_str  = ", ".join(categories) if categories else "n/a"
    prompt    = trt["input_text"]
    prompt_display = prompt[:200] + ("..." if len(prompt) > 200 else "")

    if failure == "false_positive":
        diagnosis = (
            f"FAILURE TYPE: False Positive — guard flagged a SAFE prompt as UNSAFE\n\n"
            f"ROOT CAUSE (prototype geometry):\n"
            f"  The prompt's hidden-state embedding landed in the\n"
            f"  \"{label}\" cluster (distance: {cos_dist}).\n\n"
            f"WHAT THIS PROTOTYPE CAPTURES:\n"
            f"  {proto.get('description', '')}\n\n"
            f"WHY THE GUARD OVER-TRIGGERED:\n"
            f"  {proto.get('failure_mode_fp', '')}\n\n"
            f"RECOMMENDED FIX:\n"
            f"  {proto.get('fix_fp', '')}"
        )
    else:
        diagnosis = (
            f"FAILURE TYPE: False Negative — guard missed an UNSAFE prompt (said SAFE)\n\n"
            f"ROOT CAUSE (prototype geometry):\n"
            f"  The prompt's nearest prototype is \"{label}\"\n"
            f"  (distance: {cos_dist}), but the guard did not fire.\n\n"
            f"WHAT THIS PROTOTYPE CAPTURES:\n"
            f"  {proto.get('description', '')}\n\n"
            f"WHY THE GUARD MISSED IT:\n"
            f"  {proto.get('failure_mode_fn', '')}\n\n"
            f"RECOMMENDED FIX:\n"
            f"  {proto.get('fix_fn', '')}"
        )

    return (
        f"=== PROTOTYPE-ONLY AUDIT EXPLANATION ===\n\n"
        f"PROMPT: \"{prompt_display}\"\n\n"
        f"GUARD DECISION:    [{trt['guard_decision']}]\n"
        f"CATEGORIES:        {cats_str}\n\n"
        f"MATCHED PROTOTYPE: {label} ({proto_key})\n"
        f"COSINE DISTANCE:   {cos_dist}  "
        f"(lower = stronger match to this failure pattern)\n\n"
        f"{diagnosis}"
    ).strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(INPUT_PATH) as f:
        bench = json.load(f)

    output = []
    for case in bench:
        proto_key = case["treatment"]["matched_prototype"]
        proto     = PROTOTYPE_META.get(proto_key, {})
        failure   = case["failure_type"]

        output.append({
            "case_id":      case["case_id"],
            "failure_type": failure,
            "control":      case["control"],
            "treatment_proto_only": {
                **{k: v for k, v in case["treatment"].items() if k != "explanation"},
                "explanation_type":    "prototype_only",
                "explanation":         build_prototype_explanation(case),
                "prototype_description": proto.get("description", ""),
                "failure_mode": proto.get(
                    "failure_mode_fp" if failure == "false_positive" else "failure_mode_fn", ""
                ),
                "recommended_fix": proto.get(
                    "fix_fp" if failure == "false_positive" else "fix_fn", ""
                ),
            },
            "treatment_llm": case["treatment"],
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(output)} cases → {OUTPUT_PATH}")
    print()
    print("=== Sample FP explanation ===")
    fp = next(c for c in output if c["failure_type"] == "false_positive")
    print(fp["treatment_proto_only"]["explanation"])
    print()
    print("=== Sample FN explanation ===")
    fn = next(c for c in output if c["failure_type"] == "false_negative")
    print(fn["treatment_proto_only"]["explanation"])


if __name__ == "__main__":
    main()
