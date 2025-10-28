#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import json
import time
import base64
import requests
from rdflib import Graph, RDF, RDFS, OWL

##################################
# Tunable parameters
##################################
LOCAL_VLM_MODEL = "minicpm-v"   # Recommended local model on Raspberry Pi
OLLAMA_URL_CHAT = "http://localhost:11434/api/chat"
OLLAMA_URL_GENERATE = "http://localhost:11434/api/generate"
CAPTURES_ROOT = "./captures"
OUTPUT_DIR = "./output"
ONTOLOGY_FILE = "avcc_with_reasoning_no_shacl.ttl"


##################################
# Utility: read image -> base64
##################################
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


##################################
# Build ontology summary (classes / properties) for prompt
##################################
def extract_ontology_prompt(ttl_path):
    g = Graph()
    g.parse(ttl_path, format="turtle")

    class_lines = ["Ontology Classes (and Hierarchy):"]
    for s in g.subjects(RDF.type, OWL.Class):
        label = g.value(s, RDFS.label)
        comment = g.value(s, RDFS.comment)
        subclass_of = g.value(s, RDFS.subClassOf)
        class_name = s.split("#")[-1] if "#" in s else s
        superclass = (
            subclass_of.split("#")[-1]
            if subclass_of and "#" in subclass_of
            else subclass_of
        )
        line = f"- {class_name}"
        if superclass:
            line += f" (subclass of {superclass})"
        if label:
            line += f": {label}"
        if comment:
            line += f"\n  {comment}"
        class_lines.append(line)

    property_lines = ["\nOntology Properties:"]
    for s in g.subjects(RDF.type, OWL.ObjectProperty):
        prop_name = s.split("#")[-1] if "#" in s else s
        property_lines.append(f"- {prop_name}")

    for s in g.subjects(RDF.type, OWL.DatatypeProperty):
        prop_name = s.split("#")[-1] if "#" in s else s
        property_lines.append(f"- {prop_name}")

    return "\n".join(class_lines + property_lines)


##################################
# Score 1: parse avcco:hasConfidenceScore from triples
##################################
def compute_avg_confidence_score(triples):
    confidence_scores = []
    current_subject = None
    for line in triples.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("@prefix") or line.startswith("#"):
            continue

        # Sometimes model uses avcco:confidenceScore instead of avcco:hasConfidenceScore, accept both
        if ("hasConfidenceScore" not in line) and ("confidenceScore" not in line):
            continue

        # If this line ends with ; or . we try to split as subject predicate object
        if line.endswith(";") or line.endswith("."):
            parts = line.split(" ", 2)
            if len(parts) == 3:
                s, p, o = parts
                pred_name = p.strip().split(":")[-1]
                obj_num = o.strip().split("^^")[0].strip('"; ')
                if pred_name in ["hasConfidenceScore", "confidenceScore"]:
                    try:
                        confidence_scores.append(float(obj_num))
                    except ValueError:
                        pass
            current_subject = parts[0].strip().split(":")[-1]
        else:
            # continuation line
            parts = line.split(" ", 1)
            if len(parts) == 2 and current_subject:
                p, o = parts
                pred_name = p.strip().split(":")[-1]
                obj_num = o.strip("<>").strip('"')
                if pred_name in ["hasConfidenceScore", "confidenceScore"]:
                    try:
                        confidence_scores.append(float(obj_num))
                    except ValueError:
                        pass

    if confidence_scores:
        return sum(confidence_scores) / len(confidence_scores)
    return None


##################################
# Score 2: weather weighting (optional; if classifier not available, return 1.0)
##################################
def compute_avg_classifier_score(image_paths):
    try:
        import weather_classifier_inference as uciclassifier
        classifier = uciclassifier.WeatherClassifier()
    except Exception as e:
        print("[WARN] WeatherClassifier unavailable, skipping weather weight:", e)
        return 1.0

    class_weights = {
        "Day": 1.00,
        "Night": 1.25,
        "Fog": 1.30
    }

    total_weighted_score = 0.0
    valid_image_count = 0
    for image_path in image_paths:
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                label = classifier.predict_image(image_path=image_path).strip()
                if label not in class_weights:
                    continue
                total_weighted_score += class_weights[label]
                valid_image_count += 1
            except Exception as e:
                print(f"[WARN] WeatherClassifier failed on {image_path}: {e}")
                continue

    if valid_image_count > 0:
        return total_weighted_score / valid_image_count
    return 1.0


##################################
# Clean + normalize TTL from model output
##################################
def clean_and_normalize_ttl(ttl_text: str) -> str:
    """
    - Drop high-level situation nodes (PerceptionFailureCase / EvidenceBasedAnomaly)
    - avcco:confidenceScore -> avcco:hasConfidenceScore
    - Add provenance block (VehicleA / vehicleA_obs_activity_1) if missing
    - For each avcco:Observation, ensure prov:wasGeneratedBy ex:vehicleA_obs_activity_1
    """
    if not ttl_text or not isinstance(ttl_text, str):
        return ""

    # remove code fences
    ttl_text = re.sub(r"```turtle", "", ttl_text)
    ttl_text = re.sub(r"```", "", ttl_text)

    # ex/ -> ex:
    ttl_text = ttl_text.replace("ex/", "ex:")

    # drop high-level situation blocks
    ttl_text = re.sub(
        r"ex:[^\s]+ rdf:type [^\n]*PerceptionFailureCase[^\n]*\.[\s\S]*?(?=(\nex:|$))",
        "",
        ttl_text,
        flags=re.MULTILINE,
    )
    ttl_text = re.sub(
        r"ex:[^\s]+ rdf:type [^\n]*EvidenceBasedAnomaly[^\n]*\.[\s\S]*?(?=(\nex:|$))",
        "",
        ttl_text,
        flags=re.MULTILINE,
    )

    # unify property naming
    ttl_text = ttl_text.replace("avcco:confidenceScore", "avcco:hasConfidenceScore")

    # ensure provenance block exists
    if "vehicleA_obs_activity_1" not in ttl_text:
        provenance_block = """
ex:vehicleA_obs_activity_1 a prov:Activity ;
    prov:wasAssociatedWith ex:VehicleA .

ex:VehicleA a avcco:Vehicle , prov:Agent ;
    rdfs:label "VehicleA autonomous platform" .
""".strip()
        ttl_text = provenance_block + "\n" + ttl_text

    # ensure each Observation block has prov:wasGeneratedBy
    blocks = []
    current = []
    for line in ttl_text.splitlines():
        if re.match(r"^\s*ex:[^ \t]+", line) and current:
            blocks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))

    fixed_blocks = []
    for block in blocks:
        if "avcco:Observation" in block:
            if "prov:wasGeneratedBy" not in block:
                block = block.rstrip()
                if block.endswith("."):
                    block = (
                        block[:-1]
                        + " ;\n    prov:wasGeneratedBy ex:vehicleA_obs_activity_1 ."
                    )
        fixed_blocks.append(block)

    ttl_text = "\n".join(fixed_blocks)

    return ttl_text.strip()


##################################
# Talk to Ollama (local VLM)
##################################
def get_triples_from_llm(
    image_paths,
    prompt,
    model="minicpm-v",
    url_chat=OLLAMA_URL_CHAT,
    url_generate=OLLAMA_URL_GENERATE,
):
    """
    - For llava / minicpm-v / phi style models: call /api/generate + images[]
    - For qwen2.5vl-like models: call /api/chat + messages[]
    We only send the first image path.
    """
    rgb_image_paths = [
        p for p in image_paths
        if p.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    if not rgb_image_paths:
        print("[ERR] No usable RGB image found.")
        return ""
    first_img = rgb_image_paths[0]

    try:
        img_b64 = encode_image(first_img)
    except Exception as e:
        print(f"[ERR] Failed to read image {first_img}: {e}")
        return ""

    model_lower = model.lower()
    use_generate_style = (
        "llava" in model_lower or
        "minicpm" in model_lower or
        "phi" in model_lower
    )

    if use_generate_style:
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False
        }
        url = url_generate
    else:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": img_b64},
                    ],
                }
            ],
            "stream": False
        }
        url = url_chat

    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        print(f"[ERR] Ollama HTTP request failed: {e}")
        return ""

    if "message" in result and "content" in result["message"]:
        raw_output = result["message"]["content"]
    else:
        raw_output = result.get("response", "")

    print("=== RAW OUTPUT FROM VLM ===")
    print(raw_output)

    if "```turtle" in raw_output:
        m = re.search(r"```turtle(.+?)```", raw_output, re.DOTALL)
        ttl_body = m.group(1).strip() if m else raw_output
    elif "```" in raw_output:
        m = re.search(r"```(.+?)```", raw_output, re.DOTALL)
        ttl_body = m.group(1).strip() if m else raw_output
    else:
        ttl_body = raw_output.strip()

    return ttl_body


##################################
# Get latest captured frames from Pi
##################################
def get_paths_from_raspberry_pi(root=CAPTURES_ROOT):
    latest_frames_dir = os.path.join(root, "latest", "frames")
    if not os.path.isdir(latest_frames_dir):
        print("[ERR] Cannot find captured frames directory:", latest_frames_dir)
        return []

    candidates = sorted(
        glob.glob(os.path.join(latest_frames_dir, "frame_*.jpg")) +
        glob.glob(os.path.join(latest_frames_dir, "frame_*.png")) +
        glob.glob(os.path.join(latest_frames_dir, "raw_*.jpg")) +
        glob.glob(os.path.join(latest_frames_dir, "raw_*.png"))
    )

    if len(candidates) == 0:
        print("[ERR] No jpg/png found in latest/frames")
        return []

    picked = candidates[:5]
    print("[INFO] Using images:")
    for p in picked:
        print(" -", p)
    return picked


##################################
# Local VLM pipeline (Ollama)
##################################
def run_local_vlm(prefixes, ontology_prompt, model_name=LOCAL_VLM_MODEL):
    full_prompt = f"""
{prefixes}

You are a perception sensor for an autonomous vehicle named 'VehicleA'. Your sole function is to detect corner cases and occlusions and generate low-level observational triples based on the provided images.

CRITICAL INSTRUCTIONS:
1.  USE THE PROVIDED ONTOLOGY: You have been provided with the full AV Corner Case Ontology (AVCCO) and PROV-O ontology. This is your **only allowed vocabulary**. You must strictly use only the classes, properties, and relationships defined therein.
2.  Discover any possible corner cases and map it according to the AVCCO Ontology
3.  Discover any possible occlusion cases and map it according to the AVCCO Ontology
4.  PROVENANCE IS MANDATORY: Every observation **must** be explicitly attributed to this vehicle, 'VehicleA', using the PROV-O ontology.
5.  ONLY GENERATE OBSERVATIONS: You must ONLY generate instances of `avcco:Observation` and their properties.
6.  STRICTLY FORBIDDEN: You are ABSOLUTELY FORBIDDEN from generating any instance of a high-level `avcco:Situation` or any other class that represents a fused, interpreted event.
7.  ONTOLOGY COMPLIANCE: Use only properties and classes defined in the provided ontologies.
8.  CONFIDENCE: For each observation triple, estimate a confidence score (0.0–1.0) via `avcco:hasConfidenceScore`.
9.  OUTPUT FORMAT: Return only the RDF triples in Turtle format (strict N3 notation), using the provided prefixes.

How to implement provenance:
- For the overall activity of generating observations, create an instance of `prov:Activity` (e.g., `:vehicleA_obs_activity_1`).
- This activity was associated with the agent `:VehicleA` (an instance of `prov:Agent` or `avcco:Vehicle`).
- For each individual `avcco:Observation` you generate, assert that it was `generatedBy` this provenance activity.

Ontology reference:
{ontology_prompt}
""".strip()

    image_paths = get_paths_from_raspberry_pi(CAPTURES_ROOT)
    if not image_paths:
        print("[ERR] No images available. Abort.")
        return

    raw_triples = get_triples_from_llm(
        image_paths=image_paths,
        prompt=full_prompt,
        model=model_name,
        url_chat=OLLAMA_URL_CHAT,
        url_generate=OLLAMA_URL_GENERATE,
    )
    if not raw_triples:
        print("[ERR] Local VLM returned no TTL text.")
        return

    cleaned_triples = clean_and_normalize_ttl(raw_triples)
    if not cleaned_triples.strip().startswith("@prefix"):
        cleaned_triples = prefixes + "\n" + cleaned_triples

    # scoring
    avg_conf = compute_avg_confidence_score(cleaned_triples) or 0.0
    avg_weather = compute_avg_classifier_score(image_paths) or 1.0
    adjusted = avg_conf / avg_weather if avg_weather else avg_conf
    print(f"[SCORE] avg_conf={avg_conf:.3f}  weather_weight={avg_weather:.3f}  adjusted={adjusted:.3f}")

    # try RDF parse
    g = Graph()
    parsed_ok = False
    try:
        g.parse(data=cleaned_triples, format="turtle")
        parsed_ok = True
        print(f"[INFO] RDF graph triples: {len(g)}")
    except Exception as e:
        print("[WARN] RDF parse failed. Will still write raw TTL:", e)

    # write outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ttl_out = os.path.join(OUTPUT_DIR, "vehicle_A_observations_rpi.ttl")
    score_out = os.path.join(OUTPUT_DIR, "score_summary.json")

    if parsed_ok:
        g.serialize(destination=ttl_out, format="turtle")
        print(f"[OK] Wrote RDF TTL graph: {ttl_out}")
    else:
        with open(ttl_out, "w") as f:
            f.write(cleaned_triples)
        print(f"[OK] Wrote raw TTL text (unparsed): {ttl_out}")

    with open(score_out, "w") as f:
        json.dump(
            {
                "avg_confidence": avg_conf,
                "avg_weather_weight": avg_weather,
                "adjusted_score": adjusted,
                "images_used": image_paths,
            },
            f,
            indent=2
        )
    print(f"[OK] Wrote score summary: {score_out}")


##################################
# GPT pipeline (cloud)
##################################
def connect_to_gpt():
    from dotenv import load_dotenv
    from openai import OpenAI

    env_path = os.path.join(os.path.dirname(__file__), ".env.api_key")
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("API_KEY")

    client = OpenAI(api_key=api_key)
    try:
        models = client.models.list()
        print(" GPT connection OK, sample models:", [m.id for m in models.data[:3]])
    except Exception as e:
        print(" GPT connection failed:", e)
    return client


def get_triples_from_gpt(image_paths, prompt, client):
    rgb_image_paths = [
        p for p in image_paths
        if p.lower().endswith(('.png', '.jpg', '.jpeg'))
    ][:2]

    rgb_blocks = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image(path)}"
            },
        }
        for path in rgb_image_paths
    ]

    content = [{"type": "text", "text": prompt}] + rgb_blocks

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": content}],
        max_tokens=1000
    )
    raw_output = response.choices[0].message.content
    print("=== RAW OUTPUT FROM GPT ===")
    print(raw_output)

    if "```turtle" in raw_output:
        m = re.search(r"```turtle(.+?)```", raw_output, re.DOTALL)
        ttl_body = m.group(1).strip() if m else raw_output
    elif "```" in raw_output:
        m = re.search(r"```(.+?)```", raw_output, re.DOTALL)
        ttl_body = m.group(1).strip() if m else raw_output
    else:
        ttl_body = raw_output.strip()

    return ttl_body


def run_gpt_pipeline(prefixes, ontology_prompt):
    client = connect_to_gpt()

    # images
    image_paths = get_paths_from_raspberry_pi(CAPTURES_ROOT)
    if not image_paths:
        print("[ERR] No images available. GPT branch aborted.")
        return

    # prompt body aligned with local version (provenance, Observation-only, etc.)
    full_prompt = f"""
{prefixes}

You are a perception sensor for an autonomous vehicle named 'VehicleA'. Your sole function is to detect corner cases and occlusions and generate low-level observational triples based on the provided images.

CRITICAL INSTRUCTIONS:
1.  USE THE PROVIDED ONTOLOGY: You have been provided with the full AV Corner Case Ontology (AVCCO) and PROV-O ontology. This is your **only allowed vocabulary**. You must strictly use only the classes, properties, and relationships defined therein.
2.  Discover any possible corner cases and map it according to the AVCCO Ontology
3.  Discover any possible occlusion cases and map it according to the AVCCO Ontology
4.  PROVENANCE IS MANDATORY: Every observation **must** be explicitly attributed to this vehicle, 'VehicleA', using the PROV-O ontology.
5.  ONLY GENERATE OBSERVATIONS: You must ONLY generate instances of `avcco:Observation` and their properties.
6.  STRICTLY FORBIDDEN: You are ABSOLUTELY FORBIDDEN from generating any instance of a high-level `avcco:Situation` or any other class that represents a fused, interpreted event.
7.  ONTOLOGY COMPLIANCE: Use only properties and classes defined in the provided ontologies.
8.  CONFIDENCE: For each observation triple, estimate a confidence score (0.0–1.0) via `avcco:hasConfidenceScore`.
9.  OUTPUT FORMAT: Return only the RDF triples in Turtle format (strict N3 notation), using the provided prefixes.

How to implement provenance:
- For the overall activity of generating observations, create an instance of `prov:Activity` (e.g., `:vehicleA_obs_activity_1`).
- This activity was associated with the agent `:VehicleA` (an instance of `prov:Agent` or `avcco:Vehicle`).
- For each individual `avcco:Observation` you generate, assert that it was `generatedBy` this provenance activity.

Ontology reference:
{ontology_prompt}
""".strip()

    raw_triples = get_triples_from_gpt(image_paths, full_prompt, client)
    if not raw_triples:
        print("[ERR] GPT returned no TTL text.")
        return

    cleaned_triples = clean_and_normalize_ttl(raw_triples)
    if not cleaned_triples.strip().startswith("@prefix"):
        cleaned_triples = prefixes + "\n" + cleaned_triples

    # scoring
    avg_conf = compute_avg_confidence_score(cleaned_triples) or 0.0
    avg_weather = compute_avg_classifier_score(image_paths) or 1.0
    adjusted = avg_conf / avg_weather if avg_weather else avg_conf
    print(f"[SCORE] avg_conf={avg_conf:.3f}  weather_weight={avg_weather:.3f}  adjusted={adjusted:.3f}")

    # RDF parse attempt
    g = Graph()
    parsed_ok = False
    try:
        g.parse(data=cleaned_triples, format="turtle")
        parsed_ok = True
        print(f"[INFO] RDF graph triples: {len(g)}")
    except Exception as e:
        print("[WARN] RDF parse failed (will still write TTL):", e)

    # write outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ttl_out = os.path.join(OUTPUT_DIR, "vehicle_A_observations_rpi_gpt.ttl")
    score_out = os.path.join(OUTPUT_DIR, "score_summary_gpt.json")

    # write TTL
    if parsed_ok:
        g.serialize(destination=ttl_out, format="turtle")
        print(f"[OK] Wrote GPT RDF TTL: {ttl_out}")
    else:
        with open(ttl_out, "w") as f:
            f.write(cleaned_triples)
        print(f"[OK] Wrote GPT raw TTL (unparsed): {ttl_out}")

    # write score json
    with open(score_out, "w") as f:
        json.dump(
            {
                "avg_confidence": avg_conf,
                "avg_weather_weight": avg_weather,
                "adjusted_score": adjusted,
                "images_used": image_paths,
            },
            f,
            indent=2
        )
    print(f"[OK] Wrote GPT score summary: {score_out}")


##################################
# main
##################################
def main(mode="ollama"):
    prefixes = """
@prefix avcco: <http://cornercase.org/avcco#> .
@prefix ex:    <http://cornercase.org/instances#> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
@prefix prov:  <http://www.w3.org/ns/prov#> .
@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:   <http://www.w3.org/2002/07/owl#> .
@prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
""".strip()

    ontology_path = os.path.join(os.path.dirname(__file__), ONTOLOGY_FILE)
    ontology_prompt = extract_ontology_prompt(ontology_path)

    if mode == "ollama":
        run_local_vlm(prefixes, ontology_prompt, model_name=LOCAL_VLM_MODEL)
    elif mode == "gpt":
        run_gpt_pipeline(prefixes, ontology_prompt)
    else:
        print("[ERR] mode must be 'ollama' or 'gpt'")


if __name__ == "__main__":
    print("Starting vehicle A observation process...")
    # change to "gpt" if you want to test GPT branch
    main(mode="gpt")
