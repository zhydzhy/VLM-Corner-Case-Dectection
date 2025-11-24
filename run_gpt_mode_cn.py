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
# 可调参数
##################################
LOCAL_VLM_MODEL = "minicpm-v"   # 树莓派本地推荐：minicpm-v
OLLAMA_URL_CHAT = "http://localhost:11434/api/chat"
OLLAMA_URL_GENERATE = "http://localhost:11434/api/generate"
CAPTURES_ROOT = "./captures"
OUTPUT_DIR = "./output"
ONTOLOGY_FILE = "avcc_with_reasoning_no_shacl.ttl"


##################################
# 工具函数：读图 -> base64
##################################
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


##################################
# 从本体里提取类/属性概览，拼进 prompt
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
            line += f"\n  {comment}"
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
# 打分1：从 triples 里解析 avcco:hasConfidenceScore
##################################
def compute_avg_confidence_score(triples):
    confidence_scores = []
    current_subject = None
    for line in triples.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("@prefix") or line.startswith("#"):
            continue

        # 模型有时候叫 avcco:confidenceScore，我们也接受
        if ("hasConfidenceScore" not in line) and ("confidenceScore" not in line):
            continue

        # 如果该行以 ; 或 . 结尾，按 subject predicate object 切
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
            # continuation 行
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
# 打分2：天气加权 (可选; 如果分类器不可用就当1.0)
##################################
def compute_avg_classifier_score(image_paths):
    try:
        import weather_classifier_inference as uciclassifier
        classifier = uciclassifier.WeatherClassifier()
    except Exception as e:
        print("[WARN] WeatherClassifier 不可用，跳过天气权重:", e)
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
                print(f"[WARN] WeatherClassifier 处理失败 {image_path}: {e}")
                continue

    if valid_image_count > 0:
        return total_weighted_score / valid_image_count
    return 1.0


##################################
# 清洗 + 规范化 模型输出的 TTL 文本
##################################
def clean_and_normalize_ttl(ttl_text: str) -> str:
    """
    - 去掉高层 Situation 节点 (PerceptionFailureCase / EvidenceBasedAnomaly)
    - avcco:confidenceScore -> avcco:hasConfidenceScore
    - 加 provenance (VehicleA / vehicleA_obs_activity_1) 如果缺
    - 给每个 avcco:Observation 自动补 prov:wasGeneratedBy
    """
    if not ttl_text or not isinstance(ttl_text, str):
        return ""

    # 去掉代码块标记
    ttl_text = re.sub(r"```turtle", "", ttl_text)
    ttl_text = re.sub(r"```", "", ttl_text)

    # ex/ -> ex:
    ttl_text = ttl_text.replace("ex/", "ex:")

    # 删掉高阶块
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

    # 统一属性名
    ttl_text = ttl_text.replace("avcco:confidenceScore", "avcco:hasConfidenceScore")

    # provenance 块兜底
    if "vehicleA_obs_activity_1" not in ttl_text:
        provenance_block = """
ex:vehicleA_obs_activity_1 a prov:Activity ;
    prov:wasAssociatedWith ex:VehicleA .

ex:VehicleA a avcco:Vehicle , prov:Agent ;
    rdfs:label "VehicleA autonomous platform" .
""".strip()
        ttl_text = provenance_block + "\n" + ttl_text

    # Observation 块自动补 prov:wasGeneratedBy
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
                        + " ;\n    prov:wasGeneratedBy ex:vehicleA_obs_activity_1 ."
                    )
        fixed_blocks.append(block)

    ttl_text = "\n".join(fixed_blocks)

    return ttl_text.strip()


##################################
# 跟 Ollama (本地 VLM) 交互
##################################
def get_triples_from_llm(
    image_paths,
    prompt,
    model="minicpm-v",
    url_chat=OLLAMA_URL_CHAT,
    url_generate=OLLAMA_URL_GENERATE,
):
    """
    - llava / minicpm-v / phi 系列：用 /api/generate + images[]
    - qwen2.5vl 等：用 /api/chat + messages[]
    只送第一张图
    """
    rgb_image_paths = [
        p for p in image_paths
        if p.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    if not rgb_image_paths:
        print("[ERR] 没有可用的RGB图像")
        return ""
    first_img = rgb_image_paths[0]

    try:
        img_b64 = encode_image(first_img)
    except Exception as e:
        print(f"[ERR] 无法读取图像 {first_img}: {e}")
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
        print(f"[ERR] Ollama HTTP 请求失败: {e}")
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
# 找最新采集的图片
##################################
def get_paths_from_raspberry_pi(root=CAPTURES_ROOT):
    latest_frames_dir = os.path.join(root, "latest", "frames")
    if not os.path.isdir(latest_frames_dir):
        print("[ERR] 找不到摄像头帧目录:", latest_frames_dir)
        return []

    candidates = sorted(
        glob.glob(os.path.join(latest_frames_dir, "frame_*.jpg")) +
        glob.glob(os.path.join(latest_frames_dir, "frame_*.png")) +
        glob.glob(os.path.join(latest_frames_dir, "raw_*.jpg")) +
        glob.glob(os.path.join(latest_frames_dir, "raw_*.png"))
    )

    if len(candidates) == 0:
        print("[ERR] latest/frames 里没有匹配到任何jpg/png")
        return []

    picked = candidates[:5]
    print("[INFO] Using images:")
    for p in picked:
        print(" -", p)
    return picked


##################################
# 本地 VLM 路线 (Ollama)
##################################
def run_local_vlm(prefixes, ontology_prompt, model_name=LOCAL_VLM_MODEL):
    full_prompt = f"""
{prefixes}

You are a perception sensor for an autonomous vehicle named 'VehicleA'. Your sole function is to detect corner cases and occlusions and generate low-level observational triples based on the provided images.

CRITICAL INSTRUCTIONS:
1.  USE THE PROVIDED ONTOLOGY: You have been provided with the full AV Corner Case Ontology (AVCCO) and PROV-O ontology. This is your **only allowed vocabulary**. You must strictly use only the classes, properties, and relationships defined therein.
2.  Discover any possible corner cases and map it according to the AVCCO Ontology
3.  Discover any possible occlusion cases and map it according to the AVCCO Ontology
4.  PROVENANCE IS MANDATORY: Every observation **must** be explicitly attributed to this vehicle, 'VehicleA', using the PROV-O ontology.
5.  ONLY GENERATE OBSERVATIONS: You must ONLY generate instances of `avcco:Observation` and their properties.
6.  STRICTLY FORBIDDEN: You are ABSOLUTELY FORBIDDEN from generating any instance of a high-level `avcco:Situation` or any other class that represents a fused, interpreted event.
7.  ONTOLOGY COMPLIANCE: Use only properties and classes defined in the provided ontologies.
8.  CONFIDENCE: For each observation triple, estimate a confidence score (0.0–1.0) via `avcco:hasConfidenceScore`.
9.  OUTPUT FORMAT: Return only the RDF triples in Turtle format (strict N3 notation), using the provided prefixes.

How to implement provenance:
- For the overall activity of generating observations, create an instance of `prov:Activity` (e.g., `:vehicleA_obs_activity_1`).
- This activity was associated with the agent `:VehicleA` (an instance of `prov:Agent` or `avcco:Vehicle`).
- For each individual `avcco:Observation` you generate, assert that it was `generatedBy` this provenance activity.

Ontology reference:
{ontology_prompt}
""".strip()

    image_paths = get_paths_from_raspberry_pi(CAPTURES_ROOT)
    if not image_paths:
        print("[ERR] 没有图片，终止。")
        return

    raw_triples = get_triples_from_llm(
        image_paths=image_paths,
        prompt=full_prompt,
        model=model_name,
        url_chat=OLLAMA_URL_CHAT,
        url_generate=OLLAMA_URL_GENERATE,
    )
    if not raw_triples:
        print("[ERR] VLM没有返回TTL文本")
        return

    cleaned_triples = clean_and_normalize_ttl(raw_triples)
    if not cleaned_triples.strip().startswith("@prefix"):
        cleaned_triples = prefixes + "\n" + cleaned_triples

    # 打分
    avg_conf = compute_avg_confidence_score(cleaned_triples) or 0.0
    avg_weather = compute_avg_classifier_score(image_paths) or 1.0
    adjusted = avg_conf / avg_weather if avg_weather else avg_conf
    print(f"[SCORE] avg_conf={avg_conf:.3f}  weather_weight={avg_weather:.3f}  adjusted={adjusted:.3f}")

    # 尝试解析RDF
    g = Graph()
    parsed_ok = False
    try:
        g.parse(data=cleaned_triples, format="turtle")
        parsed_ok = True
        print(f"[INFO] RDF graph triples: {len(g)}")
    except Exception as e:
        print("[WARN] RDF parse 失败，仍然会落盘原始TTL:", e)

    # 写盘
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ttl_out = os.path.join(OUTPUT_DIR, "vehicle_A_observations_rpi.ttl")
    score_out = os.path.join(OUTPUT_DIR, "score_summary.json")

    if parsed_ok:
        g.serialize(destination=ttl_out, format="turtle")
        print(f"[OK] 已写入 RDF TTL 图: {ttl_out}")
    else:
        with open(ttl_out, "w") as f:
            f.write(cleaned_triples)
        print(f"[OK] 已写入未解析TTL文本: {ttl_out}")

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
    print(f"[OK] 分数写入: {score_out}")


##################################
# GPT 路线 (云)
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
        print("✅ GPT连接OK, 示例模型:", [m.id for m in models.data[:3]])
    except Exception as e:
        print("❌ GPT连接失败:", e)
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

    # 图像
    image_paths = get_paths_from_raspberry_pi(CAPTURES_ROOT)
    if not image_paths:
        print("[ERR] 没有图片，GPT模式结束。")
        return

    # prompt（和本地一致，保持 provenance / 低层 Observation 约束）
    full_prompt = f"""
{prefixes}

You are a perception sensor for an autonomous vehicle named 'VehicleA'. Your sole function is to detect corner cases and occlusions and generate low-level observational triples based on the provided images.

CRITICAL INSTRUCTIONS:
1.  USE THE PROVIDED ONTOLOGY: You have been provided with the full AV Corner Case Ontology (AVCCO) and PROV-O ontology. This is your **only allowed vocabulary**. You must strictly use only the classes, properties, and relationships defined therein.
2.  Discover any possible corner cases and map it according to the AVCCO Ontology
3.  Discover any possible occlusion cases and map it according to the AVCCO Ontology
4.  PROVENANCE IS MANDATORY: Every observation **must** be explicitly attributed to this vehicle, 'VehicleA', using the PROV-O ontology.
5.  ONLY GENERATE OBSERVATIONS: You must ONLY generate instances of `avcco:Observation` and their properties.
6.  STRICTLY FORBIDDEN: You are ABSOLUTELY FORBIDDEN from generating any instance of a high-level `avcco:Situation` or any other class that represents a fused, interpreted event.
7.  ONTOLOGY COMPLIANCE: Use only properties and classes defined in the provided ontologies.
8.  CONFIDENCE: For each observation triple, estimate a confidence score (0.0–1.0) via `avcco:hasConfidenceScore`.
9.  OUTPUT FORMAT: Return only the RDF triples in Turtle format (strict N3 notation), using the provided prefixes.

How to implement provenance:
- For the overall activity of generating observations, create an instance of `prov:Activity` (e.g., `:vehicleA_obs_activity_1`).
- This activity was associated with the agent `:VehicleA` (an instance of `prov:Agent` or `avcco:Vehicle`).
- For each individual `avcco:Observation` you generate, assert that it was `generatedBy` this provenance activity.

Ontology reference:
{ontology_prompt}
""".strip()

    raw_triples = get_triples_from_gpt(image_paths, full_prompt, client)
    if not raw_triples:
        print("[ERR] GPT没返回TTL")
        return

    cleaned_triples = clean_and_normalize_ttl(raw_triples)
    if not cleaned_triples.strip().startswith("@prefix"):
        cleaned_triples = prefixes + "\n" + cleaned_triples

    # 分数计算
    avg_conf = compute_avg_confidence_score(cleaned_triples) or 0.0
    avg_weather = compute_avg_classifier_score(image_paths) or 1.0
    adjusted = avg_conf / avg_weather if avg_weather else avg_conf
    print(f"[SCORE] avg_conf={avg_conf:.3f}  weather_weight={avg_weather:.3f}  adjusted={adjusted:.3f}")

    # RDF parse 尝试
    g = Graph()
    parsed_ok = False
    try:
        g.parse(data=cleaned_triples, format="turtle")
        parsed_ok = True
        print(f"[INFO] RDF graph triples: {len(g)}")
    except Exception as e:
        print("[WARN] RDF parse 失败(但还是会落盘):", e)

    # 输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ttl_out = os.path.join(OUTPUT_DIR, "vehicle_A_observations_rpi_gpt.ttl")
    score_out = os.path.join(OUTPUT_DIR, "score_summary_gpt.json")

    # TTL 落盘
    if parsed_ok:
        g.serialize(destination=ttl_out, format="turtle")
        print(f"[OK] GPT RDF TTL 写入: {ttl_out}")
    else:
        with open(ttl_out, "w") as f:
            f.write(cleaned_triples)
        print(f"[OK] GPT 原始TTL写入: {ttl_out}")

    # 分数落盘
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
    print(f"[OK] GPT 分数写入: {score_out}")


##################################
# main
##################################
def main(mode="ollama"):
    prefixes = """
@prefix avcco: <http://cornercase.org/avcco#> .
@prefix ex:    <http://cornercase.org/instances#> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
@prefix prov:  <http://www.w3.org/ns/prov#> .
@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:   <http://www.w3.org/2002/07/owl#> .
@prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
""".strip()

    ontology_path = os.path.join(os.path.dirname(__file__), ONTOLOGY_FILE)
    ontology_prompt = extract_ontology_prompt(ontology_path)

    if mode == "ollama":
        run_local_vlm(prefixes, ontology_prompt, model_name=LOCAL_VLM_MODEL)
    elif mode == "gpt":
        run_gpt_pipeline(prefixes, ontology_prompt)
    else:
        print("[ERR] mode 只能是 ollama 或 gpt")


if __name__ == "__main__":
    print("Starting vehicle A observation process...")
    # 你要调试 GPT 分支的话，改成 "gpt"
    main(mode="gpt")
