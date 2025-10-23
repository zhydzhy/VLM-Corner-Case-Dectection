# This file contains the logic for the process
# Note: We need to implement a mechanism for continuous arrival of new images.
# Path of the images folder
#   images
#       - scenarios
#            - RGB
#            - LIDAR


import base64
import os
import re
import glob
from rdflib import Graph, RDF, RDFS, OWL
import weather_classifier_inference as uciclassifier
import re
import BEV_generator
import requests
import base64
import json

# === Function to encode image as base64 ===
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


# === Function to extract ontology summary as prompt ===
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


# === Function to extract confidence scores ===
def compute_avg_confidence_score(triples):
    confidence_scores = []
    current_subject = None
    for line in triples.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("@prefix") or line.startswith("#"):
            continue

        if "hasConfidenceScore" not in line:
            continue

        # Detect subject line
        # avcco:hasConfidenceScore "0.95"^^xsd:float ;
        # avcco:hasConfidenceScore "0.95"^^xsd:float .
        if line.endswith(";") or line.endswith("."):
            parts = line.split(" ", 2)
            if len(parts) == 3:
                s, p, o = parts
                s = s.strip().split(":")[-1]
                p = p.strip().split("^^")[0].strip('"')

                # Check for confidence score
                if s == "hasConfidenceScore":
                    try:
                        confidence_scores.append(float(p))
                    except ValueError:
                        pass
            current_subject = parts[0].strip().split(":")[-1]
        else:
            # Handle multiline continuation for the same subject
            parts = line.split(" ", 1)
            if len(parts) == 2 and current_subject:
                p, o = parts
                p = p.strip().split(":")[-1]
                o = o.strip("<>").strip('"')

                # Check for confidence score
                if p == "hasConfidenceScore":
                    try:
                        confidence_scores.append(float(o))
                    except ValueError:
                        pass

    # Calculate average LVLM confidence score
    return sum(confidence_scores) / len(confidence_scores) if confidence_scores else None


# === Function to average weighted score by using UCI classifier ===
# NOTE: Use the list of images we used to get the triples from the LLM
def compute_avg_classifier_score(image_paths):
    classifier = uciclassifier.WeatherClassifier()
    class_weights = {
        "Day": 1.00,
        "Night": 1.25,
        "Fog": 1.30
    }
    # The maximum weight should be less than 1.17

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
                print(f"Error processing {image_path}: {e}")
                continue

    return total_weighted_score / valid_image_count if valid_image_count > 0 else 0.0


def get_triples_from_llm(image_paths, prompt, model="qwen3-vl:235b-cloud"):
    # Prepare the image blocks
    rgb_image_paths = [
        path for path in image_paths if path.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    #print(rgb_image_paths)
    bev_image_path = None
    try:
        lidar_image_path = [
            path for path in image_paths if path.lower().endswith('.ply')
        ]
        if lidar_image_path:
            bev_image_path = BEV_generator.convert_ply_to_png(lidar_image_path[0])
    except Exception as e:
        print(f"Error generating BEV image: {e}")
        bev_image_path = None

    valid_images = rgb_image_paths.copy()
    if bev_image_path and os.path.exists(bev_image_path):
        valid_images.append(bev_image_path)

    if not valid_images:
        print("No valid image found for Ollama input.")
        return ""
    images_b64 = []
    for img_path in valid_images:
        img_b64 = encode_image(img_path)
        images_b64.append(img_b64)
    print(f"Encoded {len(images_b64)} images for Ollama input.")
    data = {
        "model": model,
        "prompt": prompt,
        "images": images_b64,
        "stream": False
    }

    # === Ollama API Call ===
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "images": [images_b64[0]],
        "stream": False
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        result = response.json()
        raw_output = result.get("response", "")
    except Exception as e:
        print(f"Ollama HTTP request failed: {e}")
        return ""

    print("=== RAW TTL ===")
    print(raw_output)

    # triples = open("raw_output.ttl", "w")
    # triples_file.write(raw_output.strip())
    # triples_file.close()

    # Extract triples from markdown code block if present
    # TODO: Handle other issues:
    # 1. ex/ to ex:
    # 2. Remove any text before or after the triples
    # 3. Handle both ```turtle and ```
    # 4. Handle spaces before comments.
    if "```turtle" in raw_output:
        triples = re.search('```turtle(.+?)```', raw_output, re.DOTALL)
        return triples.group(1).strip().replace("ex/", "ex:") if triples else raw_output.strip().replace("ex/", "ex:")
    elif "```" in raw_output:
        triples = re.search('```(.+?)```', raw_output, re.DOTALL)
        return triples.group(1).strip().replace("ex/", "ex:") if triples else raw_output.strip().replace("ex/", "ex:")
    else:
        return raw_output.strip().replace("ex/", "ex:") if raw_output else ""


# === Paths ===
scenarios_folder = r"/home/vlmteam/Qwen3-VLM-Detection/CARLA_DATASET_MULTI_AGENTS"
ttl_path = r"/home/vlmteam/Qwen3-VLM-Detection/avcc_with_reasoning_no_shacl.ttl"

main_graph = Graph()

# === Ontology & prompt setup ===
ontology_prompt = extract_ontology_prompt(ttl_path)

prefixes = """
@prefix avcco: <http://cornercase.org/avcco#> .
@prefix ex:    <http://cornercase.org/instances#> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
"""

prompt = f"""
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
"""

prompt2 = f"""
{prefixes}
We now give you RGP and LiDAR and 
You are a perception sensor for an autonomous vehicle named 'VehicleA'. Your sole function is to detect corner cases and occlusions and generate low-level observational triples based on the provided images and LiDAR.

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
- For the overall activity of generating observations, create an instance of `prov:Activity` (e.g., `:vehicleA`).
- This activity was associated with the agent `:VehicleA` (an instance of `prov:Agent` or `avcco:Vehicle`).
- For each individual `avcco:Observation` you generate, assert that it was `generatedBy` this provenance activity.

Ontology reference:
{ontology_prompt}
"""

# For each scenario in the root
for scenario in os.listdir(scenarios_folder):
    scenario_folder = os.path.join(scenarios_folder, scenario)
    if not os.path.isdir(scenario_folder):
        continue

    # For each weather in the scenario
    for weather in os.listdir(scenario_folder):
        main_graph = Graph()
        loop = 0  # first loop images start from 0 and 0 lidar images
        adjusted_score = 0.0

        weather_folder = os.path.join(scenario_folder, weather)
        if not os.path.isdir(weather_folder):
            continue

        vehicle_folder = os.path.join(weather_folder, "A")
        if not os.path.isdir(weather_folder):
            continue

        print(f"Processing scenario: {scenario}, weather: {weather}, vehicle: A")

        rgbs_folder = os.path.join(vehicle_folder, "rgb")
        if not os.path.isdir(rgbs_folder):  # Also check the folder i not empty
            continue
        lidar_images_folder = os.path.join(vehicle_folder, "lidar")
        if not os.path.isdir(lidar_images_folder):
            continue

        # Use 5 RGB from loop to get the confidence score
        rgb_images = sorted(
            glob.glob(os.path.join(rgbs_folder, "*.png")) +
            glob.glob(os.path.join(rgbs_folder, "*.jpg"))
        )
        # Use first LIDAR images upto loop to get the confidence score
        lidar_images = sorted(
            glob.glob(os.path.join(lidar_images_folder, "*.ply"))
        )

        while (loop * 5) < len(rgb_images):

            print(f"Loop: {loop}, RGB images: {len(rgb_images)}, LIDAR images: {len(lidar_images)}")

            # Get the next 5 RGB images and first loop LIDAR images
            try:
                rgb_images_selected = rgb_images[loop * 5: (loop * 5) + 5]
            except:
                rgb_images_selected = rgb_images[loop * 5:]

            # if len(rgb_images_selected) < 5:
            #     break
            lidar_images_selected = [] if loop == 0 else lidar_images[loop - 1:loop]

            selected_images = rgb_images_selected + lidar_images_selected

            # Call the LLM to process the images
            triples = get_triples_from_llm(selected_images, prompt if not lidar_images_selected else prompt2)
            print("=== TRIPLES ===")
            print(triples)

            # Add prefixes if not present
            if not triples.startswith("@prefix"):
                triples = prefixes + "\n" + triples

            avg_confidence_score = compute_avg_confidence_score(triples)
            if avg_confidence_score is None:
                avg_confidence_score = 0.0
            avg_classifier_score = compute_avg_classifier_score(selected_images)
            if avg_classifier_score is None:
                avg_classifier_score = 1.0

            print(f"Average Confidence Score: {avg_confidence_score}, Classifier Score: {avg_classifier_score}")
            # Calculate the total average score
            adjusted_score = avg_confidence_score / avg_classifier_score
            print("The adjusted score", adjusted_score)

            # parse the triples
            temp = Graph()
            temp.parse(data=triples, format='turtle')

            # Add to the main graph and exit the loop
            main_graph = main_graph + temp
            print(f"main graph has {len(main_graph)} triples.")

            loop_output_path = "/home/vlmteam/Qwen3-VLM-Detection/output"
            if not os.path.exists(loop_output_path):
                os.makedirs(os.path.join(loop_output_path))

            # Save the triples to a TTL file
            loop_output_file = os.path.join(loop_output_path, f"vehicle_A_observations_loop.ttl")
            temp.serialize(destination=loop_output_file, format='turtle')
            print(f"Loop graph with {len(temp)} triples saved to {loop_output_file}.")

            # Save the main graph to a TTL file
            main_output_file = os.path.join(loop_output_path, "vehicle_A_observations.ttl")
            main_graph.serialize(destination=main_output_file, format='turtle')
            print(f"Main graph with {len(main_graph)} triples saved to {main_output_file}.")

            # if adjusted_score >= 0.85:
            #     print(f"High confidence score: {adjusted_score}. Exiting the loop.")
            #     break
            # else:
            #     print(f"Low confidence score: {adjusted_score}. Continuing to next loop.")
            #     loop += 1
            #     continue

            print(f"Confidence score: {adjusted_score}. Continuing to next loop.")
            loop += 1
            continue

