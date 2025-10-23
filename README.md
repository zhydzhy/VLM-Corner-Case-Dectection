# Vision-Language Corner-Case Detection System

This repository implements a multimodal reasoning pipeline for **autonomous vehicle corner-case detection**, combining RGB camera and LiDAR data with ontology-based reasoning.  
The project was originally developed by **Malik@malikluti** and extended by me with **Ollama API integration** to enable local inference and offline deployment.

---

## Overview

The system processes both **RGB** and **LiDAR (.ply)** inputs from CARLA simulation or real sensors, converts LiDAR to Bird’s Eye View (BEV) images, and sends them to a multimodal vision-language model for reasoning.  
The model produces ontology-compliant observations using the AVCCO (Autonomous Vehicle Corner Case Ontology) and PROV-O vocabularies, enabling structured understanding of corner cases and occlusions.

---

## Author and Contributions

- **Original Author:** Malik Luti, Ian Harshbarger 
- **Ollama Integration and Documentation:** Haotian Zhang   
- **Base Models:** Qwen3-cloud

If you use or extend this repository, please cite Malik & Ian’s original work.

---

## Migration Summary: From ChatGPT API to Ollama API

| Component | Previous Implementation (ChatGPT API) | Updated Implementation (Ollama API) | Description |
|------------|----------------------------------------|--------------------------------------|--------------|
| Backend | `openai.ChatCompletion.create()` | `ollama.chat()` | Replaced cloud-based OpenAI API with local Ollama inference |
| Model | `"gpt-4-vision-preview"` or `"gpt-4o-mini"` | `"minicpm-v"` or `"llava-phi3"` | Migrated to open-source multimodal models |
| Image Handling | Base64-encoded image blocks | Base64-encoded image Local image  | Converted Image Block into Json Data |
| Authentication | Requires `OPENAI_API_KEY` | No key required | Ollama runs locally with no network dependency |
| Deployment | Remote API requests | Local | Enables fully offline evaluation |
| Latency | Network-dependent | Device-dependent | Eliminated cloud latency for real-time testing |

---

## Requirements

### System

- Python 3.9 or newer  
- Linux, macOS, or Windows (Ubuntu 22.04 recommended)  
- Optional GPU acceleration (8 GB VRAM or higher recommended)

### Python Dependencies

- `ollama`
- `opencv-python`
- `numpy`
- `pillow`
- `tqdm`
- `open3d`
- `requests`

These can be installed using `pip install -r requirements.txt` if provided, or manually as shown below.

```bash
pip install ollama opencv-python numpy pillow tqdm open3d requests ```
```

---

## Setup and Reproduction Guide

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/VLM-Corner-Case-Detection.git
cd VLM-Corner-Case-Detection
```

### 2. Create and Activate a Virtual Environment
``` bash
python3 -m venv env
source env/bin/activate
```
### 3. Install Dependencies

``` bash
pip install -r requirements.txt
```

If ```requirements.txt``` is not provided, you can manually install them:

``` bash
pip install ollama opencv-python numpy pillow tqdm open3d requests
```
### 4. Install and Run Ollama
Install Ollama for local model inference:
``` bash
curl -fsSL https://ollama.ai/install.sh | sh
```

Then Start the service:
```bash
ollama serve
```
You can verify the installation with:
```bash
ollama list
```

### Pull a Supported Vision-Language Model

Choose a model to use for multimodal inference:
```bash
ollama pull qwen3
```
Alternative options include:
```bash
ollama pull minicpm-v
ollama pull llava:latest
ollama pull minicpm-o
```

# Running the Pipeline
## Example Command

```bash
python vehicle_A_observation.py
```

### Process Overview

1. Converts the LiDAR .ply file to a BEV image using bev_generator.py.

2. Encodes both RGB and BEV images for model input.

3. Sends the data to the selected Ollama model via ```python get_triples_from_llm()``` fucntion.

4. Receives structured observations following AVCCO and PROV-O ontologies.

### Expected Output Example

```
@prefix avcco: <http://cornercase.org/avcco#> .
@prefix ex: <http://cornercase.org/instances#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:obs_blindspot_1 a avcco:Observation ;
    avcco:hasConfidenceScore "0.7"^^xsd:float ;
    avcco:hasSensorEvidence ex:camera_feed_1 ;
    avcco:hasTriggerEvent ex:occlusion_event_3 ;
    prov:wasGeneratedBy ex:vehicleA_obs_activity_1 .

ex:obs_clearvis_1 a avcco:Observation ;
    avcco:hasConfidenceScore "0.99"^^xsd:float ;
    avcco:hasSensorEvidence ex:camera_feed_1 ;
    prov:wasGeneratedBy ex:vehicleA_obs_activity_1 .

ex:obs_occlusion_1 a avcco:Observation ;
    avcco:hasConfidenceScore "0.85"^^xsd:float ;
    avcco:hasSensorEvidence ex:camera_feed_1 ;
    avcco:hasTriggerEvent ex:occlusion_event_1 ;
    prov:wasGeneratedBy ex:vehicleA_obs_activity_1 .

ex:obs_occlusion_2 a avcco:Observation ;
    avcco:hasConfidenceScore "0.8"^^xsd:float ;
    avcco:hasSensorEvidence ex:camera_feed_1 ;
    avcco:hasTriggerEvent ex:occlusion_event_2 ;
    prov:wasGeneratedBy ex:vehicleA_obs_activity_1 .

ex:VehicleA a avcco:AutonomousVehicle,
        prov:Agent .

ex:adjacent_crosswalk_1 a avcco:RoadElement .

ex:bus_stop_shelter_1 a avcco:RoadElement .

ex:intersection_far_right_1 a avcco:RoadElement .

ex:median_shelter_1 a avcco:RoadElement .

ex:moving_van_right_1 a avcco:Vehicle .

ex:occlusion_event_1 a avcco:OcclusionEvent ;
    avcco:hasOccludedEntity ex:sidewalk_pedestrian_zone_1 ;
    avcco:hasOccluder ex:bus_stop_shelter_1,
        ex:parked_vehicle_left_1 .

ex:occlusion_event_2 a avcco:OcclusionEvent ;
    avcco:hasOccludedEntity ex:intersection_far_right_1 ;
    avcco:hasOccluder ex:moving_van_right_1 .

ex:occlusion_event_3 a avcco:OcclusionEvent ;
    avcco:hasOccludedEntity ex:adjacent_crosswalk_1 ;
    avcco:hasOccluder ex:median_shelter_1 .

ex:parked_vehicle_left_1 a avcco:ParkedVehicle .

ex:sidewalk_pedestrian_zone_1 a avcco:RoadElement .

ex:camera_feed_1 a avcco:CameraFeed .

ex:vehicleA_obs_activity_1 a prov:Activity ;
    prov:wasAssociatedWith ex:VehicleA .

```
## Troubleshooting

| Problem | Likely Cause | Solution |
|----------|---------------|-----------|
| **BadRequest: request body too large** | Too many or large image inputs sent to Ollama | Reduce batch size or downscale images before encoding |
| **Connection refused** | Ollama service not started | Run `ollama serve` in a separate terminal before executing `main.py` |
| **Model not found** | Model not downloaded or misnamed | Run `ollama pull <model_name>` to fetch the model locally |
| **GPU not used** | Missing GPU drivers or Ollama not compiled with GPU support | Verify CUDA or ROCm installation and GPU availability |
| **Slow inference speed** | Large model or high-resolution images | Use a smaller Ollama model or reduce image dimensions |
| **Empty or malformed output** | Incorrect prompt or ontology path | Confirm ontology `.ttl` files exist and prompts follow AVCCO/PROV-O syntax |
| **FileNotFoundError (LiDAR or Image)** | Incorrect relative paths | Verify `data/images/` and `data/lidar/` directories exist and contain valid files |
| **OSError: connection timeout** | Ollama server not reachable | Check host address and firewall if running on a remote server |

---

## License

This project is distributed under the **MIT License**.  
Please credit **Malik Luti** and **Ian Harshbarger** as the original author and **Haotian Zhang** for the Ollama integration and extended documentation.

---

## Acknowledgments

- **UCI IAS Lab** for research guidance on ontology-based reasoning and CARLA integration  
- **Qwen** and **Ollama** teams for enabling local multimodal inference  

---

## Closing Note

This repository showcases a transition from **cloud-based ChatGPT inference** to **local Ollama deployment**, enabling reproducible and offline multimodal reasoning.  
It provides a reference framework for integrating **vision, LiDAR, and ontology reasoning** in autonomous vehicle perception pipelines.

Future directions include:
- Expanding ontology reasoning for dynamic multi-agent scenarios  
- Integrating real-time ROS-based sensor streams  
- Optimizing model performance for edge hardware such as Raspberry Pi 5 or NVIDIA Jetson  

By following the setup instructions and environment configuration above, other researchers and developers can clone, reproduce, and extend this project for their own applications in **autonomous systems** and **multimodal AI**.
