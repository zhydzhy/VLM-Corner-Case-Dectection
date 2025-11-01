import os
import glob
from PIL import Image
import torch
from rdflib import Graph
from transformers import (
    AutoProcessor,
    AutoConfig,
    AutoModelForVision2Seq,
    AutoModelForCausalLM
)
import torch
os.environ["FLASH_ATTENTION_DISABLE"] = "1"
os.environ["USE_FLASH_ATTENTION"] = "0"
class UniversalVisionModel:
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(model_name)

        # auto-select model type
        if "Phi3V" in self.config.__class__.__name__:
            print("Detected Phi-3 Vision model → using AutoModelForCausalLM")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model_type = "causal"
        else:
            print("Detected Vision2Seq model → using AutoModelForVision2Seq")
            self.model = AutoModelForVision2Seq.from_pretrained(model_name).to(self.device)
            self.model_type = "v2s"

    def run_prompt(self, image, text_prompt):
        if self.model_type == "v2s":
            inputs = self.processor(image, text_prompt, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=512)
            return self.processor.decode(out[0], skip_special_tokens=True)
        else:  # causal (Phi, Llava, etc.)
            inputs = self.processor(text_prompt, images=image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=512)
            return self.processor.batch_decode(out, skip_special_tokens=True)[0]
