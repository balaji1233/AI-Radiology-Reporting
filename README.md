# AI-Radiology-Reporting
MAIRA-2   multimodal transformer designed for the generation of grounded or non-grounded radiology reports from chest X-rays.


## Overview
**MAIRA‑2** is a cutting-edge, multimodal model designed to generate detailed chest X‑ray reports with spatial grounding. This project isn’t just another coding exercise—it’s a comprehensive solution addressing real clinical challenges by enhancing diagnostic efficiency and report accuracy.

## Problem Statement
- **The Challenge:** Radiology reporting is time‑consuming and error‑prone due to increasing imaging service demand and radiologist shortages.
- **The Impact:** Inconsistent report quality can delay patient care, increase staff workload, and potentially compromise clinical outcomes.

## Our Solution
- **Innovative Approach:** Integrates a frozen Rad‑DINO‑MAIRA‑2 image encoder, a custom projection layer, and a fine‑tuned Vicuna‑7B v1.5 language model.
- **Grounded Reporting:** Generates both narrative and spatially annotated (bounding box) reports to make findings verifiable and clinically actionable.
- **Comprehensive Context:** Leverages multiple inputs—current frontal and lateral images, prior studies, and clinical context (Indication, Technique, Comparison)—to mimic expert radiologist assessments.

## Key Features
- **Automated Report Generation:** Produces expert-level, first-draft radiology reports.
- **Advanced Localization:** Links each reported finding with precise spatial annotations.
- **Robust Evaluation Framework (RadFact):**
  - **RadFact Logical Precision:** Achieves >70% on key benchmarks.
  - **RadFact Grounding Recall:** Up to 90%, ensuring high coverage of relevant image areas.
- **State-of-the-Art Performance:** Outperforms previous models with up to a 30% boost in BLEU‑4 scores and significant gains on clinical metrics (RadGraph‑F1, CheXbert F1).

## Quantifiable Impact
- **Efficiency Gains:** Reduction in report processing errors and omissions.
- **Clinical Accuracy:** Expert reviews indicate over 90% of generated sentences are acceptable as a first draft.
- **Benchmark Results:** Demonstrated improvements on datasets like MIMIC‑CXR, PadChest‑GR, and IU‑Xray.

## Installation & Setup
1.To run this sample code, you will need the following packages:
```

pillow
protobuf
sentencepiece
torch
transformers

```
Note: You may temporarily need to install transformers from source since MAIRA-2 requires transformers>=4.46.0.dev0. 

## Set Up Environment:
Install required dependencies (e.g., Python, PyTorch, Transformers).
Refer to the Installation Guide for detailed instructions.

```
pip install git+https://github.com/huggingface/transformers.git@88d960937c81a32bfb63356a2e8ecf7999619681
```
First, initialise the model and put it in eval mode

```
from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path
import torch

model = AutoModelForCausalLM.from_pretrained("microsoft/maira-2", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/maira-2", trust_remote_code=True)

device = torch.device("cuda")
model = model.eval()
model = model.to(device)
```

We need to get some data to demonstrate the forward pass. For this example, we'll collect an example from the IU X-ray dataset, which has a permissive license.

```
import requests
from PIL import Image

def get_sample_data() -> dict[str, Image.Image | str]:
    """
    Download chest X-rays from IU-Xray, which we didn't train MAIRA-2 on. License is CC.
    We modified this function from the Rad-DINO repository on Huggingface.
    """
    frontal_image_url = "https://openi.nlm.nih.gov/imgs/512/145/145/CXR145_IM-0290-1001.png"
    lateral_image_url = "https://openi.nlm.nih.gov/imgs/512/145/145/CXR145_IM-0290-2001.png"

    def download_and_open(url: str) -> Image.Image:
        response = requests.get(url, headers={"User-Agent": "MAIRA-2"}, stream=True)
        return Image.open(response.raw)

    frontal_image = download_and_open(frontal_image_url)
    lateral_image = download_and_open(lateral_image_url)

    sample_data = {
        "frontal": frontal_image,
        "lateral": lateral_image,
        "indication": "Dyspnea.",
        "comparison": "None.",
        "technique": "PA and lateral views of the chest.",
        "phrase": "Pleural effusion."  # For the phrase grounding example. This patient has pleural effusion.
    }
    return sample_data

sample_data = get_sample_data()
```
## Training details
We did not originally train MAIRA-2 using the exact model class provided here, however we have checked that its behaviour is the same. We provide this class to facilitate research re-use and inference.

## Training data
MAIRA-2 was trained on a mix of public and private chest X-ray datasets. Each example comprises one or more CXR images and associated report text, with or without grounding (spatial annotations). The model is trained to generate the findings section of the report, with or without grounding.
| Dataset        | Country        | # examples (ungrounded)     | # examples (grounded) |
| :---           |     :---:      |          ---: |              ---:     |
| MIMIC-CXR      | USA            |55 218         |          595*         |
| PadChest       | Spain          |52828          |      3122             |
|USMix (Private) | USA            |118031         | 53613                 |
*We use the MS-CXR phrase grounding dataset to provide `grounding' examples from MIMIC-CXR.
## Documentation
[Maira 2](https://huggingface.co/microsoft/maira-2#how-to-get-started-with-the-model)

## Architecture Details
MAIRA-2 is composed of the image encoder RAD-DINO-MAIRA-2 (used frozen), a projection layer (trained from scratch), and the language model vicuna-7b-v1.5 (fully fine-tuned).

## Acknowledgements

Data Providers: MIMIC‑CXR, PadChest, USMix, and IU‑Xray.

