import sys
import torch
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from transformers import AutoTokenizer, AutoProcessor
from qwenvl.models import Qwen25_SigLIPForConditionalGeneration


model_path = "/path/to/model"

model = Qwen25_SigLIPForConditionalGeneration.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",
    torch_dtype="auto",
    device_map="auto",
).eval()

processor = AutoProcessor.from_pretrained(model_path)
processor.tokenizer.padding_side = "left"


batch_messages = [
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/path/to/demo1.png"},
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/path/to/demo2.png"},
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ]
]


inputs = processor.apply_chat_template(
    batch_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    padding=True
)
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_texts = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

for i, text in enumerate(output_texts):
    print(f"--- Batch Sample {i+1} ---")
    print(text)
    print("\n")
