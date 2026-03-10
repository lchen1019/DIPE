import sys
import torch
import json

from tqdm import tqdm
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from transformers import AutoTokenizer, AutoProcessor
from qwenvl.models import Qwen25_SigLIPForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_path = "/path/to/model"
question_path = "/path/to/questions"

model = Qwen25_SigLIPForConditionalGeneration.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",
    torch_dtype="auto",
    device_map="auto",
).eval()

# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_path,
#     attn_implementation="flash_attention_2",
#     torch_dtype="auto",
#     device_map="auto",
# ).eval()

processor = AutoProcessor.from_pretrained(model_path)
processor.tokenizer.padding_side = "left"

with open(question_path, 'r', encoding='utf-8') as f:
    questions = json.load(f)

cnt = 0
cnt_img2txt = 0
cnt_txt2img = 0
for item in tqdm(questions):
    question = item['messages']
    answer = item['answer']
    tag = item['tag']

    inputs = processor.apply_chat_template(
        question,
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

    pred = output_texts[0]
    print(pred, answer)

    if pred[0] == answer:
        cnt += 1
    
    if tag == 'text_to_image':
        cnt_txt2img += 1
    else:
        cnt_img2txt += 1


print('acc', cnt / len(questions))
print('text_to_image', cnt_txt2img)
print('image_to_txt', cnt_img2txt)
