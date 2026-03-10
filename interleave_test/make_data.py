import os
import random
import json
from PIL import Image

# ================== configuration ==================
IMAGE_DIR = "./color_images"
OUTPUT_JSON = "questions_between_no_numbers.json"
NUM_QUESTIONS = 1000

EASY_RATIO = 0.6 
LEN_EASY = (3, 4)
LEN_HARD = (5, 7)

COLORS = ["red", "blue", "black", "green", "yellow", "purple", "orange", "grey"]
TEXT_PREFIXES = ["Block", "Section", "Chapter", "Marker", "Segment", "Token", "Unit"]

COLOR_TO_PATH = {
    c: os.path.join(IMAGE_DIR, f"{c}_image.png") for c in COLORS
}

RGB_MAP = {
    "red": (255, 0, 0), "blue": (0, 0, 255), "black": (0, 0, 0), "green": (0, 255, 0),
    "yellow": (255, 255, 0), "purple": (128, 0, 128), "orange": (255, 165, 0), "grey": (128, 128, 128)
}

os.makedirs(IMAGE_DIR, exist_ok=True)
for color in COLORS:
    img_path = COLOR_TO_PATH[color]
    if not os.path.exists(img_path):
        img = Image.new("RGB", (128, 128), RGB_MAP.get(color, (128, 128, 128)))
        img.save(img_path)


def generate_unique_text(existing_set):
    while True:
        prefix = random.choice(TEXT_PREFIXES)
        num = random.randint(100, 999)
        text = f"[{prefix}-{num}]"
        if text not in existing_set:
            existing_set.add(text)
            return text

def get_anchor_desc(item):
    if item["type"] == "text":
        return f'the text snippet "{item["content"]}"'
    else:
        return f'the {item["color"]} image'

def build_sequence(difficulty="easy"):
    length = random.randint(*LEN_EASY) if difficulty == "easy" else random.randint(*LEN_HARD)
    
    num_images = random.randint(1, length - 1)
    num_texts = length - num_images
    
    if num_images <= len(COLORS):
        selected_colors = random.sample(COLORS, k=num_images)
    else:
        selected_colors = random.choices(COLORS, k=num_images)

    existing_texts = set()
    selected_texts = [generate_unique_text(existing_texts) for _ in range(num_texts)]

    items = []
    for c in selected_colors:
        items.append({"type": "image", "color": c})
    for t in selected_texts:
        items.append({"type": "text", "content": t})
    
    random.shuffle(items)
    return items

def check_uniqueness(target_item, all_items):
    if target_item["type"] == "text":
        count = sum(1 for x in all_items if x["type"] == "text" and x["content"] == target_item["content"])
    else:
        count = sum(1 for x in all_items if x["type"] == "image" and x["color"] == target_item["color"])
    return count == 1

def main():
    print(f"Generating {NUM_QUESTIONS} unnumbered questions...")
    dataset = []
    generated_count = 0

    while generated_count < NUM_QUESTIONS:
        difficulty = "easy" if random.random() < EASY_RATIO else "hard"
        items = build_sequence(difficulty)
        
        candidates = []
        for i in range(len(items) - 2):
            left = items[i]
            mid = items[i+1]
            right = items[i+2]
            
            if check_uniqueness(left, items) and check_uniqueness(right, items):
                candidates.append((left, mid, right))
        
        if not candidates:
            continue
            
        left_item, target_item, right_item = random.choice(candidates)

        example_text = (
            "I will show you a sequence of images and texts. Please identify the object located immediately between two specific items.\n\n"
            "--- EXAMPLE ---\n"
            "[Context]: [Image: Blue] -> Text: \"[Section-331]\" -> [Image: Green] -> Text: \"[Chapter-99]\"\n"
            "Question: What is located immediately between the blue image and the green image?\n"
            "Answer: The text \"[Section-331]\"\n"
            "--- END EXAMPLE ---\n\n"
            "Now, look at the following sequence and answer the question.\n"
        )
        
        final_user_content = [{"type": "text", "text": example_text}]
        
        for item in items:
            if item["type"] == "image":
                final_user_content.append({"type": "image", "image": COLOR_TO_PATH[item["color"]]})
                final_user_content.append({"type": "text", "text": "\n"})
            else:
                final_user_content.append({"type": "text", "text": f'Text: "{item["content"]}"\n'})

        
        if target_item["type"] == "image":
            q_target_type = "image"
            correct_answer = f'{target_item["color"].capitalize()} image'
            other_colors = [c.capitalize() + " image" for c in COLORS if c != target_item["color"]]
            random.shuffle(other_colors)
            options = [correct_answer] + other_colors[:3]
        else:
            q_target_type = "text"
            correct_answer = f'Text: "{target_item["content"]}"'
            
            context_distractors = [
                f'Text: "{x["content"]}"' 
                for x in items 
                if x["type"] == "text" and x["content"] != target_item["content"]
            ]
            
            random_distractors = []
            while len(random_distractors) < 3:
                t = generate_unique_text(set())
                random_distractors.append(f'Text: "{t}"')
                
            distractors = context_distractors + random_distractors
            distractors = list(set(distractors))[:3]
            while len(distractors) < 3:
                t = generate_unique_text(set())
                distractors.append(f'Text: "{t}"')
                
            options = [correct_answer] + distractors

        random.shuffle(options)
        answer_idx = options.index(correct_answer)
        answer_letter = chr(ord('A') + answer_idx)
        
        option_str = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
        
        question_text = (
            f"Question: What is located immediately between {get_anchor_desc(left_item)} and {get_anchor_desc(right_item)}?\n"
            f"{option_str}\n\n"
            "Answer with the option letter only."
        )
        
        final_user_content.append({"type": "text", "text": question_text})

        dataset.append({
            "id": generated_count,
            "difficulty": difficulty,
            "messages": [{"role": "user", "content": final_user_content}],
            "answer": answer_letter,
            "tag": f"no_num_{difficulty}_{q_target_type}"
        })
        generated_count += 1

    print(f"✅ Generated {len(dataset)} unnumbered questions.")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()