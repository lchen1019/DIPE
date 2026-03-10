import glob

def get_context():
    text_files = glob.glob("/path/to/PaulGrahamEssays") 
    full_text = ""
    for file in text_files:
        with open(file, 'r', encoding='utf-8') as f:
            full_text += f.read().strip().replace('\n', ' ')
    return full_text
