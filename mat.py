import json

with open("/Users/eaambartsumyan/Desktop/angry_gpt/materny_gpt_dataset.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

text = ""
for line in lines:
    obj = json.loads(line)
    q, a = obj["Q"], obj["A"]
    text += f"Q: {q}\nA: {a}\n\n"

with open("materny_gpt_dataset.txt", "w", encoding="utf-8") as f:
    f.write(text)
