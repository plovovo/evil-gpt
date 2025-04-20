import torch
from main import MiniGPT, CharTokenizer, generate, load_dataset_csv

model_path = "materny_gpt.pth"
dataset_path = "/Users/eaambartsumyan/Desktop/angry_gpt/angry_gpt_full_dataset.csv"
block_size = 64

raw_text = load_dataset_csv(dataset_path)
tokenizer = CharTokenizer(raw_text)

model = MiniGPT(tokenizer.vocab_size, block_size=block_size)
model.load_state_dict(torch.load(model_path))
model.eval()

print("Злой GPT готов. Напиши свой вопрос.\n")

chat_history = ""

print("GPT загружен. Можешь писать запросы. Напиши 'выход' для выхода.\n")

while True:
    user_input = input("Ты: ").strip()
    if user_input.lower() in ["выход", "exit", "quit", "q"]:
        print("Завершение диалога")
        break

    prompt = f"{chat_history}Q: {user_input}\nA:"
    response = generate(model, tokenizer, prompt, length=200)

    response = response.split("Q:")[0].split("\n\n")[0].strip()

    if "Q:" in response:
        response = response.split("Q:")[-1]
    response = response.split("A:")[-1].strip().split("\n")[0]

    print(f"GPT: {response}\n")

    chat_history += f"Q: {user_input}\nA: {response}\n\n"
