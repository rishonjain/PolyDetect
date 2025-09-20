import pandas as pd
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm
import torch
import os

# --- NEW, FAST CONFIGURATION ---
NUM_SAMPLES_PER_LANG = 3500
TOTAL_ENGLISH_SAMPLES = 10000

# --- 1. Load Local Generator Models ---
print("Loading local text generation models...")
device = 0 if torch.cuda.is_available() else -1
print(f"--- Using device: {'cuda (NVIDIA GPU)' if device == 0 else 'cpu'} ---")

print("Loading local models from './models' directory...")
distilgpt2_path = os.path.join("models", "distilgpt2")

if not os.path.exists(distilgpt2_path):
    print("\nERROR: Model files not found in the './models' directory.")
    print("Please run the 'model_architect.py' script first to download the models.")
    exit()

# Load ONLY the fast generator
generator_gpt2 = pipeline('text-generation', model=distilgpt2_path, device=device)

print("Models loaded successfully.")

# --- SIMPLIFIED & OPTIMIZED Helper function for text generation ---
def generate_ai_text(prompts):
    """Generates AI text using only the fast DistilGPT-2 model with batching."""
    ai_responses = []
    batch_size = 16 # We can use a larger batch size with the smaller model

    print(f"Generating {len(prompts)} samples with DistilGPT-2 (in batches of {batch_size})...")
    for i in tqdm(range(0, len(prompts), batch_size), desc="Fast Generation"):
        batch = prompts[i:i + batch_size]
        responses = generator_gpt2(batch, max_new_tokens=100, num_return_sequences=1, pad_token_id=50256)
        for response_list in responses:
            ai_responses.append(response_list[0]['generated_text'])
            
    return ai_responses

# --- Helper function to get MC4 samples ---
def get_mc4_samples(lang_code, num_samples):
    """Downloads a sample of text for a given language from the MC4 dataset."""
    print(f"Loading MC4 samples for language: {lang_code}...")
    dataset = load_dataset("mc4", lang_code, split="train", streaming=True, trust_remote_code=True)
    
    samples = []
    for example in tqdm(dataset, total=num_samples, desc=f"Streaming {lang_code} samples"):
        if len(samples) >= num_samples:
            break
        if 50 < len(example['text']) < 1000:
            samples.append(example['text'])
    
    print(f"Collected {len(samples)} samples for {lang_code}.")
    return samples

# --- 2. Process English Data ---
print("\n--- Processing English Data ---")
print("Loading HC3 English dataset...")
hc3_dataset = load_dataset("Hello-SimpleAI/HC3", "all", split="train", trust_remote_code=True)
english_data = []
for item in tqdm(hc3_dataset, desc="Processing HC3"):
    if item['human_answers'] and item['chatgpt_answers']:
        human_answer = item['human_answers'][0]
        chatgpt_answer = item['chatgpt_answers'][0]
        if human_answer and chatgpt_answer:
            english_data.append({"text": human_answer, "label": 0})
            english_data.append({"text": chatgpt_answer, "label": 1})

print("Loading English samples from MC4...")
mc4_en_samples = get_mc4_samples('en', TOTAL_ENGLISH_SAMPLES)
for text in mc4_en_samples:
    english_data.append({"text": text, "label": 0})

print("Generating paired AI text for MC4 English samples...")
ai_english_mc4 = generate_ai_text(mc4_en_samples)
for text in ai_english_mc4:
    english_data.append({"text": text, "label": 1})

df_english = pd.DataFrame(english_data)
print(f"Total English samples: {len(df_english)}")

# --- 3. Process Multilingual Data ---
print("\n--- Processing Multilingual Data ---")
languages = {"hindi": "hi", "french": "fr", "spanish": "es"}
all_multilingual_data = []

for lang_name, lang_code in languages.items():
    print(f"\n-- Processing {lang_name.title()} --")
    
    human_prompts = get_mc4_samples(lang_code, NUM_SAMPLES_PER_LANG)
    for text in human_prompts:
        all_multilingual_data.append({"text": text, "label": 0})

    ai_responses = generate_ai_text(human_prompts)
    for text in ai_responses:
        all_multilingual_data.append({"text": text, "label": 1})

df_multilingual = pd.DataFrame(all_multilingual_data)
print(f"Total multilingual samples: {len(df_multilingual)}")

# --- 4. Final Assembly & Cleaning ---
print("\n--- Assembling and Cleaning Final Dataset ---")
df_final = pd.concat([df_english, df_multilingual], ignore_index=True)

print("Cleaning and finalizing...")
df_final.dropna(inplace=True)
df_final.drop_duplicates(subset=['text'], inplace=True)
df_final = df_final[df_final['text'].str.len() > 50]

df_final = df_final.sample(frac=1).reset_index(drop=True)

df_final.to_csv("polydetect_dataset_fast.csv", index=False)

print("\n--- DATA CURATION COMPLETE ---")
print(f"Final dataset saved to 'polydetect_dataset_fast.csv'")
print(f"Total samples in final dataset: {len(df_final)}")