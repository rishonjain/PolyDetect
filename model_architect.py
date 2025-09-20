import os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# --- Configuration ---
MODEL_DIR = "./models"
NEURAL_MODEL_NAME = "distilbert-base-multilingual-cased"
GENERATOR_MODEL_NAMES = ["distilgpt2", "bigscience/bloom-560m"]

# --- Create directory to store models ---
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"Created directory: {MODEL_DIR}")

# --- Download and save the Neurological Engine (DistilBERT) ---
def download_neural_model():
    """Downloads and saves the main classification model and its tokenizer."""
    print(f"\n--- Downloading Neurological Model: {NEURAL_MODEL_NAME} ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(NEURAL_MODEL_NAME)
        model = AutoModel.from_pretrained(NEURAL_MODEL_NAME)
        
        save_path = os.path.join(MODEL_DIR, NEURAL_MODEL_NAME)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        
        print(f"Successfully downloaded and saved to {save_path}")
    except Exception as e:
        print(f"Error downloading {NEURAL_MODEL_NAME}: {e}")

# --- Download and save the Generator models ---
def download_generator_models():
    """Downloads and saves the text generation models and their tokenizers."""
    for model_name in GENERATOR_MODEL_NAMES:
        print(f"\n--- Downloading Generator Model: {model_name} ---")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            save_path = os.path.join(MODEL_DIR, model_name.replace("/", "_")) # replace slash for folder name
            tokenizer.save_pretrained(save_path)
            model.save_pretrained(save_path)
            
            print(f"Successfully downloaded and saved to {save_path}")
        except Exception as e:
            print(f"Error downloading {model_name}: {e}")

# --- Main execution ---
if __name__ == "__main__":
    print("--- Starting Model Download Process ---")
    print(f"All models will be saved in the '{MODEL_DIR}' directory.")
    
    download_neural_model()
    download_generator_models()
    
    print("\n--- MODEL ARCHITECT SETUP COMPLETE ---")
    print("All necessary models have been downloaded for local, offline use.")