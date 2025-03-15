import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics.pairwise import cosine_similarity
from config import HF_TOKEN


# Load project data
def load_projects(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


# Load LLM tokenizer and model
def load_llm():
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN).to(
        "cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    return tokenizer, model, pipe


# Generate embeddings
def generate_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits[:, -1, :].cpu().numpy()


# Compute similarity
def find_most_similar(query, projects, tokenizer, model, top_k=3):
    query_embedding = generate_embedding(query, tokenizer, model)
    project_embeddings = [generate_embedding(p["ai_summary"], tokenizer, model) for p in projects]
    similarities = [cosine_similarity(query_embedding, emb)[0][0] for emb in project_embeddings]
    sorted_projects = sorted(zip(projects, similarities), key=lambda x: x[1], reverse=True)
    return sorted_projects[:top_k]


# Retrieve-augmented generation (RAG)
def generate_rag_response(query, top_projects, pipe):
    context = "\n".join([f"{p['name']}: {p['ai_summary']}" for p, _ in top_projects])
    prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful AI assistant providing information about Web3 projects.<|eot_id|><|start_header_id|>user<|end_header_id|>

    User Query: {query}

    Context:
    {context}

    Provide a natural and informative response based on the given context.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    response = pipe(prompt, max_new_tokens=500, do_sample=True)[0]['generated_text']

    # Extract only the AI-generated response, avoiding template parts
    match = re.search(r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*(.+)", response, re.DOTALL)
    summary = match.group(1).strip() if match else response.strip()

    return summary


if __name__ == "__main__":
    projects = load_projects("projects.json")
    tokenizer, model, pipe = load_llm()

    query = "What coin should I buy and why? Provide a detailed analysis considering current market trends, historical performance, and future potential."
    top_projects = find_most_similar(query, projects, tokenizer, model)
    response = generate_rag_response(query, top_projects, pipe)

    print("\nTop Projects:")
    for project, score in top_projects:
        print(f"- {project['name']} (Similarity: {score:.4f})")

    print("\nGenerated Response:")
    print(response)
