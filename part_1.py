import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import HF_TOKEN

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load Llama-3 model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

BASE_URL = "https://cryptorank.io/all-coins-list"


def fetch_html(url):
    """Fetch HTML content from a URL."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    return response.text if response.status_code == 200 else None


def parse_projects(html):
    """Extract project details from HTML content."""
    soup = BeautifulSoup(html, "html.parser")
    projects = []
    project_rows = soup.select("tr.sc-635c3a22-0")  # Adjusted based on row structure

    for row in project_rows:
        name = row.select_one("p.sc-dec2158d-0.sc-6de4a45b-0").text.strip() if row.select_one(
            "p.sc-dec2158d-0.sc-6de4a45b-0") else None
        token_symbol = row.select_one("span.sc-dec2158d-0").text.strip() if row.select_one(
            "span.sc-dec2158d-0") else None
        price = row.select_one("td.sc-7338db8c-0.gcyhEa p").text.strip() if row.select_one(
            "td.sc-7338db8c-0.gcyhEa p") else None
        market_cap = row.select("td.sc-7338db8c-0.hakNfu p")[0].text.strip() if len(
            row.select("td.sc-7338db8c-0.hakNfu p")) > 0 else None
        volume = row.select("td.sc-7338db8c-0.hakNfu p")[1].text.strip() if len(
            row.select("td.sc-7338db8c-0.hakNfu p")) > 1 else None
        supply = row.select("td.sc-7338db8c-0.hakNfu p")[2].text.strip() if len(
            row.select("td.sc-7338db8c-0.hakNfu p")) > 2 else None

        # Construct basic description
        description = f"{name} ({token_symbol}) is a blockchain project. "
        if price:
            description += f"Current price: {price}. "
        if market_cap:
            description += f"Market cap: {market_cap}. "
        if volume:
            description += f"24h volume: {volume}. "
        if supply:
            description += f"Circulating supply: {supply}. "

        project = {
            "name": name,
            "token_symbol": token_symbol,
            "price": price,
            "market_cap": market_cap,
            "volume": volume,
            "supply": supply,
            "description": description
        }
        projects.append(project)

    return projects

def generate_summaries(projects):
    enhanced_projects = []
    for project in projects:
        # Properly formatted Llama-3 chat structure
        description = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You generate a single, concise sentence summarizing Web3 projects based on the given information.<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Name: {project['name']}
        Token Symbol: {project['token_symbol']}
        Price: {project['price']}
        Market Cap: {project['market_cap']}
        Description: {project['description']}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """

        # Generate response with higher token limit to ensure full sentence
        response = pipe(description, max_new_tokens=80, do_sample=False, truncation=True)[0]['generated_text']

        # Extract only the AI-generated sentence (remove assistant tag if present)
        match = re.search(r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*(.+)", response, re.DOTALL)
        summary = match.group(1).strip() if match else response.strip()

        # Remove any unwanted introductory text the model may generate
        summary = summary.split(":")[-1].strip()

        # Ensure it ends properly with a period
        summary = re.sub(r"\s+[\w\d,]+$", ".", summary).strip()

        # Assign clean summary
        project["ai_summary"] = summary
        enhanced_projects.append(project)

    return enhanced_projects



def scrape_projects():
    """Main function to scrape and process projects."""
    html = fetch_html(BASE_URL)
    if not html:
        print("Failed to fetch data.")
        return []

    projects = parse_projects(html)
    enhanced_projects = generate_summaries(projects)

    return enhanced_projects


def save_to_json(data, filename="projects.json"):
    """Save scraped data to JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    projects_data = scrape_projects()
    save_to_json(projects_data)
    print(f"Scraped and saved {len(projects_data)} projects.")
