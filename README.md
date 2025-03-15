## Overview
This repository provides a backend solution for indexing and analyzing Web3 projects. 
The system performs web scraping, AI-based summarization, similarity search, and Retrieval-Augmented Generation (RAG) to enhance user interactions with blockchain-related data.

## Workflow

### Part 1: Web Scraping and AI Summarization
The first part of the pipeline is responsible for gathering Web3 project data and summarizing it using an AI model.

1. **Web Scraping**: 
   - The script fetches data from `cryptorank.io`, extracting project details such as name, token symbol, price, market capitalization, trading volume, and circulating supply.
   - Data is parsed using `BeautifulSoup` to locate specific elements in the HTML structure.

2. **AI-Generated Summaries**:
   - Each project receives a concise, AI-generated summary using the Llama-3 model.
   - The model is prompted with structured input containing project details and asked to generate a single-sentence summary.
   - This summary is extracted, cleaned, and stored in a structured format.

3. **Saving Data**:
   - The processed data, including original and AI-enhanced summaries, is saved as a JSON file (`projects.json`).

### Part 2: Querying and AI Response Generation
Once project data is available, this part of the system enables users to query the database and retrieve relevant information.

1. **Loading Project Data**:
   - The script loads `projects.json`, containing structured details and AI-generated summaries.

2. **Similarity Search**:
   - When a user submits a query, the system converts it into an embedding using Llama-3.
   - The AI-generated summaries of all projects are also converted into embeddings.
   - The script computes cosine similarity between the query embedding and project embeddings, ranking the most relevant projects.

3. **Retrieval-Augmented Generation (RAG)**:
   - The top-ranked projects are used to construct a relevant context for the AI model.
   - A structured prompt is created, combining the user’s query and project details.
   - The AI model generates a natural response that summarizes relevant information from the selected projects.

4. **Output**:
   - The system outputs a ranked list of relevant projects along with an AI-generated response tailored to the user’s query.

## Example Output
The output consists of a JSON file containing structured Web3 project data and AI-enhanced summaries. When queried, the system retrieves and ranks relevant projects before generating an informative AI response.
