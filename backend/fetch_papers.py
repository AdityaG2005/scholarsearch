import arxiv
import json

client = arxiv.Client()

queries = [
    "cat:cs.LG",
    "transformers",
    "foundational models",
    "deep learning",
    "large language models"
]

papers = {}

for q in queries:
    print(f"Searching: {q}")
    search = arxiv.Search(
        query=q,
        max_results=200,
        sort_by=arxiv.SortCriterion.Relevance
    )
    for result in client.results(search):
        entry_id = result.entry_id
        if entry_id not in papers:
            papers[entry_id] = {
                "entry_id": entry_id,
                "title": result.title,
                "summary": result.summary,
                "authors": [a.name for a in result.authors],
                "link": result.pdf_url,
                "published": str(result.published)
            }
    print(f"Total unique papers so far: {len(papers)}")
    if len(papers) >= 1000:
        break

paper_list = list(papers.values())
print(f"Final count: {len(paper_list)}")

with open("backend/data/papers.json", "w") as f:
    json.dump(paper_list, f, indent=2)

print("Saved to backend/data/papers.json")

