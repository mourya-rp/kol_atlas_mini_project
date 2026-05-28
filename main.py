import json
import seaborn as sns
import matplotlib.pyplot as plt
from engine import parse_pubmed_html, generate_similarity_matrix

print("Starting extraction...")


with open("data/kol_1.html", "r") as f: html_1 = f.read()
with open("data/kol_2.html", "r") as f: html_2 = f.read()
with open("data/kol_3.html", "r") as f: html_3 = f.read()


p1 = parse_pubmed_html(html_1, "https://scholar.google.com/citations?user=0dDO3SAAAAAJ&hl=en")
p2 = parse_pubmed_html(html_2, "https://scholar.google.com/citations?user=5HX--AYAAAAJ&hl=en")
p3 = parse_pubmed_html(html_3, "https://scholar.google.com/citations?user=PS_CX0AAAAAJ&hl=en")
profiles = [p1, p2, p3]


json_output = [p.model_dump() for p in profiles]
with open("kol_profiles.json", "w") as f:
    json.dump(json_output, f, indent=4)
print(" JSON saved successfully!")


print("Generating AI Embeddings and Matrix (this takes a few seconds)...")
matrix = generate_similarity_matrix(profiles)
names = [p.name for p in profiles]

plt.figure(figsize=(6, 4))
sns.heatmap(matrix, annot=True, xticklabels=names, yticklabels=names, cmap="Blues", vmin=0, vmax=1)
plt.title("KOL Semantic Similarity Matrix")
plt.savefig("similarity_matrix.png", bbox_inches='tight')
print(" Heatmap saved as similarity_matrix.png!")