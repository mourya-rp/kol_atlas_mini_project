import numpy as np
from pydantic import BaseModel, Field
from typing import List
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class KOLProfile(BaseModel):
    name: str = Field(..., description="Full name of the researcher")
    affiliation: str = Field("Unknown", description="Current institution")
    h_index: int = Field(0, description="H-index score")
    citations: int = Field(0, description="Total citations count")
    top_keywords: List[str] = Field(default_factory=list, description="Core fields of study")
    source_url: str = Field(..., description="URL source of the data")
    confidence_scores: dict = Field(default_factory=dict, description="Extraction confidence")
    influence_score: float = Field(0.0, description="Calculated metric out of 100")


def calculate_influence(profile: KOLProfile) -> float:
    w_citations, w_hindex, w_keywords = 0.4, 0.4, 0.2
    norm_citations = min(profile.citations / 500000.0, 1.0) # Adjusted for medical giants
    norm_hindex = min(profile.h_index / 300.0, 1.0)
    norm_keywords = min(len(profile.top_keywords) / 5.0, 1.0)
    
    raw_score = (w_citations * norm_citations) + (w_hindex * norm_hindex) + (w_keywords * norm_keywords)
    return round(raw_score * 100, 2)


def parse_pubmed_html(html_content: str, url: str) -> KOLProfile:
    soup = BeautifulSoup(html_content, 'html.parser')
    confidence = {}
    
    name_tag = soup.find("h1", class_="heading-title")
    name = name_tag.text.strip() if name_tag else "Unknown"
    confidence['name'] = 1.0 if name_tag else 0.0
    
    aff_tag = soup.find("div", class_="affiliations")
    affiliation = aff_tag.text.strip() if aff_tag else "Unknown"
    confidence['affiliation'] = 0.8 if aff_tag else 0.1
    
    keywords = [kw.text.strip() for kw in soup.find_all("strong", class_="keyword-token")]
    confidence['top_keywords'] = 0.9 if keywords else 0.2
    
    h_tag = soup.find("span", class_="h-index")
    h_index = int(h_tag.text) if h_tag else 0
    
    cit_tag = soup.find("span", class_="citations")
    citations = int(cit_tag.text) if cit_tag else 0
    confidence['metrics'] = 0.9 if h_tag and cit_tag else 0.0
    
    profile = KOLProfile(
        name=name, affiliation=affiliation, h_index=h_index, citations=citations,
        top_keywords=keywords, source_url=url, confidence_scores=confidence
    )
    profile.influence_score = calculate_influence(profile)
    return profile


def generate_similarity_matrix(profiles: List[KOLProfile]):
    text_blocks = [
        f"Researcher {p.name} affiliated with {p.affiliation}. Expert fields: {', '.join(p.top_keywords)}."
        for p in profiles
    ]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_blocks)
    return cosine_similarity(embeddings)