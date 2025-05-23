import requests   # Request to download PDF from websites
import pdfplumber  # Extracts data from PDF
import re   # Reqular expression. It helps with text cleaning, finding numbers, or specific words
import torch #Implemented with AI-powered text summarization
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig   #AI-powered summarization
import textwrap  # Process and analyze text by formatting it for better readability
from textblob import TextBlob  # Process and analyze text to determine if it is positive, negative, or neutral
from collections import Counter # Counting repretitve keywords and synonyms
import nltk   # Helps with processing text
import spacy
import os

device = "cuda" if torch.cuda.is_available() else "cpu"   # If GPU is available, then use GPU. Otherwise, use CPU

nltk.download('punkt')   # Ensures a well structred text processing
nlp = spacy.load("en_core_web_sm")

# Dictionary method to include company's name with its sustainability report link
Sustainability_Report = {
    "Aramco": "https://www.aramco.com/-/media/publications/corporate-reports/sustainability-reports/report-2023/english/2023-saudi-aramco-sustainability-report-full-en.pdf",
    "STC": "https://www.stc.com/content/dam/groupsites/en/pdf/stc-sustainability-report2023englishV2.pdf",
    "Microsoft": "https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/msc/documents/presentations/CSR/Microsoft-2024-Environmental-Sustainability-Report.pdf"
}

# A combination of dictionary and list for the AI to recognize keywords and synonyms
related_keywords = {
    "climate change": ["global warming", "climate crisis"],
    "carbon emissions": ["CO2 emissions", "carbon footprint"]
    #"water waste": ["water convservation", "water management"],
    #"energy efficiency": ["energy savings", "power efficiency"],
    #"renewable energy": ["green energy", "clean energy"],
    #"fossil fuel": ["coal", "oil", "natural gas"],
    #"pollution": ["contamination", "environmental damage"],
    #"greenhouse gases": ["GHG", "carbon dioxide"],
    #"sustainability strategy": ["eco plan", "sustainable development"],
    #"environmental impact": ["eco footprint", "climate impact"],
    #"resource management": ["natural resource planning", "conservation efforts"]
}

# Combines every keywords and synonyms into a single list for better detection
keywords = set()
for key, related in related_keywords.items():
  keywords.add(key)
  keywords.update(related)


# AI model that is able to read text and generate summaries
AI_model = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(AI_model, trust_remote_code=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    AI_model,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)
# Process to download the sustainability report from the provided link
def installing_pdf(company_name, link):
  pdf_link = f"{company_name}_Sustainability_Report.pdf"
  response = requests.get(link)
  with open(pdf_link, "wb") as file:
    file.write(response.content)
  print(f"{pdf_link} has been downloaded successfully\n")
  return pdf_link

def readability(text):
  text = re.sub(r'\s+', ' ', text)
  text = re.sub(r'[^a-zA-Z0-9.,!?;\'"()\s]', '', text)
  return text.strip()

def ai_generated_text(text, chunk_size=512):
  text = readability(text)
  inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=chunk_size).to(device)
  outputs = model.generate(**inputs, max_new_tokens=150)
  summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return summary.strip()

def quantitative_data_extraction(paragraph):
  pattern = r'(\d+(?:,\d+)?(?:\.\d+)?\s?(?:percent|%|tons|barrels|MW|kg|CO2|emissions|score|rate|intensity|gigawatts|gigajoules|metric tons|MWh|KWh|GHG|kgCO2e|gCO2e|tCO2e)?)'
  Similarities = re.findall(pattern, paragraph, re.IGNORECASE)
  return list(set(Similarities))

def qualitative_data_extraction(paragraph):
  doc = nlp(paragraph)
  qualitative = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PERSON", "NORP", "FAC", "EVENT", "WORK_OF_ART"]]
  return list(set(qualitative))

def harvesting_process(pdf_link):
  similar_sections = []
  quantitative_info = []
  qualitative_info = []
  buffer, page_reference = [], []
  with pdfplumber.open(pdf_link) as pdf:
    for pagenum, page in enumerate(pdf.pages, start=1):
      text = page.extract_text()
      if text:
        paragraphs = text.split("\n")
        for p in paragraphs:
          quantfinding = quantitative_data_extraction(p)
          qualfinding = qualitative_data_extraction(p)
          if any(re.search(rf"\b{kw}\b", p, re.IGNORECASE) for kw in keywords):
            buffer.append(p)
            quantitative_info.extend(quantfinding)
            qualitative_info.extend(qualfinding)
            page_reference.append(pagenum)
            if len(buffer) >= 3:
              similar_sections.append((" ".join(buffer), sorted(set(page_reference))))
              buffer, page_reference = [], []
          else:
            quantitative_info.extend(quantfinding)
            qualitative_info.extend(qualfinding)
    if buffer:
      similar_sections.append((" ".join(buffer), sorted(set(page_reference))))
  return similar_sections, list(set(quantitative_info)), list(set(qualitative_info))

def qualitative_quantitative(quant_info, qual_info, sentiment):
  print("\nQuantitative words numbers have been detected:")
  for aspect in quant_info:
    print(f"{aspect}")
  print(f"\nQualitative words have been detected:")
  for aspect in qual_info:
    print(f"{aspect}")
  print(f"\nThe sentiment analysis reveal that the tone is: {sentiment}")
  print("\n" + "_" * 100 + "\n")
report_summary = {}
for company_name, link in Sustainability_Report.items():
  print(f"\n{'='*40} {company_name} Sustainability Report {'='*40}\n")
  pdf_link = installing_pdf(company_name, link)
  print("The program is harvesting important paragraphs...\n")
  similar_sections, quant_info, qual_info = harvesting_process(pdf_link)
  print(f"Harvested {len(qual_info)} qualitative items for {company_name}")
  if not similar_sections:
    print(f"There is no relevant sustainability info found in {company_name}'s report\n")
    continue
  comprhensivesummary = ""
  complete_text = ""
  for text, pages in similar_sections:
    brief = ai_generated_text(text)
    formatted_pages = ", ".join(map(str, pages))
    if brief:
      comprhensivesummary += f"(Pages {formatted_pages}) {brief}\n\n"
    complete_text += " " + text
  sentiment_score = TextBlob(complete_text).sentiment.polarity
  sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
  report_summary[company_name] = {
      "Quantitative information": quant_info,
      "Qualitative information": qual_info,
      "Sentiment": sentiment,
      "Summary": comprhensivesummary
  }
  if os.path.exists(pdf_link):
    os.remove(pdf_link)
for company, data in report_summary.items():
  if "Summary" in data:
    print(f"\n{'='*40} {company}: This is short summary preview: {'='*40}\n")
    print(data["Summary"][:500] + "...\n")
for company, data in report_summary.items():
  print(f"\n{'='*40} {company}: This is Quantitative Information: {'='*40}\n")
  for w in data["Quantitative information"]:
    print(f"{w}")
for company, data in report_summary.items():
  print(f"\n{'='*40} {company}: This is Qualitative information: {'='*40}\n")
  for w in data["Qualitative information"]:
    print(f"{w}")
print("\n" + "=" * 120)
print(f"\n{'='*40} This is a compiled AI Sustainability Report {'='*40}\n")

for company, data in report_summary.items():
  if "Summary" in data:
    print(f"\n{'='*40} {company}'s full Summary: {'='*40}\n{data['Summary']}")
  else:
    print(f"\n{company}: There is no summary that could be generated by AI.")
  print(f"\nSentiment: {data['Sentiment']}")
  print("\n" + "=" * 100 + "\n")

print("=" * 120)
