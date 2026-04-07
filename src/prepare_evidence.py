import json
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from dataset import load_dataset
from vlm_prompts import prompt_4 
import xml.etree.ElementTree as ET
from transformers import AutoModel, AutoTokenizer

def filter_out_text(text):
    AVOID_KEYWORDS = ["cookie", "cookies", "javascript", "image id", "permission to access"]
    if not isinstance(text, str): return False
    text_lower = text.lower().strip()
    if text_lower == "advertisement": return True
    return any(kw in text_lower for kw in AVOID_KEYWORDS)

def check_evidence(DATA_PATH, idx, data_version, t_number, type_of_evidence, low_cred):
    SOCIAL_MEDIA = ["x.com", "facebook.com", "youtube.com", "tiktok.com", "instagram.com"]
    keep = []
    folder = 'search_results_' if type_of_evidence == "t2t" else 'reverse_image_'
    
    for ev_id in range(10): 

        path_ = f"{DATA_PATH}/{folder}{data_version}/{idx}/{idx}_{t_number}/{idx}_{t_number}_result_{ev_id}.xml"
        
        try:
            tree = ET.parse(path_)
            root = tree.getroot()
            
            hostname = root.attrib.get("hostname")
            if hostname in low_cred or hostname in SOCIAL_MEDIA:
                continue

            paragraphs = [p.text for p in root.findall(".//main//p") if p.text]
            article_text = " ".join(paragraphs)

            if filter_out_text(article_text):
                continue

            keep.append((ev_id, article_text))
        except:
            continue
    return keep
        
def evidence_excerpt_extraction(DATA_PATH):
    output_dir = Path(DATA_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)        

    # Setup Model
    torch.manual_seed(100)
    model = AutoModel.from_pretrained('openbmb/MiniCPM-V-4_5', trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-4_5', trust_remote_code=True, use_fast=True)
    model = torch.compile(model) # optional
    
    # Data Loading
    train_data, valid_data, test_data = load_dataset(DATA_PATH, version="_INTERNAL")
    
    with open(DATA_PATH + 'mbfc_low_cred.json', 'r') as f:
        low_cred = json.load(f)
    
    article_limit = 20000
    
    for set_name, data in [("train", train_data), ("valid", valid_data), ("test", test_data)]:
        data["first_file"] = data["tweetMediaFiles"].str.split(",").str[0].str.strip()
        data["image_path"] = (DATA_PATH + "images/" + data["first_file"])
        
        all_outputs = []
        
        for _, row in tqdm(data.iterrows(), total=len(data), desc=f"Processing {set_name}"):
            idx = row.uniqueID
            user_post = row.caption

            # Load articles 
            article_text_t2t = check_evidence(DATA_PATH, idx, set_name, "t1", "t2t", low_cred)
            if not article_text_t2t:
                article_text_t2t = check_evidence(DATA_PATH, idx, set_name, "t2", "t2t", low_cred)
            
            article_text_i2t = check_evidence(DATA_PATH, idx, set_name, "lens", "i2t", low_cred)

            # VLM Inference
            try:
                image = Image.open(row.image_path).convert('RGB')
                sources = [("t2t", article_text_t2t), ("i2t", article_text_i2t)]
                
                for type_of_ev, article_list in sources:
                    for art_idx, art_text in article_list:
                        
                        art_text_snippet = art_text[:article_limit]
                        prompt = prompt_4(user_post, art_text_snippet)

                        generated_text = model.chat(
                            msgs=[{'role': 'user', 'content': [image, prompt]}],
                            tokenizer=tokenizer,
                            temperature=0.1)

                        if generated_text.strip().lower() != "none":
                            all_outputs.append((idx, art_idx, type_of_ev, generated_text))
                            
            except Exception as e:
                print(f"Error on {idx}: {e}")
    
        # Save results per set
        res_df = pd.DataFrame(all_outputs, columns=["id", "article_id", "evidence_type", "article_text"])
        res_df.to_csv(output_dir / f"minicpm_xpose_excerpts_{set_name}.csv", index=False)

