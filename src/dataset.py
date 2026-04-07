import os
import torch
import numpy as np
import pandas as pd
import open_clip
from tqdm import tqdm 
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.utils import resample
import torch.nn.functional as F
from collections import defaultdict
from utils import remove_urls, check_path

def load_dataset(DATA_PATH, version=''):
    train_data = pd.read_csv(DATA_PATH + 'xpose_training'+version+'.csv', index_col=0)
    valid_data = pd.read_csv(DATA_PATH + 'xpose_validation'+version+'.csv', index_col=0)   
    test_data = pd.read_csv(DATA_PATH + 'xpose_testing'+version+'.csv', index_col=0)

    if "INTERNAL" in version:
        train_data["caption"] = train_data["tweetText"]
        valid_data["caption"] = valid_data["tweetText"]
        test_data["caption"] = test_data["tweetText"]   
    
    train_data['id'] = train_data['uniqueID']
    valid_data['id'] = valid_data['uniqueID']   
    test_data['id'] = test_data['uniqueID']    
    
    train_data['image_id'] = train_data['uniqueID']
    valid_data['image_id'] = valid_data['uniqueID']   
    test_data['image_id'] = test_data['uniqueID']    
    
    train_data[['id', 'image_id']] = train_data[['id', 'image_id']].astype('str')
    valid_data[['id', 'image_id']] = valid_data[['id', 'image_id']].astype('str')
    test_data[['id', 'image_id']] = test_data[['id', 'image_id']].astype('str') 
       
    label_map = {'factually_correct': 0, 'miscaptioned': 1}
    
    train_data['falsified'] = train_data["class"]
    valid_data['falsified'] = valid_data["class"]
    test_data['falsified'] = test_data["class"]
    
    train_data.falsified = train_data.falsified.map(label_map).astype(int)
    valid_data.falsified = valid_data.falsified.map(label_map).astype(int)
    test_data.falsified = test_data.falsified.map(label_map).astype(int)

    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    
    return train_data, valid_data, test_data

def balance_dataframe(df, label_col="label", random_state=42):
    classes = df[label_col].unique()
    grouped = [df[df[label_col] == c] for c in classes]

    min_size = min(len(g) for g in grouped)
    resampled = [resample(g, 
                        replace=False, 
                        n_samples=min_size, 
                        random_state=random_state) for g in grouped]

    return pd.concat(resampled).sample(frac=1, random_state=random_state).reset_index(drop=True)

def load_embeddings(DATA_PATH, encoder, encoder_version):
    image_embeddings = np.load(DATA_PATH + "xpose_" + encoder.lower() + "_image_embeddings_" + encoder_version + ".npy").astype('float32') 
    text_embeddings = np.load(DATA_PATH + "xpose_" + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy").astype('float32') 
    
    image_ids = np.load(DATA_PATH + "xpose_image_ids_" + encoder_version + ".npy")
    text_ids = np.load(DATA_PATH + "xpose_text_ids_" + encoder_version +".npy")
        
    image_embeddings = pd.DataFrame(image_embeddings, index=image_ids).T
    text_embeddings = pd.DataFrame(text_embeddings, index=text_ids).T
    
    image_embeddings.columns = image_embeddings.columns.astype('str')
    text_embeddings.columns = text_embeddings.columns.astype('str')  

    return image_embeddings, text_embeddings

def load_evidence(DATA_PATH, encoder, encoder_version, evidence_version):
    
    evidence_embeddings = np.load(DATA_PATH + "xpose_" + evidence_version + '_' + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy")

    excerpts_train = pd.read_csv(DATA_PATH + "minicpm_excerpts_training.csv", index_col=0)
    excerpts_valid = pd.read_csv(DATA_PATH + "minicpm_excerpts_validation.csv", index_col=0)
    excerpts_test = pd.read_csv(DATA_PATH + "minicpm_excerpts_testing.csv", index_col=0)
    
    evidence_df = pd.concat([excerpts_train, excerpts_valid, excerpts_test])                                            
    evidence_df["item_id"] = evidence_df["id"]
    evidence_df.reset_index(inplace=True, drop=True)
    
    evidence_by_type = defaultdict(lambda: {"t2t": [], "i2t": []})
    for idx, row in evidence_df.iterrows():
        evidence_by_type[row.item_id][row.evidence_type].append(idx)    

    return evidence_embeddings, evidence_by_type

class DatasetIterator(torch.utils.data.Dataset):

    def __init__(self,input_data,visual_features,textual_features, evidence_emb, topk_evidence, evidence_embeddings, evidence_by_type):
        self.input_data = input_data
        self.visual_features = visual_features
        self.textual_features = textual_features
        self.dim = evidence_emb
        self.topk_evidence = topk_evidence
        self.evidence_embeddings = evidence_embeddings
        self.evidence_by_type = evidence_by_type

    def __len__(self):
        return self.input_data.shape[0] 

    def __getitem__(self, idx):

        current = self.input_data.iloc[idx]

        img = self.visual_features[current.image_id].values.astype("float32")
        txt = self.textual_features[current.id].values.astype("float32")
        label = float(current.falsified)

        t2t_embeddings = np.zeros((self.topk_evidence, self.dim), dtype="float32")
        i2t_embeddings = np.zeros((self.topk_evidence, self.dim), dtype="float32")   

        if self.topk_evidence > 0:

            t2t_indices = self.evidence_by_type[current.uniqueID]["t2t"]
            i2t_indices = self.evidence_by_type[current.uniqueID]["i2t"]

            # Section 5.2 Evidence Reranking244
            t2t_embeddings = self.rank_and_select(txt, t2t_indices)[:self.topk_evidence]
            i2t_embeddings = self.rank_and_select(img, i2t_indices)[:self.topk_evidence]   
                                                                    
        return img, txt, label, i2t_embeddings, t2t_embeddings

    def rank_and_select(self, query_vec, indices):

        if not indices:
            return np.zeros((10, self.dim), dtype="float32")

        evid = np.stack([self.evidence_embeddings[i] for i in indices], axis=0).astype("float32")

        query = torch.tensor(query_vec, dtype=torch.float32).unsqueeze(0)  
        evid_t = torch.tensor(evid, dtype=torch.float32)                   
        sims = F.cosine_similarity(query, evid_t)                  
        
        topk = torch.topk(sims, k=min(10, len(sims)))
        selected = evid_t[topk.indices].numpy()

        if selected.shape[0] < 10:
            pad = np.zeros((10 - selected.shape[0], self.dim), dtype="float32")
            selected = np.concatenate([selected, pad], axis=0)

        return selected   


def prepare_dataloader(input_data, visual_features, textual_features, evidence_emb, batch_size, num_workers, topk_evidence, evidence_embeddings, evidence_by_type, shuffle=False,):

    dg = DatasetIterator(input_data,visual_features,textual_features,evidence_emb,topk_evidence, evidence_embeddings, evidence_by_type)
    dataloader = DataLoader(dg,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True,drop_last=False)

    return dataloader

class EncoderDatasetIterator(torch.utils.data.Dataset):
    def __init__(self, input_data, vis_processors, txt_processors, use_images):
        self.input_data = input_data
        self.vis_processors = vis_processors
        self.txt_processors = txt_processors
        self.use_images = use_images

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        current = self.input_data.iloc[idx]
        idx = str(current.uniqueID)

        if self.use_images:
            img_path = current.image_path
            img = Image.open(img_path).convert('RGB')
            img = self.vis_processors(img)          
        else:
            img = torch.tensor(0)
            
        txt = current.caption 
        txt = self.txt_processors(txt)

        return idx, img, txt

def prepare_encoder_dataloader(input_data, vis_processors, txt_processors, batch_size, num_workers, shuffle, use_images=True):
    dataset = EncoderDatasetIterator(input_data, vis_processors, txt_processors, use_images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return dataloader   

def extract_features(DATA_PATH, encoder = "CLIP", encoder_version = "ViTL14", evidence_excerpts=False):

    if evidence_excerpts:
        excerpts_train = pd.read_csv(DATA_PATH + "minicpm_excerpts_training_INTERNAL.csv", index_col=0)
        excerpts_valid = pd.read_csv(DATA_PATH + "minicpm_excerpts_validation_INTERNAL.csv", index_col=0)
        excerpts_test = pd.read_csv(DATA_PATH + "minicpm_excerpts_testing_INTERNAL.csv", index_col=0)
        
        all_data = pd.concat([excerpts_train, excerpts_valid, excerpts_test])                                            
        all_data["caption"] = all_data["article_text"]
        all_data['uniqueID'] = all_data['id']
        use_images = False
    
    else:
        train_data, valid_data, test_data = load_dataset(DATA_PATH, version="_INTERNAL")  
        all_data = pd.concat([train_data, valid_data, test_data])
        
        all_data["first_file"] = all_data["tweetMediaFiles"].str.split(",").str[0].str.strip()
        all_data["image_path"] = (DATA_PATH + "images/" + all_data["first_file"]).apply(check_path)
        use_images = True

    # text pre-processing
    all_data["caption"] = all_data["caption"].apply(remove_urls)    
    
    # Load CLIP
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _, vis_processors = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    txt_processors = open_clip.get_tokenizer('ViT-L-14')
    model.to(device)
    
    # Define dataloader
    dataloader = prepare_encoder_dataloader(all_data, vis_processors, txt_processors,  256, 4, False, use_images)
    
    # Extract features
    all_image_features = []
    all_text_features = []
    all_ids = []
    
    model.eval() 
    with torch.no_grad():
        for ids, imgs, txts in tqdm(dataloader):

            if txts.ndim > 2:
                txts = txts.squeeze()
            
            txt_feats = model.encode_text(txts.to(device)).cpu().numpy()
            all_text_features.append(txt_feats)
            all_ids.extend(ids)

            if use_images:
                img_feats = model.encode_image(imgs.to(device)).cpu().numpy()
                all_image_features.append(img_feats)
    
    final_ids = np.array(all_ids)        
    final_text_embs = np.vstack(all_text_features)
    
    data_type = "excerpts_" if evidence_excerpts else ""
    prefix = f"{DATA_PATH}xpose_{encoder.lower()}_{data_type}"

    if use_images:
        final_image_embs = np.vstack(all_image_features)
        np.save(f"{prefix}image_embeddings_{encoder_version}.npy", final_image_embs) 
    
    np.save(f"{prefix}text_embeddings_{encoder_version}.npy", final_text_embs)        
    np.save(f"{prefix}ids_{encoder_version}.npy", final_ids)