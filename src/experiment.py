import os
import re
import numpy as np
import json
from sklearn.utils import resample
import pandas as pd
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
import torch.nn.functional as F
import time
from models import TRENT
from dataset import load_dataset, prepare_dataloader, balance_dataframe, load_embeddings, load_evidence
from utils import train_step, eval_step, eval_agreement_subsets, early_stop, save_results_csv

def set_seed(seed=0):
    random.seed(seed)                        
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def run_exp(DATA_PATH,
            topk_evidence, 
            learning_rate,
            model_name = "TRENT",
            TRAIN_MODEL=True,
            USE_EVIDENCE = True,
            evidence_encoder="clip_excerpts",
            evidence_fusion="re_rank_concat",
            evidence_version="excerpts",      
            results_filename = "results_xpose",       
            num_workers=4,
            epochs = 50,
            early_stop_epochs = 10,
            HELPFULNESS_RATIO = 0.0,
            encoder = "CLIP",
            encoder_version = "ViT-L/14",
            model_version = "TRENT",
            transformer_version = "",
            pooling_method = "",
            fusion_method = "relational_fusion",
            batch_size = 512,
            use_features = ["images", "texts"],
            tf_h_l = None,
            tf_dim = None,          
           ):
                        
    set_seed()  
          
    encoder_version_ = encoder_version.replace('-', '').replace('/', '') 
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
    train_data, valid_data, test_data = load_dataset(DATA_PATH)      
    train_data = balance_dataframe(train_data, label_col="falsified")   
    image_embeddings, text_embeddings = load_embeddings(DATA_PATH, encoder, encoder_version_)
    evidence_embeddings, evidence_by_type = load_evidence(DATA_PATH, encoder, encoder_version_, evidence_version)    
   
    emb_dim=text_embeddings.shape[0]
    evidence_emb=evidence_embeddings.shape[1]
                  
    train_dataloader = prepare_dataloader(input_data=train_data, visual_features=image_embeddings, textual_features=text_embeddings, evidence_emb=evidence_emb, batch_size=batch_size, num_workers=num_workers, topk_evidence=topk_evidence, evidence_embeddings=evidence_embeddings, evidence_by_type=evidence_by_type, shuffle=True)    

    valid_dataloader = prepare_dataloader(input_data=valid_data, visual_features=image_embeddings, textual_features=text_embeddings, evidence_emb=evidence_emb, batch_size=batch_size, num_workers=num_workers, topk_evidence=topk_evidence, evidence_embeddings=evidence_embeddings, evidence_by_type=evidence_by_type, shuffle=False)    
    
    test_dataloader = prepare_dataloader(input_data=test_data, visual_features=image_embeddings, textual_features=text_embeddings, evidence_emb=evidence_emb, batch_size=batch_size, num_workers=num_workers, topk_evidence=topk_evidence, evidence_embeddings=evidence_embeddings, evidence_by_type=evidence_by_type, shuffle=False)

    history = []
    has_not_improved_for = 0

    parameters = {"LEARNING_RATE": learning_rate, "EPOCHS": epochs, "BATCH_SIZE": batch_size, "MODEL_VERSION": str(model_version), "TRANSFORMER_VERSION": str(transformer_version), "POOLING_MECHANISM": str(pooling_method), "USE_EVIDENCE": USE_EVIDENCE, "TF_H_L": tf_h_l, "TF_DIM": tf_dim, "FUSION_METHOD": fusion_method, "NUM_WORKERS": num_workers, "USE_FEATURES": use_features, "EARLY_STOP_EPOCHS": early_stop_epochs, "ENCODER": encoder, "ENCODER_VERSION": encoder_version_, "model_name": model_name, 'helpfulness_ratio':HELPFULNESS_RATIO, 'topk_evidence': topk_evidence, 'evidence_encoder': evidence_encoder, 'evidence_version': evidence_version, 'evidence_fusion': evidence_fusion}
    
    PATH = "checkpoints_pt/model_" + model_name + ".pt"  
    model = TRENT(emb_dim=emb_dim, evidence_emb=evidence_emb)                
    model.to(device)
                                                
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
   
    if TRAIN_MODEL:
        for current_epoch in range(epochs):
            
            train_step(model, train_dataloader, current_epoch, optimizer, device, batches_per_epoch=train_dataloader.__len__())
            results, _ = eval_step(model, valid_dataloader, current_epoch, device, batches_per_epoch=valid_dataloader.__len__())      
            history.append(results)
            has_not_improved_for = early_stop(has_not_improved_for, model, optimizer, history, current_epoch, PATH, metrics_list=["F1_macro"])         
                
            if has_not_improved_for >= early_stop_epochs:        
                print(f"Performance has not improved for {early_stop_epochs} epochs. Stop training at epoch {current_epoch}!")
                break    
    
    print("Load Checkpoint.\n")
    checkpoint = torch.load(PATH, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
        
    res_val, _ = eval_step(model, valid_dataloader, -1, device, batches_per_epoch=valid_dataloader.__len__())  
    res_test, test_y_pred = eval_step(model, test_dataloader, -1, device, batches_per_epoch=test_dataloader.__len__())
                            
    res_val = {"valid_" + str(key.lower()): val for key, val in res_val.items()}        
    res_test = {"test_" + str(key.lower()): val for key, val in res_test.items()}
                                       
    df_y_pred = pd.DataFrame({'uniqueID': test_data.uniqueID.tolist(), 
                              'falsified': test_data.falsified.tolist(),
                              'noteId': test_data.noteId.tolist(),
                              'prediction': test_y_pred.flatten()})
         
    ratio_results_df = eval_agreement_subsets(DATA_PATH, df_y_pred, test_data)
    all_results = {**parameters, **res_test, **res_val, **ratio_results_df.to_dict()}

    all_results["path"] = PATH
    all_results["history"] = history

    if not os.path.isdir("results"):
        os.mkdir("results")
    
    if TRAIN_MODEL:
        save_results_csv("results/", results_filename, all_results)

    else:
        print("F1:", all_results['test_f1_macro'] * 100)
        print("F1 (h>=80%):", all_results['help_f1_macro'][0][-2] * 100)
        print("F1 (h>=90%):", all_results['help_f1_macro'][0][-1] * 100)