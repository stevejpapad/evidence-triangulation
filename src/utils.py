import os
import re
import time
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
import torch.nn.functional as F

def train_step(model, input_dataloader, current_epoch, optimizer, device, batches_per_epoch):
    
    epoch_start_time = time.time()

    running_loss = 0.0
    model.train()
    
    for i, data in enumerate(input_dataloader, 0):

        images = data[0].to(device, non_blocking=True)
        texts = data[1].to(device, non_blocking=True).squeeze(1)
        labels = data[2].to(device, non_blocking=True)
        
        i2t_evidence_embeddings = data[3].to(device, non_blocking=True).squeeze(1) 
        t2t_evidence_embeddings = data[4].to(device, non_blocking=True).squeeze(1) 
        
        optimizer.zero_grad()

        output = model(images, texts, i2t_evidence_embeddings, t2t_evidence_embeddings)                     
        loss = F.binary_cross_entropy_with_logits(output.float(), labels.float())                     
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
                    
        print(f"[Epoch:{current_epoch + 1}, Batch:{i + 1:5d}/{batches_per_epoch}]. Passed time: {round((time.time() - epoch_start_time) / 60, 1)} minutes. loss: {running_loss / (i+1):.3f}.",end="\r",)  

def eval_step(model, input_dataloader, current_epoch, device, batches_per_epoch):
        
    y_true = []
    y_pred = []
    
    model.eval()
    
    with torch.no_grad():
    
        for i, data in enumerate(input_dataloader, 0):

            images = data[0].to(device, non_blocking=True)
            texts = data[1].to(device, non_blocking=True).squeeze(1)
            labels = data[2].to(device, non_blocking=True)   
            i2t_evidence_embeddings = data[3].to(device, non_blocking=True).squeeze(1) 
            t2t_evidence_embeddings = data[4].to(device, non_blocking=True).squeeze(1)             

            output = model(images, texts, i2t_evidence_embeddings,t2t_evidence_embeddings)
                         
            y_pred.extend(output.cpu().detach().numpy())
            y_true.extend(labels.cpu().detach().numpy())

    y_pred = np.vstack(y_pred)          
    y_pred = 1/(1 + np.exp(-y_pred))
    
    y_true = np.array(y_true).reshape(-1,1)
 
    auc = metrics.roc_auc_score(y_true, y_pred)    
    y_pred = np.round(y_pred)        
    acc = metrics.accuracy_score(y_true, y_pred)    
    bal_acc = metrics.balanced_accuracy_score(y_true, y_pred)                        
    prec = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred) 
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
    cm = metrics.confusion_matrix(y_true, y_pred, normalize="true").diagonal()

    results = {
        "epoch": current_epoch,
        "Accuracy": round(acc, 4),
        "Balanced_Accuracy": round(bal_acc, 4),                            
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(recall, 4),
        "F1_macro": round(f1_macro, 4),                            
        'Pristine': round(cm[0], 4),
        'Falsified': round(cm[1], 4),
    }
    
    return results, y_pred

def eval_agreement_subsets(DATA_PATH, df_y_pred, test_data):
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    accuracy_list, precision_list, recall_list, f1_macro_list = [], [], [], []

    helpfulness_df = pd.read_csv(DATA_PATH + "helpfulness_scores.csv", index_col=0)
    helpfulness_df = helpfulness_df[helpfulness_df.HELPFUL > 0]

    test_data_with_help = test_data.merge(helpfulness_df, on='noteId', how='left')

    for ratio in ratios:
        filtered_test = test_data_with_help[test_data_with_help["helpfulness_ratio"] >= ratio]
        
        df_y_pred_copy = df_y_pred.merge(
            filtered_test[['uniqueID']], 
            on="uniqueID", 
            how="inner"
        )
        
        if df_y_pred_copy.empty:
            acc = prec = rec = f1 = 0.0
        else:
            y_true = df_y_pred_copy["falsified"].values
            y_pred = df_y_pred_copy["prediction"].values
            
            acc = metrics.accuracy_score(y_true, y_pred)
            prec = metrics.precision_score(y_true, y_pred, zero_division=0)
            rec = metrics.recall_score(y_true, y_pred, zero_division=0)
            f1 = metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
        
        accuracy_list.append(round(acc, 6))
        precision_list.append(round(prec, 6))
        recall_list.append(round(rec, 6))
        f1_macro_list.append(round(f1, 6))


    return pd.DataFrame({
        "help_ratios": [ratios],
        "help_accuracy": [accuracy_list],
        "help_precision": [precision_list],
        "help_recall": [recall_list],
        "help_f1_macro": [f1_macro_list],
    })
              
def topsis(xM, wV=None):
    m, n = xM.shape

    if wV is None:
        wV = np.ones((1, n)) / n
    else:
        wV = wV / np.sum(wV)

    normal = np.sqrt(np.sum(xM**2, axis=0))

    rM = xM / normal
    tM = rM * wV
    twV = np.max(tM, axis=0)
    tbV = np.min(tM, axis=0)
    dwV = np.sqrt(np.sum((tM - twV) ** 2, axis=1))
    dbV = np.sqrt(np.sum((tM - tbV) ** 2, axis=1))
    swV = dwV / (dwV + dbV)

    arg_sw = np.argsort(swV)[::-1]

    r_sw = swV[arg_sw]

    return np.argsort(swV)[::-1]

def choose_best_model(input_df, metrics, epsilon=1e-6):

    X0 = input_df.copy()
    X0 = X0.reset_index(drop=True)
    X1 = X0[metrics]
    X1 = X1.reset_index(drop=True)
    
    X1[:-1] = X1[:-1] + epsilon    
    X1["F1_macro"] = 1 - X1["F1_macro"]

    X_np = X1.to_numpy()
    best_results = topsis(X_np)
    top_K_results = best_results[:1]
    return X0.iloc[top_K_results]

def early_stop(has_not_improved_for, model, optimizer, history, current_epoch, PATH, metrics_list):

    best_index = choose_best_model(
        pd.DataFrame(history), metrics=metrics_list
    ).index[0]
        
    if not os.path.isdir(PATH.split('/')[0]):
        os.mkdir(PATH.split('/')[0])

    if current_epoch == best_index:
        
        torch.save(
            {
                "epoch": current_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            PATH,
        )

        has_not_improved_for = 0
    else:
        has_not_improved_for += 1
            
    return has_not_improved_for

def save_results_csv(output_folder_, output_file_, model_performance_):
    print("Save Results ", end=" ... ")
    exp_results_pd = pd.DataFrame(pd.Series(model_performance_)).transpose()
    if not os.path.isfile(output_folder_ + "/" + output_file_ + ".csv"):
        exp_results_pd.to_csv(
            output_folder_ + "/" + output_file_ + ".csv",
            header=True,
            index=False,
            columns=list(model_performance_.keys()),
        )
    else:
        exp_results_pd.to_csv(
            output_folder_ + "/" + output_file_ + ".csv",
            mode="a",
            header=False,
            index=False,
            columns=list(model_performance_.keys()),
        )
    print("Done\n")


def remove_urls(text):
    if not isinstance(text, str):
        return text  
    text = re.sub(r'http\S+|www\.\S+', '', text)  
    return re.sub(r'\s+', ' ', text).strip()      
    
def check_path(path):
    if os.path.exists(path):
        return path
    
    if path.lower().endswith(".jpg"):
        alt_path = path[:-4] + ".png"
        if os.path.exists(alt_path):
            return alt_path
    
    if path.lower().endswith(".png"):
        alt_path = path[:-4] + ".jpg"
        if os.path.exists(alt_path):
            return alt_path
    
    return None