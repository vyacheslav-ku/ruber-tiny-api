import logging
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
from torch import nn

import yaml
import pandas as pd
import numpy as np

from models import YesNoModel

log = logging.getLogger(__name__)



def read_config():
    with open("config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print("Read successful")
    print(config)
    log.info(config)
    return config


def read_objects():


    # with open('model_0.picle', 'rb') as f:
    #     model_0 = pickle.load(f)
    with open('model_rubertyni.picle', 'rb') as f:
        modelrubertyni = pickle.load(f)
    with open('tokenizer.picle', 'rb') as f:
        tokenizer = pickle.load(f)
    #torch.save(model_0.state_dict(), "model_0.torch")

    model_0 = YesNoModel()
    model_0.load_state_dict(torch.load("model_0.torch"))
    model_0.eval()
    return model_0, modelrubertyni, tokenizer


def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()


def make_predict(text, main_model, wv_model, wv_tokenizer):
    new_df = pd.DataFrame()
    new_df['DATA'] = [text]
    new_df['VEC'] = new_df['DATA'].apply(lambda x: embed_bert_cls(x, wv_model, wv_tokenizer))
    X_train2 = torch.from_numpy(np.stack(new_df['VEC'].values)).type(torch.float)
    test_logits = main_model(X_train2).squeeze()
    return int(torch.round(torch.sigmoid(test_logits)).tolist())
