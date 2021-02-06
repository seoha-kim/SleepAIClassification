from loader import *
import os
import pandas as pd
pd.set_option('display.max_row', None)
pd.set_option('display.max_columns', None)
from tqdm import tqdm

import random
import torch
import torch.nn as nn
import timm

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class Model(nn.Module):
    def __init__(self, model_name, pretrained=False, n_class=5):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        if model_name=='tf_efficientnet_b4_ns':
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, n_class)
        else: # 'resnet50', 'resnext50_32x4d'
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x

def inference(model, data_loader, device):
    model.eval()
    lst_preds_out = []
    for step, (images) in tqdm(enumerate(data_loader), total=len(data_loader)):
        images = images.to(device).float()
        test_preds = model(images)
        lst_preds_out += [torch.softmax(test_preds, 1).detach().cpu().numpy()]
    lst_preds_out = np.concatenate(lst_preds_out, axis=0)
    return lst_preds_out

def write_file(preds):
    dic_label = {0: 'Wake', 1: 'REM', 2: 'N1', 3: 'N2', 4: 'N3'}
    submission = pd.DataFrame(preds)
    submission = submission.applymap(lambda x: dic_label[x])
    if not (os.path.isdir('./submission')): os.mkdir('./submission')
    if not (os.path.isdir('./submission/{}'.format(day))): os.mkdir('./submission/{}'.format(day))
    submission.to_csv('./submission/{}/{}.csv'.format(day, title), encoding='utf-8-sig', index=False, header=False)
    print('file rows : ', submission.shape[0])
    print('submission file is saved !')

if __name__ == "__main__":
    best_models = ['tf_efficientnet_b4_ns/fold_1_epoch_3']  # modify !
    split = len(best_models)
    batch_size = 8
    num_workers = 4
    seed = 42
    day = 'thursday' # modify!
    title = 'submission_2' # modify!

    total_preds_out = []
    seed_everything(seed)
    test_dataset = TestDataset()
    test_loader = data_loader('test', test_dataset, batch_size=batch_size, num_workers=num_workers)
    for fold in range(split):
        print('---------- Fold {} is Inferring ----------'.format(fold+1))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Model(model_name=best_models[fold].split('/')[0]).to(device)
        weights = os.path.join('./best_model/', best_models[fold])
        model.load_state_dict(torch.load(weights)['model'])
        with torch.no_grad():
            preds = inference(model, test_loader, device)
        total_preds_out.append(preds)

        del model, weights, preds
        torch.cuda.empty_cache()
    mean_preds = np.mean(total_preds_out, axis=0)
    test_preds = np.argmax(mean_preds, axis=1)
    write_file(test_preds)
