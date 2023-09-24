import pytorch_lightning as pl
import pandas as pd
import torch
import wandb
from metrics import get_hi_metrics, get_lo_metrics
from torch.utils.data import Dataset


class MoleculeDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

class MolecularModel(pl.LightningModule):
    def __init__(self, params: dict):
        super().__init__()
        layers = []
        for i in range(len(params['layers']) - 1):
            layers.append(torch.nn.Linear(params['layers'][i], params['layers'][i+1]))
            layers.append(torch.nn.ReLU())
            if params['use_dropout']:
                layers.append(torch.nn.Dropout(params['dropout']))
        layers.append(torch.nn.Linear(params['layers'][-1], 1))
        layers.append(torch.nn.Sigmoid())

        self.model = torch.nn.Sequential(
            *layers
        )
        print(self.model)
        self.lr = params['lr']
        self.l2 = params['l2']
        self.save_hyperparameters()

        self.train_epoch_outputs = []
        self.train_epoch_inputs = []
        self.valid_epoch_outputs = []
        self.valid_epoch_inputs = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)


class HiModel(MolecularModel):
    def __init__(self, params: dict):
        super().__init__(params)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = torch.squeeze(y)
        y_pred = self(x)
        y_pred = torch.squeeze(y_pred)
        loss = torch.nn.MSELoss()(y_pred, y)
        self.log('train_loss', loss, prog_bar=True)
        self.train_epoch_outputs.append(y_pred.detach().cpu())
        self.train_epoch_inputs.append(y.cpu())
        return loss

    def on_train_epoch_end(self):
        all_outputs = torch.cat(self.train_epoch_outputs)

        all_inputs = torch.cat(self.train_epoch_inputs)
        inputs_df = pd.DataFrame({
            'value': all_inputs,
        })
        metrics = get_hi_metrics(inputs_df, all_outputs)
        metrics_to_log = {
            'train_roc_auc': metrics['roc_auc'],
            'train_bedroc': metrics['bedroc'],
            'train_prc_auc': metrics['prc_auc']
        }
        self.log_dict(metrics_to_log, logger=True, on_step=False, on_epoch=True)
        self.train_epoch_outputs.clear() 
        self.train_epoch_inputs.clear() 

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = torch.squeeze(y)
        y_pred = self(x)
        y_pred = torch.squeeze(y_pred)
        loss = torch.nn.MSELoss()(y_pred, y)
        self.log('val_loss', loss, prog_bar=True)
        self.valid_epoch_outputs.append(y_pred.detach().cpu())
        self.valid_epoch_inputs.append(y.cpu())
        return loss

    def on_validation_epoch_end(self):
        all_outputs = torch.cat(self.valid_epoch_outputs)
        all_inputs = torch.cat(self.valid_epoch_inputs)
        inputs_df = pd.DataFrame({
            'value': all_inputs,
        })
        metrics = get_hi_metrics(inputs_df, all_outputs)
        metrics_to_log = {
            'test_roc_auc': metrics['roc_auc'],
            'test_bedroc': metrics['bedroc'],
            'test_prc_auc': metrics['prc_auc']
        }
        self.log_dict(metrics_to_log, logger=True, on_step=False, on_epoch=True)
        self.valid_epoch_outputs.clear()
        self.valid_epoch_inputs.clear()




class LoModel(pl.LightningModule):
    # Alas, copy-pasted from the base class, but without sigmoid activation at the end
    def __init__(self, params: dict, train_data, test_data):
        super().__init__()
        layers = []
        for i in range(len(params['layers']) - 1):
            layers.append(torch.nn.Linear(params['layers'][i], params['layers'][i+1]))
            layers.append(torch.nn.ReLU())
            if params['use_dropout']:
                layers.append(torch.nn.Dropout(params['dropout']))
        layers.append(torch.nn.Linear(params['layers'][-1], 1))

        self.model = torch.nn.Sequential(
            *layers
        )
        print(self.model)
        self.lr = params['lr']
        self.l2 = params['l2']
        self.save_hyperparameters()

        self.train_epoch_outputs = []
        self.valid_epoch_outputs = []

        self.train_data = train_data.copy()
        self.test_data = test_data.copy()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = torch.squeeze(y)
        y_pred = self(x)
        y_pred = torch.squeeze(y_pred)
        loss = torch.nn.MSELoss()(y_pred, y)
        self.log('train_loss', loss, prog_bar=True)
        self.train_epoch_outputs.append(y_pred.detach().cpu())
        return loss

    def on_train_epoch_end(self):
        return

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = torch.squeeze(y)
        y_pred = self(x)
        y_pred = torch.squeeze(y_pred)
        loss = torch.nn.MSELoss()(y_pred, y)
        self.log('val_loss', loss, prog_bar=True)
        self.valid_epoch_outputs.append(y_pred.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        all_outputs = torch.cat(self.valid_epoch_outputs)
        metrics = get_lo_metrics(self.test_data, all_outputs)
        metrics_to_log = {
            'test_r2': metrics['r2'],
            'test_spearman': metrics['spearman'],
            'test_mae': metrics['mae']
        }
        self.log_dict(metrics_to_log, logger=True, on_step=False, on_epoch=True)
        self.valid_epoch_outputs.clear()