from typing import Optional, Any, Callable
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms as transform_lib
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self,
                 input_size: int  = 1,
                 seq_length: int = 4,
                 hidden_size: int = 2,
                 num_layers: int = 1,
                 pred_output_size: int = 1):
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.pred_output_size = pred_output_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, pred_output_size)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out
    
    
class LstmPredModule(pl.LightningModule):
    def __init__(self,
                 input_size: int = 1,
                 seq_length: int = 4,
                 hidden_size: int  = 2,
                 num_layers: int = 1,
                 pred_output_size: int = 1,
                 learning_rate: float = 1e-3,
                 max_nro_epochs: int = 100) -> None:
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.pred_output_size = pred_output_size
        self.learning_rate = learning_rate
        self.max_nro_peochs = max_nro_epochs
        
        self.lstm_model = LSTM(input_size, hidden_size, num_layers, pred_output_size)
        self.criterion = self.get_criterion()
        self.save_hyperparameters()
        
    def get_criterion(self):
        criterion = torch.nn.MSELoss()  # mean-squared error for regression
        return criterion
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=self.learning_rate)
        return optimizer
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        sample, target = batch
        pred = self.lstm_model.forward(sample)
        train_loss = self.criterion(pred, target)
        self.log_dict({"train_loss": train_loss}, on_step=False, on_epoch=True, prog_bar=True)
        return train_loss
    
    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        sample, target = batch
        pred = self.lstm_model.forward(sample)
        val_loss = self.criterion(pred, target)
        self.log_dict({"validation_loss": val_loss}, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        sample, target = batch
        pred = self.lstm_model.forward(sample)
        return pred


