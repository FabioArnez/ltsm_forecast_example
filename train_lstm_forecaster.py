import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import LearningRateMonitor
from pl_bolts.callbacks import TrainingDataMonitor
from airline_dataset import AirlineDataset
from airline_dataset import AirlineDataModule
from lstm_forecast_model import LSTM
from lstm_forecast_model import LstmPredModule


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    pl.seed_everything(100)
    
    checkpoint_callback = ModelCheckpoint(monitor="validation_loss",
                                          mode='min',
                                          every_n_epochs=1,
                                          save_top_k=2,
                                          save_last=True,
                                          save_on_train_epoch_end=False)

    monitor = TrainingDataMonitor(log_every_n_steps=20)
    
    progress_bar = RichProgressBar(theme=RichProgressBarTheme(description="green_yellow",
                                                              progress_bar="green1",
                                                              progress_bar_finished="green1",
                                                              batch_progress="green_yellow",
                                                              time="grey82",
                                                              processing_speed="grey82",
                                                              metrics="grey82"))
    
    # data_module = AirlineDataModule(batch_size=83, drop_last=False)
    data_module = AirlineDataModule(batch_size=120, train_size=0.8, drop_last=False)
    
    model_module = LstmPredModule(learning_rate=1e-2)
    # model_module.to(device);
    max_nro_epochs = 2000
    trainer = pl.Trainer(accelerator='gpu',
                         devices=-1,
                         max_epochs=max_nro_epochs,
                         callbacks=[progress_bar, checkpoint_callback, monitor])
    
    trainer.fit(model_module, datamodule=data_module)


if __name__ == '__main__':
    main()
