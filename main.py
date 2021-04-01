from dataloaders.dataloader import DataLoader
from models.conv_model import ConvModel
from trainers.conv_model_trainer import ConvModelTrainer
from utils.config import get_config_from_json
from utils.args import get_args


if __name__ == "__main__":
    try:
        args = get_args()
        config, _ = get_config_from_json(args.config)
    except Exception as err:
        print("missing or invalid arguments")
        exit(0)
    dataloader = DataLoader('datasets/data', config)
    train_data, valid_data = dataloader.create_datasets()
    model = ConvModel(config)
    trainer = ConvModelTrainer(config, model)
    trainer.train(train_data, valid_data)