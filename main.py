import argparse
import logging
from Model import run
import wandb

log_path = './Outputs/Logs/'
logging.basicConfig(filename=log_path + 'Run.log', encoding='utf-8', level=logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process options for KG-NN')
    parser.add_argument('--loss_criterion', type=str, default="CE",
                        choices=["SupCon", "CE"], help="Choose the loss criterion")
    parser.add_argument('--mode', type=int, default=0,
                        help="Choose if you want to 0 : pretrain, mode : Train, 2 : Test",
                        choices=[0, 1, 2])
    parser.add_argument('--batch_size', type=int, default=1024, help="Choose batch size for training")
    parser.add_argument('--model_name', type=str, help="Choose the model to evaluate")
    parser.add_argument('--epochs', type=int, help="Choose total number of epochs")
    parser.add_argument('--learning_rate', type=float, help="Choose learning rate")

    opt = parser.parse_args()

    criterion_options = dict()
    criterion_options['SupCon'] = {"epochs": 1000,
                                   "temperature": 0.5,
                                   "annealing": "cosine",
                                   "learning_rate": 0.5,
                                   "optimizer": "sgd"}

    criterion_options['CE'] = {"epochs": 500,
                               "optimizer": "sgd",
                               "learning_rate": 0.106}

    config = dict(
        epochs=criterion_options[opt.loss_criterion]['epochs'],
        criterion=opt.loss_criterion,
        optimizer=criterion_options[opt.loss_criterion]['optimizer'],
        learning_rate=criterion_options[opt.loss_criterion]['learning_rate'],
        model="ResNet-50",
        dataset="MiniImagenet"
    )

    wandb.init(project="KG-NN Transfer learning Redo", config=config, entity="thesis-yugansh")

    if opt.mode == 0:
        run.pre_training(opt, config, criterion_options)
    elif opt.mode == 1:
        run.linear_phase_training(opt, config, criterion_options)
    else:
        run.inference(opt)









