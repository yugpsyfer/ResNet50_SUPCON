import argparse
import logging
from Model import run

log_path = './Outputs/Logs/'
logging.basicConfig(filename=log_path + 'Run.log', encoding='utf-8', level=logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process options for KG-NN')
    parser.add_argument('--loss_criterion', type=str, default="SupCon",
                        choices=["SupCon", "CE"], help="Choose the loss criterion")
    parser.add_argument('--mode', type=int, default=0,
                        help="Choose if you want to 0 : pretrain, 1 : Train, 2 : Test",
                        choices=[0, 1, 2])
    parser.add_argument('--batch_size', type=int, default=1024, help="Choose batch size for training")
    parser.add_argument('--model_name', type=str, help="Choose the model to evaluate")

    opt = parser.parse_args()

    if opt.mode == 0:
        run.pre_training(opt)
    elif opt.mode == 1:
        run.linear_phase_training(opt)
    else:
        run.inference(opt)









