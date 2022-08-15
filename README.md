# ResNet50_SUPCON + KG

## TO START THE CODE USE FOLLOWING COMMAND 

### python main.py --epochs 100 --mode 0 --loss_criterion CE --learning_rate 0.1 --min_lr 0 --L2_decay 0 --batch_size 1024

## Change the mode parameter to select pretraining or linear layer training phase

### MODE - 0 PRETRAINING || 1 TRAINING || 2 INFERENCE 

### LOSS CRITERION ARE CE or SupCon

## WHILE IN TRAINING PHASE USE --model param to point the models location 