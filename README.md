# Repro of Learning Visual Models using a Knowledge Graph as a Trainer || [Paper](https://arxiv.org/abs/2102.08747)

---
## Quickstart


- Training with cross entropy loss

`python main.py --epochs 100 --mode 0 --loss_criterion CE --learning_rate 0.1 --min_lr 0 --L2_decay 0 --batch_size 1024`

- Pretraining with Cross Entropy loss

`python main.py --epochs 100 --mode 1 --loss_criterion CE --learning_rate 0.0001 --min_lr 0 --L2_decay 0 --model_name NAME_OF_MODEL`

---
Change the _mode_ parameter to select one of the following tasks:
 
* 0 - PRETRAINING 
* 1 - TRAINING 
* 2 - INFERENCE 

Loss Criterion can be changed by using parameters **CE** for cross entropy or **SupCon** for Supervised Contrastive loss



