import os
from src.parser import args
import torch
from datetime import datetime

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	folder = f'sava_results/{args.model}_{args.dataset}/checkpoints/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model'+'.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list
		}, file_path)