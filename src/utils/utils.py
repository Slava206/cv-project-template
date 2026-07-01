# src/utils/utils.py

import torch
import random
import numpy as np

def get_device():
    """Возвращает устройство для PyTorch"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    """Устанавливает seed для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False