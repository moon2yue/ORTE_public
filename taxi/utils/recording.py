import  sys
import os
import random
import torch
import time
import logging
import numpy as np
import logging
import zipfile
from datetime import datetime
from torch.utils.data import Dataset


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def set_time():
    now = int(time.time())
    timeArray = time.localtime(now)
    formstyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return formstyleTime
def set_logger(name, formstyleTime):
    logging.basicConfig(level=logging.INFO,
                        filename='preprocess.log',
                        filemode='a',)
    logger = logging.getLogger(name)
    logger.info("Start running time: {}".format(formstyleTime))
    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def check_and_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"create path {path}")
    else:
        print(f"{path} exists")
    return path

class PrintToLogger:
    def __init__(self, fileName='app.log', log_level=logging.INFO, path=None):

        if path:
            os.makedirs(path, exist_ok=True)
            log_file = os.path.join(path, fileName)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=log_file,
            filemode='a'
        )
        self.logger = logging.getLogger()
        self.original_stdout = sys.stdout

    def write(self, message):
        if message.strip():
            self.logger.info(message.strip())
        self.original_stdout.write(message)
    def flush(self):
        self.original_stdout.flush()

if __name__ == "__main__":
    log_dir = 'logs'
    print_to_logger = PrintToLogger(log_file='app.log', log_dir=log_dir)
    sys.stdout = print_to_logger
    sys.stdout = print_to_logger.original_stdout