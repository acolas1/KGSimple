from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import logging

import random
import numpy as np
import torch

from cli import args
from simplify_batched import run
# from utils import *

def main():
   

    # preprocessData()

    if args.do_train:
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
            print("Output directory () already exists and is not empty.")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
            
        log_filename = "{}log.txt".format("" if args.do_train else "eval_")
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', 
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO, 
                            handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)), logging.StreamHandler()])
        logger = logging.getLogger(__name__)
        logger.info(args)
        logger.info(args.output_dir)
    
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        args.n_gpu = torch.cuda.device_count()
    
    
    
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

        if not args.do_train and not args.do_predict:
            raise ValueError("At least one of `do_train` or `do_predict` must be True.")

        if args.do_train:
            if not args.train_file:
                raise ValueError("If `do_train` is True, then `train_file` must be specified.")
            if not args.predict_file:
                raise ValueError("If `do_train` is True, then `predict_file` must be specified.")
                
                
        if args.do_predict:
            if not args.predict_file:
                raise ValueError("If `do_predict` is True, then `predict_file` must be specified.")
        run(args, logger)
        
        
    elif args.do_sample:
        print("Sampling...")
    
    
 

if __name__ == '__main__':
    main()