#!/usr/bin/env python
# -*- coding: utf-8 -*

import time
import sys
sys.path.append("/Users/chenjiayi/Documents/DSS-basedStyleTransfer")
sys.path.append("/Users/chenjiayi/Documents/DSS-basedStyleTransfer/image_analogy")
sys.path.append("/Users/chenjiayi/Documents/DSS-basedStyleTransfer/image_analogy/losses")

import image_analogy.argparser
import image_analogy.main


if __name__ == '__main__':
    args = image_analogy.argparser.parse_args()
    if args:
        from image_analogy.models.nnf import NNFModel as model_class
        
        start_time = time.time()
        try:
            image_analogy.main.main(args, model_class)
        except KeyboardInterrupt:
            print('Shutting down...')
        print('Done after {:.2f} seconds'.format(time.time() - start_time))
