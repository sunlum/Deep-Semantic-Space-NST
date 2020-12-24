import time

import numpy as np
from keras import backend as K

from image_analogy.losses.analogy import analogy_loss
from image_analogy.losses.core import content_loss
from image_analogy.losses.mrf import mrf_loss
from image_analogy.losses.neural_style import neural_style_loss

from .base import BaseModel


class AnalogyModel(BaseModel):
    '''Model for image analogies.'''

    def build_loss(self, a_sem_image, a_image, b_sem_image,b_image):
        '''Create an expression for the loss as a function of the image inputs.'''
        print('Building loss...')
        loss = super(AnalogyModel, self).build_loss(a_sem_image, a_image, b_sem_image,b_image)
        # Precompute static features for performance
        print('Precomputing static features...')
        all_asem_features, all_a_image_features, all_bsem_features,all_b_features = self.precompute_static_features(a_sem_image, a_image, b_sem_image,b_image)
        
        print('Building DSS losses...')
        if self.args.analogy_weight:#PatchMatch   -image-analogies, a kind of content loss (if B is not a photograph but a [0,1] mask)
            for layer_name in self.args.analogy_layers:
                asem_features = all_asem_features[layer_name][0]#array
                a_image_features = all_a_image_features[layer_name][0]#array
                bsem_features = all_bsem_features[layer_name][0]#array
                # current combined output
                layer_features = self.get_layer_output(layer_name)#variable
                combination_features = layer_features[0, :, :, :]#variable
                al = nnf_analogy_loss(
                    asem_features, a_image_features, bsem_features, combination_features,
                    num_steps=self.args.analogy_nnf_steps, patch_size=self.args.patch_size,
                    patch_stride=self.args.patch_stride, jump_size=1.0)
                loss += (self.args.analogy_weight / len(self.args.analogy_layers)) * al

        print('Building Content losses...')
        if self.args.content_weight:#content loss(now b is seg)  (default=0.0)
            for layer_name in self.args.b_content_layers:
                b_features = K.variable(all_b_features[layer_name][0])
                # current combined output
                bp_features = self.get_layer_output(layer_name)
                cl = content_loss(bp_features, b_features)
                loss += self.args.content_weight / len(self.args.b_content_layers) * cl

        print('Building Style losses...')
        if self.args.neural_style_weight != 0.0:#Gram  (default=0.0)
            for layer_name in self.args.neural_style_layers:
                a_image_features = K.variable(all_a_image_features[layer_name][0])
                layer_features = self.get_layer_output(layer_name)
                layer_shape = self.get_layer_output_shape(layer_name)
                # current combined output
                combination_features = layer_features[0, :, :, :]
                nsl = neural_style_loss(a_image_features, combination_features, 3, self.output_shape[-2], self.output_shape[-1])
                loss += (self.args.neural_style_weight / len(self.args.neural_style_layers)) * nsl

        return loss
