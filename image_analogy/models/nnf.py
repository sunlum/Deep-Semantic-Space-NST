import time

import numpy as np
from keras import backend as K

from image_analogy.losses.core import content_loss
from image_analogy.losses.nnf import nnf_analogy_loss, NNFState, PatchMatcher
from image_analogy.losses.neural_style import neural_style_loss

from .base import BaseModel


class NNFModel(BaseModel):
    '''Faster model for image analogies.'''
    def build(self, a_sem_image, a_image, b_sem_image,b_image, output_shape):
        self.output_shape = output_shape
        loss = self.build_loss(a_sem_image, a_image, b_sem_image,b_image)
        # get the gradients of the generated image wrt the loss
        grads = K.gradients(loss, self.net_input)#dL/dx
        outputs = [loss]
        if type(grads) in {list, tuple}:
            outputs += grads
        else:
            outputs.append(grads) #output=[loss, dloss/dx]
        f_inputs = [self.net_input]
        for nnf in self.feature_nnfs:# self.net  VS  self.nnf_feature={conv31's NNFState, conv41's NNFState}
            f_inputs.append(nnf.placeholder)#net_input=x, f_input=[x, current reconstructed F_s(coords(j-i(F_x)))]
        self.f_outputs = K.function(f_inputs, outputs)# [loss, dl/dx]=f_outputs({x, con31_Recon_F_style(F_current_x),con41_Recon_F_style(F_current_x)})

    def eval_loss_and_grads(self, x):# let placeholder be current g(x), and do forward(x)=[loss, grads]
        x = x.reshape(self.output_shape)
        f_inputs = [x]# input is x, not {x, conv31's NNFState(reconstructed Fs), conv41's NNFState}
        # update the patch indexes
        # start_t = time.time()
        for nnf in self.feature_nnfs:
            nnf.update(x, num_steps=self.args.mrf_nnf_steps) #update coords(F_s->F_current-x) (before now, matcher belongs to F_style)
            new_target = nnf.matcher.get_reconstruction()# conv31's F_style(coords(j(F_style)->i(F_current-x)))
            f_inputs.append(new_target)
        # print('PatchMatch update in {:.2f} seconds'.format(time.time() - start_t))
        # run it through

        outs = self.f_outputs(f_inputs)
        # calculate loss and grads(aut);
        #       -f_outputs is K.function, it includes session.run(self.outputs + [self.updates_op], feed_dict={x, conv31's NNFState(reconstructed Fs), conv41's NNFState}}
        # feed_dict, or input, is {x, conv31's NNFState(reconstructed Fs), conv41's NNFState}

        loss_value = outs[0] #outs[0]=loss, outs[1]=grads
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values

    def build_loss(self, a_sem_image, a_image, b_sem_image,b_image):
        '''Create an expression for the loss as a function of the image inputs.'''
        print('Building loss...')
        loss = super(NNFModel, self).build_loss(a_sem_image, a_image, b_sem_image,b_image)#viriable
        # Precompute static features for performance
        print('Precomputing static features...')
        all_asem_features, all_a_image_features, all_bsem_features,all_b_features = self.precompute_static_features(a_sem_image, a_image, b_sem_image,b_image)
        print('Building and combining losses...')
        
        print('Building DSS losses...')
        if self.args.analogy_weight:#Deep Semantic Space-guided Loss
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
        if self.args.content_weight:#content loss
            for layer_name in self.args.b_content_layers:
                b_features = K.variable(all_b_features[layer_name][0])
                # current combined output
                bp_features = self.get_layer_output(layer_name)
                cl = content_loss(bp_features, b_features)
                loss += self.args.content_weight / len(self.args.b_content_layers) * cl

        print('Building Style losses...')
        if self.args.neural_style_weight != 0.0:#Gram-based Style loss
            for layer_name in self.args.neural_style_layers:
                a_image_features = K.variable(all_a_image_features[layer_name][0])
                layer_features = self.get_layer_output(layer_name)
                layer_shape = self.get_layer_output_shape(layer_name)
                # current combined output
                combination_features = layer_features[0, :, :, :]
                nsl = neural_style_loss(a_image_features, combination_features, 3, self.output_shape[-2], self.output_shape[-1])
                loss += (self.args.neural_style_weight / len(self.args.neural_style_layers)) * nsl

        return loss
