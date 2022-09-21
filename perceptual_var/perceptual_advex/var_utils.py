import sys
import os
import functools
from torch import nn
from operator import mul
from torch import optim
from advex_uar.common.pyt_common import get_attack as get_uar_attack
from advex_uar.attacks.attacks import InverseImagenetTransform

from .perceptual_attacks import *
from .utilities import LambdaLayer
from . import utilities

# mister_ed
from recoloradv.mister_ed import loss_functions as lf
from recoloradv.mister_ed import adversarial_training as advtrain
from recoloradv.mister_ed import adversarial_perturbations as ap
from recoloradv.mister_ed import adversarial_attacks as aa
from recoloradv.mister_ed import spatial_transformers as st

from torch.autograd import Variable

# ReColorAdv
from recoloradv import perturbations as pt
from recoloradv import color_transformers as ct
from recoloradv import color_spaces as cs
from numbers import Number
import random

PGD_ITERS = 20
DATASET_NUM_CLASSES = {
    'cifar': 10,
    'imagenet100': 100,
    'imagenet': 1000,
    'bird_or_bicycle': 2,
}

class VariationParameters(advtrain.AdversarialAttackParameters):
    """ Wrapper to store an adversarial attack object as well as some extra
        parameters for how to use it in training
    """
    def __init__(self, adv_attack_obj, proportion_attacked=1.0,
                 attack_specific_params=None):
        """ Stores params for how to use adversarial attacks in training
        ARGS:
            adv_attack_obj : AdversarialAttack subclass -
                             thing that actually does the attack
            proportion_attacked: float between [0.0, 1.0] - what proportion of
                                 the minibatch we build adv examples for
            attack_specific_params: possibly None dict, but possibly dict with
                                    specific parameters for attacks
        """
        super(VariationParameters, self).__init__(adv_attack_obj, proportion_attacked,
                 attack_specific_params)

    def attack(self, inputs, labels):
        """ Builds some adversarial examples given real inputs and labels
        ARGS:
            inputs : torch.Tensor (NxCxHxW) - tensor with examples needed
            labels : torch.Tensor (N) - tensor with the examples needed
        RETURNS:
            some sample of (self.proportion_attacked * N ) examples that are
            adversarial, and the corresponding NONADVERSARIAL LABELS
            output is a tuple with three tensors:
             (adv_examples, pre_adv_labels, selected_idxs, coupled )
             adv_examples: Tensor with shape (N'xCxHxW) [the perturbed outputs]
             pre_adv_labels: Tensor with shape (N') [original labels]
             selected_idxs : Tensor with shape (N') [idxs selected]
             adv_inputs : Tensor with shape (N') [examples used to make advs]
             perturbation: Adversarial Perturbation Object
        """
        num_elements = inputs.shape[0]

        selected_idxs = sorted(random.sample(list(range(num_elements)),
                                int(self.proportion_attacked * num_elements)))

        selected_idxs = inputs.new(selected_idxs).long()
        if selected_idxs.numel() == 0:
            return (None, None, None)

        adv_inputs = Variable(inputs.index_select(0, selected_idxs))
        pre_adv_labels = labels.index_select(0, selected_idxs)

        perturbation1, perturbation2 = self.adv_attack_obj.attack(adv_inputs.data,
                                                  pre_adv_labels,
                                                  **self.attack_kwargs)
        adv_examples1 = perturbation1(adv_inputs)
        adv_examples2 = perturbation2(adv_inputs)

        return (adv_examples1, adv_examples2, pre_adv_labels, selected_idxs, adv_inputs,
                perturbation1, perturbation2)


class PGD(aa.AdversarialAttack):

    def __init__(self, classifier_net, normalizer, threat_model, loss_fxn,
                 manual_gpu=None):
        super(PGD, self).__init__(classifier_net, normalizer, threat_model,
                                  manual_gpu=manual_gpu)
        self.loss_fxn = loss_fxn # WE MAXIMIZE THIS!!!

    def attack(self, examples, labels, step_size=1.0/255.0,
               num_iterations=20, random_init=True, signed=True,
               optimizer=None, optimizer_kwargs=None,
               loss_convergence=0.999, verbose=True,
               keep_best=True):
        """ Builds PGD examples for the given examples with l_inf bound and
            given step size. Is almost identical to the BIM attack, except
            we take steps that are proportional to gradient value instead of
            just their sign.
        ARGS:
            examples: NxCxHxW tensor - for N examples, is NOT NORMALIZED
                      (i.e., all values are in between 0.0 and 1.0)
            labels: N longTensor - single dimension tensor with labels of
                    examples (in same order as examples)
            l_inf_bound : float - how much we're allowed to perturb each pixel
                          (relative to the 0.0, 1.0 range)
            step_size : float - how much of a step we take each iteration
            num_iterations: int or pair of ints - how many iterations we take.
                            If pair of ints, is of form (lo, hi), where we run
                            at least 'lo' iterations, at most 'hi' iterations
                            and we quit early if loss has stabilized.
            random_init : bool - if True, we randomly pick a point in the
                               l-inf epsilon ball around each example
            signed : bool - if True, each step is
                            adversarial = adversarial + sign(grad)
                            [this is the form that madry et al use]
                            if False, each step is
                            adversarial = adversarial + grad
            keep_best : bool - if True, we keep track of the best adversarial
                               perturbations per example (in terms of maximal
                               loss) in the minibatch. The output is the best of
                               each of these then
        RETURNS:
            AdversarialPerturbation object with correct parameters.
            Calling perturbation() gets Variable of output and
            calling perturbation().data gets tensor of output
        """

        ######################################################################
        #   Setups and assertions                                            #
        ######################################################################

        self.classifier_net.eval()

        if not verbose:
            self.validator = lambda ex, label, iter_no: None
        else:
            self.validator = self.validation_loop

        perturbation1 = self.threat_model(examples)
        perturbation2 = self.threat_model(examples)
        
        num_examples = examples.shape[0]
        var_examples = Variable(examples, requires_grad=True)
        var_labels = Variable(labels, requires_grad=False)
        variation = Variation(self.classifier_net)

        if isinstance(num_iterations, int):
            min_iterations = num_iterations
            max_iterations = num_iterations
        elif isinstance(num_iterations, tuple):
            min_iterations, max_iterations = num_iterations

        best_perturbation1 = None
        best_perturbation2 = None
        if keep_best:
            best_loss_per_example = {i: None for i in range(num_examples)}

        prev_loss = None

        ######################################################################
        #   Loop through iterations                                          #
        ######################################################################

        self.loss_fxn.setup_attack_batch(var_examples)
        self.validator(var_examples, var_labels, iter_no="START")

        # random initialization if necessary
        if random_init:
            #print("random init")
            perturbation1.random_init()
            perturbation2.random_init()
            self.validator(perturbation1(var_examples), var_labels,
                           iter_no="RANDOM")
            self.validator(perturbation2(var_examples), var_labels,
                           iter_no="RANDOM")

        # Build optimizer techniques for both signed and unsigned methods
        optimizer = optimizer or optim.Adam
        if optimizer_kwargs is None:
            optimizer_kwargs = {'lr':0.0001}
        optimizer1 = optimizer(perturbation1.parameters(), **optimizer_kwargs)
        optimizer2 = optimizer(perturbation2.parameters(), **optimizer_kwargs)

        update_fxn = lambda grad_data: -1 * step_size * torch.sign(grad_data)
        for iter_no in range(max_iterations):
            perturbation1.zero_grad()
            perturbation2.zero_grad()
            
            loss = self.loss_fxn.forward(perturbation1(var_examples), perturbation2(var_examples),
                                         perturbation1=perturbation1, perturbation2=perturbation2,
                                         output_per_example=keep_best)
            
            loss_per_example = loss
            loss = loss.sum()
            #print('overall loss:', loss)
            #print('variation:', variation.forward(perturbation1(var_examples), perturbation2(var_examples)).sum())

            loss = -1 * loss
            torch.autograd.backward(loss.sum())

            if signed:
                perturbation1.update_params(update_fxn)
                perturbation2.update_params(update_fxn)
            else:
                optimizer1.step()
                optimizer2.step()


            if keep_best:
                mask_val = torch.zeros(num_examples, dtype=torch.uint8)
                for i, el in enumerate(loss_per_example):
                    this_best_loss = best_loss_per_example[i]
                    if this_best_loss is None or this_best_loss[1] < float(el):
                        mask_val[i] = 1
                        best_loss_per_example[i] = (iter_no, float(el))

                if best_perturbation1 is None:
                    best_perturbation1 = self.threat_model(examples)
                    best_perturbation2 = self.threat_model(examples)

                best_perturbation1 = perturbation1.merge_perturbation(
                                                            best_perturbation1,
                                                            mask_val)
                best_perturbation2 = perturbation2.merge_perturbation(
                                                            best_perturbation2,
                                                            mask_val)


            self.validator((best_perturbation1 or perturbation1)(var_examples),
                           var_labels, iter_no=iter_no)
            self.validator((best_perturbation2 or perturbation2)(var_examples),
                           var_labels, iter_no=iter_no)

            # Stop early if loss didn't go down too much
            if (iter_no >= min_iterations and
                float(loss) >= loss_convergence * prev_loss):
                if verbose:
                    print("Stopping early at %03d iterations" % iter_no)
                break
            prev_loss = float(loss)



        perturbation1.zero_grad()
        perturbation2.zero_grad()
        self.loss_fxn.cleanup_attack_batch()
        perturbation1.attach_originals(examples)
        perturbation2.attach_originals(examples)
        return perturbation1, perturbation2

class PerturbationNormLoss2(lf.PartialLoss):

    def __init__(self, **kwargs):
        super(PerturbationNormLoss2, self).__init__()

        lp = kwargs.get('lp', 2)
        assert lp in [1, 2, 'inf']
        self.lp = lp


    def forward(self, examples, *args, **kwargs):
        """ Computes perturbation norm and multiplies by scale
        There better be a kwarg with key 'perturbation' which is a perturbation
        object with a 'perturbation_norm' method that takes 'lp_style' as a
        kwarg
        """

        perturbation1 = kwargs['perturbation1']
        perturbation2 = kwargs['perturbation2']
        assert isinstance(perturbation1, ap.AdversarialPerturbation)
        assert isinstance(perturbation2, ap.AdversarialPerturbation)

        return perturbation1.perturbation_norm(lp_style=self.lp) + perturbation2.perturbation_norm(lp_style=self.lp)

class RegularizedVariation(object):
    """ Wrapper for multiple PartialLoss objects where we combine with
        regularization constants """
    def __init__(self, losses, scalars, negate=False):
        """
        ARGS:
            losses : dict - dictionary of partialLoss objects, each is keyed
                            with a nice identifying name
            scalars : dict - dictionary of scalars, each is keyed with the
                             same identifying name as is in self.losses
            negate : bool - if True, we negate the whole thing at the end
        """

        assert sorted(losses.keys()) == sorted(scalars.keys())

        self.losses = losses
        self.scalars = scalars
        self.negate = negate

    def forward(self, ex1, ex2, *args, **kwargs):
        output = None
        output_per_example = kwargs.get('output_per_example', False)
        for k in self.losses:
            loss = self.losses[k]
            scalar = self.scalars[k]

            loss_val = loss.forward(ex1, ex2, *args, **kwargs)
            # assert scalar is either a...
            assert (isinstance(scalar, float) or # number
                    scalar.numel() == 1 or # tf wrapping of a number
                    scalar.shape == loss_val.shape) # same as the loss_val

            addendum = loss_val * scalar
            if addendum.numel() > 1:
                if not output_per_example:
                    addendum = torch.sum(addendum)

            if output is None:
                output = addendum
            else:
                output = output + addendum
        if self.negate:
            return output * -1
        else:
            return output


    def setup_attack_batch(self, fix_im):
        """ Setup before calling loss on a new minibatch. Ensures the correct
            fix_im for reference regularizers and that all grads are zeroed
        ARGS:
            fix_im: Variable (NxCxHxW) - Ground images for this minibatch
                    SHOULD BE IN [0.0, 1.0] RANGE
        """
        for loss in self.losses.values():
            if isinstance(loss, lf.ReferenceRegularizer):
                loss.setup_attack_batch(fix_im)
            else:
                loss.zero_grad()


    def cleanup_attack_batch(self):
        """ Does some cleanup stuff after we finish on a minibatch:
        - clears the fixed images for ReferenceRegularizers
        - zeros grads
        - clears example-based scalars (i.e. scalars that depend on which
          example we're using)
        """
        for loss in self.losses.values():
            if isinstance(loss, lf.ReferenceRegularizer):
                loss.cleanup_attack_batch()
            else:
                loss.zero_grad()

        for key, scalar in self.scalars.items():
            if not isinstance(scalar, Number):
                self.scalars[key] = None


    def zero_grad(self):
        for loss in self.losses.values():
            loss.zero_grad() # probably zeros the same net more than once...


class Variation(lf.PartialLoss):
    def __init__(self, classifier, normalizer=None):
        super(Variation, self).__init__()
        self.classifier = classifier
        self.normalizer = normalizer
        self.nets.append(self.classifier)
    
    def forward(self, ex1, ex2, *args, **kwargs):
        if self.normalizer is not None:
            classifier_in1 = self.normalizer.forward(ex1)
            classifier_in2 = self.normalizer.forward(ex2)
        else:
            classifier_in1 = ex1
            classifier_in2 = ex2
        classifier_out1 = self.classifier.forward(classifier_in1)
        classifier_out2 = self.classifier.forward(classifier_in2)
        return torch.norm(classifier_out1 - classifier_out2, dim=-1, p=2)
        
        

class MisterEdVariation(nn.Module):
    """
    Base class for attacks using the mister_ed library.
    """

    def __init__(self, model, threat_model, randomize=False,
                 perturbation_norm_loss=False, lr=0.001, random_targets=False,
                 num_classes=None, **kwargs):
        super().__init__()

        self.model = model
        self.normalizer = nn.Identity()

        self.threat_model = threat_model
        self.randomize = randomize
        self.perturbation_norm_loss = perturbation_norm_loss
        self.random_targets = random_targets
        self.attack_kwargs = kwargs
        self.lr = lr
        self.num_classes = num_classes

        self.attack = None

    def _setup_attack(self):
        var_loss = Variation(self.model, self.normalizer)
        perturbation_loss = PerturbationNormLoss2(lp=2)
        pert_factor = 0.0
        if self.perturbation_norm_loss is True:
            pert_factor = 0.05
        elif type(self.perturbation_norm_loss) is float:
            pert_factor = self.perturbation_norm_loss
        adv_loss = RegularizedVariation({
            'var': var_loss,
            'pert': perturbation_loss
        }, {
            'var': 1.0,
            'pert': -pert_factor
        }, negate=False)

        self.pgd_attack = PGD(self.model, self.normalizer,
                                 self.threat_model(), adv_loss)

        attack_params = {
            'optimizer': optim.Adam,
            'optimizer_kwargs': {'lr': self.lr},
            'signed': True,
            'verbose': False,
            'num_iterations': 0 if self.randomize else PGD_ITERS,
            'random_init': True,
        }
        attack_params.update(self.attack_kwargs)

        self.attack = VariationParameters(
            self.pgd_attack,
            1.0,
            attack_specific_params={'attack_kwargs': attack_params},
        )
        self.attack.set_gpu(False)

    def forward(self, inputs, labels):
        if self.attack is None:
            self._setup_attack()
        assert self.attack is not None

        if self.random_targets:
            return utilities.run_attack_with_random_targets(
                lambda inputs, labels: self.attack.attack(inputs, labels)[0],
                self.model,
                inputs,
                labels,
                num_classes=self.num_classes,
            )
        else:
            return self.attack.attack(inputs, labels)[0], self.attack.attack(inputs, labels)[1]
        
class StAdvVariation(MisterEdVariation):
    def __init__(self, model, bound=0.05, **kwargs):
        kwargs.setdefault('lr', 0.01)
        super().__init__(
            model,
            threat_model=lambda: ap.ThreatModel(ap.ParameterizedXformAdv, {
                'lp_style': 'inf',
                'lp_bound': bound,
                'xform_class': st.FullSpatial,
                'use_stadv': True,
            }),
            perturbation_norm_loss=0.0025 / bound,
            **kwargs,
        )


class ReColorAdvVariation(MisterEdVariation):
    def __init__(self, model, bound=0.06, **kwargs):
        super().__init__(
            model,
            threat_model=lambda: ap.ThreatModel(pt.ReColorAdv, {
                'xform_class': ct.FullSpatial,
                'cspace': cs.CIELUVColorSpace(),
                'lp_style': 'inf',
                'lp_bound': bound,
                'xform_params': {
                  'resolution_x': 16,
                  'resolution_y': 32,
                  'resolution_z': 32,
                },
                'use_smooth_loss': True,
            }),
            perturbation_norm_loss=0.0036 / bound,
            **kwargs,
        )

