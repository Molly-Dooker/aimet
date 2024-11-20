import ipdb
import os
import torch
from Examples.common import image_net_config
from Examples.torch.utils.image_net_evaluator import ImageNetEvaluator
from Examples.torch.utils.image_net_trainer import ImageNetTrainer
from Examples.torch.utils.image_net_data_loader import ImageNetDataLoader
from torchvision.models import resnet18
from aimet_torch.model_preparer import prepare_model
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.quantsim import QuantParams
from aimet_torch.bias_correction import correct_bias


DATASET_DIR= '/data/dataset/ImageNet_small'

class ImageNetDataPipeline:

    @staticmethod
    def get_val_dataloader() -> torch.utils.data.DataLoader:
        """
        Instantiates a validation dataloader for ImageNet dataset and returns it
        """
        data_loader = ImageNetDataLoader(DATASET_DIR,
                                         image_size=image_net_config.dataset['image_size'],
                                         batch_size=image_net_config.evaluation['batch_size'],
                                         is_training=False,
                                         num_workers=image_net_config.evaluation['num_workers']).data_loader
        return data_loader

    @staticmethod
    def evaluate(model: torch.nn.Module, use_cuda: bool) -> float:
        """
        Given a torch model, evaluates its Top-1 accuracy on the dataset
        :param model: the model to evaluate
        :param iterations: the number of batches to be used to evaluate the model. A value of 'None' means the model will be
                           evaluated on the entire dataset once.
        :param use_cuda: whether or not the GPU should be used.
        """
        evaluator = ImageNetEvaluator(DATASET_DIR, image_size=image_net_config.dataset['image_size'],
                                      batch_size=image_net_config.evaluation['batch_size'],
                                      num_workers=image_net_config.evaluation['num_workers'])

        return evaluator.evaluate(model, iterations=None, use_cuda=use_cuda)
    

model = resnet18(pretrained=True)
model = prepare_model(model)
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    model.to(torch.device('cuda'))

accuracy = ImageNetDataPipeline.evaluate(model, use_cuda)
print(accuracy)

_ = fold_all_batch_norms(model, input_shapes=(1, 3, 224, 224))


dummy_input = torch.rand(1, 3, 224, 224)    # Shape for each ImageNet sample is (3 channels) x (224 height) x (224 width)
if use_cuda:
    dummy_input = dummy_input.cuda()
sim = QuantizationSimModel(model=model,
                           quant_scheme=QuantScheme.post_training_tf_enhanced,
                           dummy_input=dummy_input,
                           default_output_bw=8,
                           default_param_bw=8)

print(sim.model)
print(sim)

def pass_calibration_data(sim_model, use_cuda):
    data_loader = ImageNetDataPipeline.get_val_dataloader()
    batch_size = data_loader.batch_size

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    sim_model.eval()
    samples = 1000

    batch_cntr = 0
    with torch.no_grad():
        for input_data, target_data in data_loader:

            inputs_batch = input_data.to(device)
            sim_model(inputs_batch)

            batch_cntr += 1
            if (batch_cntr * batch_size) > samples:
                break


sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                      forward_pass_callback_args=use_cuda)
accuracy = ImageNetDataPipeline.evaluate(sim.model, use_cuda)
print(accuracy)




################
# CLE

model = resnet18(pretrained=True)
model = prepare_model(model)

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    model.to(torch.device('cuda'))

equalize_model(model, input_shapes=(1, 3, 224, 224))


sim = QuantizationSimModel(model=model,
                           quant_scheme=QuantScheme.post_training_tf,
                           dummy_input=dummy_input,
                           default_output_bw=8,
                           default_param_bw=8)

sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                      forward_pass_callback_args=use_cuda)

accuracy = ImageNetDataPipeline.evaluate(sim.model, use_cuda)
print(accuracy)

################
# BC

data_loader = ImageNetDataPipeline.get_val_dataloader()

bc_params = QuantParams(weight_bw=8, act_bw=8, round_mode="nearest",
                        quant_scheme=QuantScheme.post_training_tf_enhanced)

correct_bias(model, bc_params, num_quant_samples=16,
             data_loader=data_loader, num_bias_correct_samples=16)

sim = QuantizationSimModel(model=model,
                           quant_scheme=QuantScheme.post_training_tf_enhanced,
                           dummy_input=dummy_input,
                           default_output_bw=8,
                           default_param_bw=8)

sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                      forward_pass_callback_args=use_cuda)

accuracy = ImageNetDataPipeline.evaluate(sim.model, use_cuda)
print(accuracy)

os.makedirs('./output/', exist_ok=True)
dummy_input = dummy_input.cpu()
sim.export(path='./output/', filename_prefix='resnet18_after_cle_bc', dummy_input=dummy_input)