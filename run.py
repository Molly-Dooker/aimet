import ipdb
import torch
from Examples.common import image_net_config
from Examples.torch.utils.image_net_evaluator import ImageNetEvaluator
from Examples.torch.utils.image_net_data_loader import ImageNetDataLoader
from torchvision.models import resnet18
from aimet_torch.model_preparer import prepare_model
from aimet_torch.quant_analyzer import CallbackFunc



DATASET_DIR= 'imagenet_samples/'

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



forward_pass_callback = CallbackFunc(pass_calibration_data, use_cuda)
eval_callback = CallbackFunc(ImageNetDataPipeline.evaluate, use_cuda)

# enable mse loss per layer analysis
data_loader = ImageNetDataPipeline.get_val_dataloader()
dummy_input = torch.rand(1, 3, 224, 224)    # Shape for each ImageNet sample is (3 channels) x (224 height) x (224 width)
if use_cuda:
    dummy_input = dummy_input.cuda()