import ipdb
import os, sys
import torch
import torch.nn.functional as F
from Examples.common import image_net_config
from Examples.torch.utils.image_net_evaluator import ImageNetEvaluator
from Examples.torch.utils.image_net_trainer import ImageNetTrainer
from Examples.torch.utils.image_net_data_loader import ImageNetDataLoader
from torchvision.models import resnet50
from aimet_torch.model_preparer import prepare_model
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.qc_quantize_op import StaticGridQuantWrapper
from functools import partial
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


TEST_NUM = 100

DATASET_DIR   = '/data/dataset/ImageNet_small'
Calibrate_DIR = '/data/dataset/ImageNet_small'


def hook(name,module, input, output):
    if module not in cached_input_output:
        cached_input_output[module] = []
    # Meanwhile store data in the RAM.
    cached_input_output[module].append((input[0].detach().cpu(), output.detach().cpu()))


def adaquant(layer, cached_inps, cached_outs, test_inp, test_out, lr1=1e-4, lr2=1e-2, iters=100, progress=True, batch_size=50):
    ipdb.set_trace()
    print("\nRun adaquant")
    test_inp = test_inp.to('cuda'); test_out = test_out.to('cuda')
    layer.eval()
    
    mse_before = F.mse_loss(layer._module_to_wrap(test_inp), test_out)

    # lr_factor = 1e-2
    # Those hyperparameters tuned for 8 bit and checked on mobilenet_v2 and resnet50
    # Have to verify on other bit-width and other models
    lr_qpin = 1e-1# lr_factor * (test_inp.max() - test_inp.min()).item()  # 1e-1
    lr_qpw = 1e-3#lr_factor * (layer.weight.max() - layer.weight.min()).item()  # 1e-3
    lr_w = 1e-5#lr_factor * layer.weight.std().item()  # 1e-5
    lr_b = 1e-3#lr_factor * layer.bias.std().item()  # 1e-3

    opt_w = torch.optim.Adam([layer.weight], lr=lr_w)
    if hasattr(layer, 'bias') and layer.bias is not None: opt_bias = torch.optim.Adam([layer.bias], lr=lr_b)
    opt_qparams_in = torch.optim.Adam([layer.quantize_input.running_range,
                                       layer.quantize_input.running_zero_point], lr=lr_qpin)
    opt_qparams_w = torch.optim.Adam([layer.quantize_weight.running_range,
                                      layer.quantize_weight.running_zero_point], lr=lr_qpw)

    losses = []
    for j in (tqdm(range(iters)) if progress else range(iters)):
        idx = torch.randperm(cached_inps.size(0))[:batch_size]

        train_inp = cached_inps[idx]#.cuda()
        train_out = cached_outs[idx]#.cuda()

        qout = layer(train_inp)
        loss = F.mse_loss(qout, train_out)

        losses.append(loss.item())
        opt_w.zero_grad()
        if hasattr(layer, 'bias') and layer.bias is not None: opt_bias.zero_grad()
        opt_qparams_in.zero_grad()
        opt_qparams_w.zero_grad()
        loss.backward()
        opt_w.step()
        if hasattr(layer, 'bias') and layer.bias is not None: opt_bias.step()
        opt_qparams_in.step()
        opt_qparams_w.step()

            # if len(losses) < 10:
            #     total_loss = loss.item()
            # else:
            #     total_loss = np.mean(losses[-10:])
            # print("mse out: {}, pc mean loss: {}, total: {}".format(mse_out.item(), mean_loss.item(), total_loss))

    mse_after = F.mse_loss(layer(test_inp), test_out)
    return mse_before.item(), mse_after.item()


class ImageNetDataPipeline:

    @staticmethod
    def get_val_dataloader() -> torch.utils.data.DataLoader:
        """
        Instantiates a validation dataloader for ImageNet dataset and returns it
        """
        data_loader = ImageNetDataLoader(Calibrate_DIR,
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

    @staticmethod
    def finetune(model: torch.nn.Module, epochs, learning_rate, learning_rate_schedule, use_cuda):
        """
        Given a torch model, finetunes the model to improve its accuracy
        :param model: the model to finetune
        :param epochs: The number of epochs used during the finetuning step.
        :param learning_rate: The learning rate used during the finetuning step.
        :param learning_rate_schedule: The learning rate schedule used during the finetuning step.
        :param use_cuda: whether or not the GPU should be used.
        """
        trainer = ImageNetTrainer(DATASET_DIR, image_size=image_net_config.dataset['image_size'],
                                  batch_size=image_net_config.train['batch_size'],
                                  num_workers=image_net_config.train['num_workers'])

        trainer.train(model, max_epochs=epochs, learning_rate=learning_rate,
                      learning_rate_schedule=learning_rate_schedule, use_cuda=use_cuda)
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

model = resnet50(pretrained=True)
model = prepare_model(model)

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    model.to(torch.device('cuda'))


# accuracy = ImageNetDataPipeline.evaluate(model, use_cuda)
# print(accuracy)

_ = fold_all_batch_norms(model, input_shapes=(1, 3, 224, 224))
dummy_input = torch.rand(1, 3, 224, 224)    # Shape for each ImageNet sample is (3 channels) x (224 height) x (224 width)
if use_cuda:    dummy_input = dummy_input.cuda()
sim = QuantizationSimModel(model=model,
                           quant_scheme=QuantScheme.post_training_tf_enhanced,
                           dummy_input=dummy_input,
                           default_output_bw=8,
                           default_param_bw=8)
# ipdb.set_trace()
mm = sim.model
for node in mm.graph.nodes:
    print(f"Node: {node.name}, Op: {node.op}")
    
    # 입력 노드 확인
    print("  Inputs:")
    for arg in node.args:
        if isinstance(arg, torch.fx.Node):
            print(f"    - {arg.name}")
    
    # 출력 노드 확인
    print("  Outputs:")
    for user in node.users:
        print(f"    - {user.name}")
    

sim.compute_encodings(forward_pass_callback=pass_calibration_data, forward_pass_callback_args=use_cuda)

os.makedirs('./output/', exist_ok=True)
dummy_input = dummy_input.cpu()
sim.export(path='./output/', filename_prefix='resnet50_after_qat', dummy_input=dummy_input)


##########################################################
# handlers = []
# cached_input_output = {}

# for name,m in sim.model.named_modules():    
#     if name =='': continue       
#     if not isinstance(m,StaticGridQuantWrapper): continue
#     if not (isinstance(m._module_to_wrap, torch.nn.Linear) or isinstance(m._module_to_wrap, torch.nn.Conv2d)): continue
#     # print(name)
#     handlers.append(m.register_forward_hook(partial(hook,name)))
# accuracy = ImageNetDataPipeline.evaluate(sim.model, use_cuda)

# for handler in handlers: handler.remove()

# ipdb.set_trace()
# for i, layer in enumerate(cached_input_output):
#     data = cached_input_output[layer]
#     cached_inps  = torch.cat([x[0] for x in data])
#     cached_outs = torch.cat([x[1] for x in data])
#     idx = torch.randperm(cached_inps.size(0))[:TEST_NUM]
#     test_inp = cached_inps[idx]
#     test_out = cached_outs[idx]
#     mse_before, mse_after = adaquant(layer, cached_inps, cached_outs, test_inp, test_out, iters=100, lr1=1e-5, lr2=1e-4)

# mm = sim.model.conv1
# iq = mm.input_quantizers[0]
# oq = mm.output_quantizers[0]
# wq = mm.param_quantizers['weight']
# bq = mm.param_quantizers['bias']


