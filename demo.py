from torch.utils.data import Dataset, DataLoader
import matplotlib
import torch
from bases import LATEX_BASES, LATEX_CONSTANTS
from experiment import EPS, Experiment, data
from neural_net import OccamNet
from targets import TARGET_FUNCTIONS
from utils import get_model_equation
from video_saver import VideoSaver
from visualization import draw_model_network, visualize
import matplotlib.pyplot as plt
import inspect
import numpy as np



exp_params = {
  "name": "example",
  "architecture": "OccamNet",
  "bases":["SINE", "ADDITION", "ADDITION", "ADDITION", "SINE"],
  "constants": ["ONE"],
  "target_function": "SINE_SHIFT_OFFSET",
  "domain_type": "continuous",
  "data_domain": [[-10, 10]],
  "recurrence_depth": 1,
  "depth": 3,
  "repeat": 1,
  "record": True,
  "recording_rate": 50,
  "dataset_size": 600,
  "batch_size": 200,
  "sampling_size": 100,
  "truncation_parameter": 5,
  "number_of_inputs": 1,
  "number_of_outputs": 1,
  "learning_rate": 0.005,
  "variances": 0.01,
  "temperature": 1,
  "last_layer_temperature": 1,
  "epochs": 1000,
  "skip_connections": True,
  "temperature_evolution": "still",
  "training_method": "evolutionary",
  "visualization": ["network", "loss", "expression"]
}
exp = Experiment(**exp_params)

architecture_arguments = inspect.getfullargspec(OccamNet)[0]
architecture_params = {argname: argument for (argname, argument) in exp.__dict__.items()
                       if argname in architecture_arguments}

model = OccamNet(**architecture_params).to(exp.device)


def run():

    print("USING DEVICE", exp.device)

    x = exp.sampler(exp.dataset_size)
    y = TARGET_FUNCTIONS[exp.target_function](x)

    xnp = x[:,0].cpu().detach().numpy().flatten()
    sortindx = np.argsort(xnp)
    ynp = y[:,0].cpu().detach().numpy().flatten()
    gradient = np.gradient(ynp[sortindx], xnp[sortindx])
    inverse_gradient = 1 / (np.abs(gradient) + EPS)

    sortindx = np.argsort(ynp)
    x = x[sortindx]
    y = y[sortindx]

    window_size = 10
    variances = torch.full([x.shape[0], 1], exp.variances)

    dl = DataLoader(data(x, y, variances), batch_size=exp.batch_size, shuffle=True)

    video_saver = None
    video_saver = VideoSaver(video_name= "%s_%s_%d" % ("", exp.name, 0))

    train_params = {
              'dataset':dl,
              'loss_function':torch.nn.MSELoss(),
              'video_saver': video_saver,
              'x':x,
              'y':y,
    }

    train_params.update(exp.__dict__)

    proper_arguments = inspect.getfullargspec(model.train)[0]
    filtered_params = {argname: argument for (argname, argument) in train_params.items()
                                            if argname in proper_arguments}
    loss = model.train(**filtered_params)
    print(f"DONE WITH {exp.name}, iteration {0}")
    video_saver.save(fps=15)
    video_saver.close()
    torch.save(model.state_dict(), "/tmp/model.pth")


if __name__ == "__main__":
    run()
    model.load_state_dict(torch.load("/tmp/model.pth"))
    print(get_model_equation(model))
    # visualize(model, viz_type=['network', 'expression'], epoch=0, skip_connections=True)
    
