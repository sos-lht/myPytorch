import torch
from net import simpleconv3
mynet = simpleconv3(4)
mynet.load_state_dict(torch.load('models/model.pt',map_location=lambda storage,loc: storage))
mynet.train(False)
dummy_input = torch.randn((1,3,48,48))
torch.onnx.export(mynet, dummy_input, "model.onnx", verbose=False)
