import torch
import torch.onnx
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable

import warnings
import copy
import coremltools
import os
import sys

import onnx
from onnx_coreml import convert
from onnx import onnx_pb

from session import *

def main(input_file, output_file):
    num_classes = 8
    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Dropout(p=.5),
        nn.Linear(num_ftrs, num_classes),
        nn.Sigmoid()
    )

    checkpoint = torch.load(input_file, map_location=lambda storage, loc: storage)
    model_ft.load_state_dict(checkpoint['model'])

    imsize = 224
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)

    protofile = output_file + '.proto'

    model_ft.train(False)
    model_ft.eval()
    model_ft.cpu()
    torch.onnx.export(model_ft, dummy_input, protofile, input_names=['0'], output_names=['classification'], verbose=True)
    model = onnx.load(protofile)

    scale = 1.0 / (0.226 * 255.0)
    red_scale = 1.0 / (0.229 * 255.0)
    green_scale = 1.0 / (0.224 * 255.0)
    blue_scale = 1.0 / (0.225 * 255.0)

    args = dict(is_bgr=False, red_bias = -(0.485 * 255.0)  , green_bias = -(0.456 * 255.0)  , blue_bias = -(0.406 * 255.0))

    model_file = open(protofile, 'rb')
    model_proto = onnx_pb.ModelProto()
    model_proto.ParseFromString(model_file.read())
    coreml_model = convert(model_proto, image_input_names=['0'], preprocessing_args=args)

    spec = coreml_model.get_spec()

    # get NN portion of the spec
    nn_spec = spec.neuralNetwork
    layers = nn_spec.layers # this is a list of all the layers
    layers_copy = copy.deepcopy(layers) # make a copy of the layers, these will be added back later
    del nn_spec.layers[:] # delete all the layers

    # add a scale layer now
    # since mlmodel is in protobuf format, we can add proto messages directly
    # To look at more examples on how to add other layers: see "builder.py" file in coremltools repo
    scale_layer = nn_spec.layers.add()
    scale_layer.name = 'scale_layer'
    scale_layer.input.append('0')
    scale_layer.output.append('input1_scaled')

    params = scale_layer.scale
    params.scale.floatValue.extend([red_scale, green_scale, blue_scale]) # scale values for RGB
    params.shapeScale.extend([3,1,1]) # shape of the scale vector 

    # now add back the rest of the layers (which happens to be just one in this case: the crop layer)
    nn_spec.layers.extend(layers_copy)

    # need to also change the input of the crop layer to match the output of the scale layer
    nn_spec.layers[1].input[0] = 'input1_scaled'

    print(spec.description)

    coreml_model = coremltools.models.MLModel(spec)

    coreml_model.save(output_file)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} [input file] [output file name]")
    else:
        main(sys.argv[1], sys.argv[2])
