# deep.lib
Deep learning library built on top of PyTorch

If you use Anaconda to manage python environments (recommended), do the following to create an environment with the required dependencies.
```
conda env create -f environment.yml
activate deeplib
conda install protobuf requests opencv pytorch torchvision cuda90 -c pytorch
pip install onnx-coreml
```

If coremltools does not install correctly try the following:
```
pip install git+https://github.com/apple/coremltools
```

Additional Dependencies For GPU Support
- Cuda 9.0
- cudnn



## CSC 4996/7
- [The Road Damage Dataset](https://s3-ap-northeast-1.amazonaws.com/mycityreport/RoadDamageDataset.tar.gz)

Download the dataset and unzip it to `/path/to/data`.

### To train a model
```
python train.py [--data_dir /path/to/data] [--model_dir /path/to/save/models] [--batch_size 8]
```

After each epoch of training, the model is evaluated against a withheld validation set. `Validation Accuracy: xx` is output after validation; `xx` is the F-Measure on the validation set. The model that performs best on the validation set will be saved to `/path/to/save/models` 

### To evaluate a model
```
python evaluate.py /path/to/model.ckpt.tar [--data_dir /path/to/data] [--batch_size 8]
```

This will evaluate the model's performance against the validation set used during training.
