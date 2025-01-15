# whole-model-profiling
 This project is used to profile different DNN models. It has been tested on M2 MAC mps, Nvidia GeForce RTX 3070 Ti GPU and a Xeon CPU. Tests are not yet finished on Jetson AGX.

The profiling can be run via the terminal using the below command as an example:
```python simple_profile_model.py -dn M2MPS -d mps -m ResNet152 -pt time -i 100 -b 16```
