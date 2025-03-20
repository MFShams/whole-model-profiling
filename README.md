# whole-model-profiling
 This project is used to profile different DNN models. Testing is on-going on M2 MAC mps, Nvidia GeForce RTX 3070 Ti GPU and a Xeon CPU. Tests are not yet conducted on Jetson AGX.

DISCLAIMER: The code may provide wrong readings because of minor bugs to be solved when revising this project in the future.

The profiling can be run via the terminal using the below command as an example:

```python simple_profile_model.py -dn M2MPS -d mps -m ResNet152 -pt time -i 100 -b 16```
