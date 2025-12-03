# whole-model-profiling
This repository provides a lightweight tool for profiling end-to-end DNN models. It is a simplified extract of a component from a larger (currently closed-source) research project. Current testing covers Apple M2 (MPS), NVIDIA GeForce RTX 3070 Ti, and an Intel Xeon CPU. Jetson AGX testing is pending, though the corresponding component in the main project is fully validated.

DISCLAIMER: This code may produce inaccurate measurements due to minor unresolved bugs. It will be revised in future updates.

The profiling can be run via the terminal using the below command as an example:

```python simple_profile_model.py -dn M2MPS -d mps -m ResNet152 -pt time -i 100 -b 16```
