# Hiding-Face-into-Background-A-Proactive-Countermeasure-Against-Malicious-Face-Swapping
Codecs for "X. Shen, H. Yao*, S. Tan, C. Qin, Hiding Face into Background: A Proactive Countermeasure Against Malicious Face Swapping, IEEE Transactions on Industrial Informatics, 2024.

This project is a Python implementation of the paper “Hiding Face into Background: A Proactive Countermeasure Against Malicious Face Swapping”, built on PyTorch.

Please train CompressNet before training the whole FHNet framework with the pre-trained model parameters. Training CompressNet requires two datasets to be prepared: the original images and their corresponding JPEG-compressed images. Once the pre-trained models are saved in the FHNet project (the path and file name can be configured in config.py), the FHNet can be trained by running train_iteration.py.
