# MNIST
deep_learn_mnist.py
深度学习识别手写数据集

cnn_mnist.py:
cnn+tensorflow 识别手写数据集
urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1045)>
解决办法：
ssl._create_default_https_context = ssl._create_unverified_context

