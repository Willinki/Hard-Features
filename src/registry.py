from src.vgg import VGG11, VGG16, VGG19
from src.perceptron import TanhBetaMLP
from src.data import CIFAR10DataModule, MNISTDataModule, FashionMNISTDataModule


def model_registry():
    return {
        "vgg11": VGG11,
        "vgg16": VGG16,
        "vgg19": VGG19,
        "tanh_perceptron": TanhBetaMLP,
    }


def dataset_registry():
    return {
        "cifar10": CIFAR10DataModule,
        "mnist": MNISTDataModule,
        "fashion": FashionMNISTDataModule,
    }
