"""Baseline and monogenic models."""

from monogenet.models.resnet import BasicBlock, ResNet18
from monogenet.models.small_cnn import SmallCNN

__all__ = ["SmallCNN", "ResNet18", "BasicBlock"]
