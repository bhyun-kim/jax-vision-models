import os.path as osp
from typing import Any, Callable, List, Optional

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader


class TinyImageNet(ImageFolder):
    """Tiny ImageNet dataset
        Download and unzip dataset:
            http://cs231n.stanford.edu/tiny-imagenet-200.zip

    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        assert split in [
            "train", "val", "test"
        ], ValueError("split must be one of ['train', 'val', 'test']")

        super().__init__(
            osp.join(root, split),
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        words = osp.join(root, "words.txt")
        self.classes = self.match_class_name(self.classes, words)

    def match_class_name(self, classes: List[str], words: str) -> List[str]:
        """Match class name by words.txt

        Args:
            classes (List[str]): class names by folder name
            words (str): path to words.txt
                Example of words.txt:
                    n00001740	entity
                    n00001930	physical entity
                    ...

        Returns:
            new_classes (List[str]): class names by words.txt
        """

        with open(words, "r") as f:
            lines = f.readlines()

        words_dict = {}
        for line in lines:
            line = line.strip().split("\t")
            words_dict[line[0]] = line[1]

        new_classes = []

        for c in classes:
            new_classes.append(words_dict[c])

        return new_classes
