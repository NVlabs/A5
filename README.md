# A5

## Adversarial Augmentation Against Adversarial Attacks

<p align="justify">
Many defenses against adversarial attacks (e.g., robust classifiers, randomization, or image purification) use countermeasures put to work only after the attack has been crafted. We adopt a different perspective to introduce $A^5$ (Adversarial Augmentation Against Adversarial Attacks), a novel framework including the first certified preemptive defense against adversarial attacks. The main idea is to craft a defensive perturbation to guarantee that any attack (up to a given magnitude) towards the input in hand will fail. $A^5$ allows effective on-the-fly defensive augmentation with a robustifier network that ignores the ground truth label. It consistently beats state of the art certified defenses on MNIST, CIFAR10, FashionMNIST and Tinyimagenet, and can be used also to create certifiably robust physical objects. Our code allows experimenting on a wide range of scenarios beyond the man-in-the-middle attack tested in our CVPR paper, including the case of physical attacks.
</p>

## Getting the code

To facilitate the distribution and use of the code among researchers, we suggest using docker:

## Understanding the code

<p align="justify">
Detailed information about $A^5$ and code usage can be found in our CVPR paper. We also reccommend reading the related supplementary material for more details on the $A^5$ training recipes and the typical usage scenarios. A brief description of the recipes and usage scenarios is anyway given here.
</p>

## Using the code

## How to cite our work
Please cite our work as:
    
    @inproceedings{Fro23_a5,
      author={Frosio, Iuri and Kautz, Jan},
      year={2023},
      title={The Best Defense is a Good Offense: Adversarial Agumentation Against Adversarial Attacks},
      booktitle={CVPR},
    }
  

