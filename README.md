# A5

## Adversarial Augmentation Against Adversarial Attacks

<p align="justify">
Many defenses against adversarial attacks (e.g., robust classifiers, randomization, or image purification) use countermeasures put to work only after the attack has been crafted. We adopt a different perspective to introduce $A^5$ (Adversarial Augmentation Against Adversarial Attacks), a novel framework including the first certified preemptive defense against adversarial attacks. The main idea is to craft a defensive perturbation to guarantee that any attack (up to a given magnitude) towards the input in hand will fail. $A^5$ allows effective on-the-fly defensive augmentation with a robustifier network that ignores the ground truth label. It consistently beats state of the art certified defenses on MNIST, CIFAR10, FashionMNIST and Tinyimagenet, and can be used also to create certifiably robust physical objects. Our code allows experimenting on a wide range of scenarios beyond the man-in-the-middle attack tested in our CVPR paper, including the case of physical attacks.
</p>

## Getting the code

In your preferred folder \<dir\>, clone the git repo, e.g.:

    mkdir <dir>/A5
    cd <dir>/A5
    git clone https://github.com/NVlabs/A5.git

## Docker 

To facilitate the distribution and use of the code among researchers, we suggest using docker. Once in the A5 folder, you should first create the docker image:

    cd <dir>/A5/A5
    cd docker
    nvidia-docker image build -t A5 .
    cd ..

To run the docker image:

    nvidia-docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v <dir>/A5:/mount/A5 A5

Notice that we are launching docker with a shared volume that includes the code got from git. When running docker, you can easily navigate to this folder as:

    cd ~/mount/A5
    
From here you can launch all the $A^5$ python scripts.

## Getting the datasets

<p align="justify">
To facilitate the use of the same code across different datasets, we use for all of them the webdataset format. We provide a script to convert the desired dataset into webdataset format for MNIST, CIFAR10, FashionMNIST, Tinyimagenet, and the Fonts dataset used in our CVPR paper. If you want to add more datasets, feel free to modify the convert_dataset.py script. To get help and convert (as an example) the MNIST dataset into the webdataset format used in $A^5$, you can use:

    python convert_dataset.py --help
    python convert_dataset.py --dataset-name mnist --output-folder webdataset_mnist

The syntax for the other datasets is similar.

</p>
    

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
  

