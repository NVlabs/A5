# A5

## Adversarial Augmentation Against Adversarial Attacks

<p align="justify">
Many defenses against adversarial attacks (e.g., robust classifiers, randomization, or image purification) use countermeasures put to work only after the attack has been crafted. We adopt a different perspective to introduce $A^5$ (Adversarial Augmentation Against Adversarial Attacks), a novel framework including the first certified preemptive defense against adversarial attacks. The main idea is to craft a defensive perturbation to guarantee that any attack (up to a given magnitude) towards the input in hand will fail. $A^5$ allows effective on-the-fly defensive augmentation with a robustifier network that ignores the ground truth label. It consistently beats state of the art certified defenses on MNIST, CIFAR10, FashionMNIST and Tinyimagenet, and can be used also to create certifiably robust physical objects. Our code allows experimenting on a wide range of scenarios beyond the man-in-the-middle attack tested in our CVPR paper, including the case of physical attacks.
</p>

## Getting the code

In your preferred folder \<dir\>, clone the git repo, e.g.:

    mkdir <dir>/A5
    cd <dir>/A5
    git clone git@github.com:NVlabs/A5.git

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

*Important note: Each user is responsible for checking the content of datasets and the applicable licenses and determining if suitable for the intended use and applicable links before the script runs and the data is placed in the user machine.*

</p>
    
## Understanding the code and the recipes

<p align="justify">
Detailed information about $A^5$ and code usage can be found in our CVPR paper. We also reccommend reading the related supplementary material for more details on the $A^5$ training recipes and the typical usage scenarios. A brief description of the recipes and usage scenarios is anyway given here. The general $A^5$ framework is well represented by the following schematic representation.
</p>

![](/figs/a5.png)
___

<p align="justify">
Generally, we use $\boldsymbol{w}$ to indicate the appearance of the physical objects, that are framed by an acquisition process $A$ to generate the images (or data) $\boldsymbol{x}$. Physical adversarial attacks are crafted by physical modification of the objects $\boldsymbol{w}$ before acquiring their image (e.g., adversarial patches). Man in the Middle (MitM) adversarial attacks are crafted while transmitting data $\boldsymbol{w}$ from the acquisition device to the classifier. The attacker can see the content within the protected blocks, but cannot modify them. We work therefore under the assumption of a white box scenario, and cover with our method black, grey or white box scenarios. Since $A^5$ uses certfied bounds, it can protect against any form of attack.
</p>

<p align="justify">
- Offline data robustification, known ground truth, legacy classifier ($A^5/O$): we use $\boldsymbol{w}$ to indicate the samples of the dataset; the acquisition block boils down to simple normalization in this case. $A^5/O$ operates by modifying (offline) the samples so that they are more robust when using the legacy classifier C. This recipe has little of no practical use, apart from establishing a solid baseline.
  
  ![](/figs/a5_o_training.png)
  ![](/figs/a5_o_inference.png)
  ___
</p>

<p align="justify">
- On-the-fly data robustification, unknown ground truth, legacy classifier ($A^5/R$): at traininig time, we use $\boldsymbol{w}$ to indicate the raw samples of the dataset, and $\boldsymbol{x}$ to indicate them after normalization. A robustifier R adds defensive agumentation on top of $\boldsymbol{x}$ on-the-fly, thus it can be used in practice to protect acquired data in real time, before transmitting them to the legacy classifier C.

  ![](/figs/a5_r_training.png)
  ![](/figs/a5_r_rc_inference.png)
  ___
</p>

<p align="justify">
- On-the-fly data robustification, unknown ground truth, re-trained classifier ($A^5/RC$): at traininig time, we use $\boldsymbol{w}$ to indicate the raw samples of the dataset, and $\boldsymbol{x}$ to indicate them after normalization. A robustifier R adds defensive agumentation on top of $\boldsymbol{x}$ on-the-fly, thus it can be used in practice to protect acquired data in real time, before transmitting them to the classifier C that has been fine tuned during training, together with R.

  ![](/figs/a5_rc_training.png)
  ![](/figs/a5_r_rc_inference.png)
  ___
</p>

<p align="justify">
- Offline physical object robustification, known ground truth, legacy classifier ($A^5/P$): at traininig time, we use $\boldsymbol{w}$ to indicate the appearance of the physical objects, and $\boldsymbol{x}$ to indicate their image after acquisition with A, where A is a camera acquisition procedure. The objects' apperance is changed to make them robust against MitM attack that, at inference time, may be crafted while transmitting the images to the classifier. We use the legacy classifier C.

  ![](/figs/a5_p_training.png)
  ![](/figs/a5_p_inference.png)
</p>

<p align="justify">
- Offline physical object robustification, known ground truth, re-trained classifier ($A^5/P$): at traininig time, we use $\boldsymbol{w}$ to indicate the appearance of the physical objects, and $\boldsymbol{x}$ to indicate their image after acquisition with A, where A is a camera acquisition procedure. The objects' apperance is changed to make them robust against MitM attack that, at inference time, may be crafted while transmitting the images to the classifier. We use the a classifier C that is co-trained with $\boldsymbol{w}$.

  ![](/figs/a5_pc_training.png)
  ![](/figs/a5_pc_inference.png)
</p>

## Using the code

<p align="justify">
The script a5.py allows training a standard classifier, a robust classifier, a robustifier, or finding defensive auugmentations for physical objects or for a dataset. Training for each of these elements can be done together with the other ones, to generate different $A^5$ recipes. The syntax for a5.py is the following.
</p>

    python a5.py --help
    usage: a5.py [-h] [--train-prototypes] [--train-robustifier]
             [--train-classifier] [--test] [--no-autoattack]
             [--robustifier-arch {mnist,cifar10,tinyimagenet,identity}]
             [--acquisition-arch {identity,camera}]
             [--classifier-arch {mnist,cifar10,tinyimagenet,fonts}]
             [--training-dataset-folder TRAINING_DATASET_FOLDER]
             [--validation-dataset-folder VALIDATION_DATASET_FOLDER]
             [--test-dataset-folder TEST_DATASET_FOLDER]
             [--prototypes-dataset-folder PROTOTYPES_DATASET_FOLDER]
             [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--lr LR]
             [--lr-scheduler-milestones LR_SCHEDULER_MILESTONES [LR_SCHEDULER_MILESTONES ...]]
             [--lr-scheduler-gamma LR_SCHEDULER_GAMMA]
             [--x-epsilon-attack-scheduler-name {LinearScheduler,AdaptiveScheduler,SmoothedScheduler,FixedScheduler}]
             [--x-epsilon-attack-scheduler-opts X_EPSILON_ATTACK_SCHEDULER_OPTS]
             [--x-augmentation-mnist] [--x-augmentation-cifar10]
             [--save-interval SAVE_INTERVAL]
             [--batch-multiplier BATCH_MULTIPLIER]
             [--test-multiplier TEST_MULTIPLIER]
             [--load-classifier LOAD_CLASSIFIER]
             [--load-robustifier LOAD_ROBUSTIFIER] [--log-dir LOG_DIR]
             [--x-epsilon-attack-training X_EPSILON_ATTACK_TRAINING]
             [--x-epsilon-attack-testing X_EPSILON_ATTACK_TESTING]
             [--w-epsilon-attack-training W_EPSILON_ATTACK_TRAINING]
             [--w-epsilon-attack-testing W_EPSILON_ATTACK_TESTING]
             [--x-epsilon-defense X_EPSILON_DEFENSE]
             [--w-epsilon-defense W_EPSILON_DEFENSE]
             [--bound-type {IBP,CROWN-IBP,CROWN,CROWN-FAST}] [--verbose]

And here is a detailed explanation of all the parameters that can be passed as input.

<p align="justify">
- [--train-prototypes] [--train-robustifier] [--train-classifier] Indicate the task for $A^5$. All of these can be used together to create different $A^5$ recipes. For instance, for $A^5/RC$, one should train a robustifier and a classifier at the same time. 
</p>

<p align="justify">
- [--test] [--no-autoattack] The test option is used to test the trained elements on the test dataset. It can be used together with the training task, keep in mind that the test will run after training is complete. The no-autoattack option does not compute the autoattack error (faster).
</p>

<p align="justify">
- [--robustifier-arch {mnist,cifar10,tinyimagenet,identity}] [--acquisition-arch {identity,camera}] [--classifier-arch {mnist,cifar10,tinyimagenet,fonts}] These parameters indicate the architectures of the modules in $A^5$. An identity module is a simple way of indicating that the module is not used; e.g., when traininig a simple robust classifier, the robustifier should be identity as this is not used. The architecture of the neural networks are in the /models subfolder. If one wants to add different architectures, the code has to be modified accordingly. The acquisition arch "camera" can be seen as a special form of augmentation of the physical objects $\boldsymbol{w}$ that simulates camera acquisition.
</p>

<p align="justify">
- [--training-dataset-folder TRAINING_DATASET_FOLDER] [--validation-dataset-folder VALIDATION_DATASET_FOLDER] [--test-dataset-folder TEST_DATASET_FOLDER] [--prototypes-dataset-folder PROTOTYPES_DATASET_FOLDER] These indicate the folders with the datasets. Notice that the prototypes-dataset-folder is used to indicate a pre-trained robustified dataset, that generally lies in the logdir/w folder.
</p>

<p align="justify">
- [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--lr LR] [--lr-scheduler-milestones LR_SCHEDULER_MILESTONES [LR_SCHEDULER_MILESTONES ...]] [--lr-scheduler-gamma LR_SCHEDULER_GAMMA] [--batch-multiplier BATCH_MULTIPLIER] Training parameters. The batch multiplier is used to save GPU memory while training: the gradient is accumulated --batch-multipler times before performing the updating steps. This allow training with larger batches at minimum memory cost. It is however slower than using an equivalent larger batch.
</p>

<p align="justify">
- [--x-epsilon-attack-scheduler-name {LinearScheduler,AdaptiveScheduler,SmoothedScheduler,FixedScheduler}] [--x-epsilon-attack-scheduler-opts X_EPSILON_ATTACK_SCHEDULER_OPTS] Training schedulers for $\epsilon$ as described in the auto_LiRPA documentation.
</p>

<p align="justify">
- [--x-augmentation-mnist] [--x-augmentation-cifar10] use augmentation for mnist or cifar10. If other augmentation strategies are needed, they have to be added to the code.
</p>

<p align="justify">
- [--save-interval SAVE_INTERVAL] Interval (epochs) to save the models while training.
</p>        
             
<p align="justify">
- [--test-multiplier TEST_MULTIPLIER] Increase the size of the test dataset. This is generally not used, but comes in hand when the dataset is small. For instance, when testing $A^5/P$ on the font dataset, that contains only 62 characters, this allows increasing the size of the dataset and considering different random augmentation for testing.
</p>        

<p align="justify">
- [--load-classifier LOAD_CLASSIFIER] [--load-robustifier LOAD_ROBUSTIFIER] Load a classifier / robustifier before training or testing. Please notice that to load a defended set of physical objects one has to use --prototypes-dataset-folder.
</p>        

<p align="justify">
- [--log-dir LOG_DIR] folder use to store the training and testing results.
</p>        

<p align="justify">
- [--x-epsilon-attack-training X_EPSILON_ATTACK_TRAINING] [--x-epsilon-attack-testing X_EPSILON_ATTACK_TESTING] [--w-epsilon-attack-training W_EPSILON_ATTACK_TRAINING] [--w-epsilon-attack-testing W_EPSILON_ATTACK_TESTING] [--x-epsilon-defense X_EPSILON_DEFENSE] [--w-epsilon-defense W_EPSILON_DEFENSE] These are all the attack magnitudes. Please notice the correct interpretation may be a function of the adopted recipe.
</p>        

<p align="justify">
[--bound-type {IBP,CROWN-IBP,CROWN,CROWN-FAST}] Bound type used when calling the auto_LiRPA functions.
</p>        

<p align="justify">
- [--verbose] Mostly used for profiling.
</p>        


## Pretrained model

<p align="justify">
To save time and energy, our intention is to share pre-trained the models (robustifies, classifiers) mentioned in our CVPR paper to all researchers that need them. These will be published here on-demand. If you need one of our models to be made public, please send your requst to:
</p>

[ifrosio@nvidia.com](mailto:ifrosio@nvidia.com).


## How to cite our work
Please cite our work as:
    
    @inproceedings{Fro23_a5,
      author={Frosio, Iuri and Kautz, Jan},
      year={2023},
      title={The Best Defense is a Good Offense: Adversarial Agumentation Against Adversarial Attacks},
      booktitle={CVPR},
    }
  

## License

### NVIDIA License

#### 1. Definitions

“Licensor” means any person or entity that distributes its Work.
“Work” means (a) the original work of authorship made available under this license, which may include software, documentation, or other files, and (b) any additions to or derivative works  thereof  that are made available under this license.
The terms “reproduce,” “reproduction,” “derivative works,” and “distribution” have the meaning as provided under U.S. copyright law; provided, however, that for the purposes of this license, derivative works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work.
Works are “made available” under this license by including in or with the Work either (a) a copyright notice referencing the applicability of this license to the Work, or (b) a copy of this license.

#### 2. License Grant

2.1 Copyright Grant. Subject to the terms and conditions of this license, each Licensor grants to you a perpetual, worldwide, non-exclusive, royalty-free, copyright license to use, reproduce, prepare derivative works of, publicly display, publicly perform, sublicense and distribute its Work and any resulting derivative works in any form.

#### 3. Limitations

3.1 Redistribution. You may reproduce or distribute the Work only if (a) you do so under this license, (b) you include a complete copy of this license with your distribution, and (c) you retain without modification any copyright, patent, trademark, or attribution notices that are present in the Work.

3.2 Derivative Works. You may specify that additional or different terms apply to the use, reproduction, and distribution of your derivative works of the Work (“Your Terms”) only if (a) Your Terms provide that the use limitation in Section 3.3 applies to your derivative works, and (b) you identify the specific derivative works that are subject to Your Terms. Notwithstanding Your Terms, this license (including the redistribution requirements in Section 3.1) will continue to apply to the Work itself.

3.3 Use Limitation. The Work and any derivative works thereof only may be used or intended for use non-commercially. Notwithstanding the foregoing, NVIDIA Corporation and its affiliates may use the Work and any derivative works commercially. As used herein, “non-commercially” means for research or evaluation purposes only.

3.4 Patent Claims. If you bring or threaten to bring a patent claim against any Licensor (including any claim, cross-claim or counterclaim in a lawsuit) to enforce any patents that you allege are infringed by any Work, then your rights under this license from such Licensor (including the grant in Section 2.1) will terminate immediately.

3.5 Trademarks. This license does not grant any rights to use any Licensor’s or its affiliates’ names, logos, or trademarks, except as necessary to reproduce the notices described in this license.

3.6 Termination. If you violate any term of this license, then your rights under this license (including the grant in Section 2.1) will terminate immediately.

#### 4. Disclaimer of Warranty.

THE WORK IS PROVIDED “AS IS” WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES OR CONDITIONS OF 
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT. YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER THIS LICENSE. 

#### 5. Limitation of Liability.

EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE SHALL ANY LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR RELATED TO THIS LICENSE, THE USE OR INABILITY TO USE THE WORK (INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS INTERRUPTION, LOST PROFITS OR DATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY OTHER DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

*Important note: Each user is responsible for checking the content of datasets and the applicable licenses and determining if suitable for the intended use and applicable links before the script runs and the data is placed in the user machine.*
