# DoctorNet Pytorch Implementation
Unofficial pytorch implementation of DoctorNet from "Who Said What: Modeling Individual Labelers Improves Classification"

Paper: https://arxiv.org/abs/1703.08774

## Experiments
We used [LabelMe](http://labelme.csail.mit.edu/Release3.0/) for validating implementation. LabelMe is an image classification task that was labeled by 77 annotators in AMT(Amazon Mechanical Turk).
Original data is [here](http://fprodrigues.com/deep_LabelMe.tar.gz).
However in the real data, 18 of the annotators didn't labeled any image.
So total 59 annotators' labels were used for training.
You can download the preprocessed version [here](https://postechackr-my.sharepoint.com/:f:/g/personal/shhj1998_postech_ac_kr/EiLGvgBa7fZLtEgFGoFGG-YBvTDLe2DqrB1TYgCClRUoBg?e=YJT2hU).

### Training
Training is done in two-stage.
First, we train annotator classifiers with shared feature extractor - Inception v3.
After the classifiers converged, we fix them and train weights used for averaging decisions of annotators.

#### Annotator classifier

#### Weights

### Testing
Model | DoctorNet(paper)[1] | doctornet-pytorch
--- | --- | --- 
Accuracy | 82.12 | 77.61

## 참조문헌
[1] https://arxiv.org/abs/2012.13052
