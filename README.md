# Student-Attendance-Tracking-Facial-Recognition-for-faces-with-without-maks-

The problem that we choose to tackle is tracking classroom attendance. Being present in the classroom, allows students to both learn and grow with their peers.Attendance is an important part of daily classroom evaluation. It allows students to learn and grow alongside their peers, enabling different perspectives and ways of understanding various concepts (which self-study do not capture).

Our goal here is to Manage classroom attendance in a more efficient manner. This could help better understand correlations between attendance and student performance.

However, there’s 1 challenge that exists, which is that many of the datasets does not contain mask images. We aim to train our model to record students attendance who wear and do not wear masks. Thus, we mask half of the images in train, validation and test sets. For example: if an individual has 500 images in the dataset, 250 of them would be masked. The open source tool 'Mask The Face'(Reference: https://github.com/aqeelanwar/MaskTheFace) is used here. This tool simulates masks of different combination of color, pattern and mask type on faces to train deep NNs.

Datasets used
- Knowing the problem in hand, we know many datasets exist that can be used for facial recognition. We use the following datasets for training and validation:

- Train and Test Datasets used: VGGFace2 (link)-The dataset used for training is VGGFace2 that has more than a million train images of 9000+ subjects. The images have large variations in pose, age, illumination etc.

- Validation Dataset used: LabeledFacesInTheWild (link)-LFW Dataset containing 13k images of 5000+ subjects.

- Final model evaluation(demonstration) will be on unseen data from classroom where the models record presence/absence of a student

First, for all 3 datasets(LFW, VGGFace2-Train, VGGFace2-Test), the images are first masked using an Open-source tool called "MaskTheFace". We considered different combination of color, pattern and mask type when generating the mask faces. Next all the images are cropped and ready to be processed by the model. Multiple image pairs are provided to the network which then maps them to embedding vectors and calculates triplet loss. The triplets consist of two image pairs of same(positive) and different(negative)persons. It separates the pair of same identity (positive pair) from the pair of different identities (negative pair) by a distance margin calculated as Euclidean(L2) distance. The best model is then evaluated on the test data
