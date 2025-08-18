**Title:** Semi-automated evaluation of perivascular fibrosis in histological sections

**Authors:** Abuzar Mahmood, Alex Manour, Kiet A. Duong, Lakshmi Pulakat

Molecular Cardiology Research Institute, Tufts Medical Center, Boston, MA

Department of Biology, Brandeis University, Waltham, MA

[**Abstract	1**](#abstract)

[**Introduction	2**](#introduction)

[**Results	5**](#results)

[Labelling Benchmark	5](#labelling-benchmark)

[CNN Benchmark	5](#cnn-benchmark)

[Interstitial and Perivascular FIbrosis in Healthy Wistar vs. Zucker Diabetic Fatty	5](#interstitial-and-perivascular-fibrosis-in-healthy-wistar-vs.-zucker-diabetic-fatty)

[**Discussion	5**](#discussion)

[**Conclusion	7**](#conclusion)

[**Methods	7**](#methods)

[Overview of CNNs for segmentation	7](#overview-of-cnns-for-segmentation)

[2.2. Proposed 11-layer CNN	7](#2.2.-overview-of-mask-r-cnn-for-segmentation)

[2.3. Experiments	9](#2.3.-experiments)

[2.3.1. Dataset	9](#2.3.1.-dataset)

[2.3.2. Manual thresholding and ground truth	9](#2.3.2.-manual-labelling-and-ground-truth)

[2.3.3. Data augmentation	9](#2.3.3.-data-augmentation)

[2.3.4. CNN training	10](#2.3.4.-cnn-training)

[2.3.5. Evaluation	11](#2.3.5.-evaluation)

[2.3.6. Comparison with previous methods	12](#2.3.6.-comparison-with-previous-methods)

[**References	12**](#references)

#

# Abstract {#abstract}

Annotation of histological images is a crucial task for many biomedical analyses. However, variability and complexity of structural features in histological images poses significant challenges for quantitative analyses. In much cardiac research, measurement of interstitial fibrosis is a crucial step in determining structural health of cardiac tissue; however, contamination from perivascular collagen significantly confounds this calculation, requiring removal of blood vessels prior to measurement. The conventional approach of manual annotation is 1\) labor-intensive, 2\) biased by user-specificity (not reproducible), and 3\) brittle (due to manual tuning), making it simply unscalable. Machine learning-based pipelines hold promise in providing a robust and reproducible solution to this issue, and while there is active research on use of such pipelines for measurement of renal pathology and detection of cancerous tissue, no work has been done on applying these models to cardiac assessment. Here, we present an end-to-end pipeline for quantification of interstitial and perivascular fibrosis in Mason’s Trichrome-stained heart tissue that combines image processing with deep learning for tissue detection, vessel demarcation, and fibrosis quantification. At the core of the pipeline is a mask-Convolutional Neural Network for segmentation of blood vessels. Cross-validation benchmarking of model predictions showed a median Dice Coefficient of 0.85 and a median pixel-wise prediction accuracy of 0.989, indicating robust prediction of blood vessel boundaries by the neural network. Time spent for inference of vessel boundaries per section by the model was 0.78s/image on a standard commercial machine, signifying the feasibility of running the pipeline on commercial hardware. As a “real-world” test of the pipeline, we analyzed fibrosis in trichrome-stained heart sections of healthy Wistar vs. Zucker Diabetic Fatty (ZDF) male rats. In concordance with established results, the pipeline was able to quantitatively show significantly higher inter-stitial fibrosis in ZDF rats, as compared to the Wistar controls.

Heterogeneity of samples

Inter and intra rater variability

Manual is semi quantitative at best

Interstitial fibrosis is associated with disease conditions and deteriorating heart health and function

Although it is not difficult to identify interstitial fibrosis and blood vessels, quantification is difficulty

Straw man : multi staining can be done to rigorously remove vessels but this is a more expensive and time consuming approach than trichrome

To some extent, predictions by machine learning algorithms are more robust than humans

One can expect that other architectures can be applied to the same problem

Stopping of training criterion using validation

Final performance evaluation using a test set

Details specs of training and inference machines

Since it is difficult to gather a large number of training images to train a network from scratch, fine tuning a pretrained network was chosen as the strategy

See paper for nice figs:

Microvascularity detection and quantification in glioma: a novel deep-learning \-based framework

Xieli Li¹². Qisheng Tang³ Jinhua Yu124. Yuanyuan Wang

Color standardization of slide prior to processing??

Reinhardt stain color normalization

Automated assessment can also overcome the tedious nature of visual assessment, which can be a limiting factor in large studies.

Digital image analysis has been studied widely to enable high-throughput, accurate, and reproducible assessment of digitized microscopic images of kidney tissue sections.

Lower size limit for detection \- jayapandian

Use of transfer learning? Vs pretrained models

Provides foundational model for future work on heart, similar to bouteldga 2019 and Holscher 2023

Future work could also incorporate unsurprised extraction of visual features

Gaussian noise+ blur as added augmentations

Multiple runs of model training

Slowing learning rate

Stain normalization is inferior to data augmentation

Bouteldja N, Hölscher DL, Bülow RD, Roberts ISD, Coppo R, Boor P. Tackling stain variability using CycleGAN-based stain augmentation. J Pathol Inform. 2022 Sep 13;13:100140. doi: 10.1016/j.jpi.2022.100140. PMID: 36268102; PMCID: PMC9577138.

But stain normalization is likely still useful for the fibrosis quantification step

Grid distortion/ elastic deformation for data augmentation

Discussion of mask rcnn as an 2-step anchor based model / technique

Number of params and gflops for model

GradCAM for visualizing model's focus

For discussion:

Currently, a major focus in computational pathology is the development of end-to-end DL solutions, which mostly provide qualitative results, e.g., a disease class or mutational status12,33–35. On the contrary, NGM and FLASH use segmentation as a basis for subsequent large-scale quantitative data mining. Compared to end-to-end pipelines, NGM provides an alternative approach with several advantages. The results are visually verifiable, can be easily checked by pathologists, and are therefore interpretable. This is often not the case in endto-end DL solutions, which remain a black-box in terms of explainability. Therefore, quantitative histology features remain comprehensible, even if clustered in a lower dimensional space. This can help reduce potential scepticism towards DL based systems that might hinder clinical application.

# Introduction {#introduction}

Accurate segmentation of biomedical images is fundamental for quantitative analysis. However, this task is challenging due to the characteristically high inhomogeneity and complexity of features in such images. Further, inter- and intra-plane artifacts and inconsistencies may be introduced into the acquired image as a result of the imaging procedure and methodology, such as variable lighting and stain color. Currently, to demarcate relevant image features, there are limited automated or semi-automated pipelines, leaving pathologists and researchers to rely heavily on manual annotation.

To address this critical need, we have developed AutoSlide, a comprehensive pipeline that transforms how researchers analyze histological slides by combining computer vision with deep learning to identify tissues, detect vessels, and quantify fibrosis with unprecedented precision. The AutoSlide pipeline includes modules for automated tissue recognition, smart region selection, advanced vessel detection, and fibrosis quantification, all within a reproducible workflow framework.

Manual thresholding remains the most prevalent method for segmenting biomedical images, for example to identify interstitial fibrosis or scarring in histology \\\[1,2\\\]. Whilst straightforward in principle, this approach is labor-intensive, time-consuming, and highly subjective, with usually no good way to aggregate differing labels across experimentalists. It may involve tedious re-adjustments of thresholds \\\[3,4\\\]. Thresholds are also highly sensitive to subject-dependent biases, as well as inter- and intra-image intensity variations (since spatial information is not accounted for) \\\[5,6\\\]. Thus, a variety of thresholds is commonly necessary for one image set. In addition, the time-intensivity of this process results in only small sections of the stained tissues being analyzed, leading to incomplete and biased results. For such reasons, manual annotation is not feasible for large datasets.

Interpretation of raw pixel intensities to image meaning or context is no trivial task for algorithms. A slight difference in image features such as illumination may be negligible to humans, but can result in a disparate algorithmic outcome. Numerous methods have been established to separate an image into groups displaying similar features, and thereby identify the class object of each pixel. Earlier segmentation techniques rely on distinguishing edges, regions, or textures \\\[6\\\]. However, for image data with highly irregular structural features, heterogeneous illumination, or variable coloring of similar objects, considerable pre- or post-processing is required, thus rendering such techniques unattractive and largely unsuitable.

The AutoSlide pipeline addresses these challenges through a multi-stage approach that includes:

1\. Initial annotation through intelligent thresholding and region identification

2\. Final annotation with precise tissue labeling and mask generation

3\. Region suggestion for strategic selection of analysis-ready sections

4\. Pixel clustering for advanced segmentation of tissue components

5\. Vessel detection using deep learning-based identification of vascular structures

Robust and automated segmentation methods that can overcome the inherent challenges of biomedical image segmentation are in great demand, especially for applications conventionally relying on a manual approach. An example is fibrosis identification in histology, a critical task in many key fields of clinical research including kidney failure \\\[7\\\], lung injury \\\[8\\\], hepatitis B \\\[9\\\], sinoatrial node \\\[1\\\], and atrial fibrillation \\\[10\\\].

Atrial fibrillation is the most common type of cardiac arrhythmia, associated with significant healthcare costs, reduced quality of life, morbidity and mortality \\\[11\\\]. The basic mechanisms behind its initiation and maintenance remain elusive, but accumulating recent evidence indicates that diabetes mellitus (DM) is a strong independent risk factor \\\[10–13\\\], and that atrial fibrosis or scarring (characterized by excessive extracellular matrix proteins including collagen) induced under diabetic conditions contributes considerably to arrhythmogenicity \\\[10,14,15\\\]. Quantification and comparison of atrial fibrotic remodeling in DM against controls will assist in illuminating the precise mechanisms underlying DM-induced atrial fibrillation. This requires segmentation of fibrosis from myocytes and background in a cardiac histological section, differentiated via the well-accepted Masson's trichrome stain (Fig. 1).

In recent years, machine learning for computer vision has advanced extensively, emerging as a powerful tool in a wide range of image recognition problems \\\[16–19\\\]. Machine learning methods can be generally classed as unsupervised or supervised. In the former, the algorithm identifies patterns in the input without learning from example data annotated with desired outputs (ground truth). Unsupervised methods, including k-means clustering \\\[20,21\\\], mean-shift clustering \\\[22\\\], and Markov random fields \\\[23\\\], as well as earlier supervised approaches such as support vector machines \\\[19\\\], have previously been employed to segment histological images. However, these methods typically suffer from the requirement for supplementary algorithms (e.g. manual thresholding and smoothing functions for post-processing \\\[21\\\]) to complete the segmentation objective, or additional domain expertise to define and extract suitable features from images, which are often based on strong assumptions about the data. In contrast to unsupervised methods, supervised models are trained on labelled data, learning rules to produce outputs from inputs.

Convolutional neural networks (CNNs) are generating great enthusiasm particularly in computer vision. Conceptualized in the 1980s \\\[24\\\], CNNs were biologically inspired by the visual cortex, where neurons fire in response to certain features or patterns in their local receptive fields, thereby acting as spatial filters \\\[25\\\]. CNNs effectively map highly complex relationships between the input and desired output (such as those of shapes and colors present in images), through interconnected stacks of various nonlinear functions. The most fundamental function is convolution. Contrary to alternative supervised learning approaches such as support vector machines, there is no manual hand-crafting or fine-tuning of useful features in the input. CNNs can achieve impressive performance and directly handle complex data with minimal manual effort. A CNN-based approach is fully automated and trained models are reusable after establishment.

Although the inception of neural networks was a few decades ago, deep networks with multiple stacked layers are a relatively recent development. This has been catalyzed by progress in multiple areas such as parallelized computation using GPUs, solutions to hindrances associated with training deep neural networks (such as rectified linear units (ReLUs) for the vanishing gradient problem \\\[26\\\]), and the availability of very large datasets. Deep CNNs have proved to be powerful tools in a wide array of image-related applications, excelling in image classification \\\[16,27,28\\\], handwriting recognition \\\[29\\\], object localization \\\[30\\\], and scene understanding \\\[31,32\\\]. This technique has also been successfully extended to semantic pixel-wise labeling and the biomedical domain in tasks such as image segmentation \\\[33–35\\\], detection \\\[36\\\], cell tracking \\\[17\\\], and computer-aided diagnosis \\\[18\\\]. In cardiac imaging, CNNs have been applied to identify various features including left/right ventricles, endocardium/myocardium, pathologies, and coronary artery calcification from magnetic resonance imaging, computed tomography, and ultrasound \\\[37–40\\\].

During a forward pass through a CNN, characteristics specific to certain structures in an input image (such as intensity and spatial information) are discerned by trainable filters in convolutional layers. Convolutional filters (typically 3  3 pixels) sweep across the entire visual field of the input volume by a constant stride, thus allowing the CNN to detect features in the input regardless of their exact position. Convolutional layers compute dot products between learnable filter weights and a corresponding region of the input slice, generating activation or feature maps. Thus, activations contain contextual representations of the image, with larger values indicating better resemblance between weights and intensity patterns in the receptive field of a filter. Stacking of many convolutional layers can amplify the capacity of a CNN to identify features of greater complexity from a larger field of view in the input. This ability may be further enhanced by the inclusion of other nonlinear operations such as ReLUs. Typically, pooling layers are inserted to subsample outputs from convolutions, and feature maps progressively increase in abstraction through the network as low-level information is compounded over many subsampling operations. Weights are automatically adapted during supervised learning to optimize sensitivity to certain features of relevance and maximize the accuracy of predictions compared to ground truth. A loss function quantifying the disparity between predictions and ground truth is minimized via gradient descent (e.g. Adam optimizer \\\[41\\\]), and errors are backpropagated through the network to modify weights accordingly.

In this paper, we present AutoSlide, a comprehensive pipeline for automated segmentation of histology images into a required number of tissue types, with a particular focus on fibrosis identification in cardiac histological images. The deep CNN at the core of AutoSlide displays state-of-the-art segmentation accuracy with drastically fewer parameters and substantially greater efficiency than comparable approaches. The pipeline leverages powerful Python libraries including slideio for efficient slide image handling, PyTorch for deep learning model training and inference, scikit-learn for clustering and dimensionality reduction, OpenCV for advanced image processing, and matplotlib/pandas for visualization and data handling.

To ensure reproducibility and version control of models and data, AutoSlide implements Data Version Control (DVC), allowing researchers to track large files and model artifacts efficiently. This approach enables precise versioning of models, facilitating collaboration and ensuring consistent results across multiple samples and studies.

More importantly, our proposed pipeline can potentially be extended to other similar segmentation tasks to facilitate understanding of certain diseases and to aid targeted clinical treatment. We make our source code freely available online at https://github.com/pulakatlab/auto\_slide for the benefit of potential users.

# Results {#results}

## Labelling Benchmark {#labelling-benchmark}

Inter-User Reliability
Positive and negative images
Distribution of labels per training image

## CNN Benchmark {#cnn-benchmark}

Accuracy and Speed
Intersection over union

## Interstitial and Perivascular FIbrosis in Healthy Wistar vs. Zucker Diabetic Fatty {#interstitial-and-perivascular-fibrosis-in-healthy-wistar-vs.-zucker-diabetic-fatty}

# Discussion {#discussion}

First, to illuminate the shortfalls of k-means clustering for the image segmentation problem, we visualized segmented class clusters according to the original pixel RGB values (Fig. 10). The clusters segmented by the proposed CNN were highly irregular and interlaced in some regionsContrastingly, clusters partitioned by k-means were well-separated and roughly equal in shape and volume. Fig. 10 indicates that the images do not satisfy key assumptions of the k-means algorithm; namely spherical cluster distributions of similar variance, and equal class probabilities for every data point \[64\]. As well as lacking recognition of contextual and spatial information (unlike CNNs), k-means uses the same rigid Euclidean distance minimization for all classes, despite inequality of cluster shape and size, or ill-defined boundaries. These limitations may be overcome by special transformations serving as preprocessing. However, determining suitable transformations is highly challenging. Since their success depends heavily on exact intensities, unique transformations tailored for different local regions are likely required, rendering this approach labor-intensive, and sensitive to inter- and intra-image intensity variations (similar to manual thresholding). Further, intensities are perceived by the human visual system in a highly nonlinear mapping, so although separation of the classes may be trivial by eye, the same task performed with only pixel values is much more difficult. In contrast to k-means, CNNs map highly complex relationships between intensities and desired outputs automatically, are robust to intensity variations (Fig. 6), and produce markedly superior results (Table 1 and Fig. 7). Despite being fully automated, these results strongly suggest that k-means clustering is unattractive for this application, primarily due to its inadequacy in segmenting similar images in their raw form to a satisfactory standard. Our proposed 11-layer CNN designed for per-pixel classification deviates from conventional forms, particularly in its exclusion of subsampling layers (such as pooling) and upsampling layers, and its consistent number of intermediate feature channels through the network. We use only three types of functions (convolution, ReLU, and batch normalization) arranged in an uncomplicated configuration. The network's state-of-the-art results (Table 1\) indicate that subsampling is not essential for data characterized by high variance of features between adjacent pixels and between different local regions, such as cardiac histological images. The aforementioned distinctions of the proposed CNN give it several advantages over state-of-the-art architectures designed for segmentation: maintains image resolution, reduces the total number of learnable parameters, improves efficiency, prevents overfitting to training data, and allows fine-grained details to be captured accurately. We speculate that the last two benefits may have contributed to the strong segmentation performance of our proposed method. Designing a suitable neural network for an application can be difficult, as it involves selections from a vast number of possibilities, and there is no set of universal guidelines. Further, assessment of architectures and hyperparameters can be tedious. Many recent networks are characterized by substantial depth, or a diverse range of arrangements and connections between intermediate layers. Our work shows that shallower and less complicated networks can offer impressive performance. It may be worthwhile to start simple. Although our CNN performs strongly against the state-of-the-art, our dataset consists of one type of biomedical image. The inclusion of other types of images (e.g. of other tissues or stains) will facilitate performance benchmarking and further enhancements to the proposed CNN. A high variability of image features in the dataset will be advantageous. One improvement to the proposed method would be more robust tolerance of variable brightness and contrast. This could involve the use of geometric or spatial context. Further, our results indicate that it is valuable to investigate alternative or novel methods to counter issues imposed by class imbalance in training data.

# Conclusion  {#conclusion}

In this paper, we proposed a novel 11-layer CNN and demonstrated the supervised learning-based approach for the application of histological image segmentation, particularly for fibrosis identification via Masson's trichrome staining. With an elegant configuration, drastically fewer parameters, and superior efficiency, the CNN outperformed the state-ofthe-art on our image set. This approach is also robust to typical variations in image illumination and stain color. For best results, learning data should capture a rough representation of the characteristics in the total image set, including proportions of each class, and variations in color of certain structures.

# Methods {#methods}

## Overview of CNNs for segmentation  {#overview-of-cnns-for-segmentation}

In contrast to CNNs for image classification where the output is a single class (for example “dog”), CNNs for image segmentation require dense per-pixel classification and spatial localization of classes in the form of an output segmentation map. The recent advancement of deep CNNs for segmentation was pioneered by the development of a fully convolutional network (FCN) by Long et al. \[42\]. Currently, the most prevalent and successful CNNs for segmentation are inspired by the scheme of FCN, adapting configurations originally designed for classification to perform per-pixel labeling by substituting fully connected layers with convolutions \[17,32,43,44\]. Such architectures consist of downsampling and upsampling stages, also known respectively as an encoder and decoder. The image is first downsampled by a series of convolutions and max pooling to obtain low resolution feature maps, which are then upsampled by deconvolution (also named convolutional transpose or fractionally strided convolution), generating the localization of classes desired. In some cases, concatenation between intermediate feature maps, or unpooling \[45\] is incorporated to improve final resolution. However, additional upsampling layers introduce more trainable parameters in such architectures, which are commonly in the order of tens to hundreds of millions \[17,32,42,44\]. Thus, they are susceptible to overfitting, particularly when available data is scarce, and may be difficult to train end-to-end. Additionally, results may be too coarse due to subsampling by multiple max pooling operations. These configurations may not be optimal for all types of images and segmentation goals due to unique feature properties.

## 2.2. Overview of Mask R-CNN for segmentation  {#2.2.-overview-of-mask-r-cnn-for-segmentation}

## 2.3. Experiments  {#2.3.-experiments}

Supervised learning in a CNN is typically based on a large dataset consisting of images and corresponding labels. Training is followed by an evaluation phase where predictions are produced from the trained model, and performance is analyzed. The following subsections present the detailed steps we implemented to address the image segmentation problem for vessel identification, and to benchmark segmentation performance of the proposed approach.

### 2.3.1. Dataset  {#2.3.1.-dataset}

The dataset consists of XXX total images, 36 each from control and DM groups, all 2064  1536 pixels (width  height) in size with 3 color channels (RGB). The histological images are courtesy of Fu et al., using the methods detailed in Ref. \[10\]. Briefly, left atrial sections of control and DM (induced with alloxan monohydrate) Japanese rabbits were stained with Masson's trichrome and imaged with an Olympus DP72 at 40 objective magnification. Each pixel has a spatial resolution of 161.25 nm  161.25 nm.

### 2.3.2. Manual labelling and ground truth  {#2.3.2.-manual-labelling-and-ground-truth}

Manual thresholding is presently the conventional approach for identifying tissue type and fibrosis in histology sections with Masson's trichrome stain. In these images, fibrosis is colored blue and myocytes red, while white is regarded as background. We first meticulously applied manual thresholding to all the original images by using manually toned regional thresholds and detailed touch-ups, generating segmentation ground truth. To ensure our ground truth was precise, we validated the segmentation results with experts in the field \[10\]. To facilitate segmentation, the images were preprocessed prior to thresholding via histogram normalization, to standardize the minimum and maximum intensities in each RGB channel to 0 and 255, respectively. We employed multiband thresholding for higher segmentation accuracy, involving combinations of different thresholds across channels to isolate classes, while ensuring no pixel was unclassed or multi-classed. Thresholds were selected through repeated empirical trials and visual validation for each image.

### 2.3.3. Data augmentation  {#2.3.3.-data-augmentation}

We randomly selected 24 original images (12 each from control and DM groups) to construct the training set. Since the performance of neural networks is generally improved with more training data, the amount of available data for training was amplified by augmentation, a technique popular in the field of deep learning for image classification \[16,49\]. Augmentation also helps to reduce overfitting, improving model invariance to adjustments negligible to classification outcome. We applied the following independent transformations identically to each original image and its corresponding labels, then randomly sampled 48  48 patches from augmented forms (number of patches in brackets):  Rotation by 90 (450), 180 (900), or 270 (450);  Flipping along the horizontal (450) or vertical axis (450);  Sinusoidal warping (900);  Shearing affine transformation (900). A maximum of 900 patches were obtained from one augmented version, which is roughly two-thirds the image area in pixel count. To ensure that each patch captured adequate information and avoided extremely biased proportions of pixels for any classes, we excluded from the training set the highest 4/9 of patches as ranked by the standard deviation of their class proportions. We then randomly discarded 96 patches so that the total number of patches was divisible by 128, the size of the mini-batches during gradient descent (details of training processes are in Section 2.3.4). The training set consisted of 59,904 48  48 patches in total, containing about 138 million pixels. Prior to training, we randomized the order of training data with the intent of achieving smoother convergence during gradient descent. Due to random sampling, the proportions of each class in the training set roughly reflected those in the test set, with myocytes showing the greatest prevalence (44%), followed by background (32%), and fibrosis the least prevalent (24%). Data class imbalance is a major problem in supervised classifiers, detrimentally impacting minority classes in particular \[50,51\]. A vast array of strategies has been devised for overcoming this recurrent problem (although with variable success), including oversampling, undersampling, retaining natural proportions in learning examples, data synthesis, and class-weighted loss functions \[50\]. With the intent of improving segmentation performance for fibrosis, we carried out another training of the proposed CNN with a different training set consisting of approximately balanced class proportions (35% myocyte, 34% background, and 31% fibrosis), obtained by oversampling fibrosis. This training set consisted of the same number of 48  48 patches as the standard training set. We also assessed the ability of our approach to withstand the variations in color and brightness typical of Masson's trichrome-stained histology sections, by evaluating the performance of the proposed CNN on color-adjusted test images. The maximum extents of the alterations are shown in Fig. 3\. Further, we performed an independent training of the CNN using a color-augmented version of the standard training set. We randomly selected 40% of patches to undergo contrast adjustment via the red channel, and a different 40% via the blue channel. Training images were color-adjusted by a random degree up to the predefined maximum limits.

### 2.3.4. CNN training  {#2.3.4.-cnn-training}

Training is an iterative process in which (i) training data are fed into the model in batches, (ii) predictions are produced by the current model in a forward pass, (iii) errors between predictions and ground truth labels are computed, (iv) errors are backpropagated through the network, (v) parameter corrections for all neurons are computed, and (vi) parameters are updated to minimize errors. A single cycle over the entire training set is an epoch, and typically multiple epochs are needed for convergence. We use the TensorFlow framework \[52\] to implement the CNN, and perform end-to-end training from scratch. For the first iteration of training, we initialize convolutional weights according to the recommendations of He et al. \[53\]. We sample weights from a normal distribution centered on zero, truncated at two standard deviations from the mean, with variance 2/n, where n is the product of the number of input feature channels, filter height, and filter width for a given layer. For preprocessing during model training and evaluation, we subtract training set RGB means from the corresponding channel, then normalize each channel to unit variance via division of its standard deviation. During learning, we compute cross-entropy as the loss function, which measures dissimilarity between per-pixel class distributions of ground truth and estimated probabilities. First, we normalize predicted probabilities to unit sum for each pixel via the softmax function in Eq. (2), where z contains predicted probabilities for K classes, and fj corresponds to the j-th element in the vector of softmax probabilities f. fjðzÞ ¼ ezj PK k¼1ezk (2) We then determine the loss of each pixel point Li by Eq. (3), where yi is the correct class, and j is the index along vector f. We compute loss as the mean loss across all pixels in a mini-batch. Li ¼ log efyi PJ j¼1efj \! (3) Training is carried out via gradient descent with training images in mini-batches set at size 128\. Weights are updated every mini-batch, and one revised model is produced every epoch. We minimize cross-entropy loss during stochastic optimization using Adam \[41\], which uses adaptive learning rates for smooth convergence. We set the three parameters of Adam – the upper bound learning rate, and exponential decay rates for the first and second moment estimates – to their default values of 0.001, 0.9, and 0.999, respectively. The order of mini-batches is re-randomized every epoch except for the last batch, to simplify tracking for the visualization of training progress. We end training when the mean Dice similarity coefficient (DSC, defined in Section 2.3.5) for the test set does not increase by at least 1% after 20 further epochs from its current best epoch. The best model is the one producing the highest test mean DSC.

### 2.3.5. Evaluation  {#2.3.5.-evaluation}

Our primary segmentation performance metric is the widely-adopted DSC, which assesses spatial overlap by combining precision and recall in the form of a harmonic mean \[54\]. We also measure intersection over union (IoU), a common metric for semantic segmentation, and the evaluation standard in Pascal VOC2012 \[55\]. Both are measures of overlapping areas of mutual class assignment but differ slightly in formulation. DSC and IoU scores are within (0, 1), with higher values indicating better performance. Preliminary to computation of the two metrics is the construction of a confusion matrix between ground truth and predictions, allowing the tallying of true positive (TP), false positive (FP), false negative (FN), and true negative (TN) outcomes predicted by the classifier for each class. We compute DSC and IoU scores for class c using Eqs. (4) and (5), respectively: DSCc ¼ 2  TPc 2  TPc þ FPc þ FNc (4) IoUc ¼ TPc TPc þ FPc þ FNc (5) Class scores are subsequently averaged for each image, yielding image DSC and image IoU. We distill the overall performance of a classifier into a single value for each metric by averaging respective image scores across the 48-image test set, yielding mean DSC and mean IoU. We also report mean scores for each class, determined in an analogous manner. To select the top-performing model from each training instance, each model produced predictions for the entire test set, and the basis for model selection was its overall mean DSC. We visually scrutinized all segmentation outputs from machine learning approaches to confirm their accuracy.

### 2.3.6. Comparison with previous methods  {#2.3.6.-comparison-with-previous-methods}

We compared the performance of the proposed architecture for the per-pixel classification task at hand against well-established machine learning methods, including two well-adopted CNNs for segmentation, FCN-8 \[42\] and U-Net \[17\], and the unsupervised k-means clustering algorithm. FCN-8 is the most refined version of the FCNs, and a landmark development in recent progress of CNNs for image segmentation, achieving a 20% improvement in performance against traditional approaches for standard datasets. U-Net was designed specifically for biomedical image segmentation, and has proved its superiority by winning several contests \[17,56–59\]. The network can achieve high performance with very few training data and has become widely popular, with adaptations designed for many applications \[60–62\]. For learning and testing, we used identical data to our proposed architecture, trained all networks from scratch as outlined in Section 2.3.4, and followed the same evaluation procedures presented in Section 2.3.5. The only modification we made to the original U-Net was the use of zero padding during convolutions, to preserve the lateral size of the input volume at such steps. For FCN-8 and U-Net, we initialized all biases to zero. We employed dropout during training for the two fully connected layers in FCN-8, and all convolutional layers except the last in U-Net. We experimentally determined dropout rates of 0.5 and 0 (no dropout) respectively for FCN-8 and U-Net to yield the best performance on the test set. We present their performance scores accordingly. We also compared segmentation performance against that of k-means clustering, a widely-used unsupervised machine learning algorithm,which partitions N unlabeled observations (pixels in the case of images) into K groups \[63\]. In the case of RGB image segmentation into K ¼ 3 classes, three centroids exist, each located at RGB intensities deemed optimal by the algorithm. We also performed k-means in Lab space after conversion from RGB. The performed iterative process is as follows: i. Random initialization of cluster centroids (Ck, k ¼ 1 … K); ii. Assignment of data points (xn, n ¼ 1 … N) to clusters with current centroids of minimum Euclidean distance jjxn – Ckjj away; iii. Computation of new centroids using updated groupings; iv. Repetition of steps ii and iii until convergence. To avoid results in local minima during minimization of the total Euclidean distance between points and centroids, the algorithm was performed three times. The final partitions with the lowest sum of distances from all points to centroids was selected. Since k-means is unsupervised, the classes of segmentations are unknown. Our strategy was to produce all six permutations possible with three classes and select the one which scored highest in overall test mean DSC.

# References  {#references}

\[1\] T.A. Csepe, J. Zhao, B.J. Hansen, N. Li, L.V. Sul, P. Lim, Y. Wang, O.P. Simonetti, A. Kilic, P.J. Mohler, P.M.L. Janssen, V.V. Fedorov, Human sinoatrial node structure: 3D microanatomy of sinoatrial conduction pathways, Prog. Biophys. Mol. Biol. 120 (2016) 164–178, https://doi.org/10.1016/j.pbiomolbio.2015.12.011.

\[2\] B.J. Hansen, J. Zhao, V.V. Fedorov, Fibrosis and atrial fibrillation: computerized and optical mapping: a view into the human atria at submillimeter resolution, JACC Clin. Electrophysiol. 3 (2017) 531–546, https://doi.org/10.1016/ j.jacep.2017.05.002.

\[3\] J. Zhao, T.D. Butters, H. Zhang, I.J. Legrice, G.B. Sands, B.H. Smaill, Image-based model of atrial anatomy and electrical activation: a computational platform for investigating atrial arrhythmia, IEEE Trans. Med. Imag. 32 (2013) 18–27, https:// doi.org/10.1109/TMI.2012.2227776.

\[4\] J. Zhao, T.D. Butters, H. Zhang, A.J. Pullan, I.J. LeGrice, G.B. Sands, B.H. Smaill, An image-based model of atrial muscular architecture effects of structural anisotropy on electrical activation, Circ. Arrhythmia Electrophysiol. 5 (2012) 361–370, https://doi.org/10.1161/CIRCEP.111.967950.

\[5\] M.N. Gurcan, L.E. Boucheron, A. Can, A. Madabhushi, N.M. Rajpoot, B. Yener, Histopathological image analysis: a review, IEEE Rev. Biomed. Eng. 2 (2009) 147–171, https://doi.org/10.1109/RBME.2009.2034865.

\[6\] D.L. Pham, C. Xu, J.L. Prince, Current methods in medical image segmentation, Annu. Rev. Biomed. Eng. 2 (2000) 315–337, https://doi.org/10.1146/ annurev.bioeng.2.1.315.

\[7\] E.M. Zeisberg, S.E. Potenta, H. Sugimoto, M. Zeisberg, R. Kalluri, Fibroblasts in kidney fibrosis emerge via endothelial-to-mesenchymal transition, J. Am. Soc. Nephrol. 19 (2008) 2282–2287, https://doi.org/10.1681/ASN.2008050513.

\[8\] L.B. Marks, X. Yu, Z. Vujaskovic, W. Small, R. Folz, M.S. Anscher, Radiation-induced lung injury, Semin. Radiat. Oncol. (2003) 333–345, https://doi.org/10.1016/ S1053-4296(03)00034-1.

\[9\] T.T. Chang, Y.F. Liaw, S.S. Wu, E. Schiff, K.H. Han, C.L. Lai, R. Safadi, S.S. Lee, W. Halota, Z. Goodman, Y.C. Chi, H. Zhang, R. Hindes, U. Iloeje, S. Beebe, B. Kreter, Long-term entecavir therapy results in the reversal of fibrosis/cirrhosis and continued histological improvement in patients with chronic hepatitis B, Hepatology 52 (2010) 886–893, https://doi.org/10.1002/hep.23785.

\[10\] H. Fu, G. Li, C. Liu, J. Li, X. Wang, L. Cheng, T. Liu, Probucol prevents atrial remodeling by inhibiting oxidative stress and TNF-α/NF-κB/TGF-β signal transduction pathway in alloxan-induced diabetic rabbits, J. Cardiovasc. Electrophysiol. 26 (2015) 211–222, https://doi.org/10.1111/jce.12540.

\[11\] W.B. Kannel, P.A. Wolf, E.J. Benjamin, D. Levy, Prevalence, incidence, prognosis, and predisposing conditions for atrial fibrillation: population-based estimates, Am. J. Cardiol. 82 (1998) 2N–9N, https://doi.org/10.1016/S0002-9149(98)00583-9.

\[12\] H. Fu, G. Li, C. Liu, J. Li, L. Cheng, W. Yang, G. Tse, J. Zhao, T. Liu, Probucol prevents atrial ion channel remodeling in an alloxan-induced diabetes rabbit model, Oncotarget 7 (2016) 83850–83858, https://doi.org/10.18632/oncotarget.13339.

\[13\] T. Kato, T. Yamashita, A. Sekiguchi, T. Tsuneda, K. Sagara, M. Takamura, S. Kaneko, T. Aizawa, L.T. Fu, AGEs-RAGE system mediates atrial structural remodeling in the diabetic rat, J. Cardiovasc. Electrophysiol. 19 (2008) 415–420, https://doi.org/ 10.1111/j.1540-8167.2007.01037.x.

\\\[14\\\] S. Nattel, B. Burstein, D. Dobrev, Atrial remodeling and atrial fibrillation: mechanisms and implications, Circ. Arrhythm. Electrophysiol. 1 (2008) 62–73, https://doi.org/10.1161/CIRCEP.107.754564.

\\\[15\\\] J. Zhao, B.J. Hansen, Y. Wang, T.A. Csepe, L. V Sul, A. Tang, Y. Yuan, N. Li, A. Bratasz, K.A. Powell, A. Kilic, P.J. Mohler, P.M.L. Janssen, R. Weiss, O.P. Simonetti, J.D. Hummel, V. V Fedorov, Three-dimensional integratedfunctional, structural, and computational mapping to define the structural “fingerprints” of heart-specific atrial fibrillation drivers in human heart ex vivo, J. Am. Heart Assoc. (2017), https://doi.org/10.1161/JAHA.117.005922.

\\\[16\\\] A. Krizhevsky, I. Sutskever, G.E. Hinton, ImageNet classification with deep convolutional neural networks, Adv. Neural Inf. Process. Syst. (2012) 1–9, https:// doi.org/10.1016/j.protcy.2014.09.007.

\\\[17\\\] O. Ronneberger, P. Fischer, T. Brox, U-Net: convolutional networks for biomedical image segmentation, Med. Image Comput. Comput. Interv. – MICCAI 2015 (2015) 234–241, https://doi.org/10.1007/978-3-319-24574-4\\\_28.

\\\[18\\\] D.C. Ciresan, A. Giusti, L.M. Gambardella, J. Schmidhuber, Mitosis detection in breast cancer histology images using deep neural networks, Proc. Med. Image Comput. Comput. Assist. Interv. (2013) 411–418, https://doi.org/10.1007/978-3- 642-40763-5\\\_51.

\\\[19\\\] J.M. Chen, A.P. Qu, L.W. Wang, J.P. Yuan, F. Yang, Q.M. Xiang, N. Maskey, G.F. Yang, J. Liu, Y. Li, New breast cancer prognostic factors identified by computer-aided image analysis of HE stained histopathology images, Sci. Rep. 5 (2015), https://doi.org/10.1038/srep10690.

\\\[20\\\] O. Sertel, J. Kong, G. Lozanski, A. Shana ’Ah, U. Catalyurek, J. Saltz, M. Gurcan, Texture classification using nonlinear color quantization: application to histopathological image analysis, in: ICASSP, IEEE Int. Conf. Acoust. Speech Signal Process. \\- Proc., 2008, pp. 597–600, https://doi.org/10.1109/ ICASSP.2008.4517680.

\\\[21\\\] J.C. Sieren, J. Weydert, A. Bell, B. De Young, A.R. Smith, J. Thiesse, E. Namati, G. McLennan, An automated segmentation approach for highlighting the histological complexity of human lung cancer, Ann. Biomed. Eng. 38 (2010) 3581–3591, https://doi.org/10.1007/s10439-010-0103-6.

\\\[22\\\] G. Wu, X. Zhao, S. Luo, H. Shi, Histological image segmentation using fast mean shift clustering method, Biomed. Eng. Online 14 (2015), https://doi.org/10.1186/ s12938-015-0020-x.

\\\[23\\\] V. Meas-Yedid, S. Tilie, J.-C. Olivo-Marin, Color image segmentation based on Markov random field clustering for histological image analysis, Object Recognit. Support. by User Interact. Serv. Robot. 1 (2002) 796–799, https://doi.org/ 10.1109/ICPR.2002.1044879.

\\\[24\\\] K. Fukushima, Neocognitron: a self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position, Biol. Cybern. 36 (1980) 193–202, https://doi.org/10.1007/BF00344251.

\\\[25\\\] D.H. Hubel, T.N. Wiesel, Receptive fields and functional architecture of monkey striate cortex, J. Physiol. 195 (1968) 215–243, https://doi.org/10.1113/ jphysiol.1968.sp008455.

\\\[26\\\] V. Nair, G.E. Hinton, Rectified linear units improve restricted Boltzmann machines, in: Proc. 27th Int. Conf. Mach. Learn., 2010, pp. 807–814 doi:10.1.1.165.6419.

\\\[27\\\] K. Simonyan, A. Zisserman, Very deep convolutional networks for large-scale image recognition, Int. Conf. Learn. Represent. (2015) 1–14, https://doi.org/10.1016/ j.infsof.2008.09.005.

\\\[28\\\] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in: 2016 IEEE Conf. Comput. Vis. Pattern Recognit., 2016, pp. 770–778, https:// doi.org/10.1109/CVPR.2016.90.

\\\[29\\\] A. Poznanski, L. Wolf, CNN-N-Gram for handwriting word recognition, in: 2016 IEEE Conf. Comput. Vis. Pattern Recognit., 2016, pp. 2305–2314, https://doi.org/ 10.1109/CVPR.2016.253.

\\\[30\\\] S. Ren, K. He, R. Girshick, J. Sun, Faster R-CNN: towards real-time object detection with region proposal networks, IEEE Trans. Pattern Anal. Mach. Intellect. 39 (2017) 1137–1149, https://doi.org/10.1109/TPAMI.2016.2577031.

\\\[31\\\] C. Farabet, C. Couprie, L. Najman, Y. Lecun, Learning hierarchical features for scene labeling, IEEE Trans. Pattern Anal. Mach. Intellect. 35 (2013) 1915–1929, https:// doi.org/10.1109/TPAMI.2012.231.

\\\[32\\\] V. Badrinarayanan, A. Kendall, R. Cipolla, SegNet: a deep convolutional encoderdecoder architecture for image segmentation, IEEE Trans. Pattern Anal. Mach. Intellect. 39 (2017) 2481–2495, https://doi.org/10.1109/TPAMI.2016.2644615.

\\\[33\\\] P. Liskowski, K. Krawiec, Segmenting retinal blood vessels with deep neural networks, IEEE Trans. Med. Imag. 35 (2016) 2369–2380, https://doi.org/10.1109/ TMI.2016.2546227.

\\\[34\\\] S. Pereira, A. Pinto, V. Alves, C.A. Silva, Brain tumor segmentation using convolutional neural networks in MRI images, IEEE Trans. Med. Imag. 35 (2016) 1240–1251, https://doi.org/10.1109/TMI.2016.2538465.

\\\[35\\\] J. Xu, X. Luo, G. Wang, H. Gilmore, A. Madabhushi, A deep convolutional neural network for segmenting and classifying epithelial and stromal regions in histopathological images, Neurocomputing 191 (2016) 214–223, https://doi.org/ 10.1016/j.neucom.2016.01.034.

\\\[36\\\] A. Cruz-Roa, A. Basavanhally, F. Gonzalez, H. Gilmore, M. Feldman, S. Ganesan, N. Shih, J. Tomaszewski, A. Madabhushi, Automatic detection of invasive ductal carcinoma in whole slide images with convolutional neural networks, in: Proc. SPIE, 2014, https://doi.org/10.1117/12.2043872, p. 904103\\\_1-904103\\\_15.

\\\[37\\\] Z. Xiong, V. Fedorov, X. Fu, E. Cheng, R. Macleod, J. Zhao, Fully automatic left atrium segmentation from late gadolinium enhanced magnetic resonance imaging using a dual fully convolutional neural network, IEEE Trans. Med. Imag. (2018) (Under revision).

\\\[38\\\] Phi Vu Tran, A Fully Convolutional Neural Network for Cardiac Segmentation in Short-Axis Mri, 2016\\. ArXiv Prepr. ArXiv1604.00494, https://arxiv.org/abs/1604. 00494\\.

\\\[39\\\] O. Oktay, E. Ferrante, K. Kamnitsas, M. Heinrich, W. Bai, J. Caballero, S.A. Cook, A. De Marvao, T. Dawes, D.P. O'Regan, B. Kainz, B. Glocker, D. Rueckert, Anatomically constrained neural networks (ACNNs): application to cardiac image enhancement and segmentation, IEEE Trans. Med. Imag. 37 (2018) 384–395, https://doi.org/10.1109/TMI.2017.2743464.

\\\[40\\\] J.M. Wolterink, T. Leiner, B.D. de Vos, R.W. van Hamersvelt, M.A. Viergever, I. Isgum, Automatic coronary artery calcium scoring in cardiac CT angiography using paired convolutional neural networks, Med. Image Anal. 34 (2016) 123–136, https://doi.org/10.1016/j.media.2016.04.004.

\\\[41\\\] D.P. Kingma, J.L. Ba, Adam: a method for stochastic optimization, Int. Conf. Learn. Representation 2015 (2015) 1–15. http://doi.acm.org.ezproxy.lib.ucf.edu/10. 1145/1830483.1830503.

\\\[42\\\] J. Long, E. Shelhamer, T. Darrell, Fully convolutional networks for semantic segmentation, IEEE Trans. Pattern Anal. Mach. Intellect. 39 (2017) 640–651, https://doi.org/10.1109/TPAMI.2016.2572683.

\\\[43\\\] V. Badrinarayanan, A. Handa, R. Cipolla, SegNet: a Deep Convolutional EncoderDecoder Architecture for Robust Semantic Pixel-Wise Labelling, 2015\\. ArXiv Prepr. ArXiv1505.07293, http://arxiv.org/abs/1505.07293.

\\\[44\\\] H. Noh, S. Hong, B. Han, Learning deconvolution network for semantic segmentation, in: Proc. IEEE Int. Conf. Comput. Vis., 2015, pp. 1520–1528, https:// doi.org/10.1109/ICCV.2015.178.

\\\[45\\\] M.D. Zeiler, R. Fergus, Visualizing and understanding convolutional networks, in: Lect. Notes Comput. Sci. (Including Subser. Lect. Notes Artif. Intell. Lect. Notes Bioinformatics), 2014, pp. 818–833, https://doi.org/10.1007/978-3-319-10590-1\\\_ 53\\.

\\\[46\\\] S. Ioffe, C. Szegedy, Batch normalization: accelerating deep network training by reducing internal covariate shift, in: 32nd Int. Conf. Mach. Learn., Lille, 2015, pp. 448–456.

\\\[47\\\] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov, Dropout: a simple way to prevent neural networks from overfitting, J. Mach. Learn. Res. 15 (2014) 1929–1958, https://doi.org/10.1214/12-AOS1000.

\\\[48\\\] F. Yu, V. Koltun, Multi-Scale Context Aggregation by Dilated Convolutions, 2015\\. ArXiv Prepr. ArXiv1511.07122, http://arxiv.org/abs/1511.07122.

\\\[49\\\] A.G. Howard, Some Improvements on Deep Convolutional Neural Network based Image Classification, 2013\\. ArXiv Prepr. ArXiv1312.5402, http://arxiv.org/abs/ 1312.5402.

\\\[50\\\] N. Japkowicz, S. Stephen, The class imbalance problem: a systematic study, Intell. Data Anal. 6 (2002) 429–449 doi:10.1.1.711.8214.

\\\[51\\\] G. Weiss, F. Provost, The Effect of Class Distribution on Classifier Learning: an Empirical Study, Rutgers Univ., 2001\\. https://pdfs.semanticscholar.org/c68e/ 3c07ac9374fdd6f93f919d57e398e99c7977.pdf.

\\\[52\\\] TensorFlow, (2018). https://www.tensorflow.org (accessed March 14, 2018).

\\\[53\\\] K. He, X. Zhang, S. Ren, J. Sun, Delving deep into rectifiers: surpassing human-level performance on imagenet classification, in: Proc. IEEE Int. Conf. Comput. Vis., 2015, pp. 1026–1034, https://doi.org/10.1109/ICCV.2015.123.

\\\[54\\\] K.H. Zou, S.K. Warfield, A. Bharatha, C.M.C. Tempany, M.R. Kaus, S.J. Haker, W.M. Wells, F.A. Jolesz, R. Kikinis, Statistical validation of image segmentation quality based on a spatial overlap index, Acad. Radiol. 11 (2004) 178–189, https:// doi.org/10.1016/S1076-6332(03)00671-8.

\\\[55\\\] M. Everingham, S.M.A. Eslami, L. Van Gool, C.K.I. Williams, J. Winn, A. Zisserman, The Pascal visual object classes challenge: a retrospective, Int. J. Comput. Vis. 111 (2014) 98–136, https://doi.org/10.1007/s11263-014-0733-5.

\\\[56\\\] I. Arganda-Carreras, S.C. Turaga, D.R. Berger, D. Cires¸an, A. Giusti, L.M. Gambardella, J. Schmidhuber, D. Laptev, S. Dwivedi, J.M. Buhmann, T. Liu, M. Seyedhosseini, T. Tasdizen, L. Kamentsky, R. Burget, V. Uher, X. Tan, C. Sun, T.D. Pham, E. Bas, M.G. Uzunbas, A. Cardona, J. Schindelin, H.S. Seung, Crowdsourcing the creation of image segmentation algorithms for connectomics, Front. Neuroanat. 9 (2015), https://doi.org/10.3389/fnana.2015.00142.

\\\[57\\\] V. Ulman, M. Maska, K.E.G. Magnusson, O. Ronneberger, C. Haubold, N. Harder, P. Matula, P. Matula, D. Svoboda, M. Radojevic, I. Smal, K. Rohr, J. Jalden, H.M. Blau, O. Dzyubachyk, B. Lelieveldt, P. Xiao, Y. Li, S.Y. Cho, A.C. Dufour, J.C. Olivo-Marin, C.C. Reyes-Aldasoro, J.A. Solis-Lemus, R. Bensch, T. Brox, J. Stegmaier, R. Mikut, S. Wolf, F.A. Hamprecht, T. Esteves, P. Quelhas, O. Demirel, € L. Malmstrom, F. Jug, P. Tomancak, E. Meijering, A. Mu € noz-Barrutia, M. Kozubek, \\\~ C. Ortiz-De-Solorzano, An objective comparison of cell-tracking algorithms, Br. J. Pharmacol. 14 (2017) 1141–1152, https://doi.org/10.1038/nmeth.4473.

\\\[58\\\] C.W. Wang, C.T. Huang, J.H. Lee, C.H. Li, S.W. Chang, M.J. Siao, T.M. Lai, B. Ibragimov, T. Vrtovec, O. Ronneberger, P. Fischer, T.F. Cootes, C. Lindner, A benchmark for comparison of dental radiography analysis algorithms, Med. Image Anal. 31 (2016) 63–76, https://doi.org/10.1016/j.media.2016.02.004.

\\\[59\\\] SpaceNet, (2017). http://explore.digitalglobe.com/spacenet (accessed April 12, 2018).

\\\[60\\\] K. Sirinukunwattana, J.P.W. Pluim, H. Chen, X. Qi, P.A. Heng, Y.B. Guo, L.Y. Wang, B.J. Matuszewski, E. Bruni, U. Sanchez, A. Bohm, O. Ronneberger, B. Ben Cheikh, € D. Racoceanu, P. Kainz, M. Pfeiffer, M. Urschler, D.R.J. Snead, N.M. Rajpoot, Gland segmentation in colon histology images: the GlaS challenge contest, Med. Image Anal. 35 (2017) 489–502, https://doi.org/10.1016/j.media.2016.08.008.

\\\[61\\\] S.S. Mohseni Salehi, D. Erdogmus, A. Gholipour, Auto-context convolutional neural network (Auto-Net) for brain extraction in magnetic resonance imaging, IEEE Trans. Med. Imag. 36 (2017) 2319–2330, https://doi.org/10.1109/TMI.2017.2721362.

\\\[62\\\] A. Jansson, E. Humphrey, N. Montecchio, R. Bittner, A. Kumar, T. Weyde, Singing voice separation with deep U-Net convolutional networks, in: ISMIR, Int. Soc. Music Inf. Retr. Conf., 2017, pp. 745–751. https://ismir2017.smcnus.org/wp-content/ uploads/2017/10/171\\\_Paper.pdf.

\\\[63\\\] J. Macqueen, Some methods for classification and analysis of multivariate observations, Proc. Fifth Berkeley Symp. Math. Stat. Probab. 1 (1967) 281–297 doi: citeulike-article-id:6083430.

\\\[64\\\] A.K. Jain, Data clustering: 50 years beyond K-means, Pattern Recogn. Lett. 31 (2010) 651–666, https://doi.org/10.1016/j.patrec.2009.09.011.
