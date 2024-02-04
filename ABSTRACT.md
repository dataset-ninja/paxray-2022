The authors build the **PAXRAY++: Projected Anatomy for X-Ray Dataset** containing over two million annotated instances across 15 thousand images for 157 anatomical classes for both ***lateral*** and ***frontal*** thoracic views. PAXRay++ allows them to train data-intensive 2D chest X-Ray anatomy segmentation
(CXAS) models that accurately delineate the fine-grained thoracic anatomy.

## Motivation

Chest radiographs (CXRs) are widely used for diagnosing respiratory or cardiovascular conditions, with over 130 million studies conducted annually in Germany alone. Employing ionizing radiation for penetration, CXRs offer a visual depiction of the body's organs, tissues, and cavities. However, interpreting these images poses challenges due to overlapping structures that can obscure potential pathological changes. Despite these complexities, CXRs remain a standard diagnostic procedure, but their quantitative analysis can be time-consuming. With the escalating demand for imaging procedures, the substantial workload can lead to errors from rushed examinations or burnout among doctors.

Recent advances in Computer Vision, such as convolutional neural networks (CNNs) and vision transformers (ViTs), have the potential to alleviate radiologists' workload in both image analysis and reporting. Various computer-aided diagnostic approaches automatically predict relevant findings using study-level labels from medical reports for training. However, these approaches lack interpretability and usability, as they cannot precisely define the location or provide quantitative information. Dense pixel-wise predictors can address these issues by offering concrete delineations of relevant structures. However, obtaining box or mask annotations is challenging due to the effort involved and the summation nature of CXR images.

To tackle these challenges, the authors propose a method to collect large amounts of CXR annotations without manual labeling, showcasing how these annotations can train networks to identify thoracic anatomy in a detailed manner. Inspired by the similarity between CT scans and CXRs in their origin signal, the authors leverage the advantages of CT scans in identifying structures due to their volumetric nature. The consistent body structure across modalities allows semantic translation from CT to CXR. While many datasets in other imaging domains focus on specific subsets, the authors hypothesize that combining different volumetric annotations can provide a holistic view of the human body, transferable to a large CT corpus.

## Dataset creation

The authors build the PAXRAY++ dataset containing over two million annotated instances across 15 thousand images for 157 anatomical classes for both ***lateral*** and ***frontal*** thoracic views. PAXRay++ allows them to train data-intensive 2D chest X-Ray anatomy segmentation (CXAS) models that accurately delineate the fine-grained thoracic anatomy. The authors utilized a carefully curated set of projected X-rays as a validation set for selecting hyperparameters, including model architecture. The optimal model from the validation set was then compared with annotations by two radiologists. In a direct assessment involving 157 anatomical classes across 30 frontal and lateral images, the study demonstrates a strong agreement between the model and annotators, as evidenced by high Intersection-over-Union (IoU) scores. Specifically, for the frontal view (anterior-posterior/posterior-anterior), the average IoU scores were 0.93 and 0.96, with an inter-annotator agreement (IAA) of 0.93. In the case of the lateral view, the model-annotator agreement (MAA) yielded scores of 0.83 and 0.87, accompanied by an IAA of 0.83.

<img src="https://github.com/dataset-ninja/paxray-2022/assets/120389559/e6ec38c1-8e75-4011-beb2-dc8677a5b3f6" alt="image" width="1000">

<span style="font-size: smaller; font-style: italic;">A flowchart describing the PAX-Ray++ dataset generation process. (a) Collect publicly available datasets containing different anatomical structures. (b) Train an ensemble of nnUNets of each expert domain and infer them on a shared dataset. \(c\) Merge the 3D predictions, apply anatomical priors and retrain. (d) Infer the final nnUNet on 10K chest CTs. (e) Apply a CT and label projection to generate a chest X-Ray dataset which we apply anatomical-prior-based postprocessing to collect the final dataset.</span>

## Image sources

To generate the PAX-Ray++ dataset, the authors require CT volumes as imaging data. They choose large unlabeled CT datasets focusing on the thoracic region, which is displayed in the bottom part of table.

<img src="https://github.com/dataset-ninja/paxray-2022/assets/120389559/0605c34f-8c41-4ba0-b6cd-63b5e5c64fe5" alt="image" width="1000">

<span style="font-size: smaller; font-style: italic;">Comparison of considered CT datasets for PAXRAY++ regarding size, number of labels, label domain, volume spacing, number of slices and segmentation performance of a nnUNet in 5-fold cross validation. Top half displays datasets used for label aggregation, while the bottom part was used as imaging source for PAXRAY++.</span>

The datasets include RSNA PE, RibFrac, LIDC-IDRI, COVID-19-AR, COVID19-1 and COVID-19-2.
1. The RNSA Pulmonary Embolism dataset, a contrast CT dataset for identifying pulmonary embolisms, is the largest, with more than 7000 volumes with an average
axial spacing of 1.25 mm and 237 slices.
2. The RibFrac dataset for detecting rib fractures, which we used for the initial PAXRay, consists of 660 volumes with the smallest average spacing of 1.13 mm and 359 slices.
3. The LIDC-IDRI dataset was collected to detect pulmonary nodules and consists of 1036 volumes with a spacing of 1.74 and, on average, 237 slices.
4. The COVID-19-AR dataset has, on average, the most slices of 438 and 176 volumes. However, it has a relatively large coronal and sagittal spacing.
5. The COVID-19-1 and -2 datasets contain 215 and 632 volumes but have the smallest amount of slices and the largest axial spacing.

The authors infer the nnUNet ensemble trained on the aggregated pseudo-labels on Autopet on these datasets. They apply the same postprocessing methods as done
with the initial predictions and filter volumes with predictions with anatomical deviations, i.e., too few predicted ribs. They then project the image and label files to a frontal and lateral view and resize to a uniform size of 512 × 512 using nearest interpolation for masks and Lanczos for images.

<img src="https://github.com/dataset-ninja/paxray-2022/assets/120389559/15e8695b-2cd8-4f3d-b45a-e4660c803e9a" alt="image" width="1000">

<span style="font-size: smaller; font-style: italic;">Sample of frontal projected x-rays from the RibFrac, COVID-19-AR, COVID-19-1, COVID19-2, RSNA PE, and LIDC-IDRI dataset. The authors show labels belonging to the categories Respiratory System, Vascular System, Bones, and Abdomen.</span>

The authors see that the different characteristics of each CT dataset have a direct influence on the resulting projections. Thus, the resolution of each mask depends on the resolution of the original CT scan.

## Chest X-Ray Segmentation

In developing their segmentation models, the authors opted for the UNet architecture with a ResNet-50 and ViTB backbone. They chose to leverage pre-training on ImageNet, observing comparable performance to other pre-training methods. Notably, pre-training demonstrated significantly improved results when contrasted with randomly initialized weights. The network was trained using binary cross-entropy and an additional binary dice loss, with optimization performed using AdamW at a learning rate of 0.001 over 110 epochs. The learning rate decayed by a factor of 10 at epochs 60, 90, and 100. For the foundational augmentation, they implemented random resize-and-cropping within the range [0.8, 1.2] for an image size of 512.