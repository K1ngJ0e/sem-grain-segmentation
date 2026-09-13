# Supporting Information method paragraph — conditional draft

Use this paragraph only after the actual film records confirm the stated split
and a new experiment has been completed with configs/manuscript.json. Fill in
the actual film counts if included; do not use synthetic test results.

The dataset comprised 44 annotated SEM images and was partitioned at the level
of independently fabricated films into training, validation, and test sets
containing 28, 7, and 9 images, respectively. All images originating from the
same film were assigned to a single subset. Images were resized to 512 × 512
pixels, replicated across three channels, and normalized using the ImageNet
mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225).
Binary masks were resized using nearest-neighbor interpolation. During
training only, horizontal and vertical flips were independently applied with
a probability of 0.5, and the rotation angle was sampled uniformly from
0°, 90°, 180°, and 270°. Each image and its corresponding mask underwent
identical spatial transformations. No augmentation was applied to the
validation or test sets.

The U-Net++ model incorporated a ResNet18 encoder and squeeze-and-excitation
blocks applied to encoder feature maps. The model was trained for 50 epochs
using the AdamW optimizer with a learning rate of 1 × 10−3, a weight decay of
1 × 10−4, and a batch size of 2. Tversky loss was used with alpha = 0.3 and
beta = 0.7, weighting false positives and false negatives, respectively.
The random seed was set to 42. The checkpoint with the highest mean
per-image validation Dice score, calculated at a probability threshold of
0.5, was selected. This checkpoint was evaluated on the held-out test set
without manual correction or further parameter selection. For the default
configuration, test Dice and IoU were calculated from semantic predictions
at the fixed threshold of 0.5 and averaged across images; per-film summaries
were also recorded.

Author notes: report actual independent-film counts and hardware/software
versions from the completed run. The default model requests ImageNet encoder
weights; state that initialization only after successful loading is confirmed.
If a different pre-test threshold/post-processing protocol is selected, revise
this paragraph accordingly. Describe morphology-based instance separation
separately; semantic Dice/IoU do not validate instance-level grain statistics.
