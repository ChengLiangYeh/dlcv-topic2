# Deep Learning for Computer Vision
## HW2 Project 1 ― Image Classification
- In project 1, you will need to implement an image classification model and answer some questions in the report.
- Dataset: 25000 colour images with size 32*32 pixels of 50 classes. 22500 images belong to training set. 2500 images belong to validation set.

![1](./pic/image_classification.png)

## HW2 Project 2 ― Semantic Segmentation
- In project 2, you will need to implement two semantic segmentation models and answer some questions in the report.
- Dataset: 2000 satellite images with ground truthes(masks) in training set. 257 images-masks pair in validation set. There are 7 possible classes for each pixel.

![1](./pic/semantic_segmentation.png)

## Dataset:
- Contact me for Dataset. 
- Email: chengliang.yeh@gmail.com

## Implementation:
- For image classification, I used pre-trained VGG16 as a backbone CNN model to fulfill the task.
![1](./pic/image_classification2.png)

- For semantic segmentation, I used VGG16-FCN32s and VGG16-FCN8s to fulfill the task.  
![1](./pic/semantic_segmentation2.png)

### Evaluation
To evaluate your semantic segmentation model, you can run the provided evaluation script provided in the starter code by using the following command.

    python3 mean_iou_evaluate.py <--pred PredictionDir> <--labels GroundTruthDir>

 - `<PredictionDir>` should be the directory to your predicted semantic segmentation map (e.g. `hw2_data/prediction/`)
 - `<GroundTruthDir>` should be the directory of ground truth (e.g. `hw2_data/val/seg/`)

Note that your predicted segmentation semantic map file should have the same filename as that of its corresponding ground truth label file (both of extension ``.png``).

### Visualization
To visualization the ground truth or predicted semantic segmentation map in an image, you can run the provided visualization script provided in the starter code by using the following command.

    python3 viz_mask.py <--img_path xxxx_sat.jpg> <--seg_path xxxx_mask.png>

# Submission Rules
### Deadline
109/11/10 (Tue.) 02:00 AM

### Late Submission Policy
You have a three-day delay quota for the whole semester. Once you have exceeded your quota, the credit of any late submission will be deducted by 30% each day.

Note that while it is possible to continue your work in this repository after the deadline, **we will by default grade your last commit before the deadline** specified above. If you wish to use your quota or submit an earlier version of your repository, please contact the TAs and let them know which commit to grade.

### Academic Honesty
-   Taking any unfair advantages over other class members (or letting anyone do so) is strictly prohibited. Violating university policy would result in an **F** grade for this course (**NOT** negotiable).    
-   If you refer to some parts of the public code, you are required to specify the references in your report (e.g. URL to GitHub repositories).      
-   You are encouraged to discuss homework assignments with your fellow class members, but you must complete the assignment by yourself. TAs will compare the similarity of everyone’s submission. Any form of cheating or plagiarism will not be tolerated and will also result in an **F** grade for students with such misconduct.


### Submission Format
Aside from your own Python scripts and model files, you should make sure that your submission includes *at least* the following files in the root directory of this repository:
 1.   `hw2_<StudentID>.pdf`  
The report of your homework assignment. Refer to the "*Submission*" section in the slides for what you should include in the report. Note that you should replace `<StudentID>` with your student ID, **NOT** your GitHub username.
 2.   `hw2_1.sh`  
The shell script file for running your classification model.
 3.   `hw2_2.sh`  
The shell script file for running your baseline semantic segmentation model.
 4.   `hw2_2_best.sh`  
The shell script file for running your improved segmentation model.

We will run your code in the following manner:

    bash hw2_1.sh $1 $2
where `$1` is the testing images directory (e.g. `test/images/`), and `$2` is the path of folder where you want to output your prediction file (e.g. `test/label_pred/` ). Please do not create the output prediction directory in your bash script or python codes.

    bash hw2_2.sh $1 $2
    bash hw2_2_best.sh $1 $2
where `$1` is the testing images directory (e.g. `test/images/`), and `$2` is the output prediction directory for segmentation maps (e.g. `test/label_pred/` ). Please do not create the output prediction directory in your bash script or python codes.

### Packages
This homework should be done using python3.6. For a list of packages you are allowed to import in this assignment, please refer to the requirments.txt for more details.

You can run the following command to install all the packages listed in the requirements.txt:

    pip3 install -r requirements.txt

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.

### Remarks
- If your model is larger than GitHub’s maximum capacity (100MB), you can upload your model to another cloud service (e.g. Dropbox). However, your shell script files should be able to download the model automatically. For a tutorial on how to do this using Dropbox, please click [this link](https://goo.gl/XvCaLR).
- **DO NOT** hard code any path in your file or script, and the execution time of your testing code should not exceed an allowed maximum of **10 minutes**.
- **Please refer to HW2 slides for details about the penalty that may incur if we fail to run your code or reproduce your results.**

# Q&A
If you have any problems related to HW2, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under hw2 FAQ section in FB group
