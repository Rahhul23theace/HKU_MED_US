pip install --upgrade pip && pip install -r requirements.txt
gunicorn --bind 0.0.0.0:5000 wsgi:app

1. Artefact: This label refers to ultrasound images where unwanted artifacts are present, affecting the overall image quality. Artifacts can result from various factors, such as acoustic shadows, reverberations, or other distortions. In the context of kidney ultrasound images, these artifacts may hinder proper assessment of the kidney's structure and function.

2. Incorrect Gain/Inappropriate gain: The "Incorrect_Gain" label is assigned to ultrasound images where the gain settings are not properly adjusted. Gain refers to the amplification of the returning ultrasound signals, which affects the brightness of the image. When the gain is either too high or too low, it can lead to over- or under-exposure of the image, making it difficult to accurately evaluate the kidney's condition.


3. Incorrect Position/ Suboptimal positioning: Images labeled as "Incorrect_Position" are those where the ultrasound probe or patient's positioning is not optimal for capturing a clear and accurate image of the kidney. Proper positioning is crucial for obtaining high-quality ultrasound images, as it ensures that the ultrasound beam is appropriately aligned with the target anatomy. Inaccurate positioning may result in poor visualization of the kidney or the inclusion of irrelevant structures in the image.

4. Optimal: The "Optimal" label is given to ultrasound images that exhibit high quality and are free from any significant artifacts, positioning errors, or gain issues. These images provide a clear and accurate representation of the kidney, allowing for precise assessment of its structure and function. Such images are ideal for medical professionals to use when evaluating a patient's kidney health.

5. Wrong: The "Wrong" label is assigned to ultrasound images where the captured anatomy is not the intended kidney or the image is of an entirely different organ or body part. This may occur if the ultrasound probe is placed incorrectly or if the operator is not familiar with the correct scanning technique for kidney imaging. Images with this label are not suitable for kidney quality evaluation, as they do not accurately represent the target organ.










1. Artefact: There are rib/gas/movement artifacts that cover the kidney. Please adjust the prob angular position or position. You can also try asking patient to hold breath for a while.

2. Incorrect Gain/Inappropriate gain: The image is too dark/bright so that kidney is not well presented. Please adjust the brightness.

3. Incorrect Position/ Suboptimal positioning:  Image does not cover the whole kidney, which is out of the edge. Please adjust the prob position or zoom out.
Or kidney is too small in image, please zoom in to the kidney.


4. Optimal: The kidney is well presented.

5. Wrong: No kidney is presented in the image.








1. Artefact:  There are artefacts on the image. These could be due to acoustic shadowing from the ribs (having the kidneys in the upper retroperitoneum) or ring-down artefacts from gas bubbles in the colon. If you are interested in reading more on ultrasound artefacts, please refer to explanation on Moodle MBBS Clinical Skills Platform under Radiological, Ultrasound Section I 2.3. You can avoid these arfefacts from the ribs by asking the subject to take a deep inspiration, which will lower the diaphragm and hence the position of the kidney away from the ribcage, or by positioning the probe in between the ribs to avoid the artefact. To overcome ring-down artefacts, having the subject fasted will help or gently applying pressure to displace the gas away from the area (but sometimes even with these efforts, nothing work!). 


2. Incorrect Gain/Inappropriate gain:  The image is either too "bright" or too "dark". The Gain controls the brightness of the image (see Moodle MBBS Clinical Skills Platform under Radiological, Ultrasound Section I 3.5). Another good reference for explanation can be found on https://123sonography.com/blog/ultrasound-101-part-5-gain-and-time-gain-compensation#:~:text=What%20is%20gain%3F,much%20each%20echo%20is%20amplified.


3. Incorrect Position/ Suboptimal positioning: The organ of interest, i.e. kidney, is not correctly positioned in the captured imaged. It is not centrally placed or incompletely imaged (cut-off, cropped). Having the subject in decubitus position helps to get good access to image the kidney as explained in the video on Moodle MBBS Clinical Skills Platform under Radiological, Ultrasound Section IIC 7.1. If this is not possible, then try positioning the probe anteriorly and posteriorly to find an ideal view of the kidney maybe required as every subject is different. 

4. Optimal: Well done/good work for obtaining optimal image quality of the kidney. 


5. Wrong: The image acquired is of the wrong organ. Please refer to Moodle MBBS Clinical Skills Platform under Radiological, Ultrasound Section IIC 7.1, where there is a video demonstrating how to perform and acquire image of the kidney using ultrasound. https://moodle.hku.hk/mod/page/view.php?id=2748898"# US_Project" 
"# US_Project" 
"# US_Renal" 
"# US_Renal" 
"# HKU_MED_US" 
