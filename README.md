# 3D-reconstruction-of-Shandong-University-and-its-target-detection
The project was completed in **2022** and took about more than six months.

Three-dimensional reconstruction has become a popular research direction in the field of computer vision, and has great application potential in medical imaging, relic reconstruction, VR and other fields. The main task of 3D reconstruction is to use single or multiple images and other information such as depth 2D images are converted to 3D models.

## 1.Three-dimensional reconstruction of two buildings based on deep learning

The two methods used in this article are SFM and MVS. SFM is the upstream of MVS. One can be understood as sparse reconstruction and the other as dense reconstruction. SFM calculates the pose, internal parameters, sparse points and their co-view relationship for MVS, and MVS uses this information and the color map to estimate the depth map and do the final operation.

### step1 get cloud point information
Feature extraction

The purpose of feature extraction is to extract camera parameters from the EXIF of the picture. Basically, the Exif file format is the same as the JPEG file format. Exif inserts some image/digital camera information data and thumbnail images into JPEG according to JPEG specifications. You can then view the image file in Exif format with some software such as JPEG compatible Internet browser/picture browser/image processing. Just like browsing a regular JPEG image file.

COLMAP implements different camera models with different levels of complexity. If there are no inherent parameters known a priori, it is usually best to use the simplest camera model that is complex enough to model the distortion effect.

The estimated inherent parameters can be checked by double-clicking a specific image in the model viewer or exporting the model and opening the camera's.txt file. To get the best reconstruction results, it may be necessary to try different camera models for the problem. Often, when reconstruction fails and the estimated focal length value/distortion factor is seriously wrong, this is a sign of using an overly complex camera model. Conversely, if COLMAP uses many iterations of local and global beam adjustments, it indicates that too simple a camera model is being used to fully model the distortion effect.

Feature matching

In feature matching, there are roughly four methods commonly used in the industry:
Exhaustive: Official documentation shows that Exhaustive takes the longest, images are matched in pairs, and theoretically the best results. However, in practice, this method takes a long time and the effect is not necessarily good, and the judgment of the scene location is wrong from time to time. For example, the location of some pictures cannot be determined when the scene is similar, and misjudgment will occur.

Sequential: When taking photos in chronological order, adjacent scenes are often in one piece, which is suitable for using this matching method. By matching x images next to each other, the above problem is not encountered. But it can happen when the Angle is too large, or when the shot starts and ends from the same Angle
Error.

Spatial: This uses geolocation information, meaning each image must have its own location, such as gps. This method needs to be set in what range to match, so you need to know the approximate size of the reconstructed scene in advance in order to select the appropriate parameters.

Custom: indicates the custom matching mode. The parameters can be adjusted as required. This paper combines the actual running effect, running speed and its own situation. The first two matching methods are mainly used in order to combine speed with accuracy
Fine.

Sparse reconstruction

Incremental SfM is used to gradually increase the view Angle, and iteratively optimize the reprojection error to calculate different views
Camera parameters, the sparse point cloud of the scene and the visual relationship between different views and the point cloud can be determined, and finally the sparse point cloud of the scene and the camera attitude of each Angle can be obtained. If the results are very bad, then reconsider the matching method. The process is outlined below:

![image](https://github.com/supersteve2001/3D-reconstruction-of-Shandong-University-and-its-target-detection/assets/69947525/5c340a6d-7ad5-4c15-843d-2598471128d7)

### step2 Three-dimensional reconstruction of point cloud information based on MVS
The following 3D modeling results are shown below, Wen Tianlou:

![image](https://github.com/supersteve2001/3D-reconstruction-of-Shandong-University-and-its-target-detection/assets/69947525/218b5bc1-3e18-4ca4-b754-d3078fe94fbe)

School of Arts and Science Laboratory Building:

![image](https://github.com/supersteve2001/3D-reconstruction-of-Shandong-University-and-its-target-detection/assets/69947525/83623e27-31f8-4977-a232-b84c0eac380f)

Imagine, for the same point in two pictures. Now going back to the moment the picture was taken, in the three-dimensional world, there is a ray of light from this point on the picture, passing through the imaging center of the camera that took the picture, and finally reaching a three-dimensional point in space, this three dimensional points are also projected in the same way in another photo.

This process looks very ordinary, just like a normal camera projection. But because of the two images, there is a connection between them, and the proof of this connection is beyond our ability, but we just need to know that in this case, there is a natural constraint between the two images.

The reconstruction result of SFM is sparse 3D point cloud. In order to enter deeper field and get better results, we enter MVS.

## 2.Computer vision calculates data such as the number of floors, the number of Windows, and the area of the building
First, the test images are connected, and then the gradient and Angle of the images are calculated by the Sobel operator. The next step is image binarization. By setting the gradient threshold, it is calculated in horizontal direction and vertical direction respectively. Then the process of filling the gap is carried out. The horizontal binary image is expanded along the x direction, and then the vertical direction is filled in the y direction. After a logical "OR" (OR) operation on two images in different directions, the ImageThinning() function is used to refine the operation. The results are as follows:

Here is how to convert the image to grayscale:

![image](https://github.com/supersteve2001/3D-reconstruction-of-Shandong-University-and-its-target-detection/assets/69947525/f648656e-21bd-4c9b-8761-c83f87509e0f)

Secondly, edge detection is carried out and then de-noise and filling are carried out:

![image](https://github.com/supersteve2001/3D-reconstruction-of-Shandong-University-and-its-target-detection/assets/69947525/5fc20d36-cb81-4751-b757-6d1da227276c)

According to the point cloud results of the first section of 3D modeling, we tested that the area gap was ideal (within 100 square meters). The following is the point coordinate extraction:

![image](https://github.com/supersteve2001/3D-reconstruction-of-Shandong-University-and-its-target-detection/assets/69947525/23264a3b-310e-4fac-aeb2-4a4b1b3a9c92)

### Local Otsu algorithm
Otsu method (Otsu method) is an important threshold segmentation method in the field of image processing (local OTSu method), which is suitable for processing bimodal images.
However, most developers are not familiar with its principle, so it is necessary to explain and analyze it in detail.

A fully automatic global threshold algorithm usually consists of the following steps:

1. Preprocess the input image.
2. Obtain the image histogram (pixel distribution).
3. Calculate the threshold TT.
4. Replace the areas of the image with pixels larger than T TT with white and the rest with black.
   
![image](https://github.com/supersteve2001/3D-reconstruction-of-Shandong-University-and-its-target-detection/assets/69947525/b5e9fdc8-df5b-4caa-99f2-0cd690959d34)

### Target detection under yolo algorithm
In the actual calculation process, all the code of the model is pytorch version, and it is trained and tested on the GPU server of the Moment Pool Cloud, so the path configuration in the code is based on the path configuration in the Moment Pool cloud as the standard (/mnt), and the folder here contains the official COCO data set the complete network data after training. See the instructions for use (attached) for details.

The COCO dataset provides distinguishing data for 80 object classes, and a yolo model trained on this dataset can recognize eight types of proprietary class tags within the Wen Tian Tower and the Academy of Arts building: person , bicycle , car , motorbike , traffic light , fire hydrant ,stop sigh, bench.
The effect of the actual algorithm is shown as follows:

![image](https://github.com/supersteve2001/3D-reconstruction-of-Shandong-University-and-its-target-detection/assets/69947525/01afd8dc-64f3-48dc-876d-f7b730a26723)

## 3.cloud point and white model
Through the processes of dedistortion, depth estimation and depth map generation, the estimated depth map is fused to obtain dense reconstruction results:

![image](https://github.com/supersteve2001/3D-reconstruction-of-Shandong-University-and-its-target-detection/assets/69947525/589169d4-ade1-4762-a849-69c35021a81c)

The Delaunay refinement algorithm builds triangular or tetrahedral (" element ") meshes for applications such as interpolation, rendering, terrain databases, geographic information systems, and the solution of partial differential equations with the most demanding finite element methods. The Delaunay optimization algorithm operates by maintaining a Delaunay, or constrained Delaunay triangulation, which is optimized by inserting additional vertices until the grid meets the limitations on the quality and size of the elements. These algorithms provide theoretical boundaries for spatial grading of element mass, side length, and element size. after Topological and geometric fidelity of hybrid domains, including curved domains with internal boundaries; And really satisfactory performance in practice.

![image](https://github.com/supersteve2001/3D-reconstruction-of-Shandong-University-and-its-target-detection/assets/69947525/974390a4-9ae5-4e9e-a8f7-58c38ef2fa5c)

Sketchup is an intelligent 3D model drawing software. It can draw 3D models through simple operations such as drawing lines, extending planes, etc., and can render materials for models.

![image](https://github.com/supersteve2001/3D-reconstruction-of-Shandong-University-and-its-target-detection/assets/69947525/71b3acca-d252-4f5b-8447-611cda57c9c8)

### 4. Web display
Once you have the pcd point cloud of the target building, you can implement the web loading point cloud by debugging threejs.

The results are shown as follows:

![image](https://github.com/supersteve2001/3D-reconstruction-of-Shandong-University-and-its-target-detection/assets/69947525/57abf1f7-b79b-480e-b367-25974f812509)

![image](https://github.com/supersteve2001/3D-reconstruction-of-Shandong-University-and-its-target-detection/assets/69947525/7e4def51-f728-4b72-984b-6a6b313b6115)

Once you have a more complete white model, use threejs to load the white model and display information about the building through threejs click events.

The results are shown as follows:

![image](https://github.com/supersteve2001/3D-reconstruction-of-Shandong-University-and-its-target-detection/assets/69947525/714f9538-9ada-4068-a364-90d061ffdb1b)

![image](https://github.com/supersteve2001/3D-reconstruction-of-Shandong-University-and-its-target-detection/assets/69947525/179fa9d2-be06-43d5-b7d6-a9df3d51110f)



