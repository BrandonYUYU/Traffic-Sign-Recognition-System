<h1 align="center">✨ Traffic Sign Detection & Recognition System ✨</h1>

Assignment's Course : Mini Project
Description : A C++/OpenCV application that detects, segments, and classifies road traffic signs

✨✨✨

- **Color Segmentation:** HSV thresholding to isolate red, blue, and yellow sign regions.  
- **Shape Filtering:** Douglas–Peucker contour approximation to detect triangles and circles.  
- **Region Extraction & Preprocessing:** Crop, resize to 64×64 px, convert to HSV and gray scale.  
- **Feature Extraction:** HOG descriptor with one-hot color encoding.  
- **Classification:** SVM and Random Forest
- **Testing:** Confusion matrix and accuracy.



---


## 🔧 Run
``` 
Choose from the menu:

   1. Train SVM
   2. Test SVM
   3. Train Random Forest
   4. Test Random Forest
   5. Predict Single Image (SVM)

   For option 5, enter the filename (e.g. `005.png`) located in `Inputs/input/`.
```

