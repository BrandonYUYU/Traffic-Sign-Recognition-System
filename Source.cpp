#include	<opencv2/opencv.hpp>
#include	<opencv2/highgui/highgui.hpp>
#include	<opencv2/imgproc.hpp>
#include	<opencv2/ml.hpp>
#include	<iostream>
#include	<stdlib.h>
#include	<string>
#include	<random>
#include	<algorithm>
#include	"Supp.h"
#include	<math.h>

using namespace cv;
using namespace cv::ml;
using namespace std;


// label mapping
map<int, string> trafficSignMapping = {
	{0, "Speed limit (5km/h)"},
	{1, "Speed limit (15km/h)"},
	{2, "Speed limit (30km/h)"},
	{3, "Speed limit (40km/h)"},
	{4, "Speed limit (50km/h)"},
	{5, "Speed limit (60km/h)"},
	{6, "Speed limit (70km/h)"},
	{7, "Speed limit (80km/h)"},
	{8, "No straight or left turn"},
	{9, "No straight or right turn"},
	{10, "No straight"},
	{11, "No left turn"},
	{12, "No left or right turn"},
	{13, "No right turn"},
	{14, "No takeover"},
	{15, "No U - turn"},
	{16, "No car"},
	{17, "No horn"},
	{18, "Speed limit (40km/h) end"},
	{19, "Speed limit (50km/h) end"},
	{20, "Ahead or right turn"},
	{21, "Straight ahead"},
	{22, "Turn left ahead"},
	{23, "Turn left or right"},
	{24, "Turn right ahead"},
	{25, "Keep left"},
	{26, "Keep right"},
	{27, "Roundabout"},
	{28, "Car allowed"},
	{29, "Horn"},
	{30, "Bicycle (blue)"},
	{31, "U-turn"},
	{32, "Must turn left or right"},
	{33, "Traffic signals"},
	{34, "Warning"},
	{35, "Pedestrian crossing"},
	{36, "Cycle crossing"},
	{37, "School crossing"},
	{38, "Sharp bend to right"},
	{39, "Sharp bend to left"},
	{40, "Down"},
	{41, "Road work"},
	{42, "Slow"},
	{43, "T junction right"},
	{44, "T junction left"},
	{45, "Village"},
	{46, "Double bend"},
	{47, "Railway with no guard"},
	{48, "Construction"},
	{49, "Many Bends"},
	{50, "Railway with guard"},
	{51, "High accident"},
	{52, "Stop"},
	{53, "No entry (Both direction)"},
	{54, "No stopping or parking"},
	{55, "No entry"},
	{56, "Give way"},
	{57, "Check"}
};


// retrive mapping
string getLabelFromValue(float value) {
	// Round the input value to the nearest integer
	int roundedValue = round(value);

	// Check if the rounded value exists in the mapping
	auto it = trafficSignMapping.find(roundedValue);

	if (it != trafficSignMapping.end()) {
		// If found, return the corresponding label
		return it->second;
	}
	else {
		// If not found, return an "Unknown" label
		return "Unknown sign";
	}
}


// generate color mask
vector<Mat> colorMask(Mat src) {
	Mat hsv, blur;
	GaussianBlur(src, blur, Size(3, 3), 3);
	cvtColor(blur, hsv, COLOR_BGR2HSV); //convert to hsv

	Mat redMask, redMask1, redMask2;
	Scalar red_lower1(0, 120, 0);
	Scalar red_upper1(10, 255, 255);
	inRange(hsv, red_lower1, red_upper1, redMask1);
	Scalar red_lower2(170, 120, 0); //170 120 0
	Scalar red_upper2(220, 255, 255);
	inRange(hsv, red_lower2, red_upper2, redMask2);
	redMask = redMask1 | redMask2;
	threshold(redMask, redMask, 0, 255, THRESH_BINARY + THRESH_OTSU);

	Mat yellowMask;
	Scalar yellow_lower(10, 120, 0);//10 120 0
	Scalar yellow_upper(40, 255, 255);//40 255 255
	inRange(hsv, yellow_lower, yellow_upper, yellowMask);
	threshold(yellowMask, yellowMask, 0, 255, THRESH_BINARY + THRESH_OTSU);

	Mat blueMask;
	Scalar blue_lower(90, 120, 0);
	Scalar blue_upper(130, 255, 255);
	inRange(hsv, blue_lower, blue_upper, blueMask);
	threshold(blueMask, blueMask, 0, 255, THRESH_BINARY + THRESH_OTSU);

	return { redMask, blueMask, yellowMask };

}


// detect shape
void detectShape(vector<Point> contour, Mat& show) { //show is the mask after shape detection --> no change name --> avoid error
	vector<Point> approx;
	vector<Point> tri;
	vector<Point> hull;
	Point2i		center;
	convexHull(contour, hull, false);
	approxPolyDP(hull, approx, 0.06 * arcLength(hull, true), true);
	if (approx.size() == 3) {
		minEnclosingTriangle(hull, tri);
		vector<double> side_lengths;
		for (int i = 0; i < 3; i++) {
			side_lengths.push_back(norm(tri[i] - tri[(i + 1) % 3]));
		}
		if ((side_lengths[0] / side_lengths[1] < 1.4 && side_lengths[0] / side_lengths[1] > 1 / 1.4)
			&& (side_lengths[1] / side_lengths[2] < 1.4 && side_lengths[1] / side_lengths[2] > 1 / 1.4)) {
			fillPoly(show, tri, Scalar(255, 255, 255));


		}
		else {
			fillPoly(show, contour, Scalar(255, 255, 255));
		}
	}
	else {
		//th for circle
		approxPolyDP(hull, approx, 0.01 * arcLength(hull, true), true);
		if (approx.size() > 10) {
			fillPoly(show, hull, Scalar(255, 255, 255));

		}
		else {
			fillPoly(show, contour, Scalar(255, 255, 255));
		}
	}

}


// one hot encoding
void colorFeatureOneHotEncoding(Mat HSV, vector<vector<int>>& colorFeatureOneHot) {
	vector<Mat> newMask = colorMask(HSV);
	int redCount = countNonZero(newMask[0]);
	int yellowCount = countNonZero(newMask[1]);
	int blueCount = countNonZero(newMask[2]);

	if (blueCount > 5 * redCount && blueCount > 5 * yellowCount) {
		colorFeatureOneHot.push_back({ 1,0,0 }); //represent blue
	}
	else if (redCount > yellowCount) {
		colorFeatureOneHot.push_back({ 0,1,0 }); //represent yellow
	}
	else if (yellowCount > redCount) {
		colorFeatureOneHot.push_back({ 0,0,1 }); //represent blue
	}
	else {
		colorFeatureOneHot.push_back({ 0,0,0 });
	}
}


// Function to extract HOG features from an image
void extractHOGFeatures(vector<Mat>& img, vector<vector<float>>& HOG, vector<vector<int>>& colorFeature) {

	HOGDescriptor hog;
	hog.winSize = Size(64, 64);

	cout << "FEATURE EXTRACTION" << endl;

	for (int i = 0; i < img.size(); i++) {
		vector<float> descriptor;
		hog.compute(img[i], descriptor);
		descriptor.push_back(colorFeature[i][0]);
		descriptor.push_back(colorFeature[i][1]);
		descriptor.push_back(colorFeature[i][2]);

		HOG.push_back(descriptor);
	}
}


// convertion for classification
void convertVectortoMatrix(vector<vector<float>>& HOG, Mat& Mat) {
	int descriptor_size = HOG[0].size();

	for (int i = 0;i < HOG.size();i++) {
		for (int j = 0;j < descriptor_size;j++) {
			Mat.at<float>(i, j) = HOG[i][j];
		}
	}
}


// SVM
void trainSVM(vector<int>& trainLabels, Mat& trainMat, vector<Mat> images) {

	Ptr<SVM> svm = SVM::create();
	svm->setKernel(SVM::LINEAR);
	svm->setC(1);
	svm->setGamma(0.5);
	svm->setType(SVM::C_SVC);
	svm->train(trainMat, ROW_SAMPLE, trainLabels);
	svm->save("traffic_sign_svm.xml"); // Save trained model
}


// Random Forest
void trainRandomForest(vector<int>& trainLabels, Mat& trainMat, vector<Mat> images) {

	Ptr<RTrees> randomForest = RTrees::create();
	randomForest->setMaxDepth(10);
	randomForest->setMinSampleCount(2);
	randomForest->setRegressionAccuracy(0);
	randomForest->setUseSurrogates(false);
	randomForest->setPriors(cv::Mat());
	randomForest->setCalculateVarImportance(true);
	randomForest->setActiveVarCount(4);
	randomForest->getVarImportance();
	randomForest->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 50, 0.01));
	randomForest->train(trainMat, ROW_SAMPLE, trainLabels);
	randomForest->save("traffic_sign_rf.xml"); // Save trained model
}



// model evaluation
void evaluate(Mat& response, float& count, float& accuracy, vector<int>& testLabels)
{

	// Initialize confusion matrix
	Mat confusionMatrix = Mat::zeros(58, 58, CV_32S);

	// Fill the confusion matrix
	for (int i = 0; i < response.rows; i++) {
		int trueLabel = testLabels[i];
		int predictedLabel = static_cast<int>(response.at<float>(i, 0));
		confusionMatrix.at<int>(trueLabel, predictedLabel)++;
	}

	// Print confusion matrix
	system("cls");
	cout << "Confusion Matrix:" << endl;
	for (int i = 0; i < 58; i++) {
		for (int j = 0; j < 58; j++) {
			cout << confusionMatrix.at<int>(i, j) << " ";
		}
		cout << endl;
	}

	cout << endl << "Classes with low precision / recall" << endl << endl;
	for (int i = 0; i < 58; i++) {
		int tp = confusionMatrix.at<int>(i, i); // True positives
		int fn = sum(confusionMatrix.row(i))[0] - tp; // False negatives
		int fp = sum(confusionMatrix.col(i))[0] - tp; // False positives

		float precision = tp / static_cast<float>(tp + fp);
		float recall = tp / static_cast<float>(tp + fn);
		float f1Score = 2 * (precision * recall) / (precision + recall);

		if ((precision == 1 && recall == 1) || isnan(precision)) {
			continue;
		}
		cout << "Class " << getLabelFromValue(i) << " - Precision: " << precision
			<< ", Recall: " << recall
			<< ", F1-Score: " << f1Score << endl;
	}
	cout << endl;

	for (int i = 0; i < response.rows; i++)
	{
		if (response.at<float>(i, 0) == testLabels[i])
			count = count + 1;
	}
	accuracy = (count / response.rows) * 100;
	cout << "THE ACCURACY IS :" << accuracy << endl;
	cout << "THE NUMBER SEGMENTED :" << count << " / " << response.rows << endl;

}


// result visualization
void visualize(Mat response, vector<Mat> oriImages, vector<Mat> segmentedImages, vector<Mat> testImages, vector<vector<float>>& HOG) {
	///////////////// Window
	for (int i = 0; i < response.total();i++) {
		int const	noOfImagePerCol = 1, noOfImagePerRow = 2;
		Mat			detailResultWin, win[noOfImagePerRow * noOfImagePerCol], legend[noOfImagePerRow * noOfImagePerCol];
		createWindowPartition(oriImages[i], detailResultWin, win, legend, noOfImagePerCol, noOfImagePerRow);

		putText(legend[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

		oriImages[i].copyTo(win[0]);

		putText(legend[1], getLabelFromValue(response.at<float>(i, 0)), Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

		resize(segmentedImages[i], segmentedImages[i], oriImages[i].size());
		segmentedImages[i].copyTo(win[1]);

		imshow("Classification", detailResultWin);
		imshow("Preprocessed images", testImages[i]);


		int plotHeight = 500;
		int binWidth = 1;
		int plotWidth = HOG[0].size() * binWidth + 20;
		Mat plot = Mat::zeros(plotHeight, plotWidth, CV_8UC3);
		int scaleFactor = 380;

		for (int k = 0; k < HOG[0].size(); k++) {
			int binHeight = static_cast<int>(HOG[i][k] * scaleFactor);  // Calculate height of the bar

			//blue color range
			Scalar color = Scalar(
				235,
				235,
				(k * 80) % 240
			);

			rectangle(plot,
				Point(k * binWidth + 10, plotHeight - 30),					// Bottom-left corner
				Point((k + 1) * binWidth + 10, plotHeight - binHeight - 30), // Top-right corner
				color,
				FILLED);                                          // Fill the rectangle
		}

		resize(plot, plot, Size(plotWidth / 3, 400));

		imshow("HOG Feature Plot", plot);
		waitKey();
		destroyAllWindows();
	}
}


// Process Flow
int main(int argc, char** argv) {
	Mat			src;
	Mat			croppedSign, resizedSign;
	char		str[256];
	vector<Scalar>	colors;
	int const	MAXfPt = 200;
	int			t1, t2, t3, t4;
	RNG			rng(0);
	String		imgPattern("Inputs/Traffic signs/signs/*.png"); //test
	String		imgPatternTrain("Inputs/Traffic signs/train/*.png");
	vector<string>	imageNames;
	vector<string>	imageNamesTrain;

	vector<Mat> oriImages;
	vector<Mat> segmentedImages;
	vector<Mat> trainImages; //resized
	vector<Mat> testImages;
	vector<int> trainLabels;
	vector<int> testLabels;

	vector<vector<int>> trainColorFeatureOneHot;
	vector<vector<int>> testColorFeatureOneHot;

	string firstThreeDigits;
	int firstThree;

	// get MAXfPt random but brighter colors for drawing later
	for (int t = 0; t < MAXfPt; t++) {
		for (;;) {
			t1 = rng.uniform(0, 255); // blue
			t2 = rng.uniform(0, 255); // green
			t3 = rng.uniform(0, 255); // red
			t4 = t1 + t2 + t3;
			// Below get random colors that is not dim
			if (t4 > 255) break;
		}
		colors.push_back(Scalar(t1, t2, t3));
	}

	int choice = -1;

	while (true) {
		imageNames.clear();
		imageNamesTrain.clear();
		cout << "MODEL LOADING......" << endl;
		Ptr<SVM> svm = SVM::load("traffic_sign_svm.xml");
		Ptr<RTrees> rf = RTrees::load("traffic_sign_rf.xml");

		cout << "1: trainSVM, 2: testSVM\n3: trainRandomForest, 4: testRandomForest\n5: predictInputImage(SVM)	:";
		choice = -1;
		cin >> choice;
		///train
		if (choice == 1 || choice == 3) {
			cout << "MODEL TRAINING....." << endl;
			cv::glob(imgPatternTrain, imageNamesTrain, true);
			for (int i = 0; i < imageNamesTrain.size(); ++i) {
				src = imread(imageNamesTrain[i]);

				if (src.empty()) { // found no such file?
					cout << "cannot open image for reading" << endl;
					return -1;
				}
				//color mask - return blue red yellow mask			
				vector<Mat> masks = colorMask(src);

				//combined mask
				Mat red, blue, green;
				Mat combined_mask;
				red = masks[0] | masks[2];
				blue = masks[1];
				green = masks[2];
				vector<Mat> channels = { blue, green, red };
				merge(channels, combined_mask);
				cvtColor(combined_mask, combined_mask, COLOR_BGR2GRAY);


				// shape detection
				vector<vector<Point>> contours;
				Mat show;
				show.create(src.rows, src.cols, src.type());
				show = 0;

				Mat segmented;
				segmented.create(src.rows, src.cols, src.type());
				segmented = 0;

				double maxArea = 0;
				int maxIndex = -1;

				findContours(combined_mask, contours, RETR_CCOMP, CHAIN_APPROX_NONE);

				for (int index = 0; index < contours.size(); index++) {
					double area = contourArea(Mat(contours[index]), false);
					if (area > maxArea) {
						maxArea = area;
						maxIndex = index;
					}
				}
				Rect boundingBox;
				// No contour found
				if (maxIndex > contours.size())
				{
					continue;
				}
				else {
					detectShape(contours[maxIndex], show);
					segmented = src & show;
					boundingBox = boundingRect(contours[maxIndex]);
					croppedSign = segmented(boundingBox);
					//Resize the cropped sign 
					resize(croppedSign, resizedSign, Size(64, 64), 0, 0, INTER_LINEAR);
					Mat newHSV;
					cvtColor(resizedSign, newHSV, COLOR_BGR2HSV); //convert to hsv
					cvtColor(resizedSign, resizedSign, COLOR_BGR2GRAY);

					colorFeatureOneHotEncoding(newHSV, trainColorFeatureOneHot);

				}
				int lastSlashPos = imageNamesTrain[i].find_last_of('\\');
				firstThreeDigits = imageNamesTrain[i].substr(lastSlashPos + 1, 3);
				firstThree = stoi(firstThreeDigits);

				trainLabels.push_back(firstThree);
				trainImages.push_back(resizedSign);

			}
			// FEATURE EXTRACTION & CLASSIFICATION
			vector<vector<float>> trainHOG;
			extractHOGFeatures(trainImages, trainHOG, trainColorFeatureOneHot);
			size_t descriptor_size_train = trainHOG[0].size();
			Mat trainMat(trainHOG.size(), descriptor_size_train, CV_32FC1);
			convertVectortoMatrix(trainHOG, trainMat);
			if (choice == 1) {
				trainSVM(trainLabels, trainMat, trainImages);
			}
			else {
				trainRandomForest(trainLabels, trainMat, trainImages);
			}
		}
		else if (choice == 2 || choice == 4 || choice == 5)
		{
			if (choice == 5) {
				string name;
				cout << "Key in image name in the directory [Inputs/Traffic signs/input/]	(eg. xxx.png)	::";
				cin >> name;
				String	imgDir = "Inputs/Traffic signs/input\\" + name;
				imageNames.push_back(imgDir);
			}
			else {
				cv::glob(imgPattern, imageNames, true);
			}
			for (int j = 0; j < imageNames.size(); ++j)
			{
				src = imread(imageNames[j]);

				if (src.empty()) { // found no such file?
					cout << "cannot open image for reading" << endl;
					return -1;
				}

				resize(src, src, Size(200, 200));
				oriImages.push_back(src);

				int lastSlashPos = imageNames[j].find_last_of('\\');
				firstThreeDigits = imageNames[j].substr(lastSlashPos + 1, 3);
				firstThree = stoi(firstThreeDigits);

				testLabels.push_back(firstThree);

				vector<Mat> masks = colorMask(src);



				//combined mask
				Mat red, blue, green;
				Mat combined_mask;
				red = masks[0] | masks[2];
				blue = masks[1];
				green = masks[2];
				vector<Mat> channels = { blue, green, red };
				merge(channels, combined_mask);
				cvtColor(combined_mask, combined_mask, COLOR_BGR2GRAY);

				vector<vector<Point>> contours;

				// shape detection
				Mat show;
				show.create(src.rows, src.cols, src.type());
				show = 0;

				Mat segmented;
				segmented.create(src.rows, src.cols, src.type());
				segmented = 0;

				double maxArea = 0;
				int maxIndex = -1;

				findContours(combined_mask, contours, RETR_CCOMP, CHAIN_APPROX_NONE);
				for (int index = 0; index < contours.size(); index++) {
					double area = contourArea(Mat(contours[index]), false);
					if (area > maxArea) {
						maxArea = area;
						maxIndex = index;
					}
				}
				Rect boundingBox;
				detectShape(contours[maxIndex], show);
				segmented = src & show;
				boundingBox = boundingRect(contours[maxIndex]);
				croppedSign = segmented(boundingBox);

				//Resize the cropped sign 
				resize(croppedSign, resizedSign, Size(64, 64), 0, 0, INTER_LINEAR);
				Mat newHSV;
				cvtColor(resizedSign, newHSV, COLOR_BGR2HSV); //convert to hsv
				cvtColor(resizedSign, resizedSign, COLOR_BGR2GRAY);

				colorFeatureOneHotEncoding(newHSV, testColorFeatureOneHot);

				segmentedImages.push_back(segmented);
				testImages.push_back(resizedSign);



			}
			// FEATURE EXTRACTION & CLASSIFICATION
			vector<vector<float>> testHOG;
			extractHOGFeatures(testImages, testHOG, testColorFeatureOneHot);
			int descriptor_size_test = testHOG[0].size();
			Mat testMat(testHOG.size(), descriptor_size_test, CV_32FC1);
			convertVectortoMatrix(testHOG, testMat);
			Mat response;
			if (choice == 2 || choice == 5) { // svm
				if (svm->empty()) {
					cerr << "Error: Could not load the SVM model from file!" << endl;
					return -1;
				}
					svm->predict(testMat, response);
			}
			else { // rf
				if (rf->empty()) {
					cerr << "Error: Could not load the RF model from file!" << endl;
					return -1;
				}
					rf->predict(testMat, response);

			}


			float count = 0;
			float accuracy = 0;
			evaluate(response, count, accuracy, testLabels);

			visualize(response, oriImages, segmentedImages, testImages, testHOG);

			return 0;
		}
		else {
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(), '\n');
			cout << "invalid input" << endl;
		}
	}
	return 0;

}

