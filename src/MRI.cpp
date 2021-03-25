#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

int main()
{
	std::string origPath = R"(registration_data\0_032_S_0147___2006-09-19___S19427.nii.png)";

	cv::Mat tmplt = cv::imread(origPath);

	for (auto& p : std::filesystem::recursive_directory_iterator(R"(registration_data\)"))
	{
		std::string path{ p.path().u8string() };

		if (origPath == path)
			continue;

		cv::Mat image = cv::imread(path);
		cv::Mat res0;

		cv::matchTemplate(tmplt, image, res0, cv::TM_CCORR_NORMED);

		double min0, max0;
		cv::Point min_loc0, max_loc0;
		cv::minMaxLoc(res0, &min0, &max0, &min_loc0, &max_loc0);

		cv::Mat tmpltG;
		applyColorMap(tmplt, tmpltG, cv::COLORMAP_DEEPGREEN);

		cv::Mat imageR;
		applyColorMap(image, imageR, cv::COLORMAP_HOT);
		cv::Mat sum;
		sum = tmpltG + imageR;

		cv::putText(sum, std::to_string(max0), cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 0.5, cv::LINE_AA);
		cv::imshow("Overlay", sum);

		auto detector = cv::ORB::create();

		std::vector<cv::KeyPoint> keypoints1, keypoints2;
		cv::Mat descriptors1, descriptors2;

		detector->detectAndCompute(tmplt, cv::Mat(), keypoints1, descriptors1);
		detector->detectAndCompute(image, cv::Mat(), keypoints2, descriptors2);

		cv::Mat img_keypoints1;
		cv::drawKeypoints(tmplt, keypoints1, img_keypoints1);

		cv::Mat img_keypoints2;
		cv::drawKeypoints(image, keypoints2, img_keypoints2);

		cv::imshow("Keypoints 1", img_keypoints1);
		cv::imshow("Keypoints 2", img_keypoints2);

		cv::BFMatcher matcher(cv::NORM_HAMMING, true);

		std::vector<cv::DMatch> matches;

		matcher.match(descriptors1, descriptors2, matches, cv::Mat());

		cv::Mat viz;
		cv::drawMatches(tmplt, keypoints1, image, keypoints2, { matches }, viz);

		cv::imshow("Matches", viz);

		cv::Mat dst;
		std::vector<cv::Point2f> obj;
		std::vector<cv::Point2f> scene;

		for (size_t i = 0; i < matches.size(); ++i)
		{
			obj.push_back(keypoints1[matches[i].queryIdx].pt);
			scene.push_back(keypoints2[matches[i].trainIdx].pt);
		}

		cv::Mat homography = cv::findHomography(scene, obj, cv::RANSAC);

		cv::warpPerspective(image, image, homography, cv::Size(image.cols, image.rows));

		cv::Mat res;
		cv::matchTemplate(tmplt, image, res, cv::TM_CCORR_NORMED);

		double min, max;
		cv::Point min_loc, max_loc;
		cv::minMaxLoc(res, &min, &max, &min_loc, &max_loc);

		cv::Mat imageR2;
		applyColorMap(image, imageR2, cv::COLORMAP_HOT);
		cv::Mat sum2;
		sum2 = tmpltG + imageR2;

		cv::putText(sum2, std::to_string(max), cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 0.5, cv::LINE_AA);
		cv::imshow("Alligned overlay", sum2);

		cv::waitKey();
	}
}
