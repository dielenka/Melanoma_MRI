#include <iostream>
#include <filesystem>
#include <numeric>
#include <functional>
#include <opencv2/opencv.hpp>

void CalculateHistogram(cv::Mat& src, cv::Mat& histogram)
{
    cv::Mat image;
    cv::cvtColor(src, image, cv::COLOR_BGR2HSV);

    int channels[] = { 0, 1, 2 };
    int histSize[] = { 256, 256, 256 };
    float sranges[] = { 0, 256 };
    const float* ranges[] = { sranges, sranges, sranges };

    cv::calcHist(&image, 1, channels, cv::Mat(), histogram, 3, histSize, ranges, true, true);
}

cv::Mat SumMatrices(std::string path)
{
    cv::Mat sumhist;

    for (auto& entry : std::filesystem::recursive_directory_iterator(path))
    {
        std::filesystem::path file = entry.path();
        std::string path{ file.u8string() };

        cv::Mat src = cv::imread(path);

        cv::Mat hist;
        CalculateHistogram(src, hist);

        if (sumhist.empty())
            sumhist = hist;
        else
            cv::add(sumhist, hist, sumhist);
    }

    return sumhist;
}

std::unordered_map<std::string, bool> Correlate(std::string path, cv::Mat& negatives, cv::Mat& positives)
{
    std::unordered_map<std::string, bool> results;

    for (auto& entry : std::filesystem::recursive_directory_iterator(path))
    {
        std::filesystem::path file = entry.path();
        std::string path{ file.u8string() };

        cv::Mat src = cv::imread(path);

        cv::Mat hist;
        CalculateHistogram(src, hist);

        cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);

        if (cv::compareHist(hist, negatives, cv::HISTCMP_CORREL) > cv::compareHist(hist, positives, cv::HISTCMP_CORREL))
            results.emplace(path, 0);
        else
            results.emplace(path, 1);

        std::cout << path << " " << results.at(path) << std::endl;
    }

    return results;
}

int main()
{
    cv::Mat negativeSum = SumMatrices(R"(melanoma_data\data\train\0)");
    cv::Mat positiveSum = SumMatrices(R"(melanoma_data\data\train\1)");

    cv::normalize(negativeSum, negativeSum, 1.0, 0.0, cv::NORM_L1);
    cv::normalize(positiveSum, positiveSum, 1.0, 0.0, cv::NORM_L1);

    std::unordered_map<std::string, bool> negatives = Correlate(R"(melanoma_data\data\test\0)", negativeSum, positiveSum);
    std::unordered_map<std::string, bool> positives = Correlate(R"(melanoma_data\data\test\1)", negativeSum, positiveSum);
    
    size_t trueNegatives = negatives.size() - (std::accumulate(negatives.begin(), negatives.end(), 0, [](const size_t previous, decltype(*negatives.begin()) p) { return previous + p.second; }));
    size_t truePositives = std::accumulate(positives.begin(), positives.end(), 0, [](const size_t previous, decltype(*positives.begin()) p) { return previous + p.second; });
    
    std::cout << "true negatives: " << trueNegatives << "/" << negatives.size() << std::endl;
    std::cout << "false positives: " << negatives.size() - trueNegatives << "/" << negatives.size() << std::endl;
    std::cout << "true positives: " << truePositives << "/" << negatives.size() << std::endl;
    std::cout << "false negatives: " << positives.size() - truePositives << "/" << positives.size() << std::endl;
    
    double accuracy = (double)(truePositives + trueNegatives) / (double)(negatives.size() + positives.size());

    std::cout << "accuracy: " << accuracy * 100 << "%" << std::endl;
}
