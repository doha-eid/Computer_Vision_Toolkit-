#ifndef AGGLOMERATIVE_H
#define AGGLOMERATIVE_H
#include <iostream>
#include <opencv2/opencv.hpp>


 using namespace cv;
 using namespace std;

double euclidean_distance(const std::vector<double> &point1, const std::vector<double> &point2);

double clusters_distance(const vector<vector<double>> &cluster1, const vector<vector<double>> &cluster2);

vector<vector<vector<double>>> initial_clusters(Mat PixelsMatrix);


void fit(int k, Mat &pixels, vector<int> &labels, map<int, vector<double>> &centers, map<vector<double>, int> &cluster_map);


int predict_cluster(map<vector<double>, int> &cluster_map, vector<double> &point);

vector<double> predict_center(map<int, vector<double>> &centers, map<vector<double>, int> &cluster_map, vector<double> &point);

std::pair<Mat, Mat> image_preperation(Mat &image);

Mat image_color_segmentation(int k, Mat &pixels, Mat &resized_image);


#endif // AGGLOMERATIVE_H
