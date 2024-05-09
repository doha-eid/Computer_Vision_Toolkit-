#include"Kmeans.h"

Mat K_means(Mat image){
    // Convert the image to a vector
        std::vector<cv::Vec3f> pixels;
        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                pixels.push_back(image.at<cv::Vec3b>(y, x));
            }
        }

        // Choose the value of K
        int K = 3;

        // Initialize the centroids
        std::vector<cv::Vec3f> centroids;
        for (int i = 0; i < K; i++) {
            centroids.push_back(pixels[rand() % pixels.size()]);
        }

        // Assign each pixel to a cluster
        std::vector<int> labels(pixels.size());
        for (int i = 0; i < pixels.size(); i++) {
            float min_distance = FLT_MAX;
            for (int j = 0; j < centroids.size(); j++) {
                float distance = cv::norm(pixels[i], centroids[j]);
                if (distance < min_distance) {
                    min_distance = distance;
                    labels[i] = j;
                }
            }
        }

        // Recalculate the centroids
        for (int i = 0; i < centroids.size(); i++) {
            cv::Vec3f sum(0, 0, 0);
            int count = 0;
            for (int j = 0; j < pixels.size(); j++) {
                if (labels[j] == i) {
                    sum += pixels[j];
                    count++;
                }
            }
            if (count > 0) {
                centroids[i] = sum / count;
            }
        }

        // Repeat steps 5 and 6 until convergence
        int max_iterations = 10;
        for (int iteration = 0; iteration < max_iterations; iteration++) {
            // Assign each pixel to a cluster
            for (int i = 0; i < pixels.size(); i++) {
                float min_distance = FLT_MAX;
                for (int j = 0; j < centroids.size(); j++) {
                    float distance = cv::norm(pixels[i], centroids[j]);
                    if (distance < min_distance) {
                        min_distance = distance;
                        labels[i] = j;
                    }
                }
            }

            // Recalculate the centroids
            for (int i = 0; i < centroids.size(); i++) {
                cv::Vec3f sum(0, 0, 0);
                int count = 0;
                for (int j = 0; j < pixels.size(); j++) {
                    if (labels[j] == i) {
                        sum += pixels[j];
                        count++;
                    }
                }
                if (count > 0) {
                    centroids[i] = sum / count;
                }
            }
        }

        // Assign colors to each cluster
        std::vector<cv::Vec3b> cluster_colors(centroids.size());
        for (int i = 0; i < centroids.size(); i++) {
            cluster_colors[i] = centroids[i];
        }

        // Display the segmented image
        cv::Mat segmented_image(image.size(), image.type());
        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                int label = labels[y * image.cols + x];
                segmented_image.at<cv::Vec3b>(y, x) = cluster_colors[label];
            }
        }
        return segmented_image;
}
