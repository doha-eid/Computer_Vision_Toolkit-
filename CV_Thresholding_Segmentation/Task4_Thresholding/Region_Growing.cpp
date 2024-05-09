#include"Region_Growing.h"

void region_growing(Mat& img, Point seed, Mat& mask, int threshold) {
    // Create a queue for the pixels to be processed
    queue<Point> q;

    // Set the seed pixel to 255 in the mask
    mask.at<uchar>(seed) = 255;

    // Add the seed pixel to the queue
    q.push(seed);

    // Get the height and width of the image
    int height = img.rows;
    int width = img.cols;

    // Loop until the queue is empty
    while (!q.empty()) {
        // Get the next pixel from the queue
        Point p = q.front();
        q.pop();

        // Loop through the neighboring pixels
        for (int r = p.y-1; r <= p.y+1; r++) {
            for (int c = p.x-1; c <= p.x+1; c++) {
                // Check if the neighboring pixel is within the image bounds
                if (r < 0 || c < 0 || r >= height || c >= width)
                    continue;

                // Check if the neighboring pixel has already been labeled
                if (mask.at<uchar>(r, c) != 0)
                    continue;

                // Check if the difference in intensity between the current pixel and the neighboring pixel is below the threshold
                if (abs(img.at<uchar>(p) - img.at<uchar>(r, c)) < threshold) {
                    // Set the neighboring pixel to 255 in the mask
                    mask.at<uchar>(r, c) = 255;

                    // Add the neighboring pixel to the queue
                    q.push(Point(c, r));
                }
            }
        }
    }
}
