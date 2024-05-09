#include"Thresholding.h"

/******************************************Function definition**********************************************************/
Mat otsu_thresholding (Mat image){
    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);
    map<int, int> histogram = histo(gray_image);

    int size = gray_image.rows* gray_image.cols;


    float max_sigma_b = 0;

    int threshold = 0;

    //looping on the map
    auto it2 = histogram.begin();
    for (auto it = histogram.begin(); it != histogram.end(); it++){
        float w_b = 0;
        float mu_b = 0;
        float w_f = 0;
        float mu_f = 0;
        float sigma_b = 0;
        if (it == histogram.begin()){
            continue;
        }
        for (it2 ; it2 != it; it2++){
            w_b = w_b + it2->second;
            mu_b = mu_b + (it2->first)*(it2->second);
        }
        mu_b = mu_b/w_b;
        w_b = w_b/(size);

        for (auto it3 = it2; it3 != histogram.end(); it3++){
            w_f = w_f + it3->second;
            mu_f = mu_f + (it3->first)*(it3->second);
        }
        it2 = histogram.begin();
        mu_f = mu_f/w_f;
        w_f = w_f/(size);

        sigma_b = w_b*w_f*(mu_b-mu_f)*(mu_b-mu_f);

        if (sigma_b >= max_sigma_b){
            max_sigma_b = sigma_b;
            threshold = it->first;
        }

    }

    Mat threshold_image = thresholding(gray_image, threshold);
    return threshold_image;
}

map<int, int> histo (Mat image){
    map<int, int> histogram;
    for (int i = 0; i < image.rows; i++){
        for (int j = 0; j < image.cols; j++){
            histogram[image.at<uchar>(i, j)]++;
        }
    }
    return histogram;
}

Mat thresholding(Mat gray_image, int threshold){
    Mat threshold_image;
    threshold_image = gray_image.clone();
    for (int row = 0; row < gray_image.rows; row++)
    {
        for (int col = 0; col < gray_image.cols; col++)
        {
            if (gray_image.at<uchar>(row, col) > threshold)
            {
                threshold_image.at<uchar>(row, col) = 255;
            }
            else
            {
                threshold_image.at<uchar>(row, col) = 0;
            }
        }

    }
    return threshold_image;

}


Mat optimal_thresholding (Mat image){
    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);
    float threshold_previous;
    float threshold_next;
    vector<int> background;
    vector<int> foreground;
    threshold_next = ((gray_image.at<uchar>(0,0))+ (gray_image.at<uchar>(0,image.rows-1))+(gray_image.at<uchar>(image.rows-1,image.cols-1))+(gray_image.at<uchar>(image.cols-1,0)))/4.0;
    do
    {
        //loop on image
        for (int row = 0; row < gray_image.rows; row++)
        {
            for (int col = 0; col < gray_image.cols; col++)
            {
                if (gray_image.at<uchar>(row, col) > threshold_next)
                {
                    foreground.push_back(gray_image.at<uchar>(row, col));
                }
                else
                {
                    background.push_back(gray_image.at<uchar>(row, col));
                }
            }

        }
        //calculate the mean of background and foreground
        float mean_background = 0;
        float mean_foreground = 0;
        for (int i = 0; i < background.size(); i++){
            mean_background = mean_background + background[i];
        }
        mean_background = mean_background/background.size();
        for (int i = 0; i < foreground.size(); i++){
            mean_foreground = mean_foreground + foreground[i];
        }
        mean_foreground = mean_foreground/foreground.size();
        threshold_previous = threshold_next;
        threshold_next = (mean_background + mean_foreground)/2.0;
        background.clear();
        foreground.clear();
    }while ( abs(threshold_previous - threshold_next) > 0.02 );
    Mat threshold_img = thresholding(gray_image, (int)threshold_next);
    return threshold_img;
}



Mat spectral_thresholding (Mat image){
    // gray imag
    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);
    int size = gray_image.rows* gray_image.cols;
    // histogram
    map<int, int> histogram = histo(gray_image);
    // iterate on the map using pointer
    auto it_first_threshold = histogram.begin();
    it_first_threshold++;
    // creat a pointe rpoints to the next record thr map
    auto it_second_threshold = it_first_threshold;
    it_second_threshold++;
    it_second_threshold++;

    auto last_loop_second_threshold = histogram.end();
    last_loop_second_threshold--;
    last_loop_second_threshold--;

    auto last_loop_first_threshold = histogram.end();
    last_loop_first_threshold--;
    last_loop_first_threshold--;
    last_loop_first_threshold--;
    last_loop_first_threshold--;

    // main variables

    float max_sigma_b = -1;
    int first_threshold;
    int second_threshold;


    while (it_first_threshold != last_loop_first_threshold)
    {
        while (it_second_threshold != last_loop_second_threshold)
        {
            float p_1 = 0;
            float m_1 = 0;
            float p_2 = 0;
            float m_2 = 0;
            float p_3 = 0;
            float m_3 = 0;
            float m_G = 0;
            float sigma_b = 0;
            for (auto it = histogram.begin(); it != it_first_threshold; it++){
                p_1 = p_1 + (it->second)/(float)size;
                m_1 = m_1 + (it->first)*((it->second)/(float)size);
            }
            m_1 = m_1/p_1;

            for (auto it = it_first_threshold; it != it_second_threshold; it++){
                p_2 = p_2 + (it->second)/(float)size;
                m_2 = m_2 + (it->first)*((it->second)/(float)size);
            }
            m_2 = m_2/p_2;

            for (auto it = it_second_threshold; it != histogram.end(); it++){
                p_3 = p_3 + (it->second)/(float)size;
                m_3 = m_3 + (it->first)*((it->second)/(float)size);
            }
            m_3 = m_3/p_3;

            m_G = (p_1*m_1 + p_2*m_2 + p_3*m_3);

            sigma_b = p_1*(m_1-m_G)*(m_1-m_G) + p_2*(m_2-m_G)*(m_2-m_G) + p_3*(m_3-m_G)*(m_3-m_G);


            if (sigma_b >= max_sigma_b){
                max_sigma_b = sigma_b;
                first_threshold = it_first_threshold->first;
                second_threshold = it_second_threshold->first;
            }
            it_second_threshold++;

        }
        it_first_threshold++;
    }
    Mat threshold_image = double_thresholding(gray_image, first_threshold, second_threshold);

    return threshold_image;
}

Mat double_thresholding(Mat gray_image, int min_threshold, int max_threshold){
    Mat threshold_image;
    threshold_image = gray_image.clone();
    for (int row = 0; row < gray_image.rows; row++)
    {
        for (int col = 0; col < gray_image.cols; col++)
        {
            if (gray_image.at<uchar>(row, col) > max_threshold)
            {
                threshold_image.at<uchar>(row, col) = 255;
            }
            else if(gray_image.at<uchar>(row, col) < min_threshold)
            {
                threshold_image.at<uchar>(row, col) = 128;
            }else{
                threshold_image.at<uchar>(row, col) = 0;
            }
        }

    }
    return threshold_image;

}

Mat spectral_localThresholding (Mat image) {
    int rows = image.rows;
    int cols = image.cols;
    Rect roil(0, 0, cols/2, rows/2);
    Rect roi2(cols/2, 0, cols/2, rows /2);
    Rect roi3(0, rows/2, cols/2, rows/2);
    Rect roi4(cols/2, rows/2, cols/2, rows/2);
    Mat part1 = image (roil);
    part1 = spectral_thresholding (part1);
    Mat part2 = image (roi2);
    part2 = spectral_thresholding (part2);
    Mat part3 = image (roi3);
    part3 = spectral_thresholding (part3) ;
    Mat part4 = image (roi4);
    part4 = spectral_thresholding (part4) ;
    Mat thresholded_image = image. clone ();
//    convert to gray scale
    cvtColor (thresholded_image, thresholded_image, COLOR_BGR2GRAY) ;
    part1. copyTo (thresholded_image (roil)); part2.copyTo(thresholded_image(roi2)); part3.copyTo(thresholded_image(roi3)); part4. copyTo(thresholded_image (roi4));
    return thresholded_image;
}

