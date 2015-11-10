// Caution!
// !! don't compile it with NDEBUG =(
// !! don't compile it with _GLIBCXX_DEBUG, or be sure that opencv was compiled with this flag
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cmath> 
#include <exception>
#include <functional>
#include <ctime>
#include <stdlib.h>
#include <unordered_map>
#include <sys/resource.h>

using namespace cv;
using namespace std;

namespace {
    // change kPrefixPath before running or create folder `results/`
    const string kPrefixPath = "";

class ImageWrapper {
public:
    explicit ImageWrapper(const string& image_name)
        : image_name_(image_name)
    {
        image_ = imread(image_name);
        if (!image_.data) {
            cerr << "can't find: " << image_name << endl;
            throw runtime_error("bad image's name");
        }
    }

    explicit ImageWrapper(const Mat& image)
        : image_(image)
    {
        assert(image.data);
    }

    ImageWrapper(const Mat& image, const string& image_name)
        : image_name_(image_name)
        , image_(image)
    {
        assert(image_.data && image_name.size() > 0);
    }

    ImageWrapper Show() const {
        cerr << "show: " << id_ << endl; 
        namedWindow(to_string(id_), WINDOW_NORMAL);
        if (image_.type() == CV_32F) {
            Mat temp;
            image_.convertTo(temp, CV_8U, 255);
            imshow(to_string(id_), temp);
        } else {
            imshow(to_string(id_), image_);
        }
        id_++;

        return *this;
    }

    static void Wait() {
        waitKey(0);
    }

    ImageWrapper Save(const string& name = "") const {
        Mat image = image_;
        if (image.type() == CV_32F) {
            image.convertTo(image, CV_8U, 255);
        }

        if (name.size() > 0) {
            assert(imwrite(kPrefixPath + name, image));
        } else if (image_name_.size() > 0) {
            assert(imwrite(kPrefixPath + image_name_, image));
        } else {
            assert(false);
        }

        return *this;
    }

    ImageWrapper FindFruits(const int threshold_ = 140) const {
        assert(threshold_ > 40); // =(, dfs is so sad

        Mat lab_image;
        GaussianBlur(image_, lab_image, Size(5, 5), 0);
        cvtColor(lab_image, lab_image, CV_BGR2Lab);

        Mat planes[3];
        split(lab_image, planes);

        Mat& a_plane = planes[1];
        threshold(a_plane, a_plane, threshold_, 255, THRESH_BINARY);

        const int anchor_x = 5;
        const int anchor_y = 5;
        Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * anchor_x + 1, 2 * anchor_y + 1), Point(anchor_x, anchor_y));
        morphologyEx(a_plane, a_plane, MORPH_CLOSE, element);
        morphologyEx(a_plane, a_plane, MORPH_OPEN, element);

        int cnt_components;
        vector<vector<int>> conn_matrix;
        // get connected components and count of components
        tie(conn_matrix, cnt_components) = ImageWrapper(a_plane).ConnMatrix();

        Mat grad_x;
        Sobel(a_plane, grad_x, CV_32F, 1, 0); // x = 1, 0
        pow(grad_x, 2, grad_x);

        Mat grad_y;
        Sobel(a_plane, grad_y, CV_32F, 0, 1); // y = 0, 1
        pow(grad_y, 2, grad_y);

        Mat grad; // grad = sqrt(grad_x ** 2 + grad_y ** 2)
        sqrt(grad_x + grad_y, grad);
        grad.convertTo(grad, CV_8U);
        threshold(grad, grad, 100, 255, THRESH_BINARY); 

        vector<pair<vector<int>, vector<int>>> color2coord(cnt_components);
        for (int i = 0; i < grad.rows; ++i) {
            for (int j = 0; j < grad.cols; ++j) {
                if (grad.at<uchar>(i, j) && conn_matrix[i][j]) {
                    color2coord[conn_matrix[i][j]].first.push_back(i);
                    color2coord[conn_matrix[i][j]].second.push_back(j);
                }
            }
        }

        for (int i = 0; i < cnt_components; ++i) {
            sort(color2coord[i].first.begin(), color2coord[i].first.end());
            sort(color2coord[i].second.begin(), color2coord[i].second.end());
        }

        Mat circles = Mat::zeros(image_.rows, image_.cols, CV_8U);
        int cnt = 0;
        for (int i = 0; i < cnt_components; ++i) {
            vector<int>& xs = color2coord[i].first;
            vector<int>& ys = color2coord[i].second;
            if (xs.size() == 0 || ys.size() == 0) continue;
            int mid_x = xs.size() / 2;
            int mid_y = ys.size() / 2;

            Point center(ys[mid_y], xs[mid_x]);
            int radius = sqrt(pow(xs.front() - xs[mid_x], 2) + pow(ys.back() - ys[mid_y], 2));
            //if (radius < 20) continue;

            // draw circle on the Mat `circles`
            circle(circles, center, radius, Scalar(255), 6);
            cnt++;
        }

        cerr << "count = " << cnt << endl;

        //without circles
        //
        //const int anchor_x2 = 1;
        //const int anchor_y2 = 1;
        //Mat element2 = getStructuringElement(MORPH_ELLIPSE, Size(2 * anchor_x + 1, 2 * anchor_y2 + 1), Point(anchor_x2, anchor_y2));
        //dilate(grad, grad, element2);
        //grad *= 255;
        //Mat grad_arr[] = {grad, grad, Mat::zeros(image_.rows, image_.cols, CV_8U)};

        // with circles
        Mat grad_arr[] = {circles, circles, Mat::zeros(image_.rows, image_.cols, CV_8U)};

        merge(grad_arr, 3, grad); 

        return ImageWrapper(image_ + grad);
    }

    // returns labelled matrix and count of colors
    pair<vector<vector<int>>, int> ConnMatrix() const {
        assert(image_.channels() == 1);
        assert(image_.type() == CV_8U);

        vector<vector<int>> res(image_.rows, vector<int>(image_.cols));
        for (int i = 0; i < image_.rows; ++i) {
            for (int j = 0; j < image_.cols; ++j) {
                res[i][j] = image_.at<uchar>(i, j);
            }
        }

        int conn_num = 1;

        // vector<bool> is slow I know it...
        vector<vector<bool>> used(image_.rows, vector<bool>(image_.cols, false));
        for (int i = 0; i < image_.rows; ++i) {
            for (int j = 0; j < image_.cols; ++j) {
                if (!used[i][j] && res[i][j] > 0) {
                    PaintIt(res, conn_num++, i, j, used);
                }
            }
        }

        return {res, conn_num};
    }

    ImageWrapper operator + (const ImageWrapper& oth) const {
        return ImageWrapper(image_ + oth.image_);
    }

    ImageWrapper operator - (const ImageWrapper& oth) const {
        return ImageWrapper(image_ - oth.image_);
    }

    ImageWrapper operator * (double c) const {
        return ImageWrapper(image_ * c);
    }

    bool operator == (Scalar_<uchar> s) const {
        for (int i = 0; i < (int) size(); ++i) {
            if (image_.data[i] != s[0]) {
                return false;
            }
        }
        return true;
    }

    size_t size() const {
        return image_.rows * image_.cols * image_.channels();
    }

private:

    static bool Good(int i, int j, vector<vector<int>>& matr) {
        return i >= 0 && j >= 0 && 
               i < (int) matr.size() && j < (int) matr[i].size() && 
               matr[i][j] > 0;
    }
        
    static void PaintIt(vector<vector<int>>& matr, int conn_num, int i, int j, vector<vector<bool>>& used) {
        assert(i < (int) matr.size() && j < (int) matr[i].size());
        assert(i < (int) used.size() && j < (int) used[i].size());
        static const int di[] = {0, 0, 1, -1};
        static const int dj[] = {1, -1, 0, 0};
        static const int cnt_d = 4;

        used[i][j] = true; 
        matr[i][j] = conn_num;

        for (int k = 0; k < cnt_d; ++k) {
            int ni = i + di[k];
            int nj = j + dj[k];
            if (Good(ni, nj, matr) && !used[ni][nj]) {
                PaintIt(matr, conn_num, ni, nj, used);
            }
        }
    }

private:
    string image_name_;
    Mat image_;
    static int id_;
}; // ImageWrapper

int ImageWrapper::id_ = 0;
} // unnamed namespace

int main(int argc, char *argv[]) {
    string path_to_image = "./tree3.jpg";

#ifdef DEBUG
    if (argc > 1 && strlen(argv[1]) > 0) {
        path_to_image = string(argv[1]);
    }
#endif

    cerr << path_to_image << endl;

    ImageWrapper(path_to_image)
        .Show()
        .FindFruits(140)
        .Show()
        .Save("tree3_res.jpg");

    //ImageWrapper("tree1.jpg")
        //.Show()
        //.FindFruits(130)
        //.Show()
        //.Save("tree1_res.jpg");

    ImageWrapper::Wait();
    

    return 0;
}
