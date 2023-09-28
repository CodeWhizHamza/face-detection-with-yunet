#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include <chrono>
#include <iostream>
#include <thread>
#include <string>

// using namespace cv;
using namespace std;

#include <iostream>

static cv::Mat visualize(cv::Mat input, cv::Mat faces, bool print_flag = false, double fps = -1, int thickness = 2)
{
    cv::Mat output = input.clone();

    if (fps > 0) {
        cv::putText(output, cv::format("FPS: %.2f", fps), cv::Point2i(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
    }

    for (int i = 0; i < faces.rows; i++)
    {
        if (print_flag) {
            cout << "Face " << i
                << ", top-left coordinates: (" << faces.at<float>(i, 0) << ", " << faces.at<float>(i, 1) << "), "
                << "box width: " << faces.at<float>(i, 2) << ", box height: " << faces.at<float>(i, 3) << ", "
                << "score: " << faces.at<float>(i, 14) << "\n";
        }

        // Draw bounding box
        cv::rectangle(
            output, 
            cv::Rect2i(
                int(faces.at<float>(i, 0)), 
                int(faces.at<float>(i, 1)), 
                int(faces.at<float>(i, 2)), 
                int(faces.at<float>(i, 3))
            ), 
            cv::Scalar(0, 255, 0), 
            thickness
        );
        cv::putText(output, cv::format("%.4f", faces.at<float>(i, 14)), cv::Point2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)) + 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
    }
    return output;
}

int faces_counter = 0;

 void save_faces(cv::Mat frame, cv::Mat faces) {
    cv::Mat copy;
    int factor = 20;


    for (int i = 0; i < faces.rows; i++) {
        try {
            copy = frame.clone();

            int x1 = int(faces.at<float>(i, 0));
            int y1 = int(faces.at<float>(i, 1));
            int w  = int(faces.at<float>(i, 2));
            int h  = int(faces.at<float>(i, 3));

            cv::Rect face(
                (x1 > factor) ? x1 - factor : x1,
                (y1 > factor) ? y1 - factor : y1,
                w + factor* 2,
                h + factor * 2
            );
            cv::Mat croppedFace = copy(face);

            cout << "Saved..." << endl;
            cv::imwrite(std::format("./faces/face-{}.jpg", faces_counter), croppedFace);
            faces_counter++;
        }
        catch (exception e) {
            cout << "HEHE I'm not gonna save it..." << endl;
        }
    }
}

int main(int argc, char** argv)
{
    cv::String modelPath = "C:/Users/muham/OneDrive/Documents/Projects/VAS-ML-Pilot/weights/face_detection_yunet_2023mar.onnx";
    
    int backendId = 5; // 3 for cpu, 5 for cuda
    int targetId = 6; // 0 for cpu, 6 for cuda

    //float scoreThreshold = 0.9;
    float scoreThreshold = 0.5;
    float nmsThreshold = 0.3;
    int topK = 5000;

    bool save = false;
    bool vis = true;
    cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create(
        modelPath,
        "", 
        cv::Size(320, 320), 
        scoreThreshold, 
        nmsThreshold, 
        topK, 
        backendId, 
        targetId
    );

    

    // for device camera
    //int deviceId = 0;
    //cv::VideoCapture cap;
    //cap.open(deviceId, cv::CAP_ANY);

    for (int i = 0; i < argc; i++) {
        std::cout << argv[i] << endl;
    }

    if (argc < 2) {
        std::cout << "Please add title of video without extension" << endl;
        return 0;
    }

    string file_name_without_ext = argv[1];

    cout << "Attempting to open stream" << endl;
    //cv::VideoCapture cap("rtsp://admin:senti4512@sentinelai.nayatel.net:554/chID=1&streamType=sub");
    cv::VideoCapture cap(std::format("./{}.mp4", file_name_without_ext));

    if (!cap.isOpened()) {
        cout << "Cannot open stream " << endl;
        return 0;
    }

    int frameWidth = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = int(cap.get(cv::CAP_PROP_FPS));
    float delay = 1.0 / float(fps);

    cv::VideoWriter writer(
        std::format("./{}-with-faces-visualized.mp4", file_name_without_ext),
        cap.get(cv::CAP_PROP_FOURCC), 
        (double)fps, 
        cv::Size(frameWidth, frameHeight)
    );

    std::cout << "Frame Dims: " << frameWidth << " " << frameHeight << std::endl;
    std::cout << "FPS: " << fps << std::endl;

    detector->setInputSize(cv::Size(frameWidth, frameHeight));

    int counter = 0;

    cv::Mat frame;
    cv::TickMeter tm;
    while (cv::waitKey(1) < 0) // Press any key to exit
    {
        if (!cap.read(frame))
        {
            cerr << "No frames grabbed!\n";
            break;
        }

        cv::Mat faces;
        tm.start();
        detector->detect(frame, faces);
        tm.stop();

        cv::Mat vis_frame = visualize(frame, faces, false, tm.getFPS());

        imshow("libfacedetection demo", vis_frame);
        
        tm.reset();
        if (counter > 10) {
            save_faces(frame, faces);
            counter = 0;
        }
        counter += 1;

        writer.write(vis_frame);

        //std::this_thread::sleep_for(std::chrono::milliseconds(int(delay * 1000)));
    }

    writer.release();
    std::cout << "Video saved" << endl;

    cv::destroyAllWindows();

    std::cout << "Press enter to exit ..." << std::endl;
    std::cin.get();
}