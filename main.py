import cv2
from deepface import DeepFace  # type: ignore
import numpy as np
from tqdm import tqdm


def main():
    scoreThreshold = 0.5
    nmsThreshold = 0.3
    backendTarget = 3
    targetId = 0
    size = (320, 320)

    detector = cv2.FaceDetectorYN.create(
        model="./face_detection_yunet_2023mar.onnx",
        config="",
        input_size=size,
        score_threshold=scoreThreshold,
        nms_threshold=nmsThreshold,
        backend_id=backendTarget,
        target_id=targetId,
    )

    # print("Loading video...")
    # # video = cv2.VideoCapture("./museum-2.mp4")
    # fps = video.get(cv2.CAP_PROP_FPS)
    # width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    image = cv2.imread("./image.jpeg")
    width = image.shape[1]
    height = image.shape[0]
    detector.setInputSize((width, height))
    # frames = []

    detections = detector.detect(image)
    faces = detections[1]

    if faces is not None:
        for face in faces:
            image = handle_face(image, face)

    cv2.imwrite("output.jpg", image)

    # print("Processing video...")
    # for _ in tqdm(range(frame_count)):
    #     ret, frame = video.read()
    #     if not ret:
    #         break

    #     detections = detector.detect(frame)
    #     faces = detections[1]

    #     if faces is not None:
    #         for face in faces:
    #             frame = handle_face(frame, face)

    #     frames.append(frame)

    #     # cv2.imshow("frame", frame)
    #     # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     #     break

    # video.release()

    # print("Saving video...")
    # fourcc = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore
    # out = cv2.VideoWriter("output-2.avi", fourcc, fps, (width, height))

    # for frame in tqdm(frames):
    #     out.write(frame)

    # out.release()


def handle_face(frame: np.ndarray, face: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2, *_ = face
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # face area with padding in all directions

    faceArea = frame[y1 - 50 : y1 + y2 + 50, x1 - 50 : x1 + x2 + 50]

    # return frame if face area is out of bounds
    if faceArea.shape[0] == 0 or faceArea.shape[1] == 0:
        return frame

    frame = cv2.rectangle(
        frame,
        pt1=(x1, y1),
        pt2=(x1 + x2, y1 + y2),
        color=(0, 255, 0),
        thickness=2,
    )
    cv2.putText(
        frame,
        f"{face[14]:.2f}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (36, 255, 12),
        2,
    )

    # face recognition
    results = DeepFace.analyze(
        faceArea,
        actions=["age", "gender"],
        enforce_detection=False,
        silent=True,
        detector_backend="retinaface",
    )

    if len(results) == 0:
        return frame

    result = results[0]

    age = result["age"]
    gender = result["dominant_gender"]

    frame = cv2.putText(
        frame,
        text="Age: " + str(int(age)),
        org=(x1, y1 + y2 + 30),
        fontFace=0,
        fontScale=0.5,
        color=(0, 255, 0),
        thickness=1,
    )

    frame = cv2.putText(
        frame,
        text="Gender: " + gender,
        org=(x1, y1 + y2 + 50),
        fontFace=0,
        fontScale=0.5,
        color=(0, 255, 0),
        thickness=1,
    )
    return frame


if __name__ == "__main__":
    main()
