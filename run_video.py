import sys
import os
import cv2
import numpy as np

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model


def build_trained_model(weights_path: str) -> Model:
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.load_weights(weights_path)
    return model


def process_video(input_video: str, weights_path: str, output_path: str) -> None:
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("[INFO] Loading model ...")
    model = build_trained_model(weights_path)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgb, (224, 224)).astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

            pred = float(model.predict(img, verbose=0)[0][0])
            is_violence = pred > 0.5

            color = (0, 0, 255) if is_violence else (0, 255, 0)
            label = f"{'VIOLENCE' if is_violence else 'SAFE'} {pred:.2f}"

            cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
            writer.write(frame)

            frame_idx += 1
            if frame_idx % 60 == 0:
                if total_frames:
                    print(f"[INFO] Processed {frame_idx}/{total_frames} frames ({(frame_idx/total_frames)*100:.1f}%)")
                else:
                    print(f"[INFO] Processed {frame_idx} frames")
    finally:
        writer.release()
        cap.release()
        print(f"[INFO] Saved output to {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_video.py <input_video_path>")
        sys.exit(1)

    input_video = sys.argv[1]
    weights_path = os.path.join(os.path.dirname(__file__), "mobilenetv2_violence.weights.h5")
    output_path = os.path.join(os.path.dirname(__file__), "output", "v_output.mp4")

    process_video(input_video, weights_path, output_path)


if __name__ == "__main__":
    main()


