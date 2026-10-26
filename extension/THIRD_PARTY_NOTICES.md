# Third-party software and models

GhostFrame packages the following components, unmodified.

| Component | Used for | Licence | Source |
|---|---|---|---|
| MediaPipe Tasks Vision (`@mediapipe/tasks-vision` 1.1.1) | Runtime for segmentation and gesture recognition | Apache-2.0 | https://github.com/google-ai-edge/mediapipe |
| MediaPipe Gesture Recognizer model | Open-palm gesture, hand landmarks | Apache-2.0 | https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer |
| MediaPipe Selfie Segmenter (landscape) model | Person segmentation | Apache-2.0 | https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter |
| ONNX Runtime Web (`onnxruntime-web` 1.30.0) | Runs the fill model (WebGPU / WASM) | MIT | https://github.com/microsoft/onnxruntime |
| MI-GAN (`migan_pipeline_v2.onnx`), Picsart AI Research | Filling in background the camera never saw | MIT (see `models/MIGAN_LICENSE`) | https://github.com/Picsart-AI-Research/MI-GAN, https://huggingface.co/andraniksargsyan/migan |
| LaMa inpainting (`inpainting_lama_2025jan.onnx`, OpenCV Zoo export) | Filling in large never-seen areas | Apache-2.0 | https://github.com/advimman/lama, https://huggingface.co/opencv/inpainting_lama |
