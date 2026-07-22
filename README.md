# fire-smoke
For each camera, every polling cycle:

Frame capture — ffmpeg connects to the camera's RTSP stream and pulls a single frame back as a JPEG.
YOLO detection — the frame is run through a custom-trained YOLO model (me (1).pt). Any box above the confidence threshold moves to the next stage.
Florence-2 captioning — for each qualifying box, the full frame is captioned by Florence-2 (<DETAILED_CAPTION> prompt), producing a natural language description of the scene.
Semantic verification — the caption is embedded with all-MiniLM-L6-v2 and compared (cosine similarity) against two banks of reference phrases (FIRE_PHRASES, SMOKE_PHRASES). If either max similarity clears SIMILARITY_THRESHOLD, the detection is accepted.
Decision — PASS detections (real fire/smoke) and REJECT detections (YOLO false positives Florence didn't confirm) are saved to separate folders with the box drawn on the frame, and every detection is logged to a CSV.
