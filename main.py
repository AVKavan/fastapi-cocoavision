import json
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from fastapi.responses import StreamingResponse
import numpy as np
from inference import get_model
import supervision as sv
import cv2

import os
from mangum import Mangum

app = FastAPI()

handler = Mangum(app)

@app.post("/predict")
async def predict(image: UploadFile):
    # load the image
    img = Image.open(image.file)
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # load a pre-trained yolov8n model
    model = get_model(model_id="merged-project-2/1",api_key="INc4g2WbMuzVOyCAXNVp")

    # run inference on our chosen image
    results = model.infer(img_array)

    # load the results into the supervision Detections api
    detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))
    # convert the numpy integer object to a Python list
    class_list = detections.data.get('class_name').tolist()
    if class_list is None:
      class_list = [] 
    # create supervision annotators
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # annotate the image with our inference results
    annotated_image = bounding_box_annotator.annotate(
        scene=img_array, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    # convert the annotated image to PIL Image
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    annotated_image = Image.fromarray(annotated_image)

    # convert the PIL Image to bytes
    buffer = io.BytesIO()
    annotated_image.save(buffer, format='JPEG')
    buffer.seek(0)

    
    return StreamingResponse(
      content=buffer,
      media_type="image/jpeg",
      headers={
           'X-Class-List': json.dumps(class_list)  # Add class ID as a custom header
      }
  )

