from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
from utils import read_categories, url_to_tensor
from model import Model

app = FastAPI()
model = Model("converted_mobilenet_model.tflite")
result_dict = read_categories('labels.txt')

def predict_image(image: UploadFile = File(...)):
    try:
        with open("temp.jpg", "wb") as temp_image:
            temp_image.write(image.file.read())
        image_tensor = url_to_tensor("temp.jpg")
        prediction = model.test(image_tensor)
        category_label = result_dict[str(prediction)]
        return {"prediction": category_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_image")
async def predict_image_endpoint(file: UploadFile = File(...)):
    return  predict_image(file)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)