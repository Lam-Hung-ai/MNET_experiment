from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
from argparse import Namespace

from model.MNetold import MNetold  # import model của bạn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # có thể đổi sang ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model khi khởi động server ---
args = Namespace(k1=5, k2=2, k3=1, srm=1, inf=1, crf=1, sharing=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNetold(args=args)

MODEL_PATH = "weight/model_best.pth.tar"  # chỉnh lại đường dẫn checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)
try:
    model.load_state_dict(checkpoint["state_dict"])
except KeyError:
    model.load_state_dict(checkpoint)
model.to(device)
model.eval()

print(">> Model loaded and ready!")

# --- Hàm tiền xử lý ảnh ---
def preprocess_pil_image(img: Image.Image, device):
    img_rgb = img.convert("RGB")
    img_resized = img_rgb.resize((256, 256))  # resize về 256x256
    tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(device)
    return tensor

# --- Hàm hậu xử lý ảnh ---
def postprocess_to_pil(restored_bg, mask, input_tensor):
    final_output_tensor = restored_bg * mask + input_tensor * (1.0 - mask)
    output_tensor = final_output_tensor.squeeze(0).cpu()
    output_numpy_rgb = output_tensor.permute(1, 2, 0).numpy()
    output_numpy_rgb = (output_numpy_rgb * 255).astype(np.uint8)
    output_pil = Image.fromarray(output_numpy_rgb)  # chuyển sang PIL Image
    return output_pil

# --- API Endpoint ---
@app.post("/process-image")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))

    # Tiền xử lý
    input_tensor = preprocess_pil_image(img, device)

    # Inference
    with torch.no_grad():
        restored_bg, mask = model(input_tensor)

    # Hậu xử lý
    processed = postprocess_to_pil(restored_bg, mask, input_tensor)

    # Convert ảnh sang bytes để trả về
    img_byte_arr = io.BytesIO()
    processed.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    print(">> Image processed successfully!")
    return Response(content=img_byte_arr, media_type="image/png")
