import os
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import uvicorn
from openai import OpenAI

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# FastAPI app
app = FastAPI()
from fastapi.responses import HTMLResponse
from fastapi import Request

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>EKG Interpreter</title>
        </head>
        <body>
            <h2>Upload an EKG Image</h2>
            <form action="/api/ekg/interpret" enctype="multipart/form-data" method="post">
                <label for="age">Age:</label><br>
                <input type="text" name="age"><br>
                <label for="sex">Sex:</label><br>
                <input type="text" name="sex"><br>
                <label for="symptoms">Symptoms:</label><br>
                <input type="text" name="symptoms"><br>
                <label for="history">History:</label><br>
                <input type="text" name="history"><br>
                <label for="meds">Medications:</label><br>
                <input type="text" name="meds"><br>
                <label for="vitals">Vitals:</label><br>
                <input type="text" name="vitals"><br><br>
                <input type="file" name="image"><br><br>
                <input type="submit" value="Interpret EKG">
            </form>
        </body>
    </html>
    """
    
@app.post("/api/ekg/interpret")
async def interpret_ekg(
    image: UploadFile = File(...),
    age: str = Form(None),
    sex: str = Form(None),
    symptoms: str = Form(None),
    history: str = Form(None),
    meds: str = Form(None),
    vitals: str = Form(None),
):
    """
    Upload an EKG image + optional clinical context.
    Sends the data to OpenAI Vision API for interpretation.
    """

    # Read and encode the image
    img_bytes = await image.read()
    b64_img = base64.b64encode(img_bytes).decode("utf-8")

    # Build the prompt
    context_text = f"""
    Patient context:
    - Age: {age}
    - Sex: {sex}
    - Symptoms: {symptoms}
    - History: {history}
    - Medications: {meds}
    - Vitals: {vitals}

    Task: Interpret the attached EKG image. 
    Provide rate, rhythm, axis, conduction, intervals, ischemia changes, 
    and overall impression. Be concise but clinically useful.
    """

    # Call OpenAI Vision model
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # fast + multimodal
        messages=[
            {"role": "system", "content": "You are an expert cardiologist interpreting EKGs."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": context_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                ],
            },
        ],
        max_tokens=500,
    )

    interpretation = response.choices[0].message.content

    return JSONResponse({
        "overall_interpretation": interpretation,
        "context_received": {
            "age": age,
            "sex": sex,
            "symptoms": symptoms,
            "history": history,
            "meds": meds,
            "vitals": vitals,
        }
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))  # Render injects $PORT
    uvicorn.run(app, host="0.0.0.0", port=port)



