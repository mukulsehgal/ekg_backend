import os
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from dotenv import load_dotenv
import uvicorn
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# FastAPI app
app = FastAPI(title="EKG Interpreter", version="1.0")

# -------------------------------------------------------------------
# Home page: Upload form
# -------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>EKG Interpreter</title>
        </head>
        <body>
            <h2>Upload an EKG Image for Interpretation</h2>
            <form action="/api/ekg/interpret" enctype="multipart/form-data" method="post">
                <label>Age:</label><br>
                <input type="text" name="age"><br>
                <label>Sex:</label><br>
                <input type="text" name="sex"><br>
                <label>Symptoms:</label><br>
                <input type="text" name="symptoms"><br>
                <label>History:</label><br>
                <input type="text" name="history"><br>
                <label>Medications:</label><br>
                <input type="text" name="meds"><br>
                <label>Vitals:</label><br>
                <input type="text" name="vitals"><br><br>
                <label>Upload EKG Image:</label><br>
                <input type="file" name="image"><br><br>
                <input type="submit" value="Interpret EKG">
            </form>
            <p>Or visit <a href="/docs">API Docs</a></p>
        </body>
    </html>
    """

# -------------------------------------------------------------------
# API Endpoint: JSON response (works for API and the form above)
# -------------------------------------------------------------------
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
        model="gpt-4o-mini",  # multimodal, fast
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

    # Return both JSON (for API clients) and HTML (for form users)
    return HTMLResponse(f"""
    <html>
        <body>
            <h2>EKG Interpretation Result</h2>
            <p><strong>Age:</strong> {age}</p>
            <p><strong>Sex:</strong> {sex}</p>
            <p><strong>Symptoms:</strong> {symptoms}</p>
            <p><strong>History:</strong> {history}</p>
            <p><strong>Medications:</strong> {meds}</p>
            <p><strong>Vitals:</strong> {vitals}</p>
            <h3>Interpretation:</h3>
            <p>{interpretation}</p>
            <p><a href="/">⬅️ Upload Another</a></p>
        </body>
    </html>
    """)


# -------------------------------------------------------------------
# Run locally
# -------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
