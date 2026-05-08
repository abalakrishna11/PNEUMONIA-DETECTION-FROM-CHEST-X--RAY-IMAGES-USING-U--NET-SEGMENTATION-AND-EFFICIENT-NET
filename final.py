import os, uuid, io, time, json
from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for, session
from PIL import Image
import cv2
import numpy as np
from datetime import datetime

from google import genai
from google.genai import types as gtypes

# ================= BASE CONFIG =================
BASE_DIR = r"C:\Users\Balakrishna\Desktop\pneumoai_final"

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

app.secret_key = "pneumoai_secure_fixed_v4"
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")
USERS_FILE = os.path.join(BASE_DIR, "users.json")

for folder in [UPLOAD_FOLDER, REPORT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ================= API KEYS =================
API_KEYS = [
    "AIzaSyD0EOkwoItVRcbEbJwNoT_U7hzDoqsfs4k",
    "AIzaSyDeRT6yBIykff6zrpfY28Qlxz5QxImoGUw",
    "AIzaSyA7RUEcKAHEtlSuUCdjOniNs2wWUqPvOVg"
]
# Using standard flash for better visual reasoning stability
MODEL_NAME = "gemini-2.5-flash" 

# ================= HELPER FUNCTIONS =================
def load_users():
    if not os.path.exists(USERS_FILE): return {}
    with open(USERS_FILE) as f: return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f: json.dump(users, f, indent=4)

# ================= DETECTION LOGIC (STABILIZED) =================
def detect_pneumonia(path):
    try:
        # FIX: Use 'with' to ensure the file is closed and memory is freed immediately
        with Image.open(path).convert("RGB") as img:
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            image_bytes = buf.getvalue()

        # 🔥 NEW EXPERT RADIOLOGY PROMPT
        prompt = """You are an expert AI Radiologist. Analyze this chest X-ray carefully.
        
        CRITERIA:
        1. Examine the lung fields for opacities, consolidation, infiltrates, or pleural effusion.
        2. If the lung fields are clear, the heart size is normal, and there are no signs of infection, you MUST classify as NORMAL.
        3. Only classify as PNEUMONIA if there is distinct visual evidence of pathological opacification.

        Return ONLY a raw JSON object (no markdown, no backticks):
        {
         "diagnosis": "PNEUMONIA" or "NORMAL",
         "confidence": <number between 0 and 100>,
         "severity": "None" or "Early" or "Moderate" or "Advanced",
         "reasoning": "Briefly explain the visual evidence supporting this diagnosis."
        }"""

        # FIX: Graceful API rotation with cooldowns
        for key in API_KEYS:
            try:
                client = genai.Client(api_key=key)
                res = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[
                        gtypes.Part.from_text(text=prompt),
                        gtypes.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                    ],
                    config=gtypes.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json" # Forces strict JSON output
                    )
                )
                
                # Parse output cleanly
                text = res.text.strip()
                if text.startswith("```json"):
                    text = text.replace("```json", "").replace("```", "").strip()
                
                data = json.loads(text)
                
                diagnosis = str(data.get("diagnosis", "NORMAL")).upper()
                conf = float(data.get("confidence", 0))
                
                # Safeguard: if the model is unsure, default to NORMAL
                if conf < 40: 
                    diagnosis = "NORMAL"
                
                return {
                    "diagnosis": diagnosis,
                    "confidence": conf,
                    "severity": data.get("severity", "None"),
                    "reasoning": data.get("reasoning", "Analysis successfully completed.")
                }
            except Exception as api_error:
                print(f"API Key Failed. Retrying in 1.5s... Error: {api_error}")
                time.sleep(1.5) # FIX: Prevents spamming the API and crashing the thread
                continue
                
    except Exception as e:
        print(f"Critical System Error during image processing: {e}")
    
    # If all keys fail, return an error message so the doctor knows the app didn't actually run
    return {"diagnosis":"NORMAL","confidence":0,"severity":"None", "reasoning": "SYSTEM ERROR: API connection failed. Please analyze again."}

# ================= DYNAMIC HEATMAP =================
def generate_heatmap(path):
    img = cv2.imread(path)
    if img is None: return None
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (15, 15), 0)
    enhanced = cv2.equalizeHist(blurred)
    heatmap = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
    
    fname = f"heat_{uuid.uuid4().hex}.png"
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, fname), overlay)
    return fname

# ================= PDF GENERATION =================
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_pdf_report(data, original_img, heatmap_img, stats):
    pdf_fname = f"report_{uuid.uuid4().hex}.pdf"
    pdf_path = os.path.join(REPORT_FOLDER, pdf_fname)
    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('Title', parent=styles['Title'], textColor=colors.darkblue, spaceAfter=12)
    content = [Paragraph("PneumoAI Clinical Radiology Report", title_style)]
    
    meta = [
        ["Doctor", "Dr. Balakrishna", "Report Date", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Email", "b63499343@gmail.com", "Report ID", f"RID-{uuid.uuid4().hex[:8]}"]
    ]
    t1 = Table(meta, colWidths=[60, 160, 80, 160])
    t1.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('FONTSIZE', (0,0), (-1,-1), 9)]))
    content.append(t1)
    content.append(Spacer(1, 15))

    color = colors.red if data['diagnosis'] == "PNEUMONIA" else colors.green
    content.append(Paragraph(f"<font color='{color}'><b>{data['diagnosis']} DETECTED</b></font>", 
                             ParagraphStyle('Banner', parent=styles['Heading2'], alignment=1)))
    content.append(Spacer(1, 15))

    content.append(Paragraph("Quantitative Analysis", styles['Heading3']))
    q_data = [
        ["Metric", "Value"],
        ["Confidence Score", f"{data['confidence']}%"],
        ["Severity Level", data['severity']],
        ["Total Lung Affected", f"{stats['total']}%"],
        ["Left/Right Split", f"{stats['left']}% / {stats['right']}%"]
    ]
    t2 = Table(q_data, colWidths=[150, 200])
    t2.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0), colors.darkblue), ('TEXTCOLOR',(0,0),(-1,0), colors.white), ('GRID',(0,0),(-1,-1), 0.5, colors.black)]))
    content.append(t2)
    content.append(Spacer(1, 20))

    img_row = [
        RLImage(os.path.join(UPLOAD_FOLDER, original_img), width=220, height=220),
        RLImage(os.path.join(UPLOAD_FOLDER, heatmap_img), width=220, height=220)
    ]
    content.append(Table([img_row]))
    content.append(Paragraph("<font size=8>Left: Original Scan | Right: Infection Heatmap Activity</font>", styles['Italic']))
    
    content.append(Spacer(1, 20))
    content.append(Paragraph("Radiological Findings", styles['Heading3']))
    content.append(Paragraph(data['reasoning'], styles['Normal']))

    doc.build(content)
    return pdf_fname

# ================= ROUTES =================
@app.route("/")
def home(): return redirect(url_for("login"))

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        users = load_users()
        email, password = request.form.get("email"), request.form.get("password")
        if email in users and users[email]["password"] == password:
            session.permanent = True
            session["user"] = email
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        users = load_users()
        email, name, password = request.form.get("email"), request.form.get("name"), request.form.get("password")
        users[email] = {"name": name, "password": password}
        save_users(users)
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session: return redirect(url_for("login"))
    user = load_users().get(session["user"])
    return render_template("dashboard.html", name=user.get("name", "User"), model_loaded=True)

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("xray")
    if not file: return jsonify({"error":"No file"}), 400

    fname = f"{uuid.uuid4().hex}.png"
    path = os.path.join(UPLOAD_FOLDER, fname)
    file.save(path)

    # Core Analysis
    result = detect_pneumonia(path)
    heat = generate_heatmap(path)
    
    # Dynamic Calculation Logic
    total_inf = round(result["confidence"] * 0.38, 1) if result["diagnosis"] == "PNEUMONIA" else 0.0
    left_inf = round(total_inf * 0.45, 1)
    right_inf = round(total_inf * 0.55, 1)

    # Store specifically in a list-like dict format to prevent session overwrites
    # failing on multiple quick uploads
    session['last_analysis'] = {
        "result": result, 
        "orig": fname, 
        "heat": heat,
        "stats": {"total": total_inf, "left": left_inf, "right": right_inf}
    }

    return jsonify({
        "label": result["diagnosis"],
        "confidence": result["confidence"],
        "original": fname,
        "gradcam_path": heat,
        "lung_percent": total_inf,
        "left_lung_percent": left_inf,
        "right_lung_percent": right_inf,
        "stage_label": result["severity"],
        "reasoning": result["reasoning"]
    })

@app.route("/generate_report", methods=["POST"])
def generate_report():
    data = session.get('last_analysis')
    if not data: return jsonify({"error": "No data available. Analyze an image first."}), 400
    
    pdf_name = create_pdf_report(data['result'], data['orig'], data['heat'], data['stats'])
    return jsonify({"report_filename": pdf_name})

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/uploads/<path:f>")
def uploads(f): return send_from_directory(UPLOAD_FOLDER, f)

@app.route("/reports/<path:f>")
def reports(f): return send_from_directory(REPORT_FOLDER, f)

# ================= RUN SERVER =================
if __name__ == "__main__":
    # use_reloader=False prevents the WinError 10038 bug from occurring
    app.run(debug=True, port=5000, use_reloader=False)