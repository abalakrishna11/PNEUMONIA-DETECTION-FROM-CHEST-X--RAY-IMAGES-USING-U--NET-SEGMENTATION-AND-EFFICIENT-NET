╔══════════════════════════════════════════════════════╗
║         PneumoAI — Setup & Run Guide                ║
║         Pneumonia Detection System                  ║
╚══════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PROJECT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
pneumoai_final/
├── app.py                     ← Flask backend
├── requirements.txt           ← Python packages
├── README.txt                 ← This file
├── users.json                 ← Auto-created on first run
├── models/
│   └── best_final_v2.keras    ← PUT YOUR MODEL HERE
├── uploads/                   ← Auto-created (temp files)
└── templates/
    ├── login.html             ← Login page
    ├── register.html          ← Register page
    └── dashboard.html         ← Main analysis page

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SETUP STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — Install Python packages
  pip install flask werkzeug tensorflow opencv-python numpy scikit-learn

STEP 2 — Add your trained model
  Copy best_final_v2.keras into models/ folder:
  pneumoai_final/models/best_final_v2.keras

STEP 3 — Run the website
  cd pneumoai_final
  python app.py

STEP 4 — Open browser
  http://localhost:5000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 MODEL FILES SUPPORTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The app automatically looks for:
  1. models/best_final_v2.keras  ← Phase 2 model (recommended)
  2. models/best_final_v3.keras  ← Phase 3 model (if trained)
  3. models/best_final.keras     ← Original model
  4. models/efficientnet_classifier.keras

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 RUN ON ANOTHER LAPTOP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Copy entire pneumoai_final/ folder
2. Install Python 3.10
3. pip install flask werkzeug tensorflow opencv-python numpy scikit-learn
4. Copy model file to models/ folder
5. python app.py
6. Open http://localhost:5000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Login / Register system
✅ Upload chest X-ray (JPG/PNG)
✅ EfficientNetB3 prediction
✅ Confidence score with bar
✅ Grad-CAM++ heatmap visualization
✅ Severity estimation (Mild/Moderate/Severe)
✅ Image validation (rejects non-X-rays)
✅ Demo mode (works without model)
✅ Mobile responsive UI
✅ Dark medical theme

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 MODEL PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Validation Accuracy : 96.77%
AUC Score           : 0.9788
Precision           : 99.66%
Recall              : 95.65%
Threshold           : 0.70
Training images     : 6,526 (balanced 1:1)
