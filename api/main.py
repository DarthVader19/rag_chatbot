import uvicorn
import json
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr

# Initialize FastAPI app
app = FastAPI()

# --- Database ---
DB_FILE = "complaints_db.json"

def load_db():
    """Loads the database from the JSON file."""
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_db(db_data):
    """Saves the database to the JSON file."""
    with open(DB_FILE, "w") as f:
        json.dump(db_data, f, indent=4)

# --- Pydantic Models for Input Validation ---
class ComplaintInput(BaseModel):
    name: str
    phone_number: str
    email: EmailStr
    complaint_details: str

# --- API Endpoints ---

@app.post("/complaints")
def create_complaint(complaint: ComplaintInput):
    """
    Creates a new complaint and stores it in the database.
    Input (JSON): name, phone_number, email, complaint_details
    Output (JSON): complaint_id, message
    """
    db = load_db()
    
    # Generate a unique complaint ID [cite: 43]
    complaint_id = f"CMP-{uuid.uuid4().hex[:8].upper()}"
    
    # Create the complaint record
    db[complaint_id] = {
        "complaint_id": complaint_id,
        "name": complaint.name,
        "phone_number": complaint.phone_number,
        "email": complaint.email,
        "complaint_details": complaint.complaint_details,
        "created_at": datetime.now().isoformat()
    }
    
    save_db(db)
    
    return {
        "complaint_id": complaint_id,
        "message": "Complaint created successfully"
    } # [cite: 29, 30]

@app.get("/complaints/{complaint_id}")
def get_complaint(complaint_id: str):
    """
    Retrieves complaint details by complaint ID.
    Output (JSON): complaint_id, name, phone_number, email, complaint_details, created_at
    """
    db = load_db()
    
    complaint = db.get(complaint_id)
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")
        
    return complaint # [cite: 35, 36, 37, 38, 39, 40]

# --- To run the API ---
# In your terminal, navigate to the 'api' directory and run:
# uvicorn main:app --reload