from fastapi import FastAPI, Form, UploadFile, File, Depends, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
import json
from db_models import *
import os
import shutil, uuid, os

import smtplib
from email.message import EmailMessage
from config import Config



UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def send_ack_email(to_email: str, applicant_name: str):
    msg = EmailMessage()
    msg["Subject"] = "Подтверждение приёма заявления"
    msg["From"]    = "sochizhov@yandex.ru"
    msg["To"]      = to_email
    msg.set_content(
        f"Здравствуйте, {applicant_name}!\n\n"
        "Ваше заявление успешно получено. "
        "Наши сотрудники свяжутся с вами в ближайшее время.\n\n"
        "С уважением!"
    )

    # SMTP_SSL для порта 465
    with smtplib.SMTP_SSL(Config.SMTP_HOST, Config.SMTP_PORT) as smtp:
        smtp.login(Config.SMTP_USER, Config.SMTP_PASSWORD)
        smtp.send_message(msg)

class FormData(BaseModel):
    email: EmailStr

app = FastAPI()

# app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/form", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("appeal_index.html", {"request": request})

@app.get("/similarity_check")
async def get_similarity_check_form(request: Request):
    return templates.TemplateResponse("similarity_check_index.html", {"request": request})

@app.post("/check_patent")
async def check_similar(description: str = Form(...)):
    
    return FileResponse(
        path="./",
        media_type="application/pdf",
        filename="highlighted_texts.pdf"
    )

@app.post("/submit")
async def submit_form(
    background_tasks: BackgroundTasks,
    applicant: str = Form(...),
    inn: Optional[str] = Form(None),
    is_rightsholder: bool = Form(False),
    is_representative: bool = Form(False),
    email: str = Form(...),
    ip_type: str = Form(...),
    registration_number: Optional[str] = Form(None),
    links: List[str] = Form(...),
    violator_name: str = Form(...),
    ogrn: Optional[str] = Form(None),
    description: str = Form(...),
    evidence_docs: Optional[UploadFile] = File(None),
    ip_docs: Optional[UploadFile] = File(None),
    authority_docs: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    # Валидация email через Pydantic можно опустить, или добавить здесь
    
    # Сохраняем файлы и получаем пути
    def save_file(file: UploadFile, subdir: str) -> Optional[str]:
        if not file:
            return None
        ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4().hex}{ext}"
        out_dir = os.path.join(UPLOAD_DIR, subdir)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return path

    evidence_path  = save_file(evidence_docs,  "evidence")
    ip_docs_path   = save_file(ip_docs,       "ip")
    authority_path = save_file(authority_docs, "authority")

    appeal = Appeal(
        applicant           = applicant,
        inn                 = inn,
        is_rightsholder     = is_rightsholder,
        is_representative   = is_representative,
        email               = email,
        ip_type             = ip_type,
        registration_number = registration_number,
        links_json          = json.dumps(links, ensure_ascii=False),
        violator_name       = violator_name,
        ogrn                = ogrn,
        description         = description,
        evidence_path       = evidence_path,
        ip_docs_path        = ip_docs_path,
        authority_path      = authority_path,
    )

    db.add(appeal)
    db.commit()
    db.refresh(appeal)
    
    background_tasks.add_task(send_ack_email, to_email=email, applicant_name=applicant)

    return {"message": "Форма успешно отправлена", "id": appeal.id}