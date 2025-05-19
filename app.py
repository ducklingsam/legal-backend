from fastapi import FastAPI, Form, UploadFile, File, Depends, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from typing import List, Optional
from sqlalchemy.orm import Session
from pathlib import Path
import json
from db_models import *
from models import PatentSearch, SBERT
import os
import shutil
import uuid
import os
import yake
import pymorphy2

import smtplib
from email.message import EmailMessage
from config import Config


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_keyword(description: str):
    morph = pymorphy2.MorphAnalyzer()

    kw_extractor = yake.KeywordExtractor(lan="ru", n=3, top=20)
    keywords = kw_extractor.extract_keywords(description)

    def is_technical(term):
        parsed = morph.parse(term)[0]
        return parsed.tag.POS == 'NOUN'

    tech_keywords = [kw for kw, score in keywords if is_technical(kw)]

    return tech_keywords[0] if tech_keywords else None


def send_ack_email(to_email: str, applicant_name: str):
    msg = EmailMessage()
    msg["Subject"] = "Подтверждение приёма заявления"
    msg["From"] = "sochizhov@yandex.ru"
    msg["To"] = to_email
    msg.set_content(
        f"Здравствуйте, {applicant_name}!\n\n"
        "Ваше заявление успешно получено. "
        "Наши сотрудники свяжутся с вами в ближайшее время.\n\n"
        "С уважением!"
    )

    with smtplib.SMTP_SSL(Config.SMTP_HOST, Config.SMTP_PORT) as smtp:
        smtp.login(Config.SMTP_USER, Config.SMTP_PASSWORD)
        smtp.send_message(msg)


app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/application_form", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("application_form.html", {"request": request})


@app.get("/similarity_check")
async def get_similarity_check_form(request: Request):
    return templates.TemplateResponse("similarity_check.html", {"request": request})


@app.post("/search_patents")
async def check_similar(description: str = Form(...)):
    patents = PatentSearch()
    keyword = get_keyword(description)
    print(keyword)
    patents.search_by_natural(
        query={"q": keyword, "pre_tag": "", "post_tag": "", "limit": 1000})
    texts = patents.get_parsed_data()

    sbert_search = SBERT()
    sbert_search.find_simmilar(description, texts)

    file_path = Path("highlighted_texts.pdf").resolve()

    return FileResponse(
        path=file_path,
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

    evidence_path = save_file(evidence_docs,  "evidence")
    ip_docs_path = save_file(ip_docs,       "ip")
    authority_path = save_file(authority_docs, "authority")

    appeal = Appeal(
        applicant=applicant,
        inn=inn,
        is_rightsholder=is_rightsholder,
        is_representative=is_representative,
        email=email,
        ip_type=ip_type,
        registration_number=registration_number,
        links_json=json.dumps(links, ensure_ascii=False),
        violator_name=violator_name,
        ogrn=ogrn,
        description=description,
        evidence_path=evidence_path,
        ip_docs_path=ip_docs_path,
        authority_path=authority_path,
    )

    db.add(appeal)
    db.commit()
    db.refresh(appeal)

    background_tasks.add_task(
        send_ack_email, to_email=email, applicant_name=applicant)

    return {"message": "Форма успешно отправлена", "id": appeal.id}
