from datetime import datetime

from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, backref, scoped_session, Session

from config import Config


engine = create_engine(Config.DATABASE_URI)

Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Appeal(Base):
    __tablename__ = "appeals"

    id             = Column(Integer, primary_key=True, index=True)
    # timestamp      = Column(TIMESTAMP, nullable=False)
    applicant      = Column(String, nullable=False)
    inn            = Column(String, nullable=True)
    is_rightsholder = Column(Boolean, default=False)
    is_representative = Column(Boolean, default=False)
    email          = Column(String, nullable=False)
    ip_type        = Column(String, nullable=False)
    registration_number = Column(String, nullable=True)

    links_json     = Column(Text, nullable=False)
    violator_name  = Column(String, nullable=False)
    # violator_store = Column(String, nullable=False)
    ogrn           = Column(String, nullable=True)
    description    = Column(Text, nullable=False)

    evidence_path  = Column(String, nullable=True)
    ip_docs_path   = Column(String, nullable=True)
    authority_path = Column(String, nullable=True)


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_all():
    Base.metadata.create_all(engine)


if __name__ == '__main__':
    create_all()