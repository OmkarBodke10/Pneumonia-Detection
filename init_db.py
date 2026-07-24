from database import engine, Base
import models

Base.metadata.create_all(bind=engine)
print("✅ Database initialized successfully: medical_history.db created!")
