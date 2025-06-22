import sys
sys.path.append('.')
from app.main import app

print('FastAPI app imported successfully')
print('Available endpoints:', [route.path for route in app.routes])
print('App title:', app.title if hasattr(app, 'title') else 'No title')
