import logging

import warnings
import secrets
import base64
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from utils import read_config, read_objects, make_predict
from constants import CONFIG

security = HTTPBasic()

warnings.filterwarnings("ignore")
config = read_config()
app = FastAPI(
    title="FastAPI",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
log = logging.getLogger(__name__)
model_0, modelrubertyni, tokenizer = read_objects()
model_0.set_config(config)

@app.get("/docs", include_in_schema=False)
# async def get_swagger_documentation(username: str = Depends(get_current_username)):
async def get_swagger_documentation():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")


@app.get("/redoc", include_in_schema=False)
async def get_redoc_documentation():
    return get_redoc_html(openapi_url="/openapi.json", title="docs")


@app.get("/openapi.json", include_in_schema=False)
async def openapi():
    return get_openapi(title=app.title, version=app.version, routes=app.routes)


@app.get("/")
async def homepage():
    return "default"


@app.get("/yesnoquery")
async def asterisk_api(txt: str = ""):
    print(f"REQUEST: '{txt}'")
    txt = base64.b64decode(txt).decode("utf-8").strip()

    print(f"REQUEST: '{txt}'")
    if txt.strip() == "":
        return config.get("empty_value", 0)
    return make_predict(txt, model_0, modelrubertyni, tokenizer)

@app.get("/yesnoquery2")
async def asterisk_api2(txt: str = ""):
    # print(f"REQUEST: '{txt}'")
    # txt = base64.b64decode(txt).decode("utf-8").strip()

    print(f"REQUEST: '{txt}'")
    if txt.strip() == "":
        return config.get("empty_value", 0)
    return make_predict(txt, model_0, modelrubertyni, tokenizer)



@app.get("/validation2")
async def val2(group: str = "", txt: str = ""):

    print(f"REQUEST: '{txt}'")
    return model_0.predict_group(group=group, txt=txt)
