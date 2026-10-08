"""转置小助手 Web 服务：本地运行，零登录、零数据库。"""
import pathlib
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response

from core.schema import FIELDS, DERIVED_FIELDS, ASSUMPTIONS, FIELD_LABELS
from core.mapping import load_templates
from core.pipeline import transpose, revalidate, list_samples
from core.export import export_xlsx
from core.llm import llm_available, llm_config

ROOT = pathlib.Path(__file__).resolve().parent

app = FastAPI(title="转置小助手")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/bootstrap")
def bootstrap():
    templates = load_templates()
    return {
        "fields": FIELDS,
        "derived": DERIVED_FIELDS,
        "labels": FIELD_LABELS,
        "assumptions": ASSUMPTIONS,
        "llm_available": llm_available(),
        "llm_model": llm_config()["model"] if llm_available() else "",
        "templates": [{"customer_id": t["customer_id"], "customer_name": t["customer_name"],
                       "notes": t.get("notes", "")} for t in templates.values()],
        "samples": list_samples(),
    }


@app.post("/api/transpose")
async def api_transpose(req: Request):
    body = await req.json()
    return transpose(body["template_id"], body["sample_id"])


@app.post("/api/validate")
async def api_validate(req: Request):
    body = await req.json()
    return {"records": revalidate(body.get("records", []), body.get("mode", "offline"))}


@app.post("/api/export")
async def api_export(req: Request):
    body = await req.json()
    template_name = body.get("template_name", "客户")
    sample_id = body.get("sample_id", "export")
    mode = body.get("mode", "offline")
    records = revalidate(body.get("records", []), mode)
    data = export_xlsx(template_name, records, mode)
    fname = f"报岗表-{sample_id}.xlsx"
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(fname)}"},
    )
