from pydantic import BaseModel


class BaseSchemaResponse(BaseModel):
    detail: str
