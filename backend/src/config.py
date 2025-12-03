from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiConfig(BaseModel):
    title: str = "Jordan RNN API"
    description: str = "Веб-интерфейс для обучения нейронной сети Джордана"
    debug: bool = True


class RunConfig(BaseModel):
    port: int = 8000
    host: str = "127.0.0.1"
    reload: bool = True

    @property
    def base_url(self) -> str:
        return f"{self.host}:{self.port}"


class FilesConfig(BaseModel):
    base: Path = Path(__file__).parent.parent
    src: Path = base / "src"
    stocks: Path = base / "stocks"


class Settings(BaseSettings):
    api: ApiConfig = ApiConfig()
    run: RunConfig = RunConfig()
    files: FilesConfig = FilesConfig()

    model_config = SettingsConfigDict(case_sensitive=False)


settings = Settings()
