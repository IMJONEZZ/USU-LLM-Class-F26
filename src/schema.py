from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Text(Strict):
    type: Literal["text"]
    text: str = Field(min_length=1)


class Em(Strict):
    type: Literal["em"]
    text: str = Field(min_length=1)


class Strong(Strict):
    type: Literal["strong"]
    text: str = Field(min_length=1)


class Br(Strict):
    type: Literal["br"]


Part = Annotated[Text | Em | Strong | Br, Field(discriminator="type")]


class Heading(Strict):
    type: Literal["heading"]
    level: Literal[1, 2, 3]
    text: str = Field(min_length=1)


class Paragraph(Strict):
    type: Literal["paragraph"]
    parts: list[Part] = Field(min_length=1)


Block = Annotated[Heading | Paragraph, Field(discriminator="type")]


class WebPage(Strict):
    title: str = Field(min_length=1)
    body: list[Block] = Field(min_length=1)
