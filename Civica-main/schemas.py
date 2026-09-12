from pydantic import BaseModel, Field

class SchemeAnalysis(BaseModel):
    summary: str = Field(description="One-line plain-language summary")
    eligibility: list[str] = Field(default_factory=list)
    benefits: list[str] = Field(default_factory=list)
    process: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)

class FraudAnalysis(BaseModel):
    result_class: str
    result_text: str
    result_details: str

class ChatResponse(BaseModel):
    response: str
