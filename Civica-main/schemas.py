from pydantic import BaseModel, Field


class SchemeAnalysis(BaseModel):
    summary: str = Field(
        description=(
            "A concrete 2-3 sentence plain-language overview of the identified Indian government scheme, "
            "including its purpose, target beneficiaries, and the main support or benefit. Do not answer with "
            "generic insufficiency statements when the scheme can be identified."
        )
    )
    eligibility: list[str] = Field(
        default_factory=list,
        description=(
            "Specific eligibility criteria supported by retrieved evidence: age, income, occupation, geography, "
            "category, family status, or other conditions. Prefer concrete thresholds and exceptions."
        ),
    )
    benefits: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete benefits supported by evidence, including amounts, subsidies, insurance cover, services, "
            "duration, frequency, or other entitlements when stated."
        ),
    )
    process: list[str] = Field(
        default_factory=list,
        description=(
            "Actionable application steps in order. Include where/how to apply, portal or office, registration, "
            "documents, verification, and what happens next when supported by evidence."
        ),
    )
    documents_required: list[str] = Field(
        default_factory=list,
        description="Documents or information applicants need, only when supported by evidence.",
    )
    important_notes: list[str] = Field(
        default_factory=list,
        description="Important deadlines, exclusions, conditions, cautions, or practical notes from the evidence.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description=(
            "Actual source URLs used for the answer. Prefer official Government of India, state government, "
            "ministry, PIB, myScheme, or official scheme portal sources. Include direct URLs, not source names alone."
        ),
    )


class FraudAnalysis(BaseModel):
    result_class: str
    result_text: str
    result_details: str


class ChatResponse(BaseModel):
    response: str
