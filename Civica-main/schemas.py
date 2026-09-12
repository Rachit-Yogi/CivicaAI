from pydantic import BaseModel, Field


class SchemeAnalysis(BaseModel):
    response: str = Field(
        description=(
            "A dynamic, detailed citizen-friendly answer about the identified Indian government scheme, "
            "program, fund, portal, or initiative. Write natural text with useful headings and numbered "
            "steps where appropriate. Include what it is, who it is for, benefits/purpose, eligibility, "
            "documents, how to apply/access, important conditions, and current status/details when supported "
            "by grounded evidence. Do not use generic insufficiency language."
        )
    )
    sources: list[str] = Field(
        default_factory=list,
        description=(
            "Actual URLs used for the answer. Prefer official Government of India, state government, ministry, "
            "PIB, myScheme, official scheme portal, or official notification/PDF URLs."
        ),
    )


class FraudAnalysis(BaseModel):
    result_class: str = Field(
        description=(
            "One of: Verified, Likely Genuine, Unverified___Needs_Caution, Likely False, Confirmed Scam / Fraud."
        )
    )
    response: str = Field(
        description=(
            "A dynamic evidence-based explanation covering what was checked, what the evidence shows, red flags "
            "or supporting evidence, and what the user should do next. Write naturally and concretely."
        )
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Actual URLs consulted for verification.",
    )


class ChatResponse(BaseModel):
    response: str = Field(
        description=(
            "A natural, practical answer in the same language and script used by the user unless the user asks "
            "for another language. For learning, doing, or achieving a goal, explain the process progressively "
            "from beginner level through actionable next steps."
        )
    )
