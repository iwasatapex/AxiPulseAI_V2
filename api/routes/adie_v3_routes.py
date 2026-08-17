"""Canonical ADIE V3 decision API route."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.encoders import jsonable_encoder

from api.models.adie_responses import ADIEDecisionResponse
from api.models.adie_v3_requests import ADIEV3DecisionRequest
from api.security.api_key import verify_api_key
from api.services.adie_v3_service import ADIEV3Service


router = APIRouter(dependencies=[Depends(verify_api_key)])
service = ADIEV3Service()


@router.post("/decision", response_model=ADIEDecisionResponse)
def create_decision(request: ADIEV3DecisionRequest) -> ADIEDecisionResponse:
    """Run the V3 probabilistic decision pipeline."""
    metadata = None
    if request.provenance is not None:
        metadata = {"provenance": request.provenance}

    if request.scenarios:
        result = service.analyze_scenarios(
            scenarios=request.scenarios,
            observations=request.observations,
            baseline=request.baseline,
            uncertainty=request.uncertainty,
            samples=request.samples,
            cutoff=request.cutoff,
            metadata=metadata,
            targets=request.targets,
            sensitivity_output=request.sensitivity_output,
            observed=request.observed,
            observed_metrics=request.observed_metrics,
            horizon=request.horizon,
        )
    else:
        result = service.analyze(
            observations=request.observations,
            baseline=request.baseline,
            uncertainty=request.uncertainty,
            samples=request.samples,
            cutoff=request.cutoff,
            metadata=metadata,
            targets=request.targets,
        )

    return {
        "status": "success",
        "data": jsonable_encoder(result),
        "metadata": {
            "engine": "ADIE",
            "version": "V3",
        },
    }
