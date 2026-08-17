# Remediation Notes

The supplied source archive did not contain the two locked production model
binaries listed in `BASELINE_MANIFEST.md`. They cannot be reconstructed
without changing the trained-model baseline, so this remediation does not
fabricate or retrain them.

The application now fails closed in production when the locked artifacts are
unavailable, validates the NPS legacy artifact against `models/manifest.json`,
and keeps explicit degraded fallback mode behind
`AXIPULSE_ALLOW_FALLBACK_MODE=1`.

Production deployment must provide:

- `models/operation_health_predictor.joblib`
- `models/nps_predictor_model.pkl`
- `AXIPULSE_API_KEY`
- `AXIPULSE_JWT_SECRET`
- `AXIPULSE_ADMIN_USERNAME`
- `AXIPULSE_ADMIN_PASSWORD_SHA256`
- `monitoring/prometheus_api_token`
