from mnemos.api.routes.providers import router as providers_router


def test_graeae_provider_recommendation_does_not_claim_core_route():
    paths = [route.path for route in providers_router.routes]

    assert "/v1/providers/graeae/recommend" in paths
    assert "/v1/providers/recommend" not in paths


def test_application_route_table_has_one_handler_per_recommendation_path():
    from mnemos.api.main import app

    paths = [route.path for route in app.routes]

    assert paths.count("/v1/providers/recommend") == 1
    assert paths.count("/v1/providers/graeae/recommend") == 1
