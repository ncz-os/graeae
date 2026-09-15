from mnemos.domain.graeae.engine import _compute_quorum


def test_matching_numeric_templates_do_not_form_quorum():
    result = _compute_quorum(
        {
            "a": {
                "status": "success",
                "response_text": "The account balance is 100 dollars.",
            },
            "b": {
                "status": "success",
                "response_text": "The account balance is 900 dollars.",
            },
        },
        0.66,
    )
    assert not result["quorum_reached"]
    assert result["contradiction_pairs"]["a:b"]
    assert result["agreement_metric"] == "lexical_complete_link"


def test_similarity_bridge_does_not_invent_three_way_agreement(monkeypatch):
    from mnemos.domain.graeae import engine

    scores = {
        frozenset(("A", "B")): 0.9,
        frozenset(("B", "C")): 0.9,
        frozenset(("A", "C")): 0.1,
    }
    monkeypatch.setattr(
        engine, "_textual_agreement", lambda a, b: scores[frozenset((a, b))]
    )
    result = _compute_quorum(
        {k: {"status": "success", "response_text": k} for k in "ABC"}, 0.8
    )
    assert not result["quorum_reached"]


async def test_agreement_metadata_survives_http_response(
    client, auth_headers, mock_graeae_engine
):
    mock_graeae_engine.consult.return_value.update(
        {
            "agreement_metric": "lexical_complete_link",
            "contradiction_pairs": {"a:b": True},
            "cost": None,
        }
    )
    response = await client.post(
        "/v1/consultations",
        json={"prompt": "compare", "mode": "single"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json()["agreement_metric"] == "lexical_complete_link"
    assert response.json()["contradiction_pairs"] == {"a:b": True}
    assert response.json()["cost"] is None
