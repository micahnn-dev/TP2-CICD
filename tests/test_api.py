import pytest
from fastapi.testclient import TestClient
from app.main import app
 
# Créer un client pour tester notre API FastAPI
client = TestClient(app)
 
 
# -----------------------------------------------------------------------------
# Cas nominaux : entrées valides et représentatives
# -----------------------------------------------------------------------------
def test_prediction_correcte():
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0]}
    )
 
    assert response.status_code == 200
    assert response.json() == {
        "predictions": [2.0, 4.0, 6.0]
    }
 
 
# -----------------------------------------------------------------------------
# Cas invalides : données ne respectant pas les préconditions attendues
# -----------------------------------------------------------------------------
def test_prediction_incorrecte():
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0]}
    )
 
    assert response.status_code == 200
 
    # Un résultat attendu volontairement faux
    with pytest.raises(AssertionError):
        assert response.json() == {
            "predictions": [10.0, 20.0, 30.0]
        }
 
 
# TEST 3 : JSON incorrect (champ features absent)
def test_json_incorrect():
    response = client.post(
        "/predict",
        json={"valeurs": [3.5, 1.2, 4.9]}
    )
 
    # FastAPI devrait rejeter la requête invalide
    assert response.status_code == 422