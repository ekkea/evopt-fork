import glob
import json
import numpy

import pytest

from evopt.app import app


@pytest.mark.parametrize('test_case', glob.glob('test_cases/*.json'))
def test_optimizer(test_case: str):
    #test_list = glob.glob('test_data/*.json')
    client = app.test_client()

    response=None
    print(test_case)
    with open(f'{test_case}') as f:
        test_data = json.load(f)
        f.close()

    request = test_data.get("request", {})
    expected_response = test_data.get("expected_response", {})

    response = client.post("/optimize/charge-schedule", json=request)

    assert response.status_code == 200
    assert response.json["status"] == expected_response.get("status", {})
    assert numpy.isclose(response.json["objective_value"],
                         expected_response.get("objective_value", {}),
                         rtol=1e-05, atol=1e-08, equal_nan=False)



