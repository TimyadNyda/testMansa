import json
from datetime import date

import requests


def test_predict():
    """
    Test the predict route with test data
    """
    test_account = {"balance": 10000, "update_date": str(date(2020, 11, 3))}
    test_transactions = [
        {"date": str(date(2020, i, j)), "amount": -100}
        for i in range(1, 10)
        for j in range(1, 25)
    ]

    test_data = {
        "account": test_account,
        "transactions": test_transactions,
    }

    print("Calling API with test data:")

    response = requests.post(
        "http://127.0.0.1:8000/predict", data=json.dumps(test_data)
    )

    print("Response: ")
    print(response.json())

    assert response.status_code == 200


if __name__ == "__main__":
    test_predict()
