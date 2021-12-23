from datetime import date
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, validator

import pandas as pd
import numpy as np 

from utils import preprocess, preprocess_values, model, MAX_LOG_TARGET



class Account(BaseModel):
    update_date: date
    balance: float


class Transaction(BaseModel):
    amount: float
    date: date


class RequestPredict(BaseModel):
    account: Account
    transactions: List[Transaction]

    @validator("transactions")
    def validate_transaction_history(cls, v, *, values):
        # validate that
        # - the transaction list passed has at least 6 months history
        # - no transaction is posterior to the account's update date
        if len(v) < 1:
            raise ValueError("Must have at least one transaction")

        update_t = values["account"].update_date

        oldest_t = v[0].date
        newest_t = v[0].date
        for t in v[1:]:
            if t.date < oldest_t:
                oldest_t = t.date
            if t.date > newest_t:
                newest_t = t.date

        assert (
            update_t - newest_t
        ).days >= 0, "Update Date Inconsistent With Transaction Dates"
        assert (update_t - oldest_t).days > 183, "Not Enough Transaction History"

        return v


class ResponsePredict(BaseModel):
    predicted_amount: float
    upper_bound: float
    lower_bound: float


def predict(transactions: List[Transaction], account: Account) -> float:

    pd_data = pd.DataFrame(map(dict, transactions))
    # compute features
    df_ = preprocess(pd_data, account.update_date, account.balance)
    # process values and keep last 180 timesteps
    df_ = preprocess_values(df_)[-180:]

    # Return predicted amount
    preds = model.predict(np.expand_dims(df_, 0))[0]
    u_bound = np.clip(preds[0], 0, preds[0])
    l_bound = np.clip(preds[1], 0, preds[1])
    if not np.equal(u_bound, 0):
        u_bound = np.exp(u_bound*MAX_LOG_TARGET)
    if not np.equal(l_bound, 0):
        l_bound = np.exp(l_bound*MAX_LOG_TARGET)

    np.equal(0., 0)
    return ResponsePredict(
        predicted_amount=(u_bound + l_bound) / 2,
        upper_bound=u_bound,
        lower_bound=l_bound,
    )


app = FastAPI()


@app.post("/predict")
async def root(predict_body: RequestPredict):
    transactions = predict_body.transactions
    account = predict_body.account

    predicted_amount = predict(transactions, account)

    return predicted_amount