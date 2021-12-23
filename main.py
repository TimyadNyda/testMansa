from datetime import date
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, validator

import tensorflow as tf
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None

# You can use this function to test your api
# Make sure the uvicorn server is running locally on `http://127.0.0.1:8000/`
# or change the hardcoded localhost below

def preprocess(sample, end_date, balance)->list:
    """ take a list of dataframes and return train 
      & test samples with computed features"""
 
    starting_balance = balance-sample.amount.sum()
    sample['running_balance'] = starting_balance + sample.amount.cumsum()
    
    #take complete date range from start to end of a sample.
    date_range = pd.DataFrame(pd.date_range(sample.date.iloc[0], 
                                            end_date).normalize()).set_index(0)
    #print(date_range)
    sample.set_index(sample.date, inplace=True)

    #join date_range with sample so that we have all days
    #missing days in the sample will be filled with NaNs
    sample = date_range.join(sample)
    #compute year timestamps
    timestamp_s = sample.index.map(pd.Timestamp.timestamp)
    day = 24*60*60
    year = (365.2425)*day
    sample['Year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    sample['Year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    #fill NaN running_balance with previous value
    sample['running_balance'].fillna(method='ffill', inplace=True)
    #fill NaN amount with 0 
    sample['amount'].fillna(0, inplace=True)
    #compute & group-by positive and negative amounts
    amounts_pos = sample.copy(deep=True)
    amounts_neg = sample.copy(deep=True)
    amounts_pos.amount[amounts_pos.amount<0] = 0
    amounts_neg.amount[amounts_neg.amount>0] = 0
    amounts_pos = amounts_pos.groupby(amounts_pos.index).sum().amount
    amounts_neg = amounts_neg.groupby(amounts_neg.index).sum().amount
    amounts = sample.groupby(sample.index).sum().amount
    #for each day, group-by and take last value (valid for all columns except amount, amount_pos and amount_neg)
    sample = sample.groupby(sample.index).last()
    #replace amounts columns by grouped values
    sample['amount'] = amounts
    sample['amount_pos'] = amounts_pos
    sample['amount_neg'] = amounts_neg
    return sample[['Year_sin', 'Year_cos', 'amount', 'running_balance', 'amount_pos', 'amount_neg']]

def normalize_amount(x):
    MEAN_AMOUNT, STD_AMOUNT, min_total, max_total = -7.719, 666.416, -5611.383, 5613.770
    x = np.clip(x, min_total, max_total)
    return ((x-MEAN_AMOUNT)/STD_AMOUNT)

def normalize_amount_pos(x):
    MEAN_AMOUNT_POS, STD_AMOUNT_POS, MIN_AMOUNT_POS, min_pos, max_pos = 5.347, 1.989, -4.012, -2.636, 13.021
    x = np.log(x)
    x = np.clip(x, min_pos, max_pos)
    return ((x-MEAN_AMOUNT_POS)/STD_AMOUNT_POS)-MIN_AMOUNT_POS


def normalize_amount_neg(x):
    MEAN_AMOUNT_NEG, STD_AMOUNT_NEG, MIN_AMOUNT_NEG, min_neg, max_neg = 4.386, 1.816, -4.009, -2.897, 11.668
    x = np.log(-x)
    x = np.clip(x, min_neg, max_neg)
    return ((x-MEAN_AMOUNT_NEG)/STD_AMOUNT_NEG)-MIN_AMOUNT_NEG


def normalize_bal(x):
    MEAN_AMOUNT_BAL, STD_AMOUNT_BAL, min_bal, max_bal = 2706.542, 7781.311, -41299.906, 47316.415
    x = np.clip(x, min_bal, max_bal)
    return ((x-MEAN_AMOUNT_BAL)/STD_AMOUNT_BAL)

def preprocess_values(df):
    AMOUNT_COL = df.columns.get_loc("amount")
    BAL_COL = df.columns.get_loc("running_balance")
    AMOUNT_POS_COL = df.columns.get_loc("amount_pos")
    AMOUNT_NEG_COL = df.columns.get_loc("amount_neg")
    df = np.array(df)
    df[:,AMOUNT_COL] = normalize_amount(df[:,AMOUNT_COL])
    df[:,BAL_COL] = normalize_bal(df[:,BAL_COL])
    a = df[:,AMOUNT_NEG_COL]
    idx_ = np.where(a!=0)
    a[idx_]= normalize_amount_neg(a[idx_])
    df[:,AMOUNT_NEG_COL] = a

    a = df[:,AMOUNT_POS_COL]
    idx_ = np.where(a!=0)
    a[idx_]= normalize_amount_pos(a[idx_])
    df[:,AMOUNT_COL] = a
    return df

# define loss fn
def qd_objective(y_true, y_pred):

    # hyperparameters
    lambda_ = 0.01 # lambda in loss fn
    alpha_ = 0.2  # capturing (1-alpha)% of samples
    soften_ = 160.
    n_ = 50 # batch size

    '''Loss_QD-soft, from algorithm 1'''
    y_true = y_true[:,0]
    y_u = y_pred[:,0]
    y_l = y_pred[:,1]
    
    K_HU = tf.maximum(0.,tf.sign(y_u - y_true))
    K_HL = tf.maximum(0.,tf.sign(y_true - y_l))
    K_H = tf.multiply(K_HU, K_HL)
    
    K_SU = tf.sigmoid(soften_ * (y_u - y_true))
    K_SL = tf.sigmoid(soften_ * (y_true - y_l))
    K_S = tf.multiply(K_SU, K_SL)
    
    MPIW_c = tf.reduce_sum(tf.multiply((y_u - y_l),K_H))/tf.reduce_sum(K_H+1e-5)
    PICP_H = tf.reduce_mean(K_H)
    PICP_S = tf.reduce_mean(K_S)
    
    Loss_S = MPIW_c + lambda_ * n_ / (alpha_*(1-alpha_)) * tf.maximum(0.,(1-alpha_) - PICP_S)**2
    
    return Loss_S

#we load the model here, so that it is not loaded at each call
model = tf.keras.models.load_model('./mansa_model.h5', custom_objects={"qd_objective": qd_objective})

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


def predict(
    transactions: List[Transaction], account: Account
) -> float:

    raise NotImplementedError()


app = FastAPI()


@app.post("/predict")
async def root(predict_body: RequestPredict):
    transactions = predict_body.transactions
    account = predict_body.account
        
    pd_data =  pd.DataFrame(map(dict, transactions))
    #compute features
    df_ = preprocess(pd_data, account.update_date, account.balance)
    #process values and keep last 180 timesteps
    df_ = preprocess_values(df_)[-180:]
    
    # Return predicted amount
    preds = model.predict(np.expand_dims(df_,0))[0]
    
    return ResponsePredict(predicted_amount=(preds[0]+preds[1])/2, 
                            upper_bound=preds[0], 
                            lower_bound=preds[1])
