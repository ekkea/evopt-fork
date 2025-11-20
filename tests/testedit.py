import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from tabulate import tabulate

from evopt.app import app

parser = argparse.ArgumentParser(prog="testedit")
parser.add_argument("action", choices=["create", "update", "run"], help="")
parser.add_argument("file", type=str, default="", help="")
parser.add_argument("-o", "--outfile", type=str, default="test_case.json", help="")
args = parser.parse_args()
print(args)
action = args.action
file_in = Path(args.file)
file_out = Path(args.outfile)

#checks
if not file_in.is_file():
    print(f"File not found: {file_in.name}")
    sys.exit(1)

if not file_out.parents[0].is_dir():
    print(f"Directory does not exist: {file_out.parents[0].name}")
    sys.exit(1)

client = app.test_client()

if action=="create":
    request = json.loads(file_in.read_text())
    response = client.post("/optimize/charge-schedule", json=request)
    test_case = {}
    test_case['request']=request
    test_case['response']=response.get_json()
    print(test_case)
    json.dump(test_case, fp=open(file_out,"w"))

if action=="run":
    test_case = json.loads(file_in.read_text())
    request = test_case["request"]
    client = app.test_client()
    response = client.post("/optimize/charge-schedule", json=request)
    if response.status_code != 200:
        print(f"Request to optimizer returned with status {response.status_code}")
    else:
        print(f"Objective Value: {response.json["objective_value"]}")
        ts_input=request["time_series"]
        dt=ts_input["dt"]
        ts_time=np.cumsum(dt)
        #ts_time=np.subtract(np.cumsum(dt), dt[0])
        ts_solar=np.divide(ts_input["ft"],ts_input["dt"])
        ts_demand=np.negative(np.divide(ts_input["gt"],ts_input["dt"]))
        # Create DataFrame
        df = pd.DataFrame({
            "time": ts_time,
            "P_solar": ts_solar,
            "P_demand": ts_demand
        })
        df['time'] = pd.to_datetime(df['time'], unit='s')

        print(tabulate(df, headers='keys', tablefmt='psql',  floatfmt=".3f" ))
        #print(df.info())

        fig, axs = plt.subplots(2, figsize=(16,8))
        axs[0].plot(df["time"], df["P_solar"], label="P_solar")
        axs[0].plot(df["time"], df["P_demand"], label="P_demand")
        axs[0].xaxis.set_minor_locator(mdates.HourLocator())
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axs[0].grid()
        axs[0].legend()
        axs[1].plot(df["time"], df["P_demand"], label="P_demand")
        #plt.plot(df["time"], df["P_solar"], label="ft")
        #plt.plot(df["time"], df["P_demand"], label="gt")
        #plt.title("Solar Gain and Demand")
        #plt.gcf().autofmt_xdate()
        #plt.legend()
        #plt.grid(True)
        #plt.tight_layout()
        plt.show()
# response = test_data.get("expected_response")
