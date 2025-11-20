import argparse
import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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
    if "request" not in test_case or "expected_response" not in test_case:
        print(f"unexpected format in {file_in.name}")
        sys.exit(1)
    request = test_case["request"]
    expected_response = test_case["expected_response"]
    client = app.test_client()
    response = client.post("/optimize/charge-schedule", json=request)
    if response.status_code != 200:
        print(f"Request to optimizer returned with status {response.status_code}")
        sys.exit(1)
    else:
        print(f"Objective Value: {response.json["objective_value"]}")

        ts_input=request["time_series"]
        dt=ts_input["dt"]
        dt0=dt.copy()
        dt0.insert(0, 0.)
        ts_time_ex=np.cumsum(dt0)
        dt0=dt0[:-1]
        ts_time=np.cumsum(dt0)
        ts_period=np.array(dt)
        ts_prc_import=np.array(ts_input["p_N"])*1000
        ts_prc_export=np.array(ts_input["p_E"])*1000
        ts_solar=np.divide(ts_input["ft"],ts_input["dt"])
        ts_demand=np.negative(np.divide(ts_input["gt"],ts_input["dt"]))
        ts_grid=np.divide(np.subtract(response.json["grid_import"], response.json["grid_export"]), ts_input["dt"])
        ts_grid_exp=np.divide(np.subtract(expected_response["grid_import"], expected_response["grid_export"]), ts_input["dt"])
        ts_grid_dev=np.divide(np.subtract(ts_grid,ts_grid_exp), ts_grid_exp)
        # Create DataFrame
        df = pd.DataFrame({
            "time": ts_time,
            "period": ts_period,
            "prc_import": ts_prc_import,
            "prc_export": ts_prc_export,
            "P_solar": ts_solar,
            "P_demand": ts_demand,
            "P_grid": ts_grid,
            "P_grid_exp": ts_grid_exp,
            "P_grid_dev": ts_grid_dev
        })

        n_batteries=len(request["batteries"])
        for i, bat in enumerate(response.json["batteries"]):
            df[f"P_bat{i}"]=np.divide(np.subtract(bat["discharging_power"], bat["charging_power"]), ts_input["dt"])
            df[f"SOC_bat{i}"]=np.divide(bat["state_of_charge"], request["batteries"][i]["s_max"])*100
            df[f"P_bat{i}_exp"]=np.divide(np.subtract(expected_response["batteries"][i]["discharging_power"],
                                                      expected_response["batteries"][i]["charging_power"]),
                                                      ts_input["dt"])
            df[f"P_bat{i}_dev"]=np.divide(np.subtract(df[f"P_bat{i}"],df[f"P_bat{i}_exp"]), df[f"P_bat{i}_exp"])


        df['time'] = pd.to_datetime(df['time'], unit='s')
        ts_time_ex=pd.to_datetime(ts_time_ex, unit="s")

        print(tabulate(df, headers='keys', tablefmt='psql',  floatfmt=".3f" ))

        fig, axs = plt.subplots(3, figsize=(16, 12), height_ratios=[1,2,1])

        axs[0].set_ylabel("SOC")
        axs[0].set
        for i, bat in enumerate(response.json["batteries"]):
            axs[0].stairs(df[f"SOC_bat{i}"], ts_time_ex, label=f"SOC_bat{i} [%]")
        axs0_r=axs[0].twinx()
        axs0_r.set_ylabel("price")
        axs0_r.plot(df["time"], df["prc_import"], 'r+', label="prc_import [€/kWh]")
        axs0_r.plot(df["time"], df["prc_export"], 'b+', label="prc_export [€/kWh]")
        axs[0].xaxis.set_minor_locator(mdates.HourLocator())
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axs[0].grid()
        axs[0].legend()

        axs[1].set_ylabel("Power")
        for i, bat in enumerate(response.json["batteries"]):
            axs[1].stairs(df[f"P_bat{i}"], ts_time_ex, label=f"P_bat{i} [kW]")
        axs[1].stairs(df["P_grid"], ts_time_ex, label="P_grid [kW]")
        axs[1].stairs(df["P_solar"], ts_time_ex, label="P_solar [kW]")
        axs[1].stairs(df["P_demand"], ts_time_ex, label="P_demand [kW]")
        axs[1].xaxis.set_minor_locator(mdates.HourLocator())
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axs[1].grid()
        axs[1].legend()

        axs[2].set_ylabel("Deviation to Expected")
        for i, bat in enumerate(response.json["batteries"]):
            axs[2].stairs(df[f"P_bat{i}_dev"], ts_time_ex, label=f"P_bat{i}_dev [1]")
        axs[2].stairs(df["P_grid_dev"], ts_time_ex, label="P_grid_dev [1]")
        axs[2].xaxis.set_minor_locator(mdates.HourLocator())
        axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axs[2].grid()
        axs[2].legend()


        plt.show()
# response = test_data.get("expected_response")
