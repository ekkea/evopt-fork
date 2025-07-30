from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import pulp
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import os
import jwt

app = Flask(__name__)

@app.before_request
def before_request_func():
    secret_key = os.environ.get('JWT_TOKEN_SECRET')
    if secret_key:
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"message": "Missing authorization header"}), 401
        
        try:
            token_type, token = auth_header.split(' ')
            if token_type.lower() != 'bearer':
                return jsonify({"message": "Invalid token type"}), 401
            
            jwt.decode(token, secret_key, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({"message": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"message": "Invalid token"}), 401
        except Exception as e:
            return jsonify({"message": str(e)}), 401

api = Api(app, version='1.0', title='EV Charging Optimization API', 
          description='Mixed Integer Linear Programming model for EV charging optimization')

# Namespace for the API
ns = api.namespace('optimize', description='EV Charging Optimization Operations')

# Input models for API documentation
battery_config_model = api.model('BatteryConfig', {
    's_min': fields.Float(required=True, description='Minimum state of charge (Wh)'),
    's_max': fields.Float(required=True, description='Maximum state of charge (Wh)'),
    's_initial': fields.Float(required=True, description='Initial state of charge (Wh)'),
    's_goal': fields.List(fields.Float, required=False, description='Goal state of charge at each time step (Wh)'),
    'c_min': fields.Float(required=True, description='Minimum charge power (W)'),
    'c_max': fields.Float(required=True, description='Maximum charge power (W)'),
    'd_max': fields.Float(required=True, description='Maximum discharge power (W)'),
    'p_a': fields.Float(required=True, description='Value per Wh at end of horizon')
})

time_series_model = api.model('TimeSeries', {
    'dt': fields.List(fields.Float, required=True, description='duration in seconds for each time step (s)'),
    'gt': fields.List(fields.Float, required=True, description='Required energy for home consumption at each time step (Wh)'),
    'ft': fields.List(fields.Float, required=True, description='Forecasted solar generation at each time step (Wh)'),
    'p_N': fields.List(fields.Float, required=True, description='Price per kWh taken from grid at each time step'),
    'p_E': fields.List(fields.Float, required=True, description='Price per kWh fed into grid at each time step'),
})

optimization_input_model = api.model('OptimizationInput', {
    'batteries': fields.List(fields.Nested(battery_config_model), required=True, description='Battery configurations'),
    'time_series': fields.Nested(time_series_model, required=True, description='Time series data'),
    'eta_c': fields.Float(required=False, default=0.95, description='Charging efficiency'),
    'eta_d': fields.Float(required=False, default=0.95, description='Discharging efficiency'),
})

# Output models
battery_result_model = api.model('BatteryResult', {
    'charging_power': fields.List(fields.Float, description='Optimal charging energy at each time step (Wh)'),
    'discharging_power': fields.List(fields.Float, description='Optimal discharging energy at each time step (Wh)'),
    'state_of_charge': fields.List(fields.Float, description='State of charge at each time step (Wh)')
})

optimization_result_model = api.model('OptimizationResult', {
    'status': fields.String(description='Optimization status'),
    'objective_value': fields.Float(description='Optimal objective function value'),
    'batteries': fields.List(fields.Nested(battery_result_model), description='Battery optimization results'),
    'grid_import': fields.List(fields.Float, description='Energy imported from grid at each time step (Wh)'),
    'grid_export': fields.List(fields.Float, description='Energy exported to grid at each time step (Wh)'),
    'flow_direction': fields.List(fields.Integer, description='Binary flow direction (1=export, 0=import)')
})

@dataclass
class BatteryConfig:
    s_min: float
    s_max: float
    s_initial: float
    c_min: float
    c_max: float
    d_max: float
    p_a: float
    s_goal: Optional[List[float]] = None  # Goal state of charge (Wh)

@dataclass
class TimeSeriesData:
    dt: List[int] # time step length
    gt: List[float]  # Required total energy
    ft: List[float]  # Forecasted production
    p_N: List[float]  # Import prices
    p_E: List[float]  # Export prices

class EVChargingOptimizer:
    def __init__(self, batteries: List[BatteryConfig], time_series: TimeSeriesData, 
                 eta_c: float = 0.95, eta_d: float = 0.95, M: float = 1e6):
        self.batteries = batteries
        self.time_series = time_series
        self.eta_c = eta_c
        self.eta_d = eta_d
        self.M = M
        self.T = len(time_series.gt)
        self.problem = None
        self.variables = {}
        
    def create_model(self):
        """Create the MILP model"""
        # Create problem
        self.problem = pulp.LpProblem("EV_Charging_Optimization", pulp.LpMaximize)
        
        # Time steps
        time_steps = range(self.T)
        
        # Decision variables
        # Charging power variables [W]
        self.variables['c'] = {}
        for i, bat in enumerate(self.batteries):
            self.variables['c'][i] = [
                pulp.LpVariable(f"c_{i}_{t}", lowBound=0, upBound=bat.c_max * self.time_series.dt[t] /3600.)
                for t in time_steps
            ]
        
        # Discharging power variables [W]
        self.variables['d'] = {}
        for i, bat in enumerate(self.batteries):
            self.variables['d'][i] = [
                pulp.LpVariable(f"d_{i}_{t}", lowBound=0, upBound=bat.d_max * self.time_series.dt[t] /3600.)
                for t in time_steps
            ]
        
        # State of charge variables [Wh]
        self.variables['s'] = {}
        for i, bat in enumerate(self.batteries):
            self.variables['s'][i] = [
                pulp.LpVariable(f"s_{i}_{t}", lowBound=bat.s_min, upBound=bat.s_max)
                for t in time_steps
            ]
        
        # Grid import/export variables [Wh]
        self.variables['n'] = [pulp.LpVariable(f"n_{t}", lowBound=0) for t in time_steps]
        self.variables['e'] = [pulp.LpVariable(f"e_{t}", lowBound=0) for t in time_steps]
        
        # Binary flow direction variables (only when p_N <= p_E)
        self.variables['y'] = []
        for t in time_steps:
            if self.time_series.p_N[t] <= self.time_series.p_E[t]:
                self.variables['y'].append(pulp.LpVariable(f"y_{t}", cat='Binary'))
            else:
                self.variables['y'].append(None)
        
        # Binary variable for charging activation
        self.variables['z_c'] = {}
        for i, bat in enumerate(self.batteries):
            if bat.c_min > 0:
                self.variables['z_c'][i] = [
                    pulp.LpVariable(f"z_c_{i}_{t}", cat='Binary')
                    for t in time_steps
                ]
            else:
                self.variables['z_c'][i] = None
            
        # Objective function (1): Maximize economic benefit
        objective = 0
        
        # Grid import cost (negative because we want to minimize cost) [EUR]
        for t in time_steps:
            objective -= self.variables['n'][t] * self.time_series.p_N[t] / 1000.
        
        # Grid export revenue [EUR]
        for t in time_steps:
            objective += self.variables['e'][t] * self.time_series.p_E[t] / 1000.
        
        # Final state of charge value [EUR]
        for i, bat in enumerate(self.batteries):
            objective += self.variables['s'][i][-1] * bat.p_a / 1000
        
        self.problem += objective
        
        # Constraints
        self._add_constraints()
        
    def _add_constraints(self):
        """Add all constraints to the model"""
        time_steps = range(self.T)
        
        # Constraint (2): Power balance
        for t in time_steps:
            battery_net_discharge = 0
            for i, bat in enumerate(self.batteries):
                battery_net_discharge += (- self.variables['c'][i][t] 
                                          + self.variables['d'][i][t])
            
            self.problem += (battery_net_discharge 
                             + self.time_series.ft[t] 
                             + self.variables['n'][t] 
                             == self.variables['e'][t] 
                             + self.time_series.gt[t])
        
        # Constraint (3): Battery dynamics
        for i, bat in enumerate(self.batteries):
            # Initial state of charge
            if len(time_steps) > 0:
                self.problem += (self.variables['s'][i][0] 
                                == bat.s_initial 
                                + self.eta_c * self.variables['c'][i][0] 
                                - (1/self.eta_d) * self.variables['d'][i][0])
            
            # State of charge evolution
            for t in range(1, self.T):
                self.problem += (self.variables['s'][i][t] 
                                == self.variables['s'][i][t-1] 
                                + self.eta_c * self.variables['c'][i][t] 
                                - (1/self.eta_d) * self.variables['d'][i][t])

            # Constraint (6): Battery SOC goal constraints (for t > 0)
            if bat.s_goal is not None:
                for t in range(1, self.T):
                    if bat.s_goal[t] > 0:
                        self.problem += (self.variables['s'][i][t] >= bat.s_goal[t])

            # Constraint (7): Minimum charge power limits
            if bat.c_min > 0:
                for t in time_steps:
                    # Lower bound: either 0 or at least c_min
                    self.problem += self.variables['c'][i][t] >= bat.c_min * self.time_series.dt[t] /3600. * self.variables['z_c'][i][t]

        # Constraints (4)-(5): Grid flow direction (only when p_N <= p_E)
        for t in time_steps:
            if self.variables['y'][t] is not None:  # Only when p_N <= p_E
                # Export constraint
                self.problem += self.variables['e'][t] <= self.M * self.variables['y'][t]
                # Import constraint
                self.problem += self.variables['n'][t] <= self.M * (1 - self.variables['y'][t])
            
    def solve(self) -> Dict:
        """Solve the optimization problem and return results"""
        if self.problem is None:
            self.create_model()
        
        # Solve the problem
        solver = pulp.PULP_CBC_CMD(msg=0)  # Silent solver
        self.problem.solve(solver)
        
        # Extract results
        status = pulp.LpStatus[self.problem.status]
        
        if status == 'Optimal':
            result = {
                'status': status,
                'objective_value': pulp.value(self.problem.objective),
                'batteries': [],
                'grid_import': [pulp.value(var) for var in self.variables['n']],
                'grid_export': [pulp.value(var) for var in self.variables['e']],
                'flow_direction': []
            }
            
            # Extract battery results
            for i, bat in enumerate(self.batteries):
                battery_result = {
                    'charging_power': [pulp.value(var) for var in self.variables['c'][i]],
                    'discharging_power': [pulp.value(var) for var in self.variables['d'][i]],
                    'state_of_charge': [pulp.value(var) for var in self.variables['s'][i]]
                }
                result['batteries'].append(battery_result)
            
            # Extract flow direction
            for y_var in self.variables['y']:
                if y_var is not None:
                    result['flow_direction'].append(int(pulp.value(y_var)))
                else:
                    result['flow_direction'].append(0)  # Default to import when constraint not active
            
            return result
        else:
            return {
                'status': status,
                'objective_value': None,
                'batteries': [],
                'grid_import': [],
                'grid_export': [],
                'flow_direction': []
            }

@ns.route('/charge-schedule')
class OptimizeCharging(Resource):
    @api.expect(optimization_input_model)
    @api.marshal_with(optimization_result_model)
    def post(self):
        """
        Optimize EV charging schedule using MILP
        
        This endpoint solves a Mixed Integer Linear Programming problem to optimize
        EV charging schedules considering battery constraints, grid prices, and energy demands.
        """
        try:
            data = request.get_json()
            
            # Validate input data
            if not data:
                api.abort(400, "No input data provided")
            
            # Parse battery configurations
            batteries = []
            for bat_data in data['batteries']:
                batteries.append(BatteryConfig(
                    s_min=bat_data['s_min'],
                    s_max=bat_data['s_max'],
                    s_initial=bat_data['s_initial'],
                    s_goal=bat_data.get('s_goal'),
                    c_min=bat_data['c_min'],
                    c_max=bat_data['c_max'],
                    d_max=bat_data['d_max'],
                    p_a=bat_data['p_a'],
                ))
            
            # Parse time series data
            time_series = TimeSeriesData(
                dt=data['time_series']['dt'],
                gt=data['time_series']['gt'],
                ft=data['time_series']['ft'],
                p_N=data['time_series']['p_N'],
                p_E=data['time_series']['p_E'],
            )

            # Validate time series lengths
            lengths = [len(time_series.gt), len(time_series.ft), 
                      len(time_series.p_N), len(time_series.p_E)]

            # Validate s_goal if provided
            for i, bat in enumerate(batteries):
                if bat.s_goal is not None:
                    lengths.append(len(bat.s_goal))

            if len(set(lengths)) > 1:
                api.abort(400, "All time series must have the same length")
            
        except KeyError as e:
            api.abort(400, f"Missing required field: {str(e)}")
        except (TypeError, ValueError) as e:
            api.abort(400, f"Invalid data format: {str(e)}")
        
        try:
            # Create and solve optimizer
            optimizer = EVChargingOptimizer(
                batteries=batteries,
                time_series=time_series,
                eta_c=data.get('eta_c', 0.95),
                eta_d=data.get('eta_d', 0.95),
                M=1e6
            )
            
            result = optimizer.solve()
            return result
            
        except Exception as e:
            api.abort(500, f"Optimization failed: {str(e)}")

@ns.route('/health')
class Health(Resource):
    def get(self):
        """Health check endpoint"""
        return {'status': 'healthy', 'message': 'EV Charging MILP API is running'}

# Example data endpoint for testing
@ns.route('/example')
class ExampleData(Resource):
    def get(self):
        """Get example input data for testing the optimization"""
        example_data = {
            "batteries": [
                {
                    "s_min": 2000,
                    "s_max": 36000,
                    "s_initial": 15000,
                    "s_goal": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "c_min": 1380, 
                    "c_max": 3680, 
                    "d_max": 0,
                    "p_a": 0.12
                },
                {
                    "s_min": 2500,
                    "s_max": 16200,
                    "s_initial": 5000,
                    "c_min": 0,
                    "c_max": 6000, 
                    "d_max": 6000, 
                    "p_a": 0.12
                }
            ],
            "time_series": {
                "dt": [900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900],
                "gt": [53.25, 19.25, 28, 27.25, 25.25, 19.75, 22.5, 41.5, 18.25, 14.75, 38.75, 44.25, 51.5, 56, 62.75, 21.5, 18.5, 17.5, 14, 70.75, 63.5, 221.5, 44.75, 46.25, 43.5, 41.25, 50.5, 49.5, 56, 26, 20.5, 33.75, 34.5, 24.5, 20, 30.25, 22.5, 16, 41.5, 69.75, 81, 75.75, 87.25, 58.75, 42, 47.75, 38, 38.25, 45.75, 55, 60.25, 50.25, 75, 71.25, 58, 60, 71, 49, 41.25, 36.75, 52, 49.5, 61, 66.5, 55.25, 54.5, 61.25, 58.5, 58.75, 71.75, 82.5, 59.25, 42.75, 40, 35.75, 35.75, 44.75, 49.75, 37, 43.25, 41.5, 59.25, 53.25, 114, 58.75, 40.5, 18, 34.5, 34.25, 35.75, 44.25, 36.5, 23, 23.25, 32, 51, 45.25, 65, 67.25, 47, 27.5, 22.75, 22.75, 23, 37, 32.5, 31, 33, 38.5, 49, 37.5, 49.75, 46.25, 28.25, 14.25, 14.75, 21.25, 33, 44, 53.75, 18, 17, 17.25, 48.75, 39.25, 64.5, 69, 47.25, 37, 23.75, 23, 22.75, 27.5, 47.25, 49, 48, 69.75, 86.75, 9.5, 13.25, 51.75, 1.75, 6.5, 1.75, 17, 9.25, 4.5, 2, 1.5, 0.5, 13.5, 0.25, 24, 23.25, 69.75, 31.75, 23.5, 3, 5.5, 5, 10.75, 40, 42.25, 49.75, 63.5, 66.75, 59.5, 55.25, 34.75, 27.5, 16, 10.25, 17, 11.5, 35.25, 49.25, 45.25, 57.25, 45.5, 40.25, 52.25, 38.5, 36.75, 32.25, 51, 47.5, 44.75, 33.5, 43.5],
                "ft": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 2, 15.25, 24, 31.25, 36.5, 41.25, 46, 50.75, 56, 60.25, 66, 85.75, 271.25, 443.25, 531.5, 598.25, 666.25, 738.75, 811.5, 840.75, 875, 896, 935.5, 937.5, 986.25, 772.75, 1024.25, 1025.25, 1015.5, 1020.25, 1033.5, 1034.25, 1024.5, 1010.5, 996.5, 988.25, 983, 963, 938, 930.75, 886.25, 363.5, 530.5, 721, 742, 590.5, 689, 628, 571.25, 524.75, 461, 399, 344.5, 299.25, 249.5, 202.25, 146, 77.25, 52.5, 42, 32.25, 23, 14, 3.5, 180.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 2.5, 18.25, 28, 37.5, 45.25, 51.75, 58.25, 64.5, 71.75, 78, 86.25, 98.5, 250.5, 415.75, 507, 567, 627.25, 681.5, 744.5, 783, 820.75, 819, 610.5, 569.75, 808.75, 659.75, 295.5, 230.75, 259.25, 345.5, 790.75, 821.5, 874.25, 887.5, 576, 900, 781.25, 644, 387.5, 274.25, 365.25, 654, 789, 428.25, 308.75, 94.5, 37.5, 10.75, 0, 0, 0, 0, 3.75, 21, 41.25, 76.25, 152, 148.75, 75.25, 41.5, 29, 19.5, 5.75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "p_N": [0.2201, 0.2176, 0.1985, 0.1972, 0.20508, 0.19509, 0.19687, 0.1851, 0.19509, 0.19577, 0.1966, 0.1801, 0.18509, 0.1977, 0.18811, 0.1926, 0.17509, 0.18509, 0.19753, 0.2057, 0.1867, 0.1935, 0.208, 0.21564, 0.2048, 0.2166, 0.2217, 0.21162, 0.2553, 0.235, 0.2109, 0.19, 0.28394, 0.2181, 0.19495, 0.1606, 0.26737, 0.2, 0.17497, 0.142, 0.25956, 0.17994, 0.14496, 0.1179, 0.20344, 0.1521, 0.15993, 0.14493, 0.1583, 0.14531, 0.13994, 0.13493, 0.1417, 0.1424, 0.13249, 0.1241, 0.1326, 0.1418, 0.1497, 0.16812, 0.13994, 0.1594, 0.1744, 0.21084, 0.15479, 0.18491, 0.1911, 0.2123, 0.1405, 0.1872, 0.2211, 0.25493, 0.13508, 0.20506, 0.2482, 0.2987, 0.2496, 0.27494, 0.3013, 0.36355, 0.3513, 0.3431, 0.67743, 0.3627, 0.3868, 0.3334, 0.35, 0.40659, 0.2912, 0.2707, 0.26491, 0.24934, 0.2351, 0.2275, 0.2295, 0.2176, 0.23, 0.22, 0.21473, 0.18508, 0.21502, 0.21002, 0.19176, 0.19001, 0.20627, 0.20009, 0.19009, 0.18509, 0.19502, 0.19501, 0.18957, 0.20002, 0.18007, 0.19008, 0.19509, 0.22277, 0.18509, 0.22706, 0.20009, 0.20507, 0.23492, 0.21077, 0.22005, 0.22502, 0.25993, 0.25494, 0.22018, 0.17008, 0.25899, 0.25, 0.19006, 0.15501, 0.25, 0.22995, 0.18333, 0.125, 0.22, 0.19107, 0.165, 0.14994, 0.2, 0.16997, 0.15676, 0.13998, 0.1599, 0.15491, 0.11073, 0.10637, 0.125, 0.1499, 0.15491, 0.1301, 0.125, 0.125, 0.15992, 0.18398, 0.1399, 0.14994, 0.1849, 0.19646, 0.14495, 0.16496, 0.18734, 0.22494, 0.125, 0.2157, 0.18508, 0.24997, 0.15005, 0.20008, 0.25993, 0.25161, 0.22008, 0.26497, 0.28129, 0.2949, 0.29499, 0.30495, 0.29498, 0.46605, 0.28515, 0.29993, 0.28992, 0.27995, 0.26906, 0.25991, 0.24492, 0.23491, 0.24991],
                "p_E": [0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.1]
            },
            "eta_c": 0.95,
            "eta_d": 0.95
        }
        return example_data

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7050)