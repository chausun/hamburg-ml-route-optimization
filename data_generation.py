import numpy as np
import pandas as pd
import random
from math import cos, radians, sqrt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# -------------------------
# 1. DEPOT LOCATION
# -------------------------
DEPOT_LAT = 53.5511
DEPOT_LON = 9.9937

# -------------------------
# 2. GENERATE CUSTOMERS
# -------------------------
def generate_customers(n_customers=20, radius_km=5):
    customers = []

    for i in range(1, n_customers + 1):
        r = random.uniform(0, radius_km)
        angle = random.uniform(0, 2*np.pi)

        lat_offset = (r * np.cos(angle)) / 111
        lon_offset = (r * np.sin(angle)) / (111 * cos(radians(DEPOT_LAT)))

        lat = DEPOT_LAT + lat_offset
        lon = DEPOT_LON + lon_offset

        demand = random.randint(5, 30)
        service_time = random.randint(5, 20)

        start_hour = random.randint(8, 14)
        end_hour = start_hour + random.randint(2, 4)

        customers.append([
            i, lat, lon, demand,
            f"{start_hour}:00",
            f"{end_hour}:00",
            service_time
        ])

    columns = [
        "Node_ID", "Latitude", "Longitude",
        "Demand", "TW_Start", "TW_End", "Service_Time"
    ]

    return pd.DataFrame(customers, columns=columns)

# -------------------------
# 3. DISTANCE FUNCTION
# -------------------------
def calculate_distance(lat1, lon1, lat2, lon2):
    return sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111

# -------------------------
# 4. CREATE DISTANCE MATRIX
# -------------------------
def create_distance_matrix(customers_df):
    nodes = [(DEPOT_LAT, DEPOT_LON)] + list(
        zip(customers_df["Latitude"], customers_df["Longitude"])
    )

    size = len(nodes)
    matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if i != j:
                matrix[i][j] = calculate_distance(
                    nodes[i][0], nodes[i][1],
                    nodes[j][0], nodes[j][1]
                )

    return matrix

# -------------------------
# RUN SCRIPT
# -------------------------
# -------------------------
# RUN SCRIPT
# -------------------------
# -------------------------
# 5. GENERATE ML TRAINING DATA
# -------------------------
def generate_travel_time_dataset(n_samples=3000):
    data = []

    for _ in range(n_samples):
        distance = random.uniform(0.5, 8)
        hour = random.randint(8, 18)
        traffic_index = random.randint(1, 5)
        weather_index = random.randint(0, 2)

        base_time = (distance / 40) * 60

        traffic_factor = 1 + (traffic_index * 0.1)
        weather_factor = 1 + (weather_index * 0.05)

        travel_time = base_time * traffic_factor * weather_factor
        noise = random.uniform(-2, 2)

        final_time = travel_time + noise

        data.append([
            distance,
            hour,
            traffic_index,
            weather_index,
            final_time
        ])

    columns = [
        "Distance_km",
        "Hour_of_Day",
        "Traffic_Index",
        "Weather_Index",
        "Travel_Time"
    ]

    return pd.DataFrame(data, columns=columns)


# -------------------------
# 6. TRAIN ML MODEL
# -------------------------
def train_ml_model(ml_data):

    X = ml_data[["Distance_km", "Hour_of_Day", "Traffic_Index", "Weather_Index"]]
    y = ml_data["Travel_Time"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    print("\nModel Performance:")
    print("Mean Absolute Error:", round(mae, 2), "minutes")

    return model


# -------------------------
# 7. CREATE PREDICTED TRAVEL TIME MATRIX
# -------------------------
def create_predicted_time_matrix(distance_matrix, model):

    size = len(distance_matrix)
    predicted_matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if i != j:
                distance = distance_matrix[i][j]

                hour = random.randint(8, 18)
                traffic_index = random.randint(1, 5)
                weather_index = random.randint(0, 2)

                input_data = pd.DataFrame([[
                    distance,
                    hour,
                    traffic_index,
                    weather_index
                ]], columns=[
                    "Distance_km",
                    "Hour_of_Day",
                    "Traffic_Index",
                    "Weather_Index"
                ])

                predicted_time = model.predict(input_data)[0]
                predicted_matrix[i][j] = predicted_time

    return predicted_matrix


# -------------------------
# 8. SOLVE ROUTING PROBLEM
# -------------------------
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def solve_routing(time_matrix):

    size = len(time_matrix)
    time_matrix_int = (time_matrix * 10).astype(int)

    manager = pywrapcp.RoutingIndexManager(size, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return time_matrix_int[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        route = []

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            index = solution.Value(routing.NextVar(index))

        route.append(0)

        return route

    else:
        return None


# -------------------------
# RUN SCRIPT
# -------------------------
def calculate_route_cost(route, time_matrix):
    total = 0
    for i in range(len(route) - 1):
        total += time_matrix[route[i]][route[i+1]]
    return total
def plot_comparison(static_time, ml_time):

    methods = ["Static Routing", "ML-Based Routing"]
    times = [static_time, ml_time]

    plt.figure()
    bars = plt.bar(methods, times)

    plt.xlabel("Routing Method")
    plt.ylabel("Total Predicted Travel Time (minutes)")
    plt.title("Static vs ML-Based Routing Performance")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                 round(height, 2),
                 ha='center', va='bottom')

    plt.show()


if __name__ == "__main__":

    customers_df = generate_customers()
    distance_matrix = create_distance_matrix(customers_df)

    ml_data = generate_travel_time_dataset()
    model = train_ml_model(ml_data)

    predicted_time_matrix = create_predicted_time_matrix(distance_matrix, model)

    # Get routes
    static_route = solve_routing(distance_matrix)
    ml_route = solve_routing(predicted_time_matrix)

    print("\nStatic Route:", static_route)
    print("ML Route:", ml_route)

    # Evaluate both routes using predicted travel time
    static_time = calculate_route_cost(static_route, predicted_time_matrix)
    ml_time = calculate_route_cost(ml_route, predicted_time_matrix)

    improvement = ((static_time - ml_time) / static_time) * 100

    print("\nFair Comparison (Using Predicted Travel Time)")
    print("Static Route Time:", round(static_time, 2))
    print("ML Route Time:", round(ml_time, 2))
    print("Improvement (%):", round(improvement, 2))
plot_comparison(static_time, ml_time)




