from rag.pipeline import ask
from ml.anomaly_detector import predict

def run_agent(user_input: str):
    if "sensor" in user_input.lower() or "reading" in user_input.lower():
        readings = [72, 80, 68]
        result = predict(readings)
        return f"Sensor status is → {result}"
    else:
        answer, sources = ask(user_input)
        return answer