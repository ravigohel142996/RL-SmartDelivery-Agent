
"""
Baseline agent using HuggingFace Inference API (FREE - no OpenAI needed!)
Run: python baseline_agent.py
"""
import requests

BASE_URL = "http://localhost:8000"

def run_baseline(task_id="task_1"):
    print(f"
{'='*50}")
    print(f"Running baseline on {task_id}")
    print('='*50)
    
    r = requests.post(f"{BASE_URL}/baseline", params={"task_id": task_id})
    result = r.json()
    
    print(f"Trajectory:")
    for i, step in enumerate(result["trajectory"]):
        print(f"  Step {i+1}: {step['action']} → {step['message']}")
    
    print(f"
Final Score : {result['final_score']}")
    print(f"Outcome     : {result['outcome']}")
    print(f"Total Steps : {result['total_steps']}")
    return result

if __name__ == "__main__":
    for task in ["task_1", "task_2", "task_3"]:
        run_baseline(task)
    print("
✅ Baseline complete!")
