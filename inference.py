"""
inference.py — Baseline inference script
Runs rule-based agent against all 3 tasks
"""
import requests
import os

BASE_URL = os.getenv("SPACE_URL",
    "https://ravigohelai-smart-delivery-env.hf.space")

def run_inference(task_id="task_1"):
    print(f"\nRunning inference on {task_id}...")
    try:
        r = requests.post(f"{BASE_URL}/baseline",
                          params={"task_id": task_id}, timeout=30)
        result = r.json()
        print(f"  Score:   {result['final_score']}")
        print(f"  Outcome: {result['outcome']}")
        print(f"  Steps:   {result['total_steps']}")
        for i, step in enumerate(result["trajectory"]):
            print(f"  Step {i+1}: {step['action']} -> {step['message']}")
        return result
    except Exception as e:
        print(f"  Error: {e}")
        return {"task_id": task_id, "final_score": 0.0,
                "outcome": "error", "total_steps": 0}

def main():
    print("="*50)
    print("Smart Delivery RL — Inference Script")
    print("="*50)
    scores = {}
    for task_id in ["task_1", "task_2", "task_3"]:
        result = run_inference(task_id)
        scores[task_id] = result.get("final_score", 0.0)
    print("\n" + "="*50)
    print("BASELINE SCORES:")
    for task_id, score in scores.items():
        print(f"  {task_id}: {score}")
    print(f"  Average: {sum(scores.values())/len(scores):.3f}")
    print("="*50)
    return scores

if __name__ == "__main__":
    main()
