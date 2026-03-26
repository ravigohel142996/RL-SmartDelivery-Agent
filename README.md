# 🚚 Smart Delivery RL Environment

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/RaviGohelAI/smart-delivery-env)

## 🏆 Hackathon: Meta x HuggingFace x Scaler — OpenEnv Track

An RL environment where an AI agent learns optimal last-mile delivery decisions.

## 🎯 3 Tasks
| Task | Name | Difficulty |
|------|------|------------|
| task_1 | Standard Delivery | Easy |
| task_2 | Fragile Package in Bad Weather | Medium |
| task_3 | High-Value Critical Delivery | Hard |

## 🔗 Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| /tasks | GET | List all tasks |
| /reset | POST | Reset environment |
| /step | POST | Take an action |
| /state | GET | Current state |
| /grader | POST | Get score (0.0–1.0) |
| /baseline | POST | Rule-based agent demo |

## ⚡ Actions
`retry` `safe_drop` `locker_drop` `neighbor_handoff` `contact_support` `return`

## 🚀 Quick Start
```bash
pip install openenv-core fastapi uvicorn pydantic
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 🧪 Test It
```python
import requests
# Reset
requests.post("http://localhost:8000/reset", params={"task_id": "task_1"})
# Step
requests.post("http://localhost:8000/step", json={"action_type": "locker_drop", "task_id": "task_1"})
# Score
requests.post("http://localhost:8000/grader", params={"task_id": "task_1"})
```

## 👤 Author
Ravi Gohel — [GitHub](https://github.com/ravigohel142996) | [HF](https://huggingface.co/RaviGohelAI)
