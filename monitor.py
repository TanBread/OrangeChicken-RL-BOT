import time
import psutil

print("Monitoring CPU usage (Ctrl+C to stop)...\n")
while True:
    per_cpu = psutil.cpu_percent(interval=2, percpu=True)
    avg = sum(per_cpu) / len(per_cpu)
    busy = sum(1 for x in per_cpu if x > 50)
    max_core = max(per_cpu)
    min_core = min(per_cpu)
    print(f"Avg: {avg:5.1f}% | Cores >50%: {busy}/{len(per_cpu)} | Min: {min_core:5.1f}% | Max: {max_core:5.1f}%")
    time.sleep(1)
